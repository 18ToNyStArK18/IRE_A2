"""Q2 stage 2: the LambdaMART re-ranker.

Model: LightGBM `objective="lambdarank"` -- RankNet pairwise gradients, each
pair weighted by |ΔnDCG| from swapping it, boosted over regression trees. The
family, stated precisely: pairwise gradient weighted by a listwise metric. See
DesignChoices.md §1 for why this over pointwise or a pure listwise objective.

Two things here are driven by the recall ceiling (DesignChoices.md §0) and are
easy to get wrong:

1. **Training filters to impressions that contain a positive; evaluation does
   not.** A group whose labels are all 0 generates no pairs, so LambdaMART
   would learn nothing from it anyway -- but it still belongs in the reported
   population, because dropping it from evaluation would silently redefine the
   metric. This is the same trap `src/nrms/adapter.py` flags for
   `drop_no_click`: filter one side of a before/after comparison and the delta
   partly measures which impressions were included.

2. **Filtering happens on the candidate list, before features are built.** MIND
   train is 27.7M candidate rows, of which ~4.5% of groups can affect any
   reported number. Labelling is a cheap set-membership test, so we filter
   first and build features for ~1.2M rows instead of 27.7M.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import article_stats, candidates as candidates_module, config, feature_pipeline
from src.metrics import mrr_score, ndcg_score

NON_FEATURE_COLUMNS = ("impression_id", "article_id", "label")

DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10],
    # LightGBM defaults this to 30, which would be actively harmful here: it
    # truncates the pairs considered to the model's current top-30, and our
    # positives start at median rank 80 of 200 (DesignChoices.md §0.1). Most
    # positives would sit outside the truncation window and generate no
    # gradient at all. Groups are only 200 wide, so scoring every pair is
    # affordable.
    "lambdarank_truncation_level": config.CANDIDATE_K,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42,
}


# --------------------------------------------------------------- labelling

def true_clicks(processed_dir, split: str) -> dict[str, set[str]]:
    behaviors = pd.read_parquet(
        processed_dir / f"behaviors_{split}.parquet",
        columns=["impression_id", "candidates", "labels"],
    )
    return {
        row.impression_id: {a for a, l in zip(row.candidates, row.labels) if l == 1}
        for row in behaviors.itertuples(index=False)
    }


def label_candidates(candidates_df: pd.DataFrame, clicks: dict[str, set[str]]) -> pd.DataFrame:
    """Attach the 0/1 label without building any features -- cheap enough to
    run over the full candidate list so we can decide what to build features
    for."""
    out = candidates_df.copy()
    out["label"] = [
        int(article_id in clicks.get(impression_id, ()))
        for impression_id, article_id in zip(out["impression_id"], out["article_id"])
    ]
    return out


def impressions_with_positive(labelled: pd.DataFrame) -> np.ndarray:
    per_impression = labelled.groupby("impression_id")["label"].max()
    return per_impression[per_impression > 0].index.to_numpy()


# ------------------------------------------------------- matrix construction

def feature_columns(matrix: pd.DataFrame) -> list[str]:
    return [c for c in matrix.columns if c not in NON_FEATURE_COLUMNS]


def build_matrix(
    processed_dir,
    dataset: str,
    split: str,
    method: str,
    article_index=None,
    k: int | None = None,
    positives_only: bool = False,
) -> pd.DataFrame:
    """Candidates -> labelled feature matrix, sorted by impression so LightGBM's
    contiguous-group requirement holds.

    `positives_only=True` restricts to impressions containing a positive; use it
    for TRAINING only (see module docstring).
    """
    cands = candidates_module.load_candidates(processed_dir, method, split, k)
    clicks = true_clicks(processed_dir, split)
    cands = label_candidates(cands, clicks)

    if positives_only:
        keep = set(impressions_with_positive(cands).tolist())
        cands = cands[cands["impression_id"].isin(keep)]

    article_index = article_index or article_stats.TrainEventIndex(processed_dir, dataset)
    matrix = feature_pipeline.build_feature_matrix(
        processed_dir, dataset, split, cands, article_index=article_index
    )
    # feature_pipeline recomputes the label from the same ground truth; keep its
    # column and simply order the rows.
    return matrix.sort_values(["impression_id", "retrieval_rank"]).reset_index(drop=True)


def to_lgb_dataset(matrix: pd.DataFrame, features: list[str], reference: lgb.Dataset | None = None):
    """LightGBM needs each impression's rows contiguous plus a group-size array.
    NaNs are passed through untouched -- they are meaningful here (MIND has no
    dwell time at all, freshness is unknown for articles unseen before the
    impression) and LightGBM splits on missingness natively."""
    X = matrix[features].astype(np.float64)
    y = matrix["label"].to_numpy()
    group_sizes = matrix.groupby("impression_id", sort=False).size().to_numpy()
    return lgb.Dataset(X, label=y, group=group_sizes, reference=reference, free_raw_data=False)


# ------------------------------------------------------------- evaluation

def group_arrays(matrix: pd.DataFrame, score_column: str):
    """(labels, scores) per impression, in impression order."""
    labels, scores = [], []
    for _, block in matrix.groupby("impression_id", sort=False):
        labels.append(block["label"].to_numpy())
        scores.append(block[score_column].to_numpy())
    return labels, scores


def evaluate_over_population(
    labels_per_impression,
    scores_per_impression,
    n_impressions_total: int,
    ndcg_ks=(5, 10),
) -> dict[str, float]:
    """Metrics averaged over the FULL impression population, computed from only
    the impressions that were actually scored.

    Exact, not an approximation, provided the unscored impressions are exactly
    those with no positive. For such an impression `mrr_score` returns 0.0,
    `ndcg_score` returns 0.0 (its ideal DCG is 0), and AUC skips it entirely as
    undefined. They therefore contribute a known constant no matter how they are
    ordered -- so scoring them cannot change any reported number, and at MIND
    train scale skipping them avoids building 27.7M feature rows to compute a
    guaranteed zero.

    Verified against a brute-force full-population evaluation in
    tests/test_reranker.py.
    """
    auc_values, mrr_sum = [], 0.0
    ndcg_sums = {k: 0.0 for k in ndcg_ks}

    for y_true, y_score in zip(labels_per_impression, scores_per_impression):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_score = np.asarray(y_score, dtype=np.float64)
        if len(y_true) == 0:
            continue
        if 0 < y_true.sum() < len(y_true):
            auc_values.append(float(roc_auc_score(y_true, y_score)))
        mrr_sum += mrr_score(y_true, y_score)
        for k in ndcg_ks:
            ndcg_sums[k] += ndcg_score(y_true, y_score, k)

    results = {
        "auc": float(np.mean(auc_values)) if auc_values else float("nan"),
        "mrr": mrr_sum / n_impressions_total if n_impressions_total else float("nan"),
        "n_impressions": n_impressions_total,
        "n_impressions_scored": len(labels_per_impression),
        "n_impressions_scored_for_auc": len(auc_values),
    }
    for k in ndcg_ks:
        results[f"ndcg@{k}"] = ndcg_sums[k] / n_impressions_total if n_impressions_total else float("nan")
    return results


def evaluate_before_after(
    matrix: pd.DataFrame,
    model_scores: np.ndarray,
    n_impressions_total: int,
    ndcg_ks=(5, 10),
) -> dict:
    """Q2's "before and after re-ranking" on identical impressions.

    Before = stage 1's own ordering (its raw retrieval score). After = the
    re-ranker's. Both are averaged over the same full population, so the delta
    measures ranking quality rather than which impressions were included.
    """
    scored = matrix.copy()
    scored["model_score"] = model_scores

    labels, before = group_arrays(scored, "retrieval_score")
    _, after = group_arrays(scored, "model_score")

    return {
        "before": evaluate_over_population(labels, before, n_impressions_total, ndcg_ks),
        "after": evaluate_over_population(labels, after, n_impressions_total, ndcg_ks),
    }


# --------------------------------------------------------------- training

def train(
    train_matrix: pd.DataFrame,
    val_matrix: pd.DataFrame,
    features: list[str],
    params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    params = {**DEFAULT_PARAMS, **(params or {})}
    train_set = to_lgb_dataset(train_matrix, features)
    val_set = to_lgb_dataset(val_matrix, features, reference=train_set)

    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )


def feature_importance(model: lgb.Booster, features: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": features,
                "gain": model.feature_importance("gain"),
                "split": model.feature_importance("split"),
            }
        )
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )


def run(dataset: str, method: str, eval_split: str = "test", k: int | None = None) -> dict:
    processed_dir = candidates_module.PROCESSED_DIRS[dataset]
    article_index = article_stats.TrainEventIndex(processed_dir, dataset)

    # Training and early-stopping validation see positive-bearing groups only;
    # the reported evaluation does not (see module docstring).
    train_matrix = build_matrix(processed_dir, dataset, "train", method, article_index, k, positives_only=True)
    val_matrix = build_matrix(processed_dir, dataset, "val", method, article_index, k, positives_only=True)
    features = feature_columns(train_matrix)

    model = train(train_matrix, val_matrix, features)

    eval_matrix = build_matrix(processed_dir, dataset, eval_split, method, article_index, k, positives_only=True)
    n_total = len(true_clicks(processed_dir, eval_split))
    scores = model.predict(eval_matrix[features].astype(np.float64), num_iteration=model.best_iteration)
    report = evaluate_before_after(eval_matrix, scores, n_total)

    report["dataset"] = dataset
    report["method"] = method
    report["eval_split"] = eval_split
    report["best_iteration"] = model.best_iteration
    report["n_train_impressions"] = int(train_matrix["impression_id"].nunique())
    report["importance"] = feature_importance(model, features).to_dict("records")

    artifact_dir = processed_dir / "reranker"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(artifact_dir / f"lambdamart_{method}.txt"), num_iteration=model.best_iteration)
    (artifact_dir / f"report_{method}_{eval_split}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--method", choices=["bm25", "semantic"], default="bm25")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--k", type=int, default=None, help="slice candidates to top-k (default: all persisted)")
    args = parser.parse_args()

    report = run(args.dataset, args.method, args.eval_split, args.k)
    before, after = report["before"], report["after"]
    print(f"\n[{args.dataset}:{args.method}:{args.eval_split}] over {before['n_impressions']} impressions")
    for name in ("auc", "mrr", "ndcg@5", "ndcg@10"):
        print(f"  {name:8s} before={before[name]:.4f}  after={after[name]:.4f}")


if __name__ == "__main__":
    main()
