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

3. **With fresh-pool candidates that filter no longer shrinks anything.**
   `popular` puts the click in the top-200 for 97.0% (EB-NeRD) / 93.9% (MIND) of
   test impressions, so almost every group survives it. Training therefore
   subsamples negatives (every positive plus 49 random negatives -- see
   TRAIN_RANDOM_NEGATIVES for why not hard negatives) and caps MIND at 60k
   impressions; evaluation runs a chunk of impressions at a time and still scores
   every candidate, so reported metrics stay exact. Feature building
   costs ~1.8 KB/row transiently, which is what forces this on a ~5 GB budget.
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

# Training-only negative sampling (module docstring, point 3): every positive
# plus 49 negatives drawn uniformly at random, ~4x fewer rows than full groups.
#
# Hard negatives were tried first and are deliberately OFF. Keeping each group's
# 24 best-ranked negatives over-represents exactly the part of the list where the
# stage-1 rank separates the positive least well, so the model learns rank is a
# weak signal -- and then ranks worse than stage 1 over the full 200. Measured on
# EB-NeRD val MRR (stage 1 = 0.1690): hard24+rand25 0.1447, full groups with no
# sampling 0.2187, random-only 0.2249. Random sampling keeps the rank
# distribution representative, and matches full groups at a quarter of the rows.
TRAIN_HARD_NEGATIVES = 0
TRAIN_RANDOM_NEGATIVES = 49
# Memory, not data: DesignChoices.md §2D Finding 4 showed 31x more MIND training
# impressions did not help.
TRAIN_IMPRESSION_CAP = {"mind": 60_000}
EVAL_CHUNK_IMPRESSIONS = 5_000
SEED = 42
# Article stats are taken as-of each row's own time over EVERY earlier event,
# not Q1's train-only default -- see config.ARTICLE_STATS_SPLITS, which the NRMS
# freshness arm reads too. Still strictly before t, with the same click
# reporting lag as the fresh pool.
ARTICLE_STATS_SPLITS = config.ARTICLE_STATS_SPLITS


def _article_index(processed_dir, dataset: str):
    return article_stats.TrainEventIndex(
        processed_dir, dataset, splits=ARTICLE_STATS_SPLITS, click_lag_minutes=config.CLICK_REPORTING_LAG_MINUTES
    )


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


def sample_negatives(labelled: pd.DataFrame, n_hard: int, n_random: int, seed: int) -> pd.DataFrame:
    """Keep every positive, each impression's `n_hard` best-ranked negatives, and
    `n_random` of its remaining negatives drawn uniformly (seeded).

    Training only: evaluation metrics depend on every candidate. `rank` still
    records each row's original stage-1 position, so no rank information is lost.
    """
    is_negative = labelled["label"].to_numpy() == 0
    negatives = labelled[is_negative]
    hard = negatives.groupby("impression_id", sort=False)["rank"].rank(method="first").to_numpy() <= n_hard
    rest = negatives[~hard]
    draw = pd.Series(np.random.default_rng(seed).random(len(rest)), index=rest.index)
    drawn = draw.groupby(rest["impression_id"].to_numpy()).rank(method="first").to_numpy() <= n_random
    keep = labelled.index[~is_negative].union(negatives.index[hard]).union(rest.index[drawn])
    return labelled.loc[keep]


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
    sample_negatives_for_training: bool = False,
    impressions=None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Candidates -> labelled feature matrix, sorted by impression so LightGBM's
    contiguous-group requirement holds.

    `positives_only=True` restricts to impressions containing a positive -- exact
    for evaluation too, see `evaluate_over_population`.
    `sample_negatives_for_training=True` subsamples each group's negatives; for
    training and early-stopping validation only, never evaluation.
    `impressions` restricts to a subset of impression ids before anything is
    exploded or built, which is how sampling and chunking stay cheap.
    """
    cands = candidates_module.load_candidates(processed_dir, method, split, k, impressions=impressions)
    clicks = true_clicks(processed_dir, split)
    cands = label_candidates(cands, clicks)

    if positives_only:
        keep = set(impressions_with_positive(cands).tolist())
        cands = cands[cands["impression_id"].isin(keep)]

    if sample_negatives_for_training:
        cands = sample_negatives(cands, TRAIN_HARD_NEGATIVES, TRAIN_RANDOM_NEGATIVES, seed)

    article_index = article_index or _article_index(processed_dir, dataset)
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

    Before = stage 1's own ordering, taken from its RANK rather than its raw
    score: `popular` scores are integer click counts with ties everywhere, and
    sorting on those would scramble the ties and understate the baseline. Rank is
    the exact stage-1 order for every generator, and matches the score order for
    BM25 up to ties. After = the re-ranker's. Both are averaged over the same full
    population, so the delta measures ranking quality rather than which
    impressions were included.
    """
    labels, before, after = score_groups(matrix, model_scores)
    return {
        "before": evaluate_over_population(labels, before, n_impressions_total, ndcg_ks),
        "after": evaluate_over_population(labels, after, n_impressions_total, ndcg_ks),
    }


def score_groups(matrix: pd.DataFrame, model_scores: np.ndarray):
    """Per-impression (labels, stage-1 order, re-ranker scores). Factored out so
    chunked evaluation can accumulate groups across chunks and compute exactly
    what a single pass would."""
    scored = matrix.copy()
    scored["model_score"] = model_scores
    scored["stage1_order"] = -scored["retrieval_rank"].astype(np.float64)
    labels, before = group_arrays(scored, "stage1_order")
    _, after = group_arrays(scored, "model_score")
    return labels, before, after


def evaluate_chunked(
    processed_dir,
    dataset: str,
    split: str,
    method: str,
    model: lgb.Booster,
    features: list[str],
    article_index,
    k: int | None = None,
    chunk_impressions: int = EVAL_CHUNK_IMPRESSIONS,
    ndcg_ks=(5, 10),
) -> dict:
    """Full-population before/after, built and scored a chunk of impressions at
    a time. Every candidate of every scored impression is kept, so this equals a
    single-pass `evaluate_before_after` -- it just never holds more than one
    chunk's feature rows in memory."""
    n_total = len(true_clicks(processed_dir, split))
    impression_ids = pd.read_parquet(
        candidates_module.candidates_path(processed_dir, method, split), columns=["impression_id"]
    )["impression_id"].to_numpy()

    labels, before, after = [], [], []
    for start in range(0, len(impression_ids), chunk_impressions):
        chunk = set(impression_ids[start : start + chunk_impressions].tolist())
        matrix = build_matrix(
            processed_dir, dataset, split, method, article_index, k, positives_only=True, impressions=chunk
        )
        if matrix.empty:
            continue
        scores = model.predict(matrix[features].astype(np.float64), num_iteration=model.best_iteration)
        chunk_labels, chunk_before, chunk_after = score_groups(matrix, scores)
        labels += chunk_labels
        before += chunk_before
        after += chunk_after

    return {
        "before": evaluate_over_population(labels, before, n_total, ndcg_ks),
        "after": evaluate_over_population(labels, after, n_total, ndcg_ks),
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


def _training_impressions(processed_dir, dataset: str, method: str, seed: int):
    """A seeded subset of train impressions under TRAIN_IMPRESSION_CAP, or None."""
    cap = TRAIN_IMPRESSION_CAP.get(dataset)
    if cap is None:
        return None
    ids = pd.read_parquet(
        candidates_module.candidates_path(processed_dir, method, "train"), columns=["impression_id"]
    )["impression_id"].to_numpy()
    if len(ids) <= cap:
        return None
    return set(np.random.default_rng(seed).choice(ids, size=cap, replace=False).tolist())


def run(
    dataset: str,
    method: str,
    eval_split: str = "test",
    k: int | None = None,
    sample_negatives_for_training: bool = True,
) -> dict:
    processed_dir = candidates_module.PROCESSED_DIRS[dataset]
    article_index = _article_index(processed_dir, dataset)

    # Training and early-stopping validation see positive-bearing groups only,
    # negatives subsampled; the reported evaluation scores every candidate of the
    # full population (see module docstring).
    sample = sample_negatives_for_training
    train_ids = _training_impressions(processed_dir, dataset, method, SEED) if sample else None
    train_matrix = build_matrix(
        processed_dir, dataset, "train", method, article_index, k,
        positives_only=True, sample_negatives_for_training=sample, impressions=train_ids,
    )
    val_matrix = build_matrix(
        processed_dir, dataset, "val", method, article_index, k,
        positives_only=True, sample_negatives_for_training=sample,
    )
    features = feature_columns(train_matrix)
    n_train_impressions = int(train_matrix["impression_id"].nunique())
    n_train_rows = len(train_matrix)

    model = train(train_matrix, val_matrix, features)
    del train_matrix, val_matrix  # free before evaluation builds its own rows

    report = evaluate_chunked(processed_dir, dataset, eval_split, method, model, features, article_index, k)

    report["dataset"] = dataset
    report["method"] = method
    report["eval_split"] = eval_split
    report["best_iteration"] = model.best_iteration
    report["n_train_impressions"] = n_train_impressions
    report["n_train_rows"] = n_train_rows
    report["negatives_sampled"] = sample
    report["article_stats_splits"] = list(ARTICLE_STATS_SPLITS)
    report["importance"] = feature_importance(model, features).to_dict("records")

    artifact_dir = processed_dir / "reranker"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(artifact_dir / f"lambdamart_{method}.txt"), num_iteration=model.best_iteration)
    (artifact_dir / f"report_{method}_{eval_split}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


METRIC_NAMES = ("auc", "mrr", "ndcg@5", "ndcg@10")


def summary_table(reports: list[dict]) -> pd.DataFrame:
    """One row per (dataset, method), before vs after side by side."""
    rows = []
    for report in reports:
        before, after = report["before"], report["after"]
        row = {
            "dataset": report["dataset"],
            "method": report["method"],
            "split": report["eval_split"],
            "impressions": before["n_impressions"],
            "with_positive": before["n_impressions_scored"],
            "recall@k": before["n_impressions_scored"] / before["n_impressions"],
        }
        for name in METRIC_NAMES:
            row[f"{name}_before"] = before[name]
            row[f"{name}_after"] = after[name]
        # MRR ceiling is the share of impressions whose click was retrieved at
        # all: even a perfect ranker cannot score the rest.
        row["mrr_ceiling"] = row["recall@k"]
        row["headroom_captured"] = after["mrr"] / row["recall@k"] if row["recall@k"] else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--method", choices=[*candidates_module.METHODS, "all"], default="all")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--k", type=int, default=None, help="slice candidates to top-k (default: all persisted)")
    parser.add_argument(
        "--full-train-groups",
        action="store_true",
        help="train on every candidate of every group -- the original protocol, no negative sampling or cap",
    )
    args = parser.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    methods = list(candidates_module.METHODS) if args.method == "all" else [args.method]

    reports = []
    for dataset in datasets:
        for method in methods:
            processed_dir = candidates_module.PROCESSED_DIRS[dataset]
            missing = [
                split
                for split in ("train", "val", args.eval_split)
                if not candidates_module.candidates_path(processed_dir, method, split).exists()
            ]
            if missing:
                print(f"[{dataset}:{method}] skipped -- missing candidates for {sorted(set(missing))}")
                continue

            started = time.perf_counter()
            report = run(
                dataset, method, args.eval_split, args.k,
                sample_negatives_for_training=not args.full_train_groups,
            )
            report["seconds"] = round(time.perf_counter() - started, 1)
            reports.append(report)
            before, after = report["before"], report["after"]
            print(
                f"[{dataset}:{method}:{args.eval_split}] {report['seconds']}s  "
                + "  ".join(f"{n}: {before[n]:.4f}->{after[n]:.4f}" for n in METRIC_NAMES)
            )

    if not reports:
        print("nothing to run")
        return

    table = summary_table(reports)
    out_path = config.PROCESSED_DIR / f"reranker_summary_{args.eval_split}.csv"
    table.to_csv(out_path, index=False)
    print(f"\n{table.to_string(index=False, float_format=lambda v: f'{v:.4f}')}")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
