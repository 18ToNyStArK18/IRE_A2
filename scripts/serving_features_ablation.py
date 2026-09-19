#!/usr/bin/env python3
"""A2 Q9: re-ranker metrics with and without features unavailable at serving time.

    python scripts/serving_features_ablation.py --dataset ebnerd
    python scripts/serving_features_ablation.py --dataset mind

Writes results/serving_features_ablation_<dataset>.json.

Which features a live system would not have
-------------------------------------------
  impression_size         len() of the impression's own in-view list. On EB-NeRD
                          that list is the articles that came INTO VIEW, which
                          grows with how long the user stays on the page:
                          Spearman 0.50 with the impression's own read_time, an
                          outcome known only afterwards. On MIND it is the
                          platform's list length for the impression being
                          ranked, which a two-stage system choosing its own
                          candidates would not have either.
  has_known_publish_time  read off the dataset's later published_time snapshot;
                          flags the articles bulk re-stamped after the logs
                          (DesignChoices §2E), which no live system could know.
  session_clicks_before   (strict arm only) clicks on earlier impressions in the
                          same session. A live session store holds most of them,
                          but unlike the article stats this feature applies no
                          reporting lag, so it is dropped in a stricter arm.

A fourth arm, `no_history_content`, drops the two history-content features
(src/history_content.py) instead. They ARE available at serving time; the arm is
here because the same paired harness gives their contribution, with a CI.

Every arm is trained exactly as `reranker.run` trains the shipped model -- same
impressions, same negative sampling, same seed -- and differs only in its feature
list. The `all` arm must therefore reproduce report_<method>_test.json, which is
checked below. Differences are paired over test impressions, with MRR/nDCG
averaged over the full population (an unscored impression contributes 0 to both
arms) and AUC over impressions where it is defined, matching the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import history_content, reranker  # noqa: E402
from src.metrics import per_impression_metrics  # noqa: E402

METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")
UNAVAILABLE = ("impression_size", "has_known_publish_time")
ARMS = {
    "all": (),
    "serving_safe": UNAVAILABLE,
    "serving_safe_strict": UNAVAILABLE + ("session_clicks_before",),
    # Not a Q9 arm: the history-content features are serving-safe. This measures
    # what they add, on the same paired harness.
    "no_history_content": history_content.FEATURES,
}


def train_arms(processed_dir, dataset: str, method: str, article_index) -> tuple[dict, list[str]]:
    """One booster per arm, all fitted on the same sampled matrices."""
    train_ids = reranker._training_impressions(processed_dir, dataset, method, reranker.SEED)
    train_matrix = reranker.build_matrix(
        processed_dir, dataset, "train", method, article_index,
        positives_only=True, sample_negatives_for_training=True, impressions=train_ids,
    )
    val_matrix = reranker.build_matrix(
        processed_dir, dataset, "val", method, article_index,
        positives_only=True, sample_negatives_for_training=True,
    )
    all_features = reranker.feature_columns(train_matrix)
    boosters = {}
    for arm, dropped in ARMS.items():
        features = [f for f in all_features if f not in dropped]
        boosters[arm] = reranker.train(train_matrix, val_matrix, features)
        print(f"  trained {arm:20s} {len(features)} features, {boosters[arm].best_iteration} trees")
    return boosters, all_features


def per_impression_rows(processed_dir, dataset, method, article_index, boosters, chunk: int):
    """{arm: (n_scored, 4) metric rows}, plus the stage-1 'before' rows, in one
    pass over test chunks so features are built once for every arm."""
    impression_ids = np.array(sorted(reranker.true_clicks(processed_dir, "test")))
    rows = {arm: [] for arm in [*boosters, "stage1"]}
    for start in range(0, len(impression_ids), chunk):
        ids = set(impression_ids[start : start + chunk].tolist())
        matrix = reranker.build_matrix(processed_dir, dataset, "test", method, article_index,
                                       positives_only=True, impressions=ids)
        if matrix.empty:
            continue
        scores = {
            arm: booster.predict(matrix[booster.feature_name()].to_numpy(dtype=np.float64),
                                 num_iteration=booster.best_iteration)
            for arm, booster in boosters.items()
        }
        scores["stage1"] = -matrix["retrieval_rank"].to_numpy(dtype=np.float64)
        bounds = np.flatnonzero(np.r_[True, matrix["impression_id"].to_numpy()[1:] != matrix["impression_id"].to_numpy()[:-1], True])
        labels = matrix["label"].to_numpy(dtype=np.float64)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            for arm, s in scores.items():
                m = per_impression_metrics(labels[lo:hi], s[lo:hi])
                rows[arm].append([m[k] for k in METRICS])
        print(f"  scored {min(start + chunk, len(impression_ids)):,}/{len(impression_ids):,} impressions")
    return {arm: np.asarray(r, dtype=np.float64) for arm, r in rows.items()}, len(impression_ids)


def population_mean(values: np.ndarray, metric: str, n_total: int) -> float:
    """AUC over impressions where defined; MRR/nDCG over the full population,
    exactly as reranker.evaluate_over_population reports them."""
    if metric == "auc":
        return float(np.nanmean(values))
    return float(values.sum() / n_total)


def paired_ci(base: np.ndarray, arm: np.ndarray, metric: str, n_total: int,
              resamples: int, seed: int) -> dict:
    """Paired bootstrap of (arm - base) over impressions. For MRR/nDCG the
    unscored impressions are zero-padded pairs (difference 0), so the
    resampled population is the full split, as in the reported means."""
    if metric == "auc":
        keep = ~np.isnan(base) & ~np.isnan(arm)
        delta = (arm - base)[keep]
    else:
        delta = np.concatenate([arm - base, np.zeros(n_total - len(base))])
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for i in range(resamples):  # looped: a (resamples, n) index matrix is >1 GB on MIND
        means[i] = delta[rng.integers(0, len(delta), size=len(delta))].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"delta": float(delta.mean()), "ci_lower": float(lo), "ci_upper": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n": int(len(delta))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--method", default="popular")
    parser.add_argument("--chunk-impressions", type=int, default=reranker.EVAL_CHUNK_IMPRESSIONS)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    processed_dir = reranker.candidates_module.PROCESSED_DIRS[args.dataset]
    article_index = reranker._article_index(processed_dir, args.dataset)

    print(f"{args.dataset}/{args.method}: training {len(ARMS)} arms")
    boosters, all_features = train_arms(processed_dir, args.dataset, args.method, article_index)
    rows, n_total = per_impression_rows(processed_dir, args.dataset, args.method, article_index,
                                        boosters, args.chunk_impressions)

    means = {arm: {m: population_mean(r[:, i], m, n_total) for i, m in enumerate(METRICS)}
             for arm, r in rows.items()}

    # The `all` arm is the shipped configuration, so it must reproduce the report.
    committed = json.loads((processed_dir / "reranker" / f"report_{args.method}_test.json").read_text())["after"]
    deviation = max(abs(means["all"][m] - committed[m]) for m in METRICS)

    report = {
        "dataset": args.dataset,
        "method": args.method,
        "split": "test",
        "unavailable_at_serving": list(UNAVAILABLE),
        "arms": {arm: {"dropped": list(d), "n_features": len(all_features) - len(d),
                       "trees": boosters[arm].best_iteration, "metrics": means[arm]}
                 for arm, d in ARMS.items()},
        "stage1_before": means["stage1"],
        "deltas_vs_all": {
            arm: {m: paired_ci(rows["all"][:, i], rows[arm][:, i], m, n_total, args.resamples, args.seed)
                  for i, m in enumerate(METRICS)}
            for arm in ARMS if arm != "all"
        },
        "deltas_vs_stage1": {
            arm: {m: paired_ci(rows["stage1"][:, i], rows[arm][:, i], m, n_total, args.resamples, args.seed)
                  for i, m in enumerate(METRICS)}
            for arm in ARMS
        },
        "dropped_feature_gain_share_in_all": {
            f: float(g / sum(boosters["all"].feature_importance("gain")))
            for f, g in zip(boosters["all"].feature_name(), boosters["all"].feature_importance("gain"))
            if f in {g for dropped in ARMS.values() for g in dropped}
        },
        "check_all_arm_reproduces_report": {"max_abs_deviation": deviation, "ok": deviation < 1e-9},
        "n_impressions": n_total,
        "n_impressions_scored": int(len(rows["all"])),
        "config": {"resamples": args.resamples, "seed": args.seed, "chunk_impressions": args.chunk_impressions},
    }

    out = Path(args.out) if args.out else Path("results") / f"serving_features_ablation_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{args.dataset} test, {n_total:,} impressions ({len(rows['all']):,} scored)")
    print(f"'all' arm vs committed report: max deviation {deviation:.2e} -> {'OK' if deviation < 1e-9 else 'MISMATCH'}")
    print(f"{'arm':22s}" + "".join(f"{m:>10s}" for m in METRICS))
    for arm in ["stage1", *ARMS]:
        print(f"{arm:22s}" + "".join(f"{means[arm][m]:10.4f}" for m in METRICS))
    for arm, block in report["deltas_vs_all"].items():
        print(f"\n{arm} - all (paired 95% CI):")
        for m, d in block.items():
            print(f"  {m:8s} {d['delta']:+.4f} [{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]"
                  + ("  *" if d["excludes_zero"] else ""))
    print(f"\ngain share of dropped features in 'all': {report['dropped_feature_gain_share_in_all']}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
