#!/usr/bin/env python3
"""Paired bootstrap CIs for an NRMS ablation arm against its baseline.

    python scripts/paired_bootstrap.py --dataset ebnerd
    python scripts/paired_bootstrap.py --dataset mind --arm freshness

Reads the two `predictions_<split>.parquet` files an NRMS run writes -- the
baseline at `nrms/` and the arm at `nrms/<arm>/` -- and reports, per metric, the
mean per-impression difference with a bootstrap confidence interval.

Why paired, and what the interval does NOT cover
------------------------------------------------
Both arms score the *same* impression population (evaluate.py writes one row per
impression in the split's order), so the difference is measured within each
impression and the impression-to-impression variance cancels. Resampling
impressions with both models held fixed therefore bounds sampling noise in the
test set.

It says nothing about *training* variance. A different seed changes the weights,
not the test set, and this procedure cannot see that -- so a gain of the same
order as seed noise needs repeated runs (`--freshness --seed 43 --run-tag freshness_s43`),
not a tighter interval here. See DesignChoices.md §2F "Seed count".

Metrics come from src/metrics.py so this script cannot drift from the numbers in
`metrics_<split>.json`: in particular the tie-breaking convention in
`_descending_order` is the benchmark's, which matters for any scorer that
produces ties.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Run from anywhere: `python scripts/x.py` puts scripts/ on the path, not the
# repo root, so `src` would not import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import per_impression_metrics
from src.nrms import config as nrms_config

METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")


def per_impression(labels: np.ndarray, scores: np.ndarray) -> tuple[float, ...]:
    """One impression's four metrics, or NaN for AUC when it is undefined.

    An impression whose labels are all-0 or all-1 has no AUC; evaluate_impressions
    drops those from the AUC mean only, and pairing forces the same treatment on
    both arms, so the difference stays defined on exactly the same rows.
    """
    values = per_impression_metrics(labels, scores)
    return tuple(values[name] for name in METRICS)


def load_pair(dataset: str, arm: str, split: str) -> pd.DataFrame:
    artifact_dir = nrms_config.artifact_dir(dataset)
    baseline = pd.read_parquet(artifact_dir / f"predictions_{split}.parquet")
    treatment = pd.read_parquet(artifact_dir / arm / f"predictions_{split}.parquet")
    merged = baseline.merge(treatment, on="impression_id", suffixes=("_b", "_t"))
    if not (len(merged) == len(baseline) == len(treatment)):
        raise SystemExit(
            f"impression populations differ: baseline {len(baseline)}, "
            f"{arm} {len(treatment)}, joined {len(merged)} -- the comparison "
            "would measure which impressions were included, not which model is better"
        )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--arm", default="freshness", help="run-tag subdirectory of the treatment arm")
    parser.add_argument("--split", default="test")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    merged = load_pair(args.dataset, args.arm, args.split)

    rows = []
    for row in merged.itertuples(index=False):
        labels = np.asarray(row.labels_b, dtype=np.float64)
        if not np.array_equal(labels, np.asarray(row.labels_t, dtype=np.float64)):
            raise SystemExit(f"impression {row.impression_id}: labels differ between arms")
        rows.append(per_impression(labels, np.asarray(row.scores_b)) + per_impression(labels, np.asarray(row.scores_t)))
    table = np.asarray(rows, dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(table), size=(args.resamples, len(table)))

    print(f"{args.dataset} {args.split}: {len(table):,} impressions, baseline vs {args.arm}")
    print(f"{'metric':8s} {'baseline':>9s} {args.arm[:9]:>9s} {'delta':>9s} {'95% CI':>21s}   win/tie/loss")
    for i, name in enumerate(METRICS):
        base, arm = table[:, i], table[:, i + len(METRICS)]
        usable = ~np.isnan(base) & ~np.isnan(arm)
        delta = np.where(usable, arm - base, 0.0)
        # Resampled means over the usable rows only; the AUC column drops the
        # impressions where AUC is undefined, exactly as evaluate.py does.
        sums = delta[draws].sum(axis=1)
        counts = usable[draws].sum(axis=1)
        boots = sums / np.maximum(counts, 1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        d = delta[usable]
        print(
            f"{name:8s} {base[usable].mean():9.4f} {arm[usable].mean():9.4f} {d.mean():+9.4f} "
            f"[{lo:+.4f}, {hi:+.4f}]   "
            f"{np.mean(d > 1e-12):.1%}/{np.mean(np.abs(d) <= 1e-12):.1%}/{np.mean(d < -1e-12):.1%}"
        )


if __name__ == "__main__":
    main()
