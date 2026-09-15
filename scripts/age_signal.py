#!/usr/bin/env python3
"""How much ranking signal article age actually carries, per dataset.

    python scripts/age_signal.py --dataset ebnerd
    python scripts/age_signal.py --dataset mind
    python scripts/age_signal.py --dataset ebnerd --reference first-seen

Written to answer the question §2F's MIND null raised: the freshness arm gained
+0.069 AUC on EB-NeRD and nothing on MIND, with age coverage at 99.9% on both,
so the difference is not a coverage bug. This measures the signal itself.

Reported per test split:

  * coverage -- share of candidates and of clicks carrying a usable age;
  * median age of clicked vs non-clicked candidates;
  * median within-impression age spread -- the quantity that decides whether an
    additive age term can reorder a row at all. A term that is near-constant
    across an impression's candidates cannot, however well calibrated it is;
  * freshest-first AUC -- ranking by age alone, newest first, no model. Note
    this is NOT an upper bound on the available signal: EB-NeRD's trained curve
    is an inverted U peaking near 6 h, so a monotone recency rule sits at chance
    there while the learned curve does not;
  * censored share -- candidates whose reference sits in the first hour of the
    behaviour logs. A first-seen reference cannot date an article that already
    existed when logging began, so those ages are floors, not measurements;
  * the trained head's AUC used alone as a ranker, when a freshness checkpoint
    exists. Under `--reference first-seen` this becomes a transfer test -- the
    same fitted age->score curve, fed a different age definition -- which is the
    controlled way to ask what the proxy costs.

`--reference first-seen` forces MIND's proxy definition -- earliest sighting in
the behaviour logs -- onto EB-NeRD as well, which is the controlled comparison:
EB-NeRD has a real `published_time`, so running it both ways isolates what the
proxy costs on a dataset where the truth is known. In that mode EB-NeRD also
reports how far the proxy's ages fall below the true ones.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Run from anywhere: `python scripts/x.py` puts scripts/ on the path, not the
# repo root, so `src` would not import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import article_stats, config
from src.nrms import adapter, config as nrms_config
from src.nrms.freshness import FreshnessLookup
from src.nrms.ids import load_or_build_codec
from src.nrms.model import FreshnessHead

from sklearn.metrics import roc_auc_score

NS_PER_HOUR = 3.6e12


def first_seen_reference_times(processed_dir, splits) -> dict:
    """Earliest time each article appears as a candidate, for ANY dataset.

    `article_stats.freshness_reference_times` short-circuits to `published_time`
    for EB-NeRD, which is the right default but makes the proxy uninspectable on
    the one dataset that can validate it. This is that function's MIND branch,
    applied regardless of dataset.
    """
    first_seen: dict = {}
    for split in splits:
        frame = pd.read_parquet(
            processed_dir / f"behaviors_{split}.parquet", columns=["time", "candidates"]
        )
        for t, candidates in zip(frame["time"], frame["candidates"]):
            for article_id in candidates:
                previous = first_seen.get(article_id)
                if previous is None or t < previous:
                    first_seen[article_id] = t
    return first_seen


def log_window_start_ns(processed_dir, splits) -> int:
    """First timestamp anywhere in the behaviour logs.

    An article already in circulation at that moment gets a first-seen reference
    of roughly this value whatever its real publish date, so its age is right-
    censored -- the proxy's one structural weakness.
    """
    starts = [
        pd.read_parquet(processed_dir / f"behaviors_{split}.parquet", columns=["time"])["time"].min()
        for split in splits
    ]
    return int(pd.Timestamp(min(starts)).value)


def lookup_from_references(references: dict, codec) -> FreshnessLookup:
    reference_ns = np.full(codec.n_articles + 1, np.nan, dtype=np.float64)
    for article_id, timestamp in references.items():
        code = codec.id_to_code.get(str(article_id))
        if code is not None:
            reference_ns[code] = pd.Timestamp(timestamp).value
    return FreshnessLookup(reference_ns)


def load_head(artifact_dir, arm: str):
    checkpoint = artifact_dir / arm / "nrms.pt"
    if not checkpoint.exists():
        return None
    import torch

    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    weights = {k.replace("freshness_head.", ""): v for k, v in state.items() if k.startswith("freshness_head")}
    if not weights:
        return None
    head = FreshnessHead(nrms_config.FRESHNESS_HIDDEN_DIM)
    head.load_state_dict(weights)
    head.eval()
    return head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--arm", default="freshness")
    parser.add_argument(
        "--reference",
        choices=["default", "first-seen"],
        default="default",
        help="'default' is what the arm uses (EB-NeRD published_time, MIND first sighting); "
             "'first-seen' forces the MIND proxy onto both datasets",
    )
    parser.add_argument("--history-size", type=int, default=nrms_config.HISTORY_SIZE)
    args = parser.parse_args()

    processed_dir = nrms_config.PROCESSED_DIRS[args.dataset]
    artifact_dir = nrms_config.artifact_dir(args.dataset)
    codec = load_or_build_codec(processed_dir, artifact_dir)
    splits = config.ARTICLE_STATS_SPLITS

    if args.reference == "default":
        lookup = FreshnessLookup.build(processed_dir, args.dataset, codec, splits)
    else:
        lookup = lookup_from_references(first_seen_reference_times(processed_dir, splits), codec)
    truth = (
        lookup_from_references(article_stats.freshness_reference_times(processed_dir, args.dataset, splits), codec)
        if args.reference == "first-seen" and args.dataset == "ebnerd"
        else None
    )

    head = load_head(artifact_dir, args.arm)
    impressions = adapter.load_impressions(processed_dir, args.split, codec, args.history_size)
    times = impressions["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)

    window_start_ns = log_window_start_ns(processed_dir, splits)
    censored_before_ns = window_start_ns + NS_PER_HOUR

    n_candidates = n_dated = n_clicks = n_clicks_dated = n_censored = 0
    censored_true_ages = []
    clicked_ages, other_ages, spreads = [], [], []
    fresh_first_aucs, head_aucs = [], []
    proxy_gap = []

    for codes, labels, t in zip(impressions["candidates"], impressions["labels"], times):
        codes = np.asarray(codes, dtype=np.int64)
        labels = np.asarray(labels)
        log_age, known = lookup.ages(codes, t)
        ages = np.expm1(log_age)
        dated = known == 1

        n_candidates += len(codes)
        n_dated += int(dated.sum())
        n_clicks += int(labels.sum())
        n_clicks_dated += int(dated[labels == 1].sum())
        censored = dated & (lookup.reference_ns[codes] <= censored_before_ns)
        n_censored += int(censored.sum())
        clicked_ages.extend(ages[dated & (labels == 1)])
        other_ages.extend(ages[dated & (labels == 0)])
        if dated.sum() >= 2:
            spreads.append(float(ages[dated].std()))

        scorable = 0 < labels.sum() < len(labels)
        if scorable and dated.all():
            # Freshest first: the whole impression must be dated, so the ranking
            # never depends on how "unknown" is scored.
            fresh_first_aucs.append(float(roc_auc_score(labels, -ages)))
        if scorable and head is not None:
            import torch

            with torch.no_grad():
                adjustment = head(torch.from_numpy(log_age), torch.from_numpy(known)).numpy()
            head_aucs.append(float(roc_auc_score(labels, adjustment)))

        if truth is not None:
            true_log_age, true_known = truth.ages(codes, t)
            both = (known == 1) & (true_known == 1)
            proxy_gap.extend(np.expm1(true_log_age[both]) - ages[both])
            censored_true_ages.extend(np.expm1(true_log_age[censored & (true_known == 1)]))

    print(f"== {args.dataset} {args.split}: {len(impressions):,} impressions, reference={args.reference}")
    print(f"   candidates dated       : {n_dated / n_candidates:.1%} ({n_dated:,}/{n_candidates:,})")
    print(f"   clicks dated           : {n_clicks_dated / n_clicks:.1%} ({n_clicks_dated:,}/{n_clicks:,})")
    print(f"   median age, clicked    : {np.median(clicked_ages):.1f} h")
    print(f"   median age, not clicked: {np.median(other_ages):.1f} h")
    print(f"   within-impression spread (median std): {np.median(spreads):.1f} h")
    print(f"   censored by log window : {n_censored / max(n_dated, 1):.1%} of dated candidates")
    print(f"   freshest-first AUC     : {np.mean(fresh_first_aucs):.4f}  ({len(fresh_first_aucs):,} impressions)")
    if head_aucs:
        note = "" if args.reference == "default" else "  [transfer: head fitted on the default reference]"
        print(f"   trained head alone, AUC: {np.mean(head_aucs):.4f}  ({len(head_aucs):,} impressions){note}")
    if censored_true_ages:
        print(f"   censored candidates' true age: median {np.median(censored_true_ages):.0f} h")
    if proxy_gap:
        gap = np.asarray(proxy_gap)
        print(f"   proxy vs published_time: median understatement {np.median(gap):.1f} h, "
              f"{np.mean(gap > 24):.1%} of candidates understated by >24 h")


if __name__ == "__main__":
    main()
