"""Negative sampling, following Wu et al. (2019) as ebnerd-benchmark
implements it in `sampling_strategy_wu2019`.

For each impression: drop the positives from the in-view list, then emit one
training row per clicked article, pairing that positive with `npratio` sampled
negatives. The npratio+1 candidates are shuffled and the positive's resulting
index is recorded, which turns the objective into plain softmax
cross-entropy over the candidate axis -- exactly the categorical
cross-entropy their Keras model is compiled with.

Sampling is with replacement, matching their default: many impressions show
fewer than npratio negatives, and dropping those rows would bias training
toward users with long candidate lists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sampling_strategy_wu2019(
    impressions: pd.DataFrame,
    npratio: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Expand impressions into one row per positive click.

    Expects the frame from adapter.load_impressions. Returns columns:
    impression_id, time, history, candidates (int32[npratio+1]), label (int
    index of the positive within `candidates`).

    `time` is carried through because article age is a property of the
    (impression, candidate) pair, not of the article: the freshness arm needs
    each training row's own impression time to date its candidates against.
    """
    rng = np.random.default_rng(seed)

    impression_ids, times, histories, candidate_sets, labels = [], [], [], [], []
    for row in impressions.itertuples(index=False):
        candidates = np.asarray(row.candidates, dtype=np.int32)
        row_labels = np.asarray(row.labels, dtype=np.int8)
        negatives = candidates[row_labels == 0]
        if len(negatives) == 0:
            # No negative to contrast against; the softmax objective is
            # degenerate for this impression.
            continue

        for positive in candidates[row_labels == 1]:
            sampled = rng.choice(negatives, size=npratio, replace=True)
            group = np.concatenate(([positive], sampled)).astype(np.int32)
            order = rng.permutation(npratio + 1)
            impression_ids.append(row.impression_id)
            times.append(row.time)
            histories.append(row.history)
            candidate_sets.append(group[order])
            labels.append(int(np.argmax(order == 0)))

    return pd.DataFrame(
        {
            "impression_id": impression_ids,
            "time": times,
            "history": histories,
            "candidates": candidate_sets,
            "label": np.asarray(labels, dtype=np.int64),
        }
    )
