"""Recall@K for stage-1 retrieval (A1 Q2.4/Q3.4, A2 §2E stage-1 ablation).

Two implementations exist with deliberately different definitions, and both
feed reported numbers, so each is pinned on hand-built data:

  * `lexical_retrieval.compute_recall_at_k` (A1): per impression, the fraction
    of its clicked articles retrieved, averaged over impressions that have a
    click and were retrieved for at all.
  * `candidates.recall_at_k` (A2): impression-level hit rate -- did ANY clicked
    article make the top-k -- with every impression of the split in the
    denominator, including ones the generator failed to emit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import candidates
from src.lexical_retrieval import compute_recall_at_k


def _behaviors():
    return pd.DataFrame(
        {
            "impression_id": ["i1", "i2", "i3", "i4"],
            "candidates": [["a", "b", "c"], ["x", "y"], ["p", "q"], ["m", "n"]],
            "labels": [[1, 1, 0], [1, 0], [0, 1], [0, 0]],
        }
    )


# ------------------------------------------------------------------ A1 recall

def test_a1_recall_is_the_fraction_of_clicks_retrieved_per_impression():
    retrieved = pd.DataFrame(
        {
            "impression_id": ["i1", "i2", "i4"],
            "candidates": [["a", "z", "b"], ["z", "w", "x"], ["m"]],
        }
    )
    recall = compute_recall_at_k(_behaviors(), retrieved, ks=(1, 3))
    # i1: clicks {a, b}; @1 finds a (1/2), @3 finds both (2/2).
    # i2: click {x};     @1 misses (0),   @3 finds it (1).
    # i3 has no retrieval row and i4 has no click, so neither is averaged in.
    assert recall[1] == pytest.approx((0.5 + 0.0) / 2)
    assert recall[3] == pytest.approx((1.0 + 1.0) / 2)


# ------------------------------------------------------------------ A2 recall

def _write(tmp_path, wide: pd.DataFrame):
    _behaviors().to_parquet(tmp_path / "behaviors_test.parquet", index=False)
    wide.to_parquet(candidates.candidates_path(tmp_path, "m", "test"), index=False)
    return tmp_path


def test_a2_recall_is_an_impression_hit_rate_over_the_whole_split(tmp_path):
    wide = pd.DataFrame(
        {
            "impression_id": ["i1", "i2", "i4"],
            "candidates": [np.array(["z", "b", "a"]), np.array(["z", "x"]), np.array(["m"])],
        }
    )
    recall = candidates.recall_at_k(_write(tmp_path, wide), "m", "test", ks=(1, 2, 3))
    # i1 hits at position 2 (b), i2 at position 2 (x); i3 was never emitted and
    # i4 has no click -- both still count in the denominator of 4.
    assert recall[1] == pytest.approx(0 / 4)
    assert recall[2] == pytest.approx(2 / 4)
    assert recall[3] == pytest.approx(2 / 4)


def test_a2_recall_counts_one_hit_per_impression_not_per_click(tmp_path):
    wide = pd.DataFrame({"impression_id": ["i1"], "candidates": [np.array(["a", "b"])]})
    recall = candidates.recall_at_k(_write(tmp_path, wide), "m", "test", ks=(2,))
    assert recall[2] == pytest.approx(1 / 4)  # i1 retrieves both its clicks: still one hit
