"""TrainEventIndex event scope and click reporting lag.

The default scope (train only) freezes val/test queries at end-of-train totals.
That is leak-free but a train/serve skew: it zeroed the popularity features for
every fresh article clicked at test time, and cost the re-ranker 18-21% MRR.
The all-splits scope fixes the skew; these tests pin that it still never counts
an event at or after t, and that the lag delays clicks but not displays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import article_stats

T0 = pd.Timestamp("2024-01-01")
MIN = pd.Timedelta(minutes=1)
ALL = ("train", "val", "test")


def _write(tmp_path):
    def impression(minutes, candidates, labels):
        return {
            "impression_id": f"x{minutes}",
            "time": T0 + minutes * MIN,
            "history": np.asarray([], dtype=object),
            "candidates": np.asarray(candidates, dtype=object),
            "labels": np.asarray(labels, dtype=np.int8),
        }

    pd.DataFrame([impression(0, ["a", "b"], [1, 0])]).to_parquet(tmp_path / "behaviors_train.parquet", index=False)
    pd.DataFrame([impression(100, ["a", "c"], [0, 1])]).to_parquet(tmp_path / "behaviors_val.parquet", index=False)
    pd.DataFrame([impression(200, ["c"], [1]), impression(205, ["c"], [1])]).to_parquet(
        tmp_path / "behaviors_test.parquet", index=False
    )
    pd.DataFrame(
        {"article_id": ["a", "b", "c"], "category": ["s", "n", "s"], "published_time": [T0, T0, T0]}
    ).to_parquet(tmp_path / "articles.parquet", index=False)
    return tmp_path


def test_default_scope_freezes_later_queries_at_end_of_train(tmp_path):
    """Q1 semantics, kept as the default: 'c' only ever appears after train, so a
    test-time query reads zero -- exactly the skew the all-splits scope fixes."""
    stat = article_stats.TrainEventIndex(_write(tmp_path), "ebnerd").as_of("c", T0 + 300 * MIN)
    assert stat["display_count"] == 0 and stat["click_count"] == 0


def test_all_splits_scope_sees_every_earlier_event(tmp_path):
    stat = article_stats.TrainEventIndex(_write(tmp_path), "ebnerd", splits=ALL).as_of("c", T0 + 300 * MIN)
    assert stat["display_count"] == 3 and stat["click_count"] == 3


def test_all_splits_scope_is_still_strictly_before_t(tmp_path):
    """Widening the scope must not admit the impression being scored: the one at
    exactly t=200 is excluded, only the val impression at 100 counts."""
    stat = article_stats.TrainEventIndex(_write(tmp_path), "ebnerd", splits=ALL).as_of("c", T0 + 200 * MIN)
    assert stat["display_count"] == 1 and stat["click_count"] == 1


def test_click_lag_delays_clicks_but_not_displays(tmp_path):
    idx = article_stats.TrainEventIndex(_write(tmp_path), "ebnerd", splits=ALL, click_lag_minutes=10)
    stat = idx.as_of("c", T0 + 208 * MIN)
    assert stat["display_count"] == 3  # impressions at 100, 200 and 205 were all shown
    assert stat["click_count"] == 1  # only the one at 100 is older than the 10-min lag


def test_ctr_prior_stays_train_only_whatever_the_scope(tmp_path):
    d = _write(tmp_path)
    narrow = article_stats.TrainEventIndex(d, "ebnerd")
    wide = article_stats.TrainEventIndex(d, "ebnerd", splits=ALL)
    assert narrow.global_ctr == wide.global_ctr == pytest.approx(0.5)  # train: 1 click / 2 displays
