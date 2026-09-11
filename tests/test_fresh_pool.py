"""Stage-1 fresh-pool tests.

The as-of-time guarantees are what make these candidates legitimate -- a pool
that saw the impression's own display, or counted clicks that had not happened
yet, would be a leak -- so they are pinned exactly on hand-built logs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import fresh_pool

MIN = 60 * 10**9


def _log(rows):
    """rows: (minutes, candidates, labels)"""
    return pd.DataFrame(
        [
            {
                "t": int(minutes * MIN),
                "candidates": np.asarray(cands, dtype=object),
                "labels": np.asarray(labels, dtype=np.int8),
                "is_target": True,
                "impression_id": f"imp{i}",
            }
            for i, (minutes, cands, labels) in enumerate(rows)
        ]
    )


def _pool(rows):
    return fresh_pool.FreshPool(_log(rows), window_hours=1, backfill_hours=24, lag_minutes=10)


# ------------------------------------------------------- as-of guarantees

def test_same_timestamp_impressions_never_see_each_other():
    """Strictly-before: the impression being scored (and anything sharing its
    timestamp) contributes nothing to its own pool."""
    p = _pool([(100, ["a"], [0]), (100, ["b"], [0])])
    p.advance_to(100 * MIN)
    assert "a" not in p.pool and "b" not in p.pool


def test_earlier_displays_within_the_window_are_in_the_pool():
    p = _pool([(100, ["a"], [0]), (130, ["b"], [0])])
    p.advance_to(131 * MIN)
    assert set(p.pool) == {"a", "b"}


def test_display_older_than_the_window_moves_to_backfill_only():
    p = _pool([(0, ["old"], [0]), (100, ["new"], [0])])
    p.advance_to(100 * MIN + 1)  # 'old' is 100 min back: outside 1h, inside 24h
    assert "old" not in p.pool and "old" in p.backfill_counts
    assert "new" in p.pool


def test_clicks_inside_the_reporting_lag_are_not_counted():
    """A click lands after its impression, so one from 7 minutes ago may not have
    happened yet at t. Its display, however, did."""
    p = _pool([(100, ["a", "x"], [1, 0]), (105, ["b", "y"], [1, 0])])
    p.advance_to(112 * MIN)  # a's impression is 12 min old, b's only 7
    assert p.click_counts.get("a", 0) == 1
    assert p.click_counts.get("b", 0) == 0
    assert "b" in p.pool


def test_never_returns_an_article_first_shown_at_or_after_t():
    p = _pool([(100, ["a"], [0]), (200, ["future"], [0])])
    p.advance_to(150 * MIN)
    ranked, _ = p.popular(10)
    assert "future" not in {a for a, _ in ranked}


def test_queries_must_move_forward_in_time():
    p = _pool([(100, ["a"], [0])])
    p.advance_to(200 * MIN)
    with pytest.raises(ValueError):
        p.advance_to(150 * MIN)


def test_unsorted_log_is_rejected():
    with pytest.raises(ValueError):
        _pool([(200, ["a"], [0]), (100, ["b"], [0])])


# ------------------------------------------------------------- ranking

def test_popular_ranks_by_lagged_clicks_then_displays():
    p = _pool([
        (100, ["a", "b", "c"], [1, 0, 0]),
        (101, ["a", "b", "c"], [0, 1, 0]),
        (102, ["b", "c"], [1, 0]),
        (103, ["c"], [0]),
    ])
    p.advance_to(120 * MIN)
    ranked, used_backfill = p.popular(3)
    # b: 2 clicks, a: 1, c: 0 clicks despite the most displays (4)
    assert [a for a, _ in ranked] == ["b", "a", "c"]
    assert [s for _, s in ranked] == [2.0, 1.0, 0.0]
    assert used_backfill is False


def test_backfill_only_when_the_pool_is_short():
    p = _pool([(0, ["old1", "old2"], [0, 0]), (100, ["a"], [0])])
    p.advance_to(101 * MIN)

    ranked, used = p.popular(1)
    assert [a for a, _ in ranked] == ["a"] and used is False

    ranked, used = p.popular(3)
    assert ranked[0][0] == "a" and used is True
    assert {a for a, _ in ranked[1:]} == {"old1", "old2"}
    assert all(score == 0.0 for _, score in ranked[1:])  # always below a real hit


# --------------------------------------------- generator contract, end to end

def _write_processed(tmp_path):
    t0 = pd.Timestamp("2024-01-01 00:00:00")

    def impression(minutes, impression_id, history, candidates, labels):
        return {
            "impression_id": impression_id,
            "user_id": "u",
            "time": t0 + pd.Timedelta(minutes=minutes),
            "history": np.asarray(history, dtype=object),
            "candidates": np.asarray(candidates, dtype=object),
            "labels": np.asarray(labels, dtype=np.int8),
        }

    splits = {
        "train": [impression(0, "tr0", ["a"], ["a", "b", "c"], [1, 0, 0]),
                  impression(20, "tr1", ["b"], ["a", "b", "d"], [0, 1, 0])],
        "val": [impression(40, "va0", ["a"], ["b", "c", "d"], [0, 0, 1])],
        "test": [impression(60, "te0", ["a", "b"], ["a", "b", "c", "d"], [1, 0, 0, 0]),
                 impression(70, "te1", [], ["c", "d"], [0, 1])],
    }
    for name, rows in splits.items():
        pd.DataFrame(rows).to_parquet(tmp_path / f"behaviors_{name}.parquet", index=False)
    pd.DataFrame({
        "article_id": ["a", "b", "c", "d"],
        "title": ["football match", "election news", "football cup", "weather storm"],
        "abstract": ["", "", "", ""],
        "category": ["sports", "news", "sports", "weather"],
    }).to_parquet(tmp_path / "articles.parquet", index=False)
    return tmp_path


@pytest.mark.parametrize("method", ["popular", "bm25_fresh"])
def test_fresh_generators_honour_the_candidate_contract(tmp_path, method):
    from src import candidates

    processed_dir = _write_processed(tmp_path)
    wide = candidates.GENERATORS[method](processed_dir, "ebnerd", "test", k=3)

    assert list(wide.columns) == ["impression_id", "candidates", "scores", "used_fallback"]
    assert set(wide["impression_id"]) == {"te0", "te1"}
    for cands, scores in zip(wide["candidates"], wide["scores"]):
        assert len(cands) == len(scores) <= 3
        assert len(set(cands)) == len(cands)


def test_bm25_fresh_falls_back_to_popularity_without_history(tmp_path):
    from src import candidates

    processed_dir = _write_processed(tmp_path)
    wide = candidates.GENERATORS["bm25_fresh"](processed_dir, "ebnerd", "test", k=3).set_index("impression_id")
    assert bool(wide.loc["te1", "used_fallback"]) is True   # empty history
    assert bool(wide.loc["te0", "used_fallback"]) is False  # has history, pool >= k


def test_bm25_fresh_is_deterministic(tmp_path):
    """Ties are broken by a seeded draw, so two runs must agree exactly."""
    from src import candidates

    processed_dir = _write_processed(tmp_path)
    a = candidates.GENERATORS["bm25_fresh"](processed_dir, "ebnerd", "test", k=4)
    b = candidates.GENERATORS["bm25_fresh"](processed_dir, "ebnerd", "test", k=4)
    assert [list(c) for c in a["candidates"]] == [list(c) for c in b["candidates"]]


def test_bm25_fresh_ranks_every_scored_article_above_every_tie(tmp_path):
    """The random draw may only reorder ties -- never lift a lower BM25 score."""
    from src import candidates

    processed_dir = _write_processed(tmp_path)
    wide = candidates.GENERATORS["bm25_fresh"](processed_dir, "ebnerd", "test", k=4).set_index("impression_id")
    scores = list(wide.loc["te0", "scores"])  # te0 has history, and its pool holds all 4 articles
    assert scores == sorted(scores, reverse=True)
