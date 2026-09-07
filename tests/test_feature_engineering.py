"""Q1 feature-engineering tests: formula correctness plus the Q9 anti-gaming
checks -- article-level stats (popularity/CTR/freshness) must be strictly
train-only, and within-session counts must be strictly backward-looking
(never count the current impression or a later one in the same session)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src import article_stats, config, sessionize
from src.candidate_features import category_features, position_bias
from src.impression_features import recency_weighted_engagement

PROCESSED_DIRS = {"mind": config.MIND_PROCESSED_DIR, "ebnerd": config.EBNERD_PROCESSED_DIR}


def _require(dataset: str):
    d = PROCESSED_DIRS[dataset]
    if not d.exists():
        pytest.skip(f"{dataset} not built yet")
    return d


# --------------------------------------------------------------------- formulas

def test_position_bias_decreases_with_rank():
    values = [position_bias(r) for r in range(1, 6)]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(1.0 / math.log(3))


def test_recency_weighted_engagement_empty_history():
    assert recency_weighted_engagement([]) == 0.0


def test_recency_weighted_engagement_matches_manual_sum():
    history = ["a", "b", "c"]
    decay = 0.5
    expected = 1.0 + 0.5 + 0.25  # most recent (c) = weight 1, then b, then a
    assert recency_weighted_engagement(history, decay_rate=decay) == pytest.approx(expected)


def test_category_features_no_history():
    out = category_features("news", {})
    assert out == {"category_binary_match": 0, "category_affinity": 0.0}


def test_category_features_match_and_affinity():
    weights = {"sports": 2.0, "news": 1.0}
    out = category_features("sports", weights)
    assert out["category_binary_match"] == 1
    assert out["category_affinity"] == pytest.approx(2.0 / 3.0)


def test_category_features_no_match():
    weights = {"sports": 2.0}
    out = category_features("politics", weights)
    assert out["category_binary_match"] == 0
    assert out["category_affinity"] == 0.0


# ------------------------------------------------------------- session boundary (Q9)

def _make_behaviors(rows: list[dict]) -> pd.DataFrame:
    """rows: list of {user_id, session_id, time, n_clicks} -- builds a
    minimal behaviors-shaped frame (candidates/labels only long enough to
    encode n_clicks) for sessionize.add_session_context."""
    out = []
    for r in rows:
        candidates = [f"art{i}" for i in range(r["n_clicks"] + 1)]
        labels = [1] * r["n_clicks"] + [0]
        out.append({
            "user_id": r["user_id"], "session_id": r["session_id"], "time": pd.Timestamp(r["time"]),
            "candidates": candidates, "labels": labels,
        })
    return pd.DataFrame(out)


def test_session_counts_are_strictly_backward_looking():
    behaviors = _make_behaviors([
        {"user_id": "u1", "session_id": "s1", "time": "2024-01-01T10:00:00", "n_clicks": 1},
        {"user_id": "u1", "session_id": "s1", "time": "2024-01-01T10:05:00", "n_clicks": 2},
        {"user_id": "u1", "session_id": "s1", "time": "2024-01-01T10:10:00", "n_clicks": 3},
    ])
    out = sessionize.add_session_context(behaviors, dataset="ebnerd")
    out = out.sort_values("time").reset_index(drop=True)

    # first impression in the session: nothing has happened before it yet
    assert out.loc[0, "session_position"] == 1
    assert out.loc[0, "session_impressions_before"] == 0
    assert out.loc[0, "session_clicks_before"] == 0

    # second impression: exactly the first impression's 1 click precedes it
    assert out.loc[1, "session_impressions_before"] == 1
    assert out.loc[1, "session_clicks_before"] == 1

    # third impression: first two impressions' clicks (1 + 2 = 3) precede it,
    # and its OWN 3 clicks must not be counted in its own "before" total
    assert out.loc[2, "session_impressions_before"] == 2
    assert out.loc[2, "session_clicks_before"] == 3


def test_mind_sessionize_splits_on_gap():
    behaviors = _make_behaviors([
        {"user_id": "u1", "session_id": None, "time": "2024-01-01T10:00:00", "n_clicks": 0},
        {"user_id": "u1", "session_id": None, "time": "2024-01-01T10:05:00", "n_clicks": 0},  # same session
        {"user_id": "u1", "session_id": None, "time": "2024-01-01T12:00:00", "n_clicks": 0},  # gap > 30min -> new session
    ])
    out = sessionize.add_session_context(behaviors, dataset="mind").sort_values("time").reset_index(drop=True)
    assert out.loc[0, "session_id"] == out.loc[1, "session_id"]
    assert out.loc[1, "session_id"] != out.loc[2, "session_id"]
    assert out.loc[2, "session_impressions_before"] == 0  # new session resets the count


# -------------------------------------------------------- article-stats leakage (Q9)

@pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
def test_article_stats_click_and_display_counts_are_train_only(dataset):
    d = _require(dataset)
    train = pd.read_parquet(d / "behaviors_train.parquet", columns=["candidates", "labels"])

    expected_clicks, expected_displays = {}, {}
    for candidates, labels in zip(train["candidates"], train["labels"]):
        for article_id, label in zip(candidates, labels):
            expected_displays[article_id] = expected_displays.get(article_id, 0) + 1
            if label == 1:
                expected_clicks[article_id] = expected_clicks.get(article_id, 0) + 1

    stats = article_stats.build_article_stats(d, dataset).set_index("article_id")
    for article_id, count in expected_clicks.items():
        assert stats.loc[article_id, "click_count"] == count
    for article_id, count in expected_displays.items():
        assert stats.loc[article_id, "display_count"] == count


def test_mind_freshness_proxy_excludes_val_test_only_articles():
    d = _require("mind")
    train = pd.read_parquet(d / "behaviors_train.parquet", columns=["candidates"])
    val = pd.read_parquet(d / "behaviors_val.parquet", columns=["candidates"])

    train_ids = {a for cands in train["candidates"] for a in cands}
    val_ids = {a for cands in val["candidates"] for a in cands}
    val_only = val_ids - train_ids
    if not val_only:
        pytest.skip("no val-only articles in this build to test against")

    stats = article_stats.build_article_stats(d, "mind").set_index("article_id")
    sample = next(iter(val_only))
    assert stats.loc[sample, "has_known_publish_time"] == False  # noqa: E712
    assert pd.isna(stats.loc[sample, "freshness_reference_time"])
