"""Q1 feature-engineering tests: formula correctness plus the Q9 anti-gaming
checks -- article-level stats (popularity/CTR/freshness) must be queried
as-of strictly before each impression's own time (never see that impression's
own outcome, or a later train event), and within-session counts must be
strictly backward-looking."""

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
    score, has_ts = recency_weighted_engagement([], None, pd.Timestamp("2024-01-01"))
    assert score == 0.0
    assert has_ts is False


def test_recency_weighted_engagement_uses_real_elapsed_time_not_just_length():
    """The bug this guards against: a positional-only score gives identical
    output for any two same-length histories regardless of how long ago the
    clicks were. With real timestamps, a history clicked minutes ago must
    score higher than an equally-long history clicked weeks ago."""
    now = pd.Timestamp("2024-01-08T00:00:00")
    recent_history = ["a", "b"]
    recent_times = [pd.Timestamp("2024-01-07T23:00:00"), pd.Timestamp("2024-01-07T23:30:00")]
    stale_history = ["a", "b"]
    stale_times = [pd.Timestamp("2024-01-01T00:00:00"), pd.Timestamp("2024-01-01T01:00:00")]

    recent_score, recent_has_ts = recency_weighted_engagement(recent_history, recent_times, now)
    stale_score, stale_has_ts = recency_weighted_engagement(stale_history, stale_times, now)

    assert recent_has_ts and stale_has_ts
    assert recent_score > stale_score  # same length, different recency -> must differ


def test_recency_weighted_engagement_matches_manual_exponential_sum():
    now = pd.Timestamp("2024-01-01T12:00:00")
    history = ["a", "b"]
    times = [pd.Timestamp("2024-01-01T00:00:00"), pd.Timestamp("2024-01-01T06:00:00")]  # 12h, 6h before `now`
    halflife = 12.0
    expected = math.exp(-math.log(2) / halflife * 12) + math.exp(-math.log(2) / halflife * 6)
    score, has_ts = recency_weighted_engagement(history, times, now, halflife_hours=halflife)
    assert has_ts is True
    assert score == pytest.approx(expected)


def test_recency_weighted_engagement_falls_back_when_no_timestamps():
    """MIND has no per-click timestamps; must fall back to the (weaker,
    length-only) positional form and flag it, not silently pretend it's
    time-based."""
    score, has_ts = recency_weighted_engagement(["a", "b", "c"], None, pd.Timestamp("2024-01-01"))
    assert has_ts is False
    assert score == pytest.approx(1.0 + config.RECENCY_DECAY_RATE + config.RECENCY_DECAY_RATE**2)


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
def test_article_stats_asof_end_of_train_matches_manual_train_recount(dataset):
    """Querying as_of() at a time after all of train has happened must
    reproduce the exact whole-train click/display counts -- this is the
    val/test usage path, which should behave like the old static snapshot."""
    d = _require(dataset)
    train = pd.read_parquet(d / "behaviors_train.parquet", columns=["time", "candidates", "labels"])

    expected_clicks, expected_displays = {}, {}
    for candidates, labels in zip(train["candidates"], train["labels"]):
        for article_id, label in zip(candidates, labels):
            expected_displays[article_id] = expected_displays.get(article_id, 0) + 1
            if label == 1:
                expected_clicks[article_id] = expected_clicks.get(article_id, 0) + 1

    index = article_stats.TrainEventIndex(d, dataset)
    after_train = train["time"].max() + pd.Timedelta(days=1)
    for article_id, count in expected_clicks.items():
        assert index.as_of(article_id, after_train)["click_count"] == count
    for article_id, count in expected_displays.items():
        assert index.as_of(article_id, after_train)["display_count"] == count


@pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
def test_article_stats_asof_excludes_the_current_impressions_own_click(dataset):
    """Direct regression test for the training-set leak: an article clicked
    for the first (and only) time in impression I must show click_count=0
    when queried as-of I's own time -- I's own outcome must not appear in
    the feature used to score I."""
    d = _require(dataset)
    train = pd.read_parquet(d / "behaviors_train.parquet", columns=["time", "candidates", "labels"])

    first_click_time = {}
    for t, candidates, labels in zip(train["time"], train["candidates"], train["labels"]):
        for article_id, label in zip(candidates, labels):
            if label == 1 and article_id not in first_click_time:
                first_click_time[article_id] = t

    if not first_click_time:
        pytest.skip("no clicks in train to test against")

    index = article_stats.TrainEventIndex(d, dataset)
    sample_article, sample_time = next(iter(first_click_time.items()))
    stat_as_of_its_own_impression = index.as_of(sample_article, sample_time)
    assert stat_as_of_its_own_impression["click_count"] == 0

    stat_after = index.as_of(sample_article, sample_time + pd.Timedelta(seconds=1))
    assert stat_after["click_count"] == 1


def test_mind_freshness_asof_excludes_val_test_only_articles():
    d = _require("mind")
    train = pd.read_parquet(d / "behaviors_train.parquet", columns=["time", "candidates"])
    val = pd.read_parquet(d / "behaviors_val.parquet", columns=["time", "candidates"])

    train_ids = {a for cands in train["candidates"] for a in cands}
    val_ids = {a for cands in val["candidates"] for a in cands}
    val_only = val_ids - train_ids
    if not val_only:
        pytest.skip("no val-only articles in this build to test against")

    index = article_stats.TrainEventIndex(d, "mind")
    sample = next(iter(val_only))
    val_time = val["time"].max()
    stat = index.as_of(sample, val_time)
    assert stat["has_known_publish_time"] is False
    assert stat["freshness_reference_time"] is None


def test_mind_freshness_asof_future_only_article_is_unknown_not_clamped():
    """Direct regression test for the clamp bug: an article whose only train
    sighting is at time T2 must report has_known_publish_time=False (not a
    clamped freshness=0) when queried as-of an earlier time T1 < T2."""
    d = _require("mind")
    train = pd.read_parquet(d / "behaviors_train.parquet", columns=["time", "candidates"])

    first_seen = {}
    for t, candidates in zip(train["time"], train["candidates"]):
        for article_id in candidates:
            if article_id not in first_seen or t < first_seen[article_id]:
                first_seen[article_id] = t

    # an article seen only in the later half of train
    midpoint = train["time"].min() + (train["time"].max() - train["time"].min()) / 2
    late_only = next((a for a, t in first_seen.items() if t > midpoint), None)
    if late_only is None:
        pytest.skip("no train-only-in-second-half article found")

    index = article_stats.TrainEventIndex(d, "mind")
    stat_before = index.as_of(late_only, train["time"].min())
    assert stat_before["has_known_publish_time"] is False
    assert stat_before["freshness_reference_time"] is None
