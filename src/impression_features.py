"""Per-impression features: everything derivable from one impression row's
own history/session/time columns, independent of which candidate is being
scored. Reused across all of that impression's candidates.

Click-history features (Q1.1): click_count, recency-weighted engagement.
Session features (Q1.2): dwell time, hour/day, impression size, and the
session-position columns sessionize.add_session_context already attached.
Also builds `history_category_weights` -- a recency-decayed category
histogram of the user's click history -- consumed by candidate_features.py
for the per-candidate category-affinity/match features, computed once per
impression rather than once per candidate.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from src import config


def recency_weighted_engagement(
    history: list[str],
    history_impression_time: list | None,
    impression_time,
    halflife_hours: float = config.RECENCY_HALFLIFE_HOURS,
    positional_decay_rate: float = config.RECENCY_DECAY_RATE,
) -> tuple[float, bool]:
    """Sum of exp(-ln2/halflife * hours_since_click) over the history -- an
    actual time-decayed engagement score (requires per-click timestamps,
    EB-NeRD only). Returns (score, has_history_timestamps).

    A positional decay (weight by rank in the list, not real elapsed time)
    applied to a plain count -- the previous implementation -- is a pure
    function of len(history): every history of the same length produces the
    identical score regardless of how long ago those clicks actually were,
    so it carries no recency signal beyond click_count itself. Where
    per-click timestamps aren't available (MIND), we fall back to that
    positional form anyway (weaker, but still a bounded engagement summary)
    and flag it via has_history_timestamps=False so a model/analyst can tell
    the two regimes apart rather than silently trusting a degenerate number.
    """
    n = len(history)
    if n == 0:
        return 0.0, False

    if history_impression_time is not None and len(history_impression_time) == n:
        lam = math.log(2) / halflife_hours
        score = 0.0
        for ts in history_impression_time:
            hours = (impression_time - ts).total_seconds() / 3600.0
            hours = max(hours, 0.0)  # float-precision safety; history is verified to precede impression_time
            score += math.exp(-lam * hours)
        return float(score), True

    return float(sum(positional_decay_rate**i for i in range(n))), False


def history_category_weights(
    history: list[str],
    articles_lookup: dict[str, dict],
    decay_rate: float = config.RECENCY_DECAY_RATE,
) -> dict[str, float]:
    """Recency-decayed category histogram: most recent click contributes
    weight 1 to its category, each click further back contributes less."""
    weights: dict[str, float] = defaultdict(float)
    n = len(history)
    for distance_from_latest, article_id in enumerate(reversed(history)):
        record = articles_lookup.get(article_id)
        if record is None:
            continue
        category = record.get("category")
        if not category:
            continue
        weights[category] += decay_rate**distance_from_latest
    return dict(weights)


def avg_history_dwell_time(history_read_time: list[float] | None) -> tuple[float, bool]:
    """Returns (value, has_dwell_time). value is NaN when unavailable (MIND,
    or an EB-NeRD user with an empty history) so callers don't confuse
    "no data" with "zero seconds of engagement" (see parse.py docstring)."""
    if history_read_time is None or len(history_read_time) == 0:
        return float("nan"), False
    valid = [t for t in history_read_time if t is not None and not (isinstance(t, float) and np.isnan(t))]
    if not valid:
        return float("nan"), False
    return float(np.mean(valid)), True


def build_impression_features(row, articles_lookup: dict[str, dict]) -> dict:
    """`row` is one itertuples() row of a behaviors_{split} DataFrame that
    has already been through sessionize.add_session_context."""
    history = list(row.history) if row.history is not None else []
    history_impression_time = list(row.history_impression_time) if row.history_impression_time is not None else None

    dwell_mean, has_dwell = avg_history_dwell_time(
        list(row.history_read_time) if row.history_read_time is not None else None
    )
    recency_score, has_history_timestamps = recency_weighted_engagement(
        history, history_impression_time, row.time
    )

    return {
        "click_count": len(history),
        "recency_weighted_engagement": recency_score,
        "has_history_timestamps": has_history_timestamps,
        "avg_history_dwell_time": dwell_mean,
        "has_dwell_time": has_dwell,
        "hour_of_day": row.time.hour,
        "day_of_week": row.time.dayofweek,
        "impression_size": len(row.candidates),
        "session_position": int(row.session_position),
        "session_impressions_before": int(row.session_impressions_before),
        "session_clicks_before": int(row.session_clicks_before),
        "_history_category_weights": history_category_weights(history, articles_lookup),
    }
