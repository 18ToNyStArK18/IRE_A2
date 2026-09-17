"""Per-(impression, candidate) features: position bias (from the candidate's
retrieval rank), freshness, popularity/CTR, and category affinity/match --
everything that differs from one candidate to the next within the same
impression. Combined with impression_features.py's impression-level dict to
form one training row per candidate (see feature_pipeline.py).

Popularity/CTR/freshness come from an `article_stats.TrainEventIndex.as_of()`
query (as-of the impression's own time), not a static whole-train snapshot --
see article_stats.py's docstring for why that distinction matters for
train-split leakage. Because as_of() already guarantees
`freshness_reference_time < as_of_time` whenever `has_known_publish_time` is
True, freshness_features() below no longer needs to detect/clamp a
"reference is in the future" case itself -- that's resolved at the source.
"""

from __future__ import annotations

import math

import numpy as np

from src import config


def position_bias(rank: int, log_base: str = config.POSITION_BIAS_LOG_BASE) -> float:
    """1 / log(rank + 2). `rank` is the candidate's 1-indexed position in the
    stage-1 top-K ranking (whichever candidates.py method produced it) -- not a
    position in the original impression's display order, which most retrieved
    candidates never had."""
    denom = math.log2(rank + 2) if log_base == "2" else math.log(rank + 2)
    return 1.0 / denom


def freshness_features(impression_time, article_stat: dict) -> dict:
    if not article_stat["has_known_publish_time"]:
        return {"freshness_log_hours": float("nan"), "has_known_publish_time": False}
    delta_hours = (impression_time - article_stat["freshness_reference_time"]).total_seconds() / 3600.0
    delta_hours = max(delta_hours, 0.0)  # float-precision safety only; as_of() already guarantees delta > 0
    return {"freshness_log_hours": float(np.log1p(delta_hours)), "has_known_publish_time": True}


def category_features(candidate_category: str | None, history_category_weights: dict[str, float]) -> dict:
    if not candidate_category or not history_category_weights:
        return {"category_binary_match": 0, "category_affinity": 0.0}
    total = sum(history_category_weights.values())
    weight = history_category_weights.get(candidate_category, 0.0)
    return {
        "category_binary_match": int(weight > 0),
        "category_affinity": (weight / total) if total > 0 else 0.0,
    }


def build_candidate_features(
    article_id: str,
    rank: int,
    retrieval_score: float,
    impression_time,
    history_category_weights: dict[str, float],
    article_index,  # article_stats.TrainEventIndex
) -> dict:
    stat = article_index.as_of(article_id, impression_time)

    features = {
        "position_bias": position_bias(rank),
        "retrieval_rank": rank,
        # The retriever's raw score, kept alongside its rank because rank is a
        # monotone *discretisation* of it and throws away the margin: a top hit
        # scoring 495 against a runner-up at 390 and one scoring 495 against 494
        # are both just "rank 1 vs rank 2". Stage 1's ordering is exactly what
        # the re-ranker is trying to beat, so how confidently it made each call
        # is signal worth keeping.
        #
        # Scale is retriever-specific (BM25 is unbounded and ran 30-1915 on
        # EB-NeRD val; cosine is [-1, 1]), so it is only comparable within one
        # method. Train one model per candidate source, or normalise per
        # impression first, before mixing them.
        "retrieval_score": float(retrieval_score),
        "click_count_article": stat["click_count"],
        "display_count_article": stat["display_count"],
        "log_click_count_article": stat["log_click_count"],
        "ctr_article": stat["ctr"],
    }
    features.update(freshness_features(impression_time, stat))
    features.update(category_features(stat["category"], history_category_weights))
    return features
