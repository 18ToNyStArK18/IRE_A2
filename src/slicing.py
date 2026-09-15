"""Evaluation slices for A2 Q5: cold-start vs warm users, head vs tail articles.

Ported from A1's module of the same name.

Both slice *impressions*, not candidates, because every metric in
`src/metrics.py` is computed per impression and then averaged -- so a slice has
to partition the same unit the metric aggregates over, or the slice means will
not reconcile with the overall mean.
"""

from __future__ import annotations

COLD_START_THRESHOLD = 5  # <= this many historical clicks => cold-start
HEAD_FRACTION = 0.2  # top 20% of articles by train click count => head


def cold_start_or_warm(history_length: int, threshold: int = COLD_START_THRESHOLD) -> str:
    return "cold_start" if history_length <= threshold else "warm"


def build_head_article_set(
    popularity_counts: dict[str, int], head_fraction: float = HEAD_FRACTION
) -> set[str]:
    """The top `head_fraction` of articles by train click count.

    Articles absent from popularity_counts have no train clicks and are always
    tail -- which is the honest reading: an article nobody clicked in train is
    not in the head by any definition.
    """
    ranked = sorted(popularity_counts.items(), key=lambda kv: kv[1], reverse=True)
    n_head = max(1, int(len(ranked) * head_fraction)) if ranked else 0
    return {article_id for article_id, _ in ranked[:n_head]}


def head_or_tail(article_id: str, head_article_set: set[str]) -> str:
    return "head" if article_id in head_article_set else "tail"
