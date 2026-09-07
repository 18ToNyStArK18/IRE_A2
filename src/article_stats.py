"""Precomputed, train-only per-article statistics for Q1 article features:
popularity (click_count/display_count from popularity.parquet), a smoothed
CTR, and a freshness reference time.

Everything here is scoped to behaviors_train only -- same discipline as
popularity.py -- since these are exactly the aggregate, accumulating-over-time
signals A1's leakage_ablation.py demonstrated can inflate metrics if computed
over val/test too (Q9).

Freshness reference:
  - EB-NeRD ships a real `published_time` per article in articles.parquet;
    used directly.
  - MIND ships no publish date at all. We proxy it with the earliest time the
    article was ever shown as a candidate in behaviors_train -- necessarily a
    later time than the true publish date (this is a known, documented
    over-estimate, not a leakage risk: the proxy uses only train-period
    sightings). Articles never shown in train (val/test-only, i.e. genuinely
    cold-start at train time) get no proxy -- `has_known_publish_time=False`
    rather than a fabricated value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def _mind_first_seen_in_train(processed_dir) -> dict[str, pd.Timestamp]:
    train = pd.read_parquet(processed_dir / "behaviors_train.parquet", columns=["time", "candidates"])
    first_seen: dict[str, pd.Timestamp] = {}
    for t, candidates in zip(train["time"], train["candidates"]):
        for article_id in candidates:
            prev = first_seen.get(article_id)
            if prev is None or t < prev:
                first_seen[article_id] = t
    return first_seen


def build_article_stats(processed_dir, dataset: str) -> pd.DataFrame:
    articles = pd.read_parquet(processed_dir / "articles.parquet", columns=["article_id", "category"])
    pop = pd.read_parquet(processed_dir / "popularity.parquet")  # train-only by construction

    stats = articles.merge(pop, on="article_id", how="left")
    stats["click_count"] = stats["click_count"].fillna(0).astype(int)
    stats["display_count"] = stats["display_count"].fillna(0).astype(int)
    stats["ctr"] = (stats["click_count"] + config.CTR_ALPHA) / (stats["display_count"] + config.CTR_BETA)
    stats["log_click_count"] = np.log1p(stats["click_count"])

    if dataset == "ebnerd":
        published = pd.read_parquet(processed_dir / "articles.parquet", columns=["article_id", "published_time"])
        stats = stats.merge(published, on="article_id", how="left")
        stats["freshness_reference_time"] = stats["published_time"]
        stats = stats.drop(columns=["published_time"])
    else:
        first_seen = _mind_first_seen_in_train(processed_dir)
        stats["freshness_reference_time"] = stats["article_id"].map(first_seen)

    stats["has_known_publish_time"] = stats["freshness_reference_time"].notna()
    return stats[[
        "article_id", "category", "click_count", "display_count", "ctr",
        "log_click_count", "freshness_reference_time", "has_known_publish_time",
    ]]


def build_article_stats_lookup(processed_dir, dataset: str) -> dict[str, dict]:
    stats = build_article_stats(processed_dir, dataset)
    return {row.article_id: row._asdict() for row in stats.itertuples(index=False)}
