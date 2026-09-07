"""As-of-time per-article statistics for Q1 article features: popularity,
smoothed CTR, and freshness -- all queried "as of strictly before this
impression's own time", not as a single whole-train snapshot.

Why as-of and not a static snapshot: a static aggregate over the whole of
behaviors_train (the original approach) is safe for val/test rows (all of
train precedes them), but wrong for train rows themselves -- a candidate's
own click inside impression I would be one of the events counted into the
very popularity/CTR figure handed back for scoring I, which is training-set
leakage through the article-level feature rather than the row-level one Q9's
usual checks look for. TrainEventIndex.as_of() answers "what were this
article's stats using only train events strictly before time t", which for
train rows themselves excludes that row's own outcome, and for val/test rows
naturally reduces to the whole-train totals (every train event precedes
every val/test impression) -- so one query path is correct for all splits.

CTR smoothing shrinks toward `global_ctr`, the train split's own empirical
click-through rate (measured, not a hand-picked constant) with a configurable
pseudo-count (config.CTR_PRIOR_STRENGTH) -- a flat prior like 10% is a poor
fit when a dataset's real CTR is ~4% (MIND) or ~9% (EB-NeRD).

Freshness reference: EB-NeRD's real `published_time` is used directly, gated
to still require it to precede the query time (defensive: a retrieval
candidate should never be "not yet published" relative to the impression
being scored). MIND has no publish date at all, so the reference is instead
"first time this article was ever shown as a candidate in train, strictly
before the query time" -- deliberately as-of, not a whole-train minimum, so
an article whose only train sightings are all later than the impression
being scored is correctly reported as unknown (`has_known_publish_time=False`)
rather than silently clamped to "just published".
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict

import pandas as pd

from src import config


class TrainEventIndex:
    def __init__(self, processed_dir, dataset: str):
        self.dataset = dataset
        train = pd.read_parquet(processed_dir / "behaviors_train.parquet", columns=["time", "candidates", "labels"])

        display_times: dict[str, list] = defaultdict(list)
        click_times: dict[str, list] = defaultdict(list)
        total_clicks = 0
        total_displays = 0
        for t, candidates, labels in zip(train["time"], train["candidates"], train["labels"]):
            for article_id, label in zip(candidates, labels):
                display_times[article_id].append(t)
                total_displays += 1
                if label == 1:
                    click_times[article_id].append(t)
                    total_clicks += 1

        for times in display_times.values():
            times.sort()
        for times in click_times.values():
            times.sort()

        self._display_times = display_times
        self._click_times = click_times
        self.global_ctr = (total_clicks / total_displays) if total_displays else 0.0

        articles = pd.read_parquet(processed_dir / "articles.parquet", columns=["article_id", "category"])
        self._category = dict(zip(articles["article_id"], articles["category"]))

        self._published_time: dict[str, pd.Timestamp] | None = None
        if dataset == "ebnerd":
            published = pd.read_parquet(processed_dir / "articles.parquet", columns=["article_id", "published_time"])
            self._published_time = dict(zip(published["article_id"], published["published_time"]))

    def as_of(self, article_id: str, as_of_time) -> dict:
        displays = self._display_times.get(article_id, [])
        clicks = self._click_times.get(article_id, [])
        display_count = bisect_left(displays, as_of_time)  # count strictly before as_of_time
        click_count = bisect_left(clicks, as_of_time)

        ctr = (click_count + config.CTR_PRIOR_STRENGTH * self.global_ctr) / (
            display_count + config.CTR_PRIOR_STRENGTH
        )

        if self._published_time is not None:
            ref = self._published_time.get(article_id)
            has_known = ref is not None and not pd.isna(ref) and ref < as_of_time
            freshness_reference_time = ref if has_known else None
        else:
            has_known = display_count > 0
            freshness_reference_time = displays[0] if has_known else None

        return {
            "category": self._category.get(article_id),
            "click_count": click_count,
            "display_count": display_count,
            "ctr": ctr,
            "log_click_count": math.log1p(click_count),
            "freshness_reference_time": freshness_reference_time,
            "has_known_publish_time": has_known,
        }
