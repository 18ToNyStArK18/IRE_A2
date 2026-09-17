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

Event scope (`splits`): by default only behaviors_train feeds the index -- the
Q1 semantics, under which val/test queries see frozen end-of-train totals. That
is leak-free, but it is a train/serve skew: train rows get live, accumulating
counts while val/test rows get a frozen snapshot, in which the fresh articles
actually clicked at test time all read zero. Measured on EB-NeRD with the
re-ranker over `popular` candidates: clicked articles' median click count is 16
on train and 0 on val and test (99.2% of clicked test articles read exactly
zero), and the model -- having learned "many clicks -> clicked" -- fell 18-21%
below stage 1 on val/test. Passing splits=("train", "val", "test") makes every
row's stats as-of its own time over every earlier event, which is what a live
system's counters hold. It adds no future information, because as_of() only
ever counts events strictly before t. `click_lag_minutes` applies the fresh
pool's click reporting lag, since a click lands after its impression. The CTR
prior stays train-only either way, so no global constant carries later-period
information.
"""

from __future__ import annotations

import array
import math
from collections import defaultdict

import numpy as np
import pandas as pd

from src import config

# Event times are stored as int64 nanoseconds-since-epoch rather than as
# pd.Timestamp objects: MIND-small's train split alone has 5.84M (article,
# display) pairs, and one Python object per event costs ~0.5-1 GB resident,
# which does not survive MINDlarge. Raw int64 is 8 bytes/event (~49 MB for the
# same split) and np.searchsorted over a sorted int64 array is exactly
# bisect_left's "count of events strictly before t".
_EMPTY_TIMES = np.empty(0, dtype=np.int64)


def freshness_reference_times(processed_dir, dataset: str, splits=("train",)) -> dict[str, pd.Timestamp]:
    """Per-article freshness reference, by the same definition
    TrainEventIndex.as_of applies: EB-NeRD's real `published_time`, and for MIND
    -- which ships no publish date -- the earliest time the article was shown as
    a candidate anywhere in `splits`.

    Returned UNGATED. A reference only becomes usable for a particular
    impression once it precedes that impression's own time; as_of() applies that
    gate internally, so callers of this function must apply it themselves.
    tests/test_freshness.py asserts the two definitions agree.

    Separate from TrainEventIndex because the NRMS freshness arm needs one
    reference per catalogue article, not a per-query lookup, and building the
    full sorted event index to read a single value per article would scan 5.84M
    events on MIND for nothing.
    """
    if dataset == "ebnerd":
        articles = pd.read_parquet(
            processed_dir / "articles.parquet", columns=["article_id", "published_time"]
        )
        return {
            article_id: published
            for article_id, published in zip(articles["article_id"], articles["published_time"])
            if not pd.isna(published)
        }

    return first_seen_times(processed_dir, splits)


def first_seen_times(processed_dir, splits) -> dict[str, pd.Timestamp]:
    """Earliest time each article was shown as a candidate in `splits`, for any
    dataset. MIND's freshness proxy; also applied to EB-NeRD by
    scripts/age_signal.py to price that proxy where real publish times exist."""
    first_seen: dict[str, pd.Timestamp] = {}
    for split in splits:
        frame = pd.read_parquet(
            processed_dir / f"behaviors_{split}.parquet", columns=["time", "candidates"]
        )
        for t, candidates in zip(frame["time"], frame["candidates"]):
            for article_id in candidates:
                previous = first_seen.get(article_id)
                if previous is None or t < previous:
                    first_seen[article_id] = t
    return first_seen


class TrainEventIndex:
    def __init__(self, processed_dir, dataset: str, splits=("train",), click_lag_minutes: float = 0.0):
        """`splits` chooses which behaviour logs feed the index and
        `click_lag_minutes` delays when a click becomes countable -- see the
        module docstring. The defaults reproduce Q1's train-only, no-lag
        semantics exactly."""
        self.dataset = dataset
        self.splits = tuple(splits)
        self._click_lag_ns = int(click_lag_minutes * 60 * 10**9)

        # array.array("q") holds raw int64s, so accumulating an event retains no
        # Python object -- see the _EMPTY_TIMES note above on why that matters.
        display_times: dict[str, array.array] = defaultdict(lambda: array.array("q"))
        click_times: dict[str, array.array] = defaultdict(lambda: array.array("q"))
        total_clicks = 0
        total_displays = 0
        for split in self.splits:
            frame = pd.read_parquet(
                processed_dir / f"behaviors_{split}.parquet", columns=["time", "candidates", "labels"]
            )
            counts_toward_prior = split == "train" or "train" not in self.splits
            times_ns = frame["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            for t_ns, candidates, labels in zip(times_ns, frame["candidates"], frame["labels"]):
                t_ns = int(t_ns)
                for article_id, label in zip(candidates, labels):
                    display_times[article_id].append(t_ns)
                    if counts_toward_prior:
                        total_displays += 1
                    if label == 1:
                        click_times[article_id].append(t_ns)
                        if counts_toward_prior:
                            total_clicks += 1

        # behaviors_train is already time-ordered (split.py sorts before
        # slicing), so these sorts are cheap -- kept so as_of() cannot silently
        # go wrong if that ever stops holding.
        self._display_times = {a: np.sort(np.asarray(b, dtype=np.int64)) for a, b in display_times.items()}
        self._click_times = {a: np.sort(np.asarray(b, dtype=np.int64)) for a, b in click_times.items()}
        self.global_ctr = (total_clicks / total_displays) if total_displays else 0.0

        articles = pd.read_parquet(processed_dir / "articles.parquet", columns=["article_id", "category"])
        self._category = dict(zip(articles["article_id"], articles["category"]))

        self._published_time: dict[str, pd.Timestamp] | None = None
        if dataset == "ebnerd":
            published = pd.read_parquet(processed_dir / "articles.parquet", columns=["article_id", "published_time"])
            self._published_time = dict(zip(published["article_id"], published["published_time"]))

    def as_of(self, article_id: str, as_of_time) -> dict:
        as_of_ts = pd.Timestamp(as_of_time)
        as_of_ns = as_of_ts.value
        displays = self._display_times.get(article_id, _EMPTY_TIMES)
        clicks = self._click_times.get(article_id, _EMPTY_TIMES)
        # side="left" == bisect_left: count of events strictly before as_of_time,
        # so an impression's own click never enters its own feature.
        display_count = int(np.searchsorted(displays, as_of_ns, side="left"))
        # Clicks additionally wait out the reporting lag: a click from an
        # impression within the lag of t may not have happened yet at t.
        click_count = int(np.searchsorted(clicks, as_of_ns - self._click_lag_ns, side="left"))

        ctr = (click_count + config.CTR_PRIOR_STRENGTH * self.global_ctr) / (
            display_count + config.CTR_PRIOR_STRENGTH
        )

        if self._published_time is not None:
            ref = self._published_time.get(article_id)
            has_known = ref is not None and not pd.isna(ref) and ref < as_of_ts
            freshness_reference_time = ref if has_known else None
        else:
            has_known = display_count > 0
            # display_count > 0 guarantees displays[0] < as_of_time, so the
            # reference can never be in the future relative to the query.
            freshness_reference_time = pd.Timestamp(int(displays[0]), unit="ns") if has_known else None

        return {
            "category": self._category.get(article_id),
            "click_count": click_count,
            "display_count": display_count,
            "ctr": ctr,
            "log_click_count": math.log1p(click_count),
            "freshness_reference_time": freshness_reference_time,
            "has_known_publish_time": has_known,
        }
