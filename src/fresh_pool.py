"""As-of-time candidate pools over the platform's display log (A2 Q2, stage 1).

Why this exists: A1 retrieves from the WHOLE catalogue by similarity to the
user's history, and recovers the clicked article for only ~2.5-3.9% of
impressions at K=200. Measured on the logs, the clicked article is almost always
one the platform was showing right then: EB-NeRD's median clicked-article age at
click time is 3.1 hours and 92% are clicked within a day of publication, while
the catalogue reaches back to 2000. So candidates have to come from what is in
circulation, not from the archive.

The pool at impression time t is built only from events strictly before t:

  displays  every article shown in any impression with t_row in [t-1h, t).
            Strict `< t`, so impressions sharing t's timestamp never see each
            other, and the impression being scored never sees its own in-view
            list.
  clicks    counted only from impressions in [t-1h, t-lag). A click happens
            *after* its impression, so clicks from impressions a few minutes
            before t may not have happened yet at t; counting them would leak
            future clicks (Q9). Measured cost of a 10-minute lag at @200: 0pp on
            EB-NeRD, ~1pp on MIND.
  backfill  displays over [t-24h, t), used only when the 1h pool is shorter
            than K -- EB-NeRD's is, for 20-59% of impressions depending on split.

All three are things a live system knows at serving time: its own display log
and its own click counters. No label from the impression being scored, or from
any later impression, is ever used.
"""

from __future__ import annotations

import heapq
from collections import defaultdict

import numpy as np
import pandas as pd

from src import config

_NS_PER_MINUTE = 60 * 10**9
_NS_PER_HOUR = 60 * _NS_PER_MINUTE


class _Window:
    """Article counts over log rows with lo <= t_row < hi, advanced
    monotonically with two pointers -- O(total events) for a whole pass rather
    than rescanning the window per query. `clicks=True` counts clicked articles
    only; otherwise every displayed article."""

    def __init__(self, times, candidates, labels, clicks: bool):
        self._t = times
        self._c = candidates
        self._l = labels
        self._clicks = clicks
        self._add = 0
        self._rm = 0
        self.counts: dict[str, int] = defaultdict(int)

    def _articles(self, i):
        if not self._clicks:
            return self._c[i]
        return [a for a, y in zip(self._c[i], self._l[i]) if y == 1]

    def advance(self, lo: int, hi: int) -> None:
        n = len(self._t)
        while self._add < n and self._t[self._add] < hi:
            for a in self._articles(self._add):
                self.counts[a] += 1
            self._add += 1
        while self._rm < self._add and self._t[self._rm] < lo:
            for a in self._articles(self._rm):
                self.counts[a] -= 1
                if self.counts[a] == 0:
                    del self.counts[a]
            self._rm += 1


class FreshPool:
    """Answers "what was in circulation just before t" for impressions visited in
    non-decreasing time order.

    Build it over the log from `load_log`, which spans every split: the log is
    continuous across the temporal split, so the first val impressions' pools
    legitimately include the last train impressions' displays.
    """

    def __init__(
        self,
        log: pd.DataFrame,
        window_hours: float | None = None,
        backfill_hours: float | None = None,
        lag_minutes: float | None = None,
    ):
        window_hours = config.FRESH_WINDOW_HOURS if window_hours is None else window_hours
        backfill_hours = config.FRESH_BACKFILL_HOURS if backfill_hours is None else backfill_hours
        lag_minutes = config.CLICK_REPORTING_LAG_MINUTES if lag_minutes is None else lag_minutes

        times = log["t"].to_numpy(dtype=np.int64)
        if len(times) > 1 and np.any(np.diff(times) < 0):
            # Sorting here would silently desynchronise from the caller's own
            # iteration order, so refuse instead.
            raise ValueError("FreshPool needs the log sorted by time")
        candidates = log["candidates"].to_numpy()
        labels = log["labels"].to_numpy()

        self._window = int(window_hours * _NS_PER_HOUR)
        self._backfill_span = int(backfill_hours * _NS_PER_HOUR)
        self._lag = int(lag_minutes * _NS_PER_MINUTE)
        self._displays = _Window(times, candidates, labels, clicks=False)
        self._backfill = _Window(times, candidates, labels, clicks=False)
        self._clicks = _Window(times, candidates, labels, clicks=True)
        self._now: int | None = None

    def advance_to(self, t) -> None:
        t = int(t)
        if self._now is not None and t < self._now:
            raise ValueError("FreshPool must be queried in non-decreasing time order")
        self._now = t
        self._displays.advance(t - self._window, t)
        self._backfill.advance(t - self._backfill_span, t)
        self._clicks.advance(t - self._window, t - self._lag)

    @property
    def pool(self) -> dict[str, int]:
        """Articles displayed in the last window, with their display counts."""
        return self._displays.counts

    @property
    def click_counts(self) -> dict[str, int]:
        """Lagged click counts over the same window."""
        return self._clicks.counts

    @property
    def backfill_counts(self) -> dict[str, int]:
        return self._backfill.counts

    def popular(self, k: int) -> tuple[list[tuple[str, float]], bool]:
        """Top-k of the pool by (lagged clicks, displays), backfilled to k.
        Scores are the lagged click counts; returns (ranked, used_backfill)."""
        pool, clicks = self.pool, self.click_counts
        top = heapq.nlargest(k, pool, key=lambda a: (clicks.get(a, 0), pool[a]))
        return self.backfill([(a, float(clicks.get(a, 0))) for a in top], k)

    def backfill(self, ranked: list[tuple[str, float]], k: int) -> tuple[list[tuple[str, float]], bool]:
        """Pad `ranked` to k from the 24h window, most-displayed first, with score
        0.0 -- below any genuine hit, as A1's popularity padding does. A pool
        that is empty even at 24h (the first minutes of the dataset) stays short
        rather than inventing candidates."""
        if len(ranked) >= k:
            return ranked[:k], False
        seen = {a for a, _ in ranked}
        back = self.backfill_counts
        extra = heapq.nlargest(k - len(ranked), (a for a in back if a not in seen), key=back.__getitem__)
        return ranked + [(a, 0.0) for a in extra], bool(extra)


def load_log(processed_dir, split: str, with_history: bool = False, backfill_hours: float | None = None) -> pd.DataFrame:
    """Every split's impressions in time order, trimmed to what `split` needs:
    from `backfill_hours` before its first impression to its last.

    Columns: t (int64 ns), candidates, labels, is_target, and for the target
    split's rows impression_id (plus history when `with_history`). Other splits'
    rows are only there to feed the windows.
    """
    backfill_hours = config.FRESH_BACKFILL_HOURS if backfill_hours is None else backfill_hours
    frames = []
    for s in ("train", "val", "test"):
        columns = ["time", "candidates", "labels"]
        if s == split:
            columns = ["impression_id", *columns, *(["history"] if with_history else [])]
        frame = pd.read_parquet(processed_dir / f"behaviors_{s}.parquet", columns=columns)
        frame["is_target"] = s == split
        frames.append(frame)
    log = pd.concat(frames, ignore_index=True)
    log["t"] = log["time"].to_numpy().astype("datetime64[ns]").astype(np.int64)

    target_t = log.loc[log["is_target"], "t"]
    if target_t.empty:
        return log.iloc[0:0]
    lo = int(target_t.min()) - int(backfill_hours * _NS_PER_HOUR)
    log = log[(log["t"] >= lo) & (log["t"] <= int(target_t.max()))]
    return log.sort_values("t", kind="stable").reset_index(drop=True)
