"""Session context for Q1's "within-session click patterns" / "session length".

EB-NeRD ships a real `session_id` per impression (verified against the raw
schema). MIND has no session concept at all, so we construct a proxy: sort
each user's impressions by time and start a new session whenever the gap
since their previous impression exceeds SESSION_GAP_MINUTES -- a standard
web-analytics idle-timeout heuristic. This is a documented limitation (MIND's
"sessions" are inferred, EB-NeRD's are ground truth), not an attempt to
recover the platform's real session boundaries.

All within-session counts computed here are strictly backward-looking (count
of *prior* impressions/clicks in the same session, before the current row) --
never a forward-looking "total session length" including impressions that
haven't happened yet relative to the current one, which would leak future
behaviour into a feature.
"""

from __future__ import annotations

import pandas as pd

from src import config


def _assign_mind_sessions(behaviors: pd.DataFrame, gap_minutes: int) -> pd.Series:
    ordered = behaviors.sort_values(["user_id", "time"])
    gap = ordered.groupby("user_id")["time"].diff()
    new_session = gap.isna() | (gap > pd.Timedelta(minutes=gap_minutes))
    session_seq = new_session.groupby(ordered["user_id"]).cumsum().astype(int)
    session_id = ordered["user_id"] + "_s" + session_seq.astype(str)
    return session_id.reindex(behaviors.index)


def add_session_context(behaviors: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Returns a copy of `behaviors` with `session_id` filled in (MIND) and
    three new backward-looking columns: `session_position` (1-indexed rank
    of this impression within its session, by time), `session_impressions_before`,
    `session_clicks_before`."""
    out = behaviors.copy()

    if dataset == "mind":
        out["session_id"] = _assign_mind_sessions(out, config.SESSION_GAP_MINUTES)

    out["_n_clicks_this_row"] = out["labels"].apply(lambda labels: sum(1 for l in labels if l == 1))

    ordered = out.sort_values(["session_id", "time"])
    grp = ordered.groupby("session_id")
    ordered["session_position"] = grp.cumcount() + 1
    ordered["session_impressions_before"] = grp.cumcount()
    ordered["session_clicks_before"] = (
        grp["_n_clicks_this_row"].cumsum() - ordered["_n_clicks_this_row"]
    )

    out = ordered.reindex(behaviors.index)
    return out.drop(columns=["_n_clicks_this_row"])
