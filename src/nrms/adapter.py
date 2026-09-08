"""Bridge from this repo's unified schema (src/parse.py, src/split.py) to the
shape NRMS wants.

Two schema differences have to be reconciled with ebnerd-benchmark:

  * ids -- theirs are ints, ours are strings; resolved by src/nrms/ids.py.
  * clicks -- they store a list of clicked article ids per impression, we store
    parallel `candidates`/`labels` lists. `clicked` is derived here.

Splits stay OURS. Training reads behaviors_train, validation behaviors_val,
evaluation behaviors_test, all produced by src/split.py's temporal split. The
benchmark instead evaluates on EB-NeRD's official validation period, so our
figures are not directly comparable to their leaderboard -- but our split keeps
the Part 0 temporal discipline and the Q9 no-future-leakage rule, which matters
more for this assignment than leaderboard parity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.nrms.ids import PAD_CODE, ArticleCodec


def truncate_history(codes: np.ndarray, history_size: int) -> np.ndarray:
    """Keep the `history_size` most recent clicks, left-padded with PAD_CODE.

    Our `history` is most-recent-LAST (see src/parse.py), so the recent tail is
    the slice to keep -- the same convention ebnerd-benchmark's
    `truncate_history` uses. Left-padding (rather than right) keeps the most
    recent click at the final position for every user regardless of how much
    history they have.
    """
    codes = np.asarray(codes, dtype=np.int32)
    if len(codes) >= history_size:
        return codes[-history_size:] if history_size else codes[:0]
    out = np.full(history_size, PAD_CODE, dtype=np.int32)
    if len(codes):
        out[-len(codes):] = codes
    return out


def load_impressions(
    processed_dir: Path,
    split: str,
    codec: ArticleCodec,
    history_size: int,
    drop_no_click: bool = True,
) -> pd.DataFrame:
    """One row per impression, with ids encoded and history fixed-width.

    Columns out: impression_id, user_id, time, history (int32[history_size]),
    candidates (int32[n]), labels (int8[n]), clicked (int32[k]).

    Impressions with no click are dropped by default: they yield no training
    example under Wu-2019 sampling, and at evaluation time a group with no
    positive has undefined AUC/MRR/nDCG. Pass drop_no_click=False to inspect
    them.
    """
    behaviors = pd.read_parquet(
        processed_dir / f"behaviors_{split}.parquet",
        columns=["impression_id", "user_id", "time", "history", "candidates", "labels"],
    )

    histories, candidates, labels, clicked = [], [], [], []
    for row in behaviors.itertuples(index=False):
        history_codes = codec.encode(row.history if row.history is not None else [])
        candidate_codes = codec.encode(row.candidates)
        row_labels = np.asarray(row.labels, dtype=np.int8)

        histories.append(truncate_history(history_codes, history_size))
        candidates.append(candidate_codes)
        labels.append(row_labels)
        clicked.append(candidate_codes[row_labels == 1])

    out = pd.DataFrame(
        {
            "impression_id": behaviors["impression_id"].to_numpy(),
            "user_id": behaviors["user_id"].to_numpy(),
            "time": behaviors["time"].to_numpy(),
            "history": histories,
            "candidates": candidates,
            "labels": labels,
            "clicked": clicked,
        }
    )

    if drop_no_click:
        keep = out["clicked"].apply(len) > 0
        out = out[keep].reset_index(drop=True)
    return out


def load_article_text(processed_dir: Path, text_columns) -> pd.DataFrame:
    """article_id + a single concatenated `text` field, mirroring their
    `concat_str_columns`. Missing fields become empty strings so a null body
    (every MIND article) cannot poison the concatenation."""
    articles = pd.read_parquet(processed_dir / "articles.parquet")

    parts = []
    for column in text_columns:
        if column not in articles.columns:
            continue
        parts.append(articles[column].fillna("").astype(str))
    if not parts:
        raise ValueError(f"none of {text_columns} present in {processed_dir/'articles.parquet'}")

    text = parts[0]
    for part in parts[1:]:
        text = text.str.cat(part, sep=" ")

    return pd.DataFrame(
        {"article_id": articles["article_id"].astype(str), "text": text.str.strip()}
    )
