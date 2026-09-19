"""Top-level Q1/Q2 feature-matrix builder: joins a retriever's top-K output
with the impression-level (impression_features.py) and candidate-level
(candidate_features.py) feature builders, and derives training labels from
ground-truth clicks.

Candidate contract (produced by src/candidates.py -- any stage-1 method; this
module doesn't care which):
    impression_id: str
    article_id: str
    rank: int   (1-indexed position within that impression's retrieved list)
    retrieval_score: float  (stage 1's raw score; scale is method-specific)

Label: 1 if article_id is in the impression's true clicked set (from
behaviors_{split}'s own candidates/labels where label == 1), else 0 --
regardless of whether the retrieved candidate was ever shown in the
platform's own impression. The bm25/semantic arms pool the WHOLE catalog, and
even the fresh pool is broader than one impression's in-view list, so most
retrieved candidates legitimately get label 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import article_stats, config, history_content, sessionize
from src.candidate_features import build_candidate_features
from src.impression_features import build_impression_features

# Rows accumulate as dicts, ~1.8 KB each, while the finished frame is ~195 B/row.
# Flushing periodically bounds that transient: at the scale fresh-pool
# candidates produce (EB-NeRD train alone is ~4.5M rows) holding every dict at
# once would need ~8 GB.
_FLUSH_ROWS = 250_000


def load_articles_lookup(processed_dir) -> dict[str, dict]:
    articles = pd.read_parquet(processed_dir / "articles.parquet", columns=["article_id", "category"])
    return {row.article_id: {"category": row.category} for row in articles.itertuples(index=False)}


def history_content_columns(processed_dir, dataset: str, candidates_df: pd.DataFrame, histories: dict):
    """Both history_content features for every candidate row, as arrays aligned
    to candidates_df's row positions -- 16 bytes a row rather than a dict, which
    matters at MIND's 2.7M training rows. One scorer call per impression."""
    scorer = history_content.get_scorer(processed_dir, dataset)
    title = np.full(len(candidates_df), np.nan)
    cosine = np.full(len(candidates_df), np.nan)
    article_ids = candidates_df["article_id"].to_numpy()
    for impression_id, positions in candidates_df.groupby("impression_id", sort=False).indices.items():
        if impression_id in histories:
            title[positions], cosine[positions] = scorer.score(
                histories[impression_id], article_ids[positions].tolist()
            )
    return title, cosine


def build_feature_matrix(
    processed_dir,
    dataset: str,
    split: str,
    candidates_df: pd.DataFrame,
    article_index: article_stats.TrainEventIndex | None = None,
) -> pd.DataFrame:
    """`article_index` may be built once by the caller and passed in when
    building features for multiple splits back to back (train/val/test) --
    it depends only on (processed_dir, dataset), not `split`, and rebuilding
    it per call means re-scanning the whole of behaviors_train every time."""
    behaviors = pd.read_parquet(processed_dir / f"behaviors_{split}.parquet")
    # Must sessionize the WHOLE split before filtering: an impression's
    # backward-looking session counts depend on knowing every earlier
    # impression in its session, so filtering to candidates_df's impressions
    # first would silently undercount whenever an earlier same-session
    # impression isn't itself one of the candidates being scored.
    behaviors = sessionize.add_session_context(behaviors, dataset)

    # Impression-level features have no cross-row dependency once session
    # context is attached, so -- unlike sessionize above -- this only needs
    # to run for impressions candidates_df actually references, not the
    # whole split (e.g. a dev-scale candidate subsample shouldn't pay for
    # building features over impressions nothing will ever look up).
    needed_impressions = set(candidates_df["impression_id"])
    behaviors = behaviors[behaviors["impression_id"].isin(needed_impressions)]

    articles_lookup = load_articles_lookup(processed_dir)
    if article_index is None:
        # One as-of-time index for ALL splits (train/val/test) -- for val/test
        # it naturally reduces to the whole-train totals since every train
        # event precedes them; for train rows themselves it excludes each
        # row's own (and any later) outcome. See article_stats.py's docstring.
        article_index = article_stats.TrainEventIndex(processed_dir, dataset)

    true_clicks: dict[str, set[str]] = {}
    impression_feats: dict[str, dict] = {}
    impression_time: dict[str, object] = {}
    histories: dict[str, object] = {}
    for row in behaviors.itertuples(index=False):
        true_clicks[row.impression_id] = {c for c, l in zip(row.candidates, row.labels) if l == 1}
        impression_feats[row.impression_id] = build_impression_features(row, articles_lookup)
        impression_time[row.impression_id] = row.time
        histories[row.impression_id] = row.history

    content = (
        history_content_columns(processed_dir, dataset, candidates_df, histories)
        if config.HISTORY_CONTENT_FEATURES else None
    )

    frames: list[pd.DataFrame] = []
    rows = []
    for position, cand in enumerate(candidates_df.itertuples(index=False)):
        impr_id = cand.impression_id
        impr_feat = impression_feats.get(impr_id)
        if impr_feat is None:
            continue  # candidate for an impression outside this split's behaviors table

        cand_feat = build_candidate_features(
            cand.article_id,
            int(cand.rank),
            float(cand.retrieval_score),
            impression_time[impr_id],
            impr_feat["_history_category_weights"],
            article_index,
        )

        row_out = {"impression_id": impr_id, "article_id": cand.article_id}
        row_out.update({k: v for k, v in impr_feat.items() if not k.startswith("_")})
        row_out.update(cand_feat)
        if content is not None:
            row_out["history_title_bm25"] = content[0][position]
            row_out["history_embedding_cosine"] = content[1][position]
        row_out["label"] = int(cand.article_id in true_clicks[impr_id])
        rows.append(row_out)
        if len(rows) >= _FLUSH_ROWS:
            frames.append(pd.DataFrame(rows))
            rows = []

    if rows or not frames:
        frames.append(pd.DataFrame(rows))
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
