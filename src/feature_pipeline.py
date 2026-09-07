"""Top-level Q1/Q2 feature-matrix builder: joins a retriever's top-K output
with the impression-level (impression_features.py) and candidate-level
(candidate_features.py) feature builders, and derives training labels from
ground-truth clicks.

Candidate contract (from A1 -- whichever retriever ends up wired in, BM25 or
embedding-based -- this module doesn't care which):
    impression_id: str
    article_id: str
    rank: int   (1-indexed position within that impression's retrieved list)

Label: 1 if article_id is in the impression's true clicked set (from
behaviors_{split}'s own candidates/labels where label == 1), else 0 --
regardless of whether the retrieved candidate was ever shown in the
platform's own impression. A1's retrieval pools the WHOLE catalog, which is
broader than what the platform actually displayed, so most retrieved
candidates legitimately get label 0.
"""

from __future__ import annotations

import pandas as pd

from src import article_stats, sessionize
from src.candidate_features import build_candidate_features
from src.impression_features import build_impression_features


def load_articles_lookup(processed_dir) -> dict[str, dict]:
    articles = pd.read_parquet(processed_dir / "articles.parquet", columns=["article_id", "category"])
    return {row.article_id: {"category": row.category} for row in articles.itertuples(index=False)}


def build_feature_matrix(processed_dir, dataset: str, split: str, candidates_df: pd.DataFrame) -> pd.DataFrame:
    behaviors = pd.read_parquet(processed_dir / f"behaviors_{split}.parquet")
    behaviors = sessionize.add_session_context(behaviors, dataset)

    articles_lookup = load_articles_lookup(processed_dir)
    stats_lookup = article_stats.build_article_stats_lookup(processed_dir, dataset)

    true_clicks: dict[str, set[str]] = {}
    impression_feats: dict[str, dict] = {}
    impression_time: dict[str, object] = {}
    for row in behaviors.itertuples(index=False):
        true_clicks[row.impression_id] = {c for c, l in zip(row.candidates, row.labels) if l == 1}
        impression_feats[row.impression_id] = build_impression_features(row, articles_lookup)
        impression_time[row.impression_id] = row.time

    rows = []
    for cand in candidates_df.itertuples(index=False):
        impr_id = cand.impression_id
        impr_feat = impression_feats.get(impr_id)
        if impr_feat is None:
            continue  # candidate for an impression outside this split's behaviors table

        cand_feat = build_candidate_features(
            cand.article_id,
            int(cand.rank),
            impression_time[impr_id],
            impr_feat["_history_category_weights"],
            stats_lookup,
        )

        row_out = {"impression_id": impr_id, "article_id": cand.article_id}
        row_out.update({k: v for k, v in impr_feat.items() if not k.startswith("_")})
        row_out.update(cand_feat)
        row_out["label"] = int(cand.article_id in true_clicks[impr_id])
        rows.append(row_out)

    return pd.DataFrame(rows)


def build_stub_candidates_from_impression(processed_dir, split: str) -> pd.DataFrame:
    """DEV-ONLY placeholder standing in for A1's real top-K retriever, which
    isn't wired in yet. Uses each impression's own (already-labeled) shown
    candidate list, ranked in its original display order, purely so this
    pipeline can be run and sanity-checked end to end right now. Delete/
    replace once A1's candidate generator is integrated -- production
    candidates must come from full-catalog retrieval (A1 Q2/Q3), not this."""
    behaviors = pd.read_parquet(processed_dir / f"behaviors_{split}.parquet", columns=["impression_id", "candidates"])
    rows = []
    for row in behaviors.itertuples(index=False):
        for rank, article_id in enumerate(row.candidates, start=1):
            rows.append({"impression_id": row.impression_id, "article_id": article_id, "rank": rank})
    return pd.DataFrame(rows)
