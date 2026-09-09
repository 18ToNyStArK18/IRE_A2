"""BM25 lexical candidate generation: index build, retrieval, and Recall@K.

Corpus indexing is fixed to title+abstract per the assignment spec (Q2:
"inverted index over article text (title + abstract)"), but the field list is
a plain constructor argument (see inverted_index.concat_fields) so it can be
changed in one place. Query construction (which fields of the *clicked*
articles to use, and how much history) is the experimentable axis -- see
query_construction.py and run_bm25_experiments.py.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.inverted_index import InvertedIndex, concat_fields
from src.query_construction import build_query_text

CORPUS_FIELDS = ["title", "abstract"]
K_VALUES = (50, 100, 200)
K_MAX = max(K_VALUES)


def load_articles_lookup(processed_dir) -> dict[str, dict]:
    articles = pd.read_parquet(processed_dir / "articles.parquet")
    return {
        row.article_id: {"title": row.title, "abstract": row.abstract}
        for row in articles.itertuples(index=False)
    }


def build_corpus_index(processed_dir, fields: list[str] = CORPUS_FIELDS) -> InvertedIndex:
    articles = pd.read_parquet(processed_dir / "articles.parquet")
    records = articles.to_dict("records")
    texts = [concat_fields(r, fields) for r in records]
    doc_ids = [r["article_id"] for r in records]
    return InvertedIndex().build(doc_ids, texts)


def load_popularity_fallback(processed_dir, k: int = K_MAX) -> list[str]:
    pop = pd.read_parquet(processed_dir / "popularity.parquet")
    return pop.sort_values("click_count", ascending=False)["article_id"].head(k).tolist()


def retrieve_for_split(
    behaviors: pd.DataFrame,
    index: InvertedIndex,
    articles_lookup: dict[str, dict],
    popularity_fallback: list[str],
    query_fields: list[str],
    window: int | None,
    k_max: int = K_MAX,
) -> pd.DataFrame:
    """Retrieve top-k_max BM25 candidates per impression.

    Falls back straight to the popularity list when history is empty, and
    backfills with popularity items (skipping duplicates) when a non-empty
    query still returns fewer than k_max scored candidates.
    """
    out_impression_ids = []
    out_candidates = []
    out_used_fallback = []

    for row in behaviors.itertuples(index=False):
        history = row.history
        used_fallback = False

        if history is None or len(history) == 0:
            candidates = popularity_fallback[:k_max]
            used_fallback = True
        else:
            query_text = build_query_text(list(history), articles_lookup, query_fields, window)
            if not query_text:
                candidates = popularity_fallback[:k_max]
                used_fallback = True
            else:
                ranked = index.top_k(query_text, k_max)
                candidates = [aid for aid, _ in ranked]
                if len(candidates) < k_max:
                    seen = set(candidates)
                    for aid in popularity_fallback:
                        if aid not in seen:
                            candidates.append(aid)
                            seen.add(aid)
                        if len(candidates) >= k_max:
                            break

        out_impression_ids.append(row.impression_id)
        out_candidates.append(candidates)
        out_used_fallback.append(used_fallback)

    return pd.DataFrame({
        "impression_id": out_impression_ids,
        "candidates": out_candidates,
        "used_fallback": out_used_fallback,
    })


def compute_recall_at_k(
    behaviors: pd.DataFrame,
    retrieved: pd.DataFrame,
    ks: tuple[int, ...] = K_VALUES,
) -> dict[int, float]:
    merged = behaviors[["impression_id", "candidates", "labels"]].merge(
        retrieved[["impression_id", "candidates"]],
        on="impression_id",
        suffixes=("_truth", "_retrieved"),
    )

    per_k_recalls = {k: [] for k in ks}
    for row in merged.itertuples(index=False):
        ground_truth = {c for c, l in zip(row.candidates_truth, row.labels) if l == 1}
        if not ground_truth:
            continue
        for k in ks:
            retrieved_set = set(row.candidates_retrieved[:k])
            per_k_recalls[k].append(len(retrieved_set & ground_truth) / len(ground_truth))

    return {k: (sum(v) / len(v) if v else 0.0) for k, v in per_k_recalls.items()}


def run_bm25_pipeline(
    processed_dir,
    split: str,
    query_fields: list[str],
    window: int | None,
    index: InvertedIndex | None = None,
    articles_lookup: dict | None = None,
    popularity_fallback: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[int, float]]:
    if index is None:
        index = build_corpus_index(processed_dir)
    if articles_lookup is None:
        articles_lookup = load_articles_lookup(processed_dir)
    if popularity_fallback is None:
        popularity_fallback = load_popularity_fallback(processed_dir)

    behaviors = pd.read_parquet(processed_dir / f"behaviors_{split}.parquet")
    retrieved = retrieve_for_split(
        behaviors, index, articles_lookup, popularity_fallback, query_fields, window
    )
    recall = compute_recall_at_k(behaviors, retrieved)
    return retrieved, recall
