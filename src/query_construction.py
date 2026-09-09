"""Build a BM25 query string from a user's click history.

Three independent, experimentable axes:
  - `fields`: which article fields to pull from each clicked article
              (e.g. ["title"] vs ["title", "abstract"]) -- per plan.md
  - `window`: how many of the most-recent clicks to use (e.g. 3 vs 10) -- per plan.md
  - `method`: how to turn the selected articles' text into a query string
              ("direct" concatenation, "recency_weighted" repetition, or
              "tfidf_keywords" distillation) -- the extended ablation in
              run_ablation_study.py

History lists are stored oldest-first (both MIND and EB-NeRD document their
history as chronologically ordered), so "most recent" = the tail of the list.
"""

from __future__ import annotations

from collections import Counter

from src.inverted_index import concat_fields
from src.text_utils import tokenize


def build_query_text(
    history: list[str],
    articles_lookup: dict[str, dict],
    fields: list[str],
    window: int | None,
) -> str:
    """"direct" concatenation: each selected article's text appears once."""
    recent = history[-window:] if window else history
    parts = []
    for article_id in recent:
        record = articles_lookup.get(article_id)
        if record is None:
            continue
        text = concat_fields(record, fields)
        if text:
            parts.append(text)
    return " ".join(parts)


def build_query_text_recency_weighted(
    history: list[str],
    articles_lookup: dict[str, dict],
    fields: list[str],
    window: int | None,
) -> str:
    """Repeats each selected article's text proportional to recency rank
    (oldest selected = repeated once, most recent = repeated `len(selected)`
    times), so BM25's term-frequency component naturally up-weights recent
    clicks' vocabulary -- a simple recency hack for a model with no native
    notion of time."""
    recent = history[-window:] if window else history
    parts = []
    for rank, article_id in enumerate(recent, start=1):  # rank=1 oldest ... rank=len(recent) newest
        record = articles_lookup.get(article_id)
        if record is None:
            continue
        text = concat_fields(record, fields)
        if text:
            parts.extend([text] * rank)
    return " ".join(parts)


def build_query_text_tfidf_keywords(
    history: list[str],
    articles_lookup: dict[str, dict],
    fields: list[str],
    window: int | None,
    index,
    top_k: int = 10,
) -> str:
    """Distills the direct-concatenation query down to its top-`top_k`
    TF-IDF terms (term frequency within the query text x the BM25 index's
    own corpus-wide IDF), instead of using the full text verbatim."""
    full_text = build_query_text(history, articles_lookup, fields, window)
    if not full_text:
        return ""
    tf = Counter(tokenize(full_text))
    scored = [
        (term, count * index._idf.get(term, 0.0))
        for term, count in tf.items()
        if term in index._idf
    ]
    scored.sort(key=lambda t: t[1], reverse=True)
    top_terms = [term for term, _ in scored[:top_k]]
    return " ".join(top_terms)


def build_query(
    history: list[str],
    articles_lookup: dict[str, dict],
    fields: list[str],
    window: int | None,
    method: str = "direct",
    index=None,
    top_k: int = 10,
) -> str:
    """Dispatcher over the query-construction method axis."""
    if method == "direct":
        return build_query_text(history, articles_lookup, fields, window)
    if method == "recency_weighted":
        return build_query_text_recency_weighted(history, articles_lookup, fields, window)
    if method == "tfidf_keywords":
        if index is None:
            raise ValueError("tfidf_keywords method requires the BM25 `index` (for corpus IDF)")
        return build_query_text_tfidf_keywords(history, articles_lookup, fields, window, index, top_k)
    raise ValueError(f"unknown query construction method: {method}")
