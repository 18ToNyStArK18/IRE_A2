"""History-content features for the re-ranker (A2 Q1.1): how similar each
candidate is to the TITLES and EMBEDDINGS of the articles the user clicked.

  history_title_bm25        A1's history-title query (config.BM25_CANDIDATE_CONFIG:
                            titles of the last 10 clicks, per-dataset query
                            construction) scored against the candidate with A1's
                            BM25 index over title+abstract.
  history_embedding_cosine  cosine between the candidate and A1's pooled user
                            embedding (config.SEMANTIC_CANDIDATE_CONFIG: MIND
                            mean-pools, EB-NeRD max-pools the full history).

Both are exactly the similarities the `bm25` and `semantic` stage-1 arms rank by,
recomputed for whatever candidates stage 1 produced. For those arms they overlap
retrieval_score; for the shipped `popular` arm, whose retrieval_score is a click
count, they are the only title/embedding signal the re-ranker gets.

Leakage (Q9): only the impression's own `history` -- clicks strictly before it --
and static catalogue text and vectors are read, so both are available at serving
time. NaN means "no signal" (no usable history), never "dissimilar".
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from src import config
from src.lexical_retrieval import build_corpus_index, load_articles_lookup
from src.query_construction import build_query
from src.semantic_retrieval import build_user_representation, load_embeddings_lookup

FEATURES = ("history_title_bm25", "history_embedding_cosine")


class HistoryContentScorer:
    def __init__(self, processed_dir, dataset: str):
        self._bm25 = config.BM25_CANDIDATE_CONFIG[dataset]
        self._semantic = config.SEMANTIC_CANDIDATE_CONFIG[dataset]
        self._index = build_corpus_index(processed_dir)
        self._articles = load_articles_lookup(processed_dir)
        self._embeddings, self._ann = load_embeddings_lookup(processed_dir)

    def score(self, history, candidate_ids) -> tuple[np.ndarray, np.ndarray]:
        """(history_title_bm25, history_embedding_cosine), aligned with
        `candidate_ids`. One call per impression: the query and the user vector
        are built once and shared by all its candidates."""
        history = list(history) if history is not None else []
        candidate_ids = list(candidate_ids)
        n = len(candidate_ids)
        title = np.full(n, np.nan)
        cosine = np.full(n, np.nan)
        if not history or not n:
            return title, cosine

        query = build_query(
            history, self._articles, self._bm25["fields"], self._bm25["window"],
            self._bm25["method"], index=self._index,
        )
        if query:
            title[:] = self._index.score_for_ids(query, candidate_ids)

        user = build_user_representation(
            history, self._embeddings, self._semantic["window"], self._semantic["pooling"]
        )
        if user is not None:
            cosine[:] = self._ann.score_for_ids(user, candidate_ids)
            # score_for_ids reads an article with no vector as 0.0 ("orthogonal");
            # keep it as missing instead.
            cosine[[a not in self._embeddings for a in candidate_ids]] = np.nan
        return title, cosine


@lru_cache(maxsize=2)
def get_scorer(processed_dir, dataset: str) -> HistoryContentScorer:
    """One scorer per (dataset, processed dir) per process: chunked evaluation
    builds feature matrices many times, and the BM25 index build dominates."""
    return HistoryContentScorer(processed_dir, dataset)
