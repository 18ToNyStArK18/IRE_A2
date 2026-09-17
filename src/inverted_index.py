"""A from-scratch inverted index + BM25 scorer.

Built over an arbitrary, caller-chosen concatenation of unified-schema
article fields (e.g. ["title"], ["title", "abstract"], ["title", "body"]) --
the field list is a constructor argument, not hardcoded, so the corpus text
used for indexing is fully dynamic per the assignment's requirement.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from src.text_utils import tokenize


def concat_fields(record: dict, fields: list[str]) -> str:
    parts = [str(record.get(f) or "") for f in fields]
    return " ".join(p for p in parts if p)


class InvertedIndex:
    """BM25 (Okapi) index over a fixed set of documents.

    Postings are stored per-term as parallel numpy arrays (doc indices, term
    frequencies) so that scoring a query only touches the documents that
    actually contain a query term, and the per-term contribution is computed
    with a single vectorized numpy expression.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}
        self.doc_len: np.ndarray | None = None
        self.avg_doc_len: float = 0.0
        self.n_docs: int = 0

    def build(self, doc_ids: list[str], texts: list[str]) -> "InvertedIndex":
        self.doc_ids = list(doc_ids)
        self.doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        self.n_docs = len(doc_ids)

        term_postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_len = np.zeros(self.n_docs, dtype=np.float64)

        for doc_idx, text in enumerate(texts):
            tokens = tokenize(text)
            doc_len[doc_idx] = len(tokens)
            tf = defaultdict(int)
            for tok in tokens:
                tf[tok] += 1
            for term, count in tf.items():
                term_postings[term].append((doc_idx, count))

        self.doc_len = doc_len
        self.avg_doc_len = float(doc_len.mean()) if self.n_docs else 0.0

        for term, postings in term_postings.items():
            doc_idx_arr = np.array([p[0] for p in postings], dtype=np.int64)
            tf_arr = np.array([p[1] for p in postings], dtype=np.float64)
            self._postings[term] = (doc_idx_arr, tf_arr)
            df = len(postings)
            self._idf[term] = np.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

        return self

    def score_query(self, query_text: str) -> np.ndarray:
        """Return a dense (n_docs,) BM25 score array for the query.

        Weights each unique query term by its query-side term frequency
        (standard BM25 extension, e.g. used by Lucene/Elasticsearch) --
        a term repeated 3x in the query contributes 3x the single-occurrence
        score. For any query built by simple concatenation (no repeats),
        every term's query-tf is 1, so this is numerically identical to the
        old set-based behavior; it only changes results for query
        construction methods that deliberately repeat text (e.g. recency
        weighting in run_ablation_study.py, A1 repo)."""
        scores = np.zeros(self.n_docs, dtype=np.float64)
        if self.n_docs == 0:
            return scores

        query_term_counts = Counter(tokenize(query_text))
        k1, b, avgdl = self.k1, self.b, self.avg_doc_len

        for term, query_tf in query_term_counts.items():
            posting = self._postings.get(term)
            if posting is None:
                continue
            doc_idx, tf = posting
            idf = self._idf[term]
            dl = self.doc_len[doc_idx]
            denom = tf + k1 * (1 - b + b * (dl / avgdl if avgdl > 0 else 0))
            contribution = query_tf * idf * (tf * (k1 + 1)) / denom
            np.add.at(scores, doc_idx, contribution)

        return scores

    def score_for_ids(self, query_text: str, article_ids: list[str]) -> list[float]:
        """Score a specific, caller-chosen subset of documents (e.g. the
        articles actually shown in one impression) rather than the whole
        catalog -- used to re-rank an impression's candidate list for
        AUC/MRR/nDCG in the Phase 4 evaluation harness."""
        scores = self.score_query(query_text)
        return [
            float(scores[self.doc_id_to_idx[aid]]) if aid in self.doc_id_to_idx else 0.0
            for aid in article_ids
        ]

    def top_k(self, query_text: str, k: int) -> list[tuple[str, float]]:
        scores = self.score_query(query_text)
        if k >= self.n_docs:
            order = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k)[:k]
            order = part[np.argsort(-scores[part])]
        return [(self.doc_ids[i], float(scores[i])) for i in order  if scores[i] > 0]
