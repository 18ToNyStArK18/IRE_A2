"""Brute-force ANN index (cosine similarity via normalized dot product).

Both catalogs are small enough (<100k articles) that the assignment
explicitly allows brute-force in place of FAISS/ScaNN -- a single matrix
multiply against a normalized embedding matrix is exact (not approximate)
and fast at this scale.
"""

from __future__ import annotations

import numpy as np


class BruteForceANN:
    def __init__(self, article_ids: list[str], embeddings: np.ndarray):
        assert len(article_ids) == embeddings.shape[0]
        self.article_ids = list(article_ids)
        self.article_id_to_idx = {aid: i for i, aid in enumerate(self.article_ids)}
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = (embeddings / norms).astype(np.float32)
        self.n_docs = len(article_ids)

    def top_k(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        norm = np.linalg.norm(query_vector)
        if norm == 0 or self.n_docs == 0:
            return []
        q = (query_vector / norm).astype(np.float32)
        scores = self.embeddings @ q  # cosine similarity, (n_docs,)

        if k >= self.n_docs:
            order = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k)[:k]
            order = part[np.argsort(-scores[part])]
        return [(self.article_ids[i], float(scores[i])) for i in order]

    def score_for_ids(self, query_vector: np.ndarray, article_ids: list[str]) -> list[float]:
        """Score a specific, caller-chosen subset of documents (e.g. the
        articles actually shown in one impression) rather than the whole
        catalog -- used to re-rank an impression's candidate list for
        AUC/MRR/nDCG in the Phase 4 evaluation harness."""
        norm = np.linalg.norm(query_vector)
        if norm == 0:
            return [0.0] * len(article_ids)
        q = (query_vector / norm).astype(np.float32)
        return [
            float(self.embeddings[self.article_id_to_idx[aid]] @ q) if aid in self.article_id_to_idx else 0.0
            for aid in article_ids
        ]
