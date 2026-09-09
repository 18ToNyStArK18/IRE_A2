"""Semantic candidate generation: mean-pooled user embedding -> ANN top-k.

Same fallback contract as BM25 (src/lexical_retrieval.py): empty history goes
straight to the train-only popularity list, and any query that yields fewer
than k_max ANN hits is backfilled from popularity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.ann_index import BruteForceANN
from src.lexical_retrieval import K_MAX, K_VALUES, compute_recall_at_k, load_popularity_fallback


def load_embeddings_lookup(processed_dir) -> tuple[dict[str, np.ndarray], BruteForceANN]:
    emb = pd.read_parquet(processed_dir / "feature_store" / "embeddings.parquet")
    article_ids = emb["article_id"].tolist()
    matrix = np.stack(emb["embedding"].to_numpy())
    lookup = dict(zip(article_ids, matrix))
    index = BruteForceANN(article_ids, matrix)
    return lookup, index


def build_user_embedding(
    history: list[str],
    embeddings_lookup: dict[str, np.ndarray],
    window: int | None = None,
) -> np.ndarray | None:
    """Uniform mean pool -- every selected click weighted equally."""
    recent = history[-window:] if window else history
    vectors = [embeddings_lookup[a] for a in recent if a in embeddings_lookup]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def build_user_embedding_decay(
    history: list[str],
    embeddings_lookup: dict[str, np.ndarray],
    window: int | None = None,
    decay_rate: float = 0.7,
) -> np.ndarray | None:
    """Exponential recency-decay weighted pool: the most recent click gets
    weight 1, each click further back gets weight `decay_rate` less, so old
    clicks contribute progressively less to the user's direction (rather
    than being dropped outright at a hard window boundary)."""
    recent = history[-window:] if window else history
    vectors, weights = [], []
    for distance_from_latest, article_id in enumerate(reversed(recent)):
        vec = embeddings_lookup.get(article_id)
        if vec is None:
            continue
        vectors.append(vec)
        weights.append(decay_rate**distance_from_latest)
    if not vectors:
        return None
    weights = np.asarray(weights)
    return np.average(np.stack(vectors), axis=0, weights=weights)


def build_user_embedding_max(
    history: list[str],
    embeddings_lookup: dict[str, np.ndarray],
    window: int | None = None,
) -> np.ndarray | None:
    """Element-wise max pooling across the selected clicks' embeddings --
    a genuinely different aggregation from mean/decay (each output
    dimension takes whichever clicked article activated it most strongly),
    not just a reweighted average."""
    recent = history[-window:] if window else history
    vectors = [embeddings_lookup[a] for a in recent if a in embeddings_lookup]
    if not vectors:
        return None
    return np.max(np.stack(vectors), axis=0)


def build_user_representation(
    history: list[str],
    embeddings_lookup: dict[str, np.ndarray],
    window: int | None = None,
    method: str = "mean",
    decay_rate: float = 0.7,
) -> np.ndarray | None:
    """Dispatcher over the pooling-method axis."""
    if method == "mean":
        return build_user_embedding(history, embeddings_lookup, window)
    if method == "decay":
        return build_user_embedding_decay(history, embeddings_lookup, window, decay_rate)
    if method == "max":
        return build_user_embedding_max(history, embeddings_lookup, window)
    raise ValueError(f"unknown pooling method: {method}")


def retrieve_for_split(
    behaviors: pd.DataFrame,
    ann_index: BruteForceANN,
    embeddings_lookup: dict[str, np.ndarray],
    popularity_fallback: list[str],
    k_max: int = K_MAX,
    window: int | None = None,
) -> pd.DataFrame:
    out_impression_ids = []
    out_candidates = []
    out_used_fallback = []

    for row in behaviors.itertuples(index=False):
        history = row.history
        used_fallback = False

        user_vec = (
            build_user_embedding(list(history), embeddings_lookup, window)
            if history is not None else None
        )
        if user_vec is None:
            candidates = popularity_fallback[:k_max]
            used_fallback = True
        else:
            ranked = ann_index.top_k(user_vec, k_max)
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


def run_semantic_pipeline(
    processed_dir,
    split: str,
    ann_index: BruteForceANN | None = None,
    embeddings_lookup: dict[str, np.ndarray] | None = None,
    popularity_fallback: list[str] | None = None,
    window: int | None = None,
) -> tuple[pd.DataFrame, dict[int, float]]:
    if ann_index is None or embeddings_lookup is None:
        embeddings_lookup, ann_index = load_embeddings_lookup(processed_dir)
    if popularity_fallback is None:
        popularity_fallback = load_popularity_fallback(processed_dir)

    behaviors = pd.read_parquet(processed_dir / f"behaviors_{split}.parquet")
    retrieved = retrieve_for_split(behaviors, ann_index, embeddings_lookup, popularity_fallback, window=window)
    recall = compute_recall_at_k(behaviors, retrieved, ks=K_VALUES)
    return retrieved, recall
