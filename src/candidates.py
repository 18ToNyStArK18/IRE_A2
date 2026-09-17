"""Stage 1 of the two-stage pipeline (A2 Q2): run A1's retriever and persist
top-K candidates per impression.

Uses A1's ablation-winning configs (config.BM25_CANDIDATE_CONFIG /
SEMANTIC_CANDIDATE_CONFIG) rather than its Phase 2/3 defaults -- those are the
configs A1 actually submitted to both leaderboards, so the re-ranker is built
on the strongest retriever available. BM25's winning query-construction method
differs by dataset (MIND: tfidf_keywords, EB-NeRD: recency_weighted).

Storage is WIDE -- one row per impression with the candidate list nested --
because the long form at K=200 is ~27M rows for MIND's train split alone.
`load_candidates` explodes it into the
(impression_id, article_id, rank, retrieval_score) contract that
feature_pipeline.build_feature_matrix consumes.

Fallback contract is A1's: an empty or unresolvable history retrieves straight
from the train-only popularity list, and a query returning fewer than K real
hits is backfilled from it. Backfilled entries score 0.0 so they always sort
below a genuine retrieval hit.

Methods:

  bm25        A1's BM25 over the WHOLE catalogue (the original stage 1).
  semantic    A1's embedding retrieval over the whole catalogue.
  bm25_fresh  A1's BM25, unchanged, scored only over the articles the platform
              displayed in the hour before the impression (src/fresh_pool.py).
  popular     that same fresh pool, ranked by recent (lagged) click counts.

The whole-catalogue methods recover the clicked article for only 2.55% (EB-NeRD)
/ 2.90% (MIND) of test impressions at K=200. That is NOT inherent to the task, as
this docstring used to say. Clicks go to what is in circulation right now --
EB-NeRD's median clicked-article age at click time is 3.1h, 92% within a day --
while the catalogue reaches back to 2000. Two separate mistakes compound:

  wrong search space    restricting A1's own scoring to the fresh pool lifts
                        recall@200 from 2.55% to 84.2% (EB-NeRD) and from 2.90%
                        to 21.0% (MIND);
  wrong ranking signal  ranking the pool by popularity instead reaches 97.0%
                        (EB-NeRD) / 93.9% (MIND) over the full test populations.

History similarity is also too sparse to rank a fresh pool on its own: on MIND a
title query scores a median of only ~20 of the ~2,000 pool articles above zero,
so `bm25_fresh`'s top-200 is ~90% ties, and its recall is decided by how those
ties are broken (measured 10% to 92% across tie rules). It therefore breaks ties
with a seeded random draw, the only rule under which the arm measures BM25 and
nothing else. So bm25 -> bm25_fresh -> popular is the stage-1 ablation, and
`popular` is what the re-ranker consumes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, fresh_pool
from src.lexical_retrieval import (
    build_corpus_index,
    load_articles_lookup,
    load_popularity_fallback,
)
from src.query_construction import build_query
from src.semantic_retrieval import build_user_representation, load_embeddings_lookup

PROCESSED_DIRS = config.PROCESSED_DIRS
METHODS = ("bm25", "semantic", "bm25_fresh", "popular")
_BM25_FRESH_TIE_SEED = 0


def _pad_with_popularity(ranked, popularity_fallback, k):
    if len(ranked) >= k:
        return ranked[:k]
    seen = {article_id for article_id, _ in ranked}
    out = list(ranked)
    for article_id in popularity_fallback:
        if article_id not in seen:
            out.append((article_id, 0.0))
            seen.add(article_id)
        if len(out) >= k:
            break
    return out


def _to_wide(impression_ids, ranked_per_impression, used_fallback) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "impression_id": impression_ids,
            "candidates": [[a for a, _ in r] for r in ranked_per_impression],
            "scores": [np.asarray([s for _, s in r], dtype=np.float32) for r in ranked_per_impression],
            "used_fallback": used_fallback,
        }
    )


def generate_bm25_candidates(processed_dir, dataset: str, split: str, k: int = config.CANDIDATE_K) -> pd.DataFrame:
    cfg = config.BM25_CANDIDATE_CONFIG[dataset]
    index = build_corpus_index(processed_dir)
    articles_lookup = load_articles_lookup(processed_dir)
    popularity_fallback = load_popularity_fallback(processed_dir, k)

    behaviors = pd.read_parquet(processed_dir / f"behaviors_{split}.parquet", columns=["impression_id", "history"])

    impression_ids, ranked_all, fallbacks = [], [], []
    for row in behaviors.itertuples(index=False):
        history = list(row.history) if row.history is not None else []
        query_text = (
            build_query(history, articles_lookup, cfg["fields"], cfg["window"], cfg["method"], index=index)
            if history
            else ""
        )
        if not query_text:
            ranked = [(a, 0.0) for a in popularity_fallback[:k]]
            used_fallback = True
        else:
            ranked = _pad_with_popularity(index.top_k(query_text, k), popularity_fallback, k)
            used_fallback = False

        impression_ids.append(row.impression_id)
        ranked_all.append(ranked)
        fallbacks.append(used_fallback)

    return _to_wide(impression_ids, ranked_all, fallbacks)


def generate_semantic_candidates(processed_dir, dataset: str, split: str, k: int = config.CANDIDATE_K) -> pd.DataFrame:
    cfg = config.SEMANTIC_CANDIDATE_CONFIG[dataset]
    embeddings_lookup, ann_index = load_embeddings_lookup(processed_dir)
    popularity_fallback = load_popularity_fallback(processed_dir, k)

    behaviors = pd.read_parquet(processed_dir / f"behaviors_{split}.parquet", columns=["impression_id", "history"])

    impression_ids, ranked_all, fallbacks = [], [], []
    for row in behaviors.itertuples(index=False):
        history = list(row.history) if row.history is not None else []
        user_vector = (
            build_user_representation(history, embeddings_lookup, cfg["window"], cfg["pooling"])
            if history
            else None
        )
        if user_vector is None:
            ranked = [(a, 0.0) for a in popularity_fallback[:k]]
            used_fallback = True
        else:
            ranked = _pad_with_popularity(ann_index.top_k(user_vector, k), popularity_fallback, k)
            used_fallback = False

        impression_ids.append(row.impression_id)
        ranked_all.append(ranked)
        fallbacks.append(used_fallback)

    return _to_wide(impression_ids, ranked_all, fallbacks)


def generate_popular_candidates(processed_dir, dataset: str, split: str, k: int = config.CANDIDATE_K) -> pd.DataFrame:
    """The fresh pool ranked by lagged click counts, ties broken by display
    counts, backfilled from the 24h window when shorter than K."""
    log = fresh_pool.load_log(processed_dir, split)
    pool = fresh_pool.FreshPool(log)

    impression_ids, ranked_all, fallbacks = [], [], []
    for row in log[log["is_target"]].itertuples(index=False):
        pool.advance_to(row.t)
        ranked, used_fallback = pool.popular(k)
        impression_ids.append(row.impression_id)
        ranked_all.append(ranked)
        fallbacks.append(used_fallback)

    return _to_wide(impression_ids, ranked_all, fallbacks)


def generate_bm25_fresh_candidates(processed_dir, dataset: str, split: str, k: int = config.CANDIDATE_K) -> pd.DataFrame:
    """A1's BM25 -- same config, same query construction -- scored only over the
    fresh pool, so the ablation separates the search space from the ranking
    signal. An impression with no usable history falls back to the pool's
    popularity order, mirroring A1's own empty-history fallback."""
    cfg = config.BM25_CANDIDATE_CONFIG[dataset]
    index = build_corpus_index(processed_dir)
    articles_lookup = load_articles_lookup(processed_dir)
    log = fresh_pool.load_log(processed_dir, split, with_history=True)
    pool = fresh_pool.FreshPool(log)
    tie_rng = np.random.default_rng(_BM25_FRESH_TIE_SEED)

    impression_ids, ranked_all, fallbacks = [], [], []
    for row in log[log["is_target"]].itertuples(index=False):
        pool.advance_to(row.t)
        history = list(row.history) if row.history is not None else []
        query_text = (
            build_query(history, articles_lookup, cfg["fields"], cfg["window"], cfg["method"], index=index)
            if history
            else ""
        )
        ids = list(pool.pool)
        if query_text and ids:
            scores = np.asarray(index.score_for_ids(query_text, ids), dtype=np.float64)
            # Most of the pool scores exactly zero (see module docstring), so the
            # tie-break decides most of the top-K. Ties are broken at random
            # (seeded): dict insertion order would smuggle in a recency signal, and
            # a popularity tie-break would turn this arm into `popular`.
            order = np.lexsort((tie_rng.random(len(ids)), -scores))
            ranked = [(ids[i], float(scores[i])) for i in order]
            ranked, used_fallback = pool.backfill(ranked, k)
        else:
            ranked, _ = pool.popular(k)
            used_fallback = True

        impression_ids.append(row.impression_id)
        ranked_all.append(ranked)
        fallbacks.append(used_fallback)

    return _to_wide(impression_ids, ranked_all, fallbacks)


GENERATORS = {
    "bm25": generate_bm25_candidates,
    "semantic": generate_semantic_candidates,
    "bm25_fresh": generate_bm25_fresh_candidates,
    "popular": generate_popular_candidates,
}


def candidates_path(processed_dir, method: str, split: str):
    return processed_dir / f"candidates_{method}_{split}.parquet"


def build_candidates(dataset: str, method: str, split: str, k: int = config.CANDIDATE_K) -> pd.DataFrame:
    processed_dir = PROCESSED_DIRS[dataset]
    wide = GENERATORS[method](processed_dir, dataset, split, k)
    path = candidates_path(processed_dir, method, split)
    wide.to_parquet(path, index=False)
    recall = recall_at_k(processed_dir, method, split)
    print(
        f"[{dataset}:{method}:{split}] {len(wide)} impressions x top-{k} "
        f"(fallback_frac={wide['used_fallback'].mean():.4f}) -> {path}  recall "
        + " ".join(f"@{kk}={v:.4f}" for kk, v in recall.items())
    )
    return wide


def recall_at_k(processed_dir, method: str, split: str, ks=(50, 100, 200)) -> dict[int, float]:
    """Share of the split's impressions whose clicked article is in the top-k
    candidates -- an impression-level hit rate, the same definition as A1's
    Recall@K and the re-ranker's `recall@k` column. Every impression counts in
    the denominator, including any the generator failed to emit."""
    wide = pd.read_parquet(candidates_path(processed_dir, method, split), columns=["impression_id", "candidates"])
    behaviors = pd.read_parquet(
        processed_dir / f"behaviors_{split}.parquet", columns=["impression_id", "candidates", "labels"]
    )
    clicks = {
        i: {a for a, y in zip(c, l) if y == 1}
        for i, c, l in zip(behaviors["impression_id"], behaviors["candidates"], behaviors["labels"])
    }
    retrieved = dict(zip(wide["impression_id"], wide["candidates"]))
    return {
        k: float(np.mean([bool(clicked & set(retrieved.get(i, [])[:k])) for i, clicked in clicks.items()]))
        for k in ks
    }


def load_candidates(
    processed_dir, method: str, split: str, k: int | None = None, impressions=None
) -> pd.DataFrame:
    """Long-format (impression_id, article_id, rank, retrieval_score) -- the
    contract feature_pipeline.build_feature_matrix expects. `k` slices each
    impression's list, so a K=200 file can serve a K=100 experiment without
    re-running retrieval. `impressions`, if given, keeps only those ids -- on the
    WIDE frame, before exploding, since at K=200 MIND train is 27.7M long rows and
    any sampling or chunking has to happen first."""
    wide = pd.read_parquet(candidates_path(processed_dir, method, split))
    if impressions is not None:
        wide = wide[wide["impression_id"].isin(impressions)]

    candidate_lists = [np.asarray(c)[:k] if k else np.asarray(c) for c in wide["candidates"]]
    score_lists = [np.asarray(s)[:k] if k else np.asarray(s) for s in wide["scores"]]
    lengths = np.fromiter((len(c) for c in candidate_lists), dtype=np.int64, count=len(candidate_lists))

    return pd.DataFrame(
        {
            "impression_id": np.repeat(wide["impression_id"].to_numpy(), lengths),
            "article_id": np.concatenate(candidate_lists) if len(candidate_lists) else np.array([], dtype=object),
            "rank": np.concatenate([np.arange(1, n + 1) for n in lengths]) if len(lengths) else np.array([], dtype=np.int64),
            "retrieval_score": np.concatenate(score_lists) if len(score_lists) else np.array([], dtype=np.float32),
        }
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--k", type=int, default=config.CANDIDATE_K)
    args = parser.parse_args()

    datasets = list(PROCESSED_DIRS) if args.dataset == "all" else [args.dataset]
    methods = list(METHODS) if args.method == "all" else [args.method]
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for dataset in datasets:
        for method in methods:
            for split in splits:
                build_candidates(dataset, method, split, args.k)


if __name__ == "__main__":
    main()
