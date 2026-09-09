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

KNOWN CEILING -- read before interpreting any Q2 metric. Catalogue-wide
retrieval recovers very few of the articles a given impression actually
showed: measured Recall@200 is 0.035 (EB-NeRD, BM25) and 0.045 (MIND, BM25),
and no config in A1's 39-config ablation exceeded 0.046. So ~96% of
impressions carry NO positive into the re-ranker at all, and the positive rate
over the candidate rows is ~0.02%.

That is inherent to the task framing rather than a defect here: stage 1 ranks
the whole catalogue by similarity to the user's history, while the label asks
which of the ~20 articles the platform chose to display was clicked. Nothing
pushes catalogue-wide retrieval toward those specific 20.

We keep the literal A2 Q2 reading anyway -- the re-ranker scores exactly these
top-K rows -- so AUC/MRR/nDCG reported on them are bounded by that recall and
will look near-zero. They are measuring "was the answer even retrievable",
not ranking quality. Report the ceiling alongside them. (For contrast: both
Codabench leaderboards, the NRMS baseline in src/nrms/, and A1's own eval.py
all score the impression's in-view set instead, which is why their numbers are
not comparable to these.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.lexical_retrieval import (
    build_corpus_index,
    load_articles_lookup,
    load_popularity_fallback,
)
from src.query_construction import build_query
from src.semantic_retrieval import build_user_representation, load_embeddings_lookup

PROCESSED_DIRS = {"mind": config.MIND_PROCESSED_DIR, "ebnerd": config.EBNERD_PROCESSED_DIR}
METHODS = ("bm25", "semantic")


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


GENERATORS = {"bm25": generate_bm25_candidates, "semantic": generate_semantic_candidates}


def candidates_path(processed_dir, method: str, split: str):
    return processed_dir / f"candidates_{method}_{split}.parquet"


def build_candidates(dataset: str, method: str, split: str, k: int = config.CANDIDATE_K) -> pd.DataFrame:
    processed_dir = PROCESSED_DIRS[dataset]
    wide = GENERATORS[method](processed_dir, dataset, split, k)
    path = candidates_path(processed_dir, method, split)
    wide.to_parquet(path, index=False)
    print(
        f"[{dataset}:{method}:{split}] {len(wide)} impressions x top-{k} "
        f"(fallback_frac={wide['used_fallback'].mean():.4f}) -> {path}"
    )
    return wide


def load_candidates(processed_dir, method: str, split: str, k: int | None = None) -> pd.DataFrame:
    """Long-format (impression_id, article_id, rank, retrieval_score) -- the
    contract feature_pipeline.build_feature_matrix expects. `k` slices each
    impression's list, so a K=200 file can serve a K=100 experiment without
    re-running retrieval."""
    wide = pd.read_parquet(candidates_path(processed_dir, method, split))

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
