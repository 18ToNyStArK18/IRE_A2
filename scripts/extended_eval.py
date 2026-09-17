#!/usr/bin/env python3
"""A2 Q5: extended evaluation of the full two-stage pipeline.

    python scripts/extended_eval.py --dataset ebnerd --method popular
    python scripts/extended_eval.py --dataset mind --method bm25 --beyond-sample 3000

Writes results/extended_eval_<dataset>_<method>.json:

  * AUC, MRR, nDCG@5, nDCG@10, diversity, novelty, coverage
  * two slices -- cold-start vs warm users, head vs tail clicked articles
  * a bootstrap 95% CI on every reported metric

It scores with the trained re-ranker rather than reading a saved prediction
file, reusing `src.reranker`'s own matrix builder and feature list so the
accuracy figures reconcile with the ones in `report_<method>_<split>.json`.

Two populations, deliberately
-----------------------------
Accuracy metrics are averaged over EVERY impression in the split. Only
positive-bearing impressions are scored, which is exact rather than a
shortcut: an impression with no positive contributes MRR 0 and nDCG 0 whatever
order it is put in, and AUC is undefined for it, so it cannot change any of
them (`reranker.evaluate_over_population`, verified in tests).

Beyond-accuracy describes what the system *recommends*, so that shortcut does
not apply -- a no-positive impression still produces a list. Those are computed
on a random sample of impressions (`--beyond-sample`), and coverage in
particular is conditional on that sample size, since scoring more impressions
can only cover more catalogue. The count is reported next to every such figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import beyond_accuracy, bootstrap, reranker, slicing  # noqa: E402
from src.candidates import PROCESSED_DIRS  # noqa: E402
from src.metrics import per_impression_metrics  # noqa: E402
from src.semantic_retrieval import load_embeddings_lookup  # noqa: E402

ACCURACY_METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")


def load_popularity(processed_dir) -> dict[str, int]:
    pop = pd.read_parquet(processed_dir / "popularity.parquet")
    return dict(zip(pop["article_id"], pop["click_count"]))


SCORE_ROW_BLOCK = 500_000


def score_matrix(matrix: pd.DataFrame, booster, features) -> pd.DataFrame:
    """Add `model_score` in place, predicting in row blocks.

    `build_matrix` already returns a fresh frame, so copying it here only
    doubles peak memory, and `matrix[features].astype(np.float64)` materialises
    a third copy of the whole feature block. On MIND that is ~14.6M rows x 30
    features x 8 bytes per copy, which is what pushed an earlier run into the
    OOM killer on this laptop.
    """
    scores = np.empty(len(matrix), dtype=np.float64)
    for start in range(0, len(matrix), SCORE_ROW_BLOCK):
        stop = min(start + SCORE_ROW_BLOCK, len(matrix))
        block = matrix.iloc[start:stop][features].to_numpy(dtype=np.float64)
        scores[start:stop] = booster.predict(block)
    matrix["model_score"] = scores
    return matrix


def accuracy_rows_chunked(processed_dir, args, article_index, booster, features, cold_map, head_map):
    """Per-impression metrics, built a chunk of impressions at a time.

    Mirrors `reranker.evaluate_over_population`, which chunks for the same
    reason and documents that chunking is exact here: every candidate of every
    scored impression is kept, and no feature depends on rows outside its own
    impression. Only the small per-impression metric rows are retained, so peak
    memory is one chunk's feature matrix rather than the whole split's.
    """
    impression_ids = np.array(sorted(reranker.true_clicks(processed_dir, args.split)))
    rows = []
    for start in range(0, len(impression_ids), args.chunk_impressions):
        chunk = set(impression_ids[start : start + args.chunk_impressions].tolist())
        matrix = reranker.build_matrix(
            processed_dir, args.dataset, args.split, args.method, article_index,
            positives_only=True, impressions=chunk,
        )
        if matrix.empty:
            continue
        score_matrix(matrix, booster, features)
        for impression_id, block in matrix.groupby("impression_id", sort=False):
            # AUC is NaN when undefined; bootstrap_ci drops NaNs, so it leaves
            # the AUC mean without disturbing MRR or nDCG for the same impression.
            metrics = per_impression_metrics(
                block["label"].to_numpy(dtype=np.float64), block["model_score"].to_numpy()
            )
            metrics["_cold_warm"] = cold_map.get(impression_id)
            metrics["_head_tail"] = head_map.get(impression_id)
            rows.append(metrics)
        del matrix
    return pd.DataFrame(rows)


MIN_SLICE_IMPRESSIONS = 30  # below this a bootstrap CI is not worth reading


def population_slices(processed_dir, split: str, head_set) -> tuple[dict, dict]:
    """Slice membership for EVERY impression in the split, not only the scored ones.

    Both slices are computable without the model: cold-start needs the history
    length, and head/tail needs the clicked article, which the behaviour log
    records whether or not stage 1 managed to retrieve it. That makes each
    slice's denominator exact. Estimating it instead -- scaling the scored count
    by the overall scored share, as an earlier version did -- assumes retrieval
    succeeds equally often in every slice, which is precisely the kind of thing
    a cold-start slice exists to disprove.
    """
    behaviors = pd.read_parquet(
        processed_dir / f"behaviors_{split}.parquet",
        columns=["impression_id", "history", "candidates", "labels"],
    )
    cold, head = {}, {}
    for row in behaviors.itertuples(index=False):
        n_history = 0 if row.history is None else len(row.history)
        cold[row.impression_id] = slicing.cold_start_or_warm(n_history)
        clicked = [c for c, l in zip(row.candidates, row.labels) if l == 1]
        head[row.impression_id] = slicing.head_or_tail(clicked[0], head_set) if clicked else None
    return cold, head


def accuracy_section(frame: pd.DataFrame, n_total: int, cold_map, head_map, args) -> dict:
    """Overall and per-slice CIs over the per-impression metric rows."""

    def ci_block(subset: pd.DataFrame, denominator: int) -> dict:
        block = {}
        for name in ACCURACY_METRICS:
            values = subset[name].to_numpy(dtype=np.float64)
            if name == "auc":
                # Averaged only over impressions where AUC is defined, matching
                # evaluate_impressions; NaNs are dropped inside bootstrap_ci.
                block[name] = bootstrap.bootstrap_ci(values, args.resamples, seed=args.seed)
            else:
                # MRR/nDCG are averaged over the FULL population: an unscored
                # impression contributes a known 0, so pad rather than drop.
                padded = np.concatenate([values, np.zeros(max(denominator - len(values), 0))])
                block[name] = bootstrap.bootstrap_ci(padded, args.resamples, seed=args.seed)
        return block

    section = {"overall": ci_block(frame, n_total)}

    totals = {
        "cold_start_vs_warm": pd.Series([v for v in cold_map.values() if v]).value_counts().to_dict(),
        "head_vs_tail": pd.Series([v for v in head_map.values() if v]).value_counts().to_dict(),
    }
    slices = {}
    for slice_name, column in (("cold_start_vs_warm", "_cold_warm"), ("head_vs_tail", "_head_tail")):
        slices[slice_name] = {}
        for value in sorted(v for v in frame[column].dropna().unique()):
            subset = frame[frame[column] == value]
            entry = ci_block(subset, int(totals[slice_name].get(value, len(subset))))
            entry["n_scored_impressions"] = int(len(subset))
            entry["n_population_impressions"] = int(totals[slice_name].get(value, len(subset)))
            if len(subset) < MIN_SLICE_IMPRESSIONS:
                entry["warning"] = (
                    f"only {len(subset)} scored impressions in this slice; the CI is too "
                    "wide to support a claim and is reported for completeness only"
                )
            slices[slice_name][value] = entry
    section["slices"] = slices
    section["n_impressions_total"] = int(n_total)
    section["n_impressions_scored"] = int(len(frame))
    return section


def beyond_section(scored: pd.DataFrame, processed_dir, popularity, args) -> dict:
    """Diversity/novelty/coverage over the re-ranked top-K of sampled impressions."""
    try:
        embeddings_lookup, _ = load_embeddings_lookup(processed_dir)
    except FileNotFoundError:
        # Diversity is the only metric here that needs the semantic feature
        # store, which exists only on machines that built the `semantic` arm.
        # Novelty and coverage need click counts and article ids, so the run
        # still produces them rather than failing outright.
        embeddings_lookup = None
    probabilities, default_probability = beyond_accuracy.click_probabilities(popularity)
    catalogue_size = len(pd.read_parquet(processed_dir / "articles.parquet", columns=["article_id"]))

    recommended, diversity, novelty = [], [], []
    for _, block in scored.groupby("impression_id", sort=False):
        top = block.nlargest(args.top_k, "model_score")["article_id"].tolist()
        recommended.append(top)
        value = None if embeddings_lookup is None else beyond_accuracy.intra_list_diversity(top, embeddings_lookup)
        diversity.append(float("nan") if value is None else value)
        novelty.append(beyond_accuracy.novelty(top, probabilities, default_probability))

    diversity_ci = bootstrap.bootstrap_ci(diversity, args.resamples, seed=args.seed)
    random_baseline = (
        float("nan") if embeddings_lookup is None
        else beyond_accuracy.random_pair_diversity(embeddings_lookup, seed=args.seed)
    )

    return {
        "diversity": diversity_ci,
        "diversity_random_baseline": random_baseline,
        "diversity_vs_random": (
            diversity_ci["mean"] / random_baseline if random_baseline else float("nan")
        ),
        "_diversity_unavailable": embeddings_lookup is None,
        "_diversity_note": (
            "Raw diversity is not comparable across datasets: it is dominated by how "
            "anisotropic the embedding space is. A random EB-NeRD pair scores ~0.049 "
            "(provided multilingual BERT, vectors in a narrow cone) and a random MIND "
            "pair ~0.945 (MiniLM, contrastively trained). Read diversity_vs_random, "
            "where 1.0 means 'as diverse as picking at random' and below 1.0 means the "
            "recommender concentrates."
        ),
        "novelty": bootstrap.bootstrap_ci(novelty, args.resamples, seed=args.seed),
        "coverage": bootstrap.coverage_ci(recommended, catalogue_size, args.resamples, seed=args.seed),
        "top_k": args.top_k,
        "catalogue_size": int(catalogue_size),
        "n_impressions_sampled": int(len(recommended)),
        "_note": (
            "Computed on a sample of impressions, so coverage is conditional on "
            "n_impressions_sampled -- scoring more impressions can only cover more "
            "catalogue. Diversity and novelty are per-impression means and are not."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--method", default="popular")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=10, help="re-ranked list length for beyond-accuracy")
    parser.add_argument("--beyond-sample", type=int, default=5000)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--chunk-impressions", type=int, default=reranker.EVAL_CHUNK_IMPRESSIONS,
                        help="impressions per feature-matrix chunk; lower it if memory is tight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import lightgbm as lgb

    processed_dir = PROCESSED_DIRS[args.dataset]
    model_path = processed_dir / "reranker" / f"lambdamart_{args.method}.txt"
    if not model_path.exists():
        available = sorted(p.stem.replace("lambdamart_", "") for p in (processed_dir / "reranker").glob("lambdamart_*.txt"))
        raise SystemExit(f"no model for '{args.method}'; trained arms here: {available or 'none'}")

    booster = lgb.Booster(model_file=str(model_path))
    features = booster.feature_name()
    article_index = reranker._article_index(processed_dir, args.dataset)
    popularity = load_popularity(processed_dir)
    head_set = slicing.build_head_article_set(popularity)
    n_total = len(reranker.true_clicks(processed_dir, args.split))
    cold_map, head_map = population_slices(processed_dir, args.split, head_set)

    # Accuracy: positive-bearing impressions only, which is exact for the
    # full-population means (see module docstring), chunked so MIND's ~14.6M
    # candidate rows are never resident at once.
    accuracy_frame = accuracy_rows_chunked(
        processed_dir, args, article_index, booster, features, cold_map, head_map
    )

    # Beyond-accuracy: a sample of ALL impressions, positive-bearing or not.
    rng = np.random.default_rng(args.seed)
    every = np.array(sorted(reranker.true_clicks(processed_dir, args.split)))
    sample = every if len(every) <= args.beyond_sample else rng.choice(every, args.beyond_sample, replace=False)
    beyond_matrix = reranker.build_matrix(
        processed_dir, args.dataset, args.split, args.method, article_index,
        positives_only=False, impressions=set(sample.tolist()),
    )
    beyond_scored = score_matrix(beyond_matrix, booster, features)

    report = {
        "dataset": args.dataset,
        "method": args.method,
        "split": args.split,
        "model": {"path": str(model_path.name), "num_trees": booster.num_trees()},
        "config": {
            "top_k": args.top_k,
            "chunk_impressions": args.chunk_impressions,
            "beyond_sample": args.beyond_sample,
            "resamples": args.resamples,
            "seed": args.seed,
            "cold_start_threshold": slicing.COLD_START_THRESHOLD,
            "head_fraction": slicing.HEAD_FRACTION,
        },
        "accuracy": accuracy_section(accuracy_frame, n_total, cold_map, head_map, args),
        "beyond_accuracy": beyond_section(beyond_scored, processed_dir, popularity, args),
    }

    out_path = Path(args.out) if args.out else Path("results") / f"extended_eval_{args.dataset}_{args.method}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    acc = report["accuracy"]
    print(f"{args.dataset}/{args.method} {args.split}: {acc['n_impressions_scored']:,} scored "
          f"of {acc['n_impressions_total']:,} impressions")
    for name in ACCURACY_METRICS:
        m = acc["overall"][name]
        print(f"  {name:8s} {m['mean']:.4f}  [{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]")
    for slice_name, values in acc["slices"].items():
        parts = " | ".join(f"{v}: {b['ndcg@5']['mean']:.4f} (n={b['n_scored_impressions']})" for v, b in values.items())
        print(f"  {slice_name:20s} nDCG@5 -> {parts}")
    bey = report["beyond_accuracy"]
    for name in ("diversity", "novelty"):
        m = bey[name]
        extra = ""
        if name == "diversity":
            extra = (f"   random-pair baseline {bey['diversity_random_baseline']:.4f}"
                     f"  -> {bey['diversity_vs_random']:.3f}x random")
        print(f"  {name:8s} {m['mean']:.4f}  [{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]{extra}")
    c = bey["coverage"]
    print(f"  coverage {c['value']:.4f}  [{c['ci_lower']:.4f}, {c['ci_upper']:.4f}]  "
          f"(top-{bey['top_k']} of {bey['n_impressions_sampled']:,} impressions, catalogue {bey['catalogue_size']:,})")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
