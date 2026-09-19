#!/usr/bin/env python3
"""Q4: serving and scale benchmark for the two-stage pipeline.

    python scripts/serving_benchmark.py --dataset ebnerd
    python scripts/serving_benchmark.py --dataset mind --requests 2000

Writes results/serving_<dataset>.json: component memory, per-stage request
latency percentiles, throughput and cost per 1000 queries, and measured scaling
points. Every assumption it prices against is recorded in the file's `config`
block, so the numbers can be re-derived rather than taken on trust.

What counts as request latency, and what does not
------------------------------------------------
Stage 1 is a 1-hour window over the display log. `FreshPool.advance_to(t)`
replays the log forward to t, but a live system never does that inside a
request -- it maintains those counters incrementally as impressions arrive. So
`advance_to` is excluded from request latency and reported separately as ingest
cost; timing it would measure log catch-up, not serving. The same reasoning
applies to session counters, which a live system holds in a session store:
sessionizing the split is startup here, not per-request work.

What IS timed is what a request actually has to do with the state already in
memory: rank the current pool (stage 1), build features for the top-K
(stage 2a), and score them with the booster (stage 2b).

The ANN and BM25 indexes are no longer stage-1-only: since the history-content
features (src/history_content.py, config.HISTORY_CONTENT_FEATURES) the shipped
re-ranker scores every request's candidates against the user's history with
both, so they are part of the serving footprint and their per-request cost is
timed inside stage 2a.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import article_stats, config, fresh_pool, history_content, sessionize  # noqa: E402
from src.candidate_features import build_candidate_features  # noqa: E402
from src.feature_pipeline import load_articles_lookup  # noqa: E402
from src.impression_features import build_impression_features  # noqa: E402

PROCESSED_DIRS = config.PROCESSED_DIRS
PERCENTILES = (50, 95, 99)
_PAGE = os.sysconf("SC_PAGE_SIZE")


def rss_bytes() -> int:
    """Resident set size. Catches numpy's direct allocations, which tracemalloc
    misses because large arrays bypass Python's allocator."""
    with open("/proc/self/statm") as handle:
        return int(handle.read().split()[1]) * _PAGE


def build_component(name: str, processed_dir, dataset: str, split: str):
    """Build one component in isolation and report what it retains.

    Each runs in its own process (see `measure_components`). Measuring them
    sequentially in one process does not work: pandas frees large intermediate
    buffers during a build, so RSS can fall while a component is being created --
    an earlier version of this script reported the article index at *minus*
    126 MiB. Deps are built first and RSS is snapshotted after them, so the
    delta is attributable to this component alone.
    """
    explicit = None

    if name == "behaviors_split":
        gc.collect(); before = rss_bytes(); started = time.perf_counter()
        obj = sessionize.add_session_context(
            pd.read_parquet(processed_dir / f"behaviors_{split}.parquet"), dataset
        )
        explicit = int(obj.memory_usage(deep=True).sum())
    elif name == "articles_lookup":
        gc.collect(); before = rss_bytes(); started = time.perf_counter()
        obj = load_articles_lookup(processed_dir)
    elif name == "article_stats_index":
        gc.collect(); before = rss_bytes(); started = time.perf_counter()
        obj = article_stats.TrainEventIndex(
            processed_dir, dataset, splits=config.ARTICLE_STATS_SPLITS,
            click_lag_minutes=config.CLICK_REPORTING_LAG_MINUTES,
        )
        explicit = index_array_bytes(obj)
    elif name == "fresh_pool_log":
        gc.collect(); before = rss_bytes(); started = time.perf_counter()
        obj = fresh_pool.load_log(processed_dir, split)
        explicit = int(obj.memory_usage(deep=True).sum())
    elif name == "fresh_pool_windows":
        log = fresh_pool.load_log(processed_dir, split)  # dep, excluded from the delta
        gc.collect(); before = rss_bytes(); started = time.perf_counter()
        obj = fresh_pool.FreshPool(log)
    elif name == "lightgbm_booster":
        import lightgbm as lgb
        path = next((processed_dir / "reranker").glob("lambdamart_*.txt"))
        gc.collect(); before = rss_bytes(); started = time.perf_counter()
        obj = lgb.Booster(model_file=str(path))
    elif name == "bm25_inverted_index":
        from src.lexical_retrieval import build_corpus_index
        gc.collect(); before = rss_bytes(); started = time.perf_counter()
        obj = build_corpus_index(processed_dir)
        explicit = sum(d.nbytes + t.nbytes for d, t in obj._postings.values())
    elif name == "semantic_ann_index":
        from src.semantic_retrieval import load_embeddings_lookup
        gc.collect(); before = rss_bytes(); started = time.perf_counter()
        _, obj = load_embeddings_lookup(processed_dir)
        explicit = obj.embeddings.nbytes
    else:
        raise SystemExit(f"unknown component {name}")

    elapsed = time.perf_counter() - started
    gc.collect()
    entry = {"rss_mib": round((rss_bytes() - before) / 1024**2, 1), "build_seconds": round(elapsed, 2)}
    if explicit is not None:
        entry["explicit_mib"] = round(explicit / 1024**2, 1)
    return entry


COMPONENTS = (
    "behaviors_split",
    "articles_lookup",
    "article_stats_index",
    "fresh_pool_log",
    "fresh_pool_windows",
    "lightgbm_booster",
    "bm25_inverted_index",
    "semantic_ann_index",
)


def measure_components(dataset: str, split: str) -> dict:
    """One subprocess per component, so nothing contaminates anything else."""
    import subprocess

    entries: dict[str, dict] = {}
    for name in COMPONENTS:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--dataset", dataset,
             "--split", split, "--measure-component", name],
            capture_output=True, text=True,
        )
        if completed.returncode != 0:
            entries[name] = {"error": (completed.stderr.strip().splitlines() or ["failed"])[-1][:200]}
        else:
            entries[name] = json.loads(completed.stdout)
    return entries


def index_array_bytes(index: article_stats.TrainEventIndex) -> int:
    displays = sum(a.nbytes for a in index._display_times.values())
    clicks = sum(a.nbytes for a in index._click_times.values())
    return displays + clicks


def percentiles(samples_ms: list[float]) -> dict:
    array = np.asarray(samples_ms, dtype=np.float64)
    out = {f"p{p}_ms": round(float(np.percentile(array, p)), 3) for p in PERCENTILES}
    out["mean_ms"] = round(float(array.mean()), 3)
    return out


def sample_requests(behaviors: pd.DataFrame, n: int) -> pd.DataFrame:
    """A contiguous, time-ordered block. FreshPool requires non-decreasing query
    times, and a random sample would also misrepresent the pool: its size varies
    over the day, so latency has to be sampled across a real stretch of traffic."""
    ordered = behaviors.sort_values("time")
    if len(ordered) <= n:
        return ordered
    start = (len(ordered) - n) // 2  # mid-split: avoids the cold first hour
    return ordered.iloc[start : start + n]


def benchmark_latency(requests, pool, articles_lookup, article_index, booster, features, k, content_scorer=None):
    """One timed pass. Returns per-stage millisecond samples plus pool sizes."""
    stage1, stage2a, stage2b, totals, pool_sizes, ingest = [], [], [], [], [], []

    for row in requests.itertuples(index=False):
        t_ns = pd.Timestamp(row.time).value

        # Ingest, not request work: a live system advanced these counters as the
        # events arrived. Timed separately so the cost is visible, not hidden.
        started = time.perf_counter()
        pool.advance_to(t_ns)
        ingest.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        ranked, _ = pool.popular(k)
        t1 = time.perf_counter()

        impression_features = build_impression_features(row, articles_lookup)
        weights = impression_features["_history_category_weights"]
        shared = {key: value for key, value in impression_features.items() if not key.startswith("_")}
        content = (
            content_scorer.score(row.history, [article_id for article_id, _ in ranked])
            if content_scorer is not None else None
        )
        matrix = np.empty((len(ranked), len(features)), dtype=np.float64)
        for position, (article_id, score) in enumerate(ranked):
            row_features = dict(shared)
            row_features.update(
                build_candidate_features(article_id, position + 1, score, row.time, weights, article_index)
            )
            if content is not None:
                row_features["history_title_bm25"] = content[0][position]
                row_features["history_embedding_cosine"] = content[1][position]
            matrix[position] = [row_features[name] for name in features]
        t2 = time.perf_counter()

        scores = booster.predict(matrix)
        np.argsort(-scores)
        t3 = time.perf_counter()

        stage1.append((t1 - started) * 1000)
        stage2a.append((t2 - t1) * 1000)
        stage2b.append((t3 - t2) * 1000)
        totals.append((t3 - started) * 1000)
        pool_sizes.append(len(ranked))

    return {
        "stage1_candidate_generation": percentiles(stage1),
        "stage2a_feature_building": percentiles(stage2a),
        "stage2b_model_scoring": percentiles(stage2b),
        "total_request": percentiles(totals),
        "ingest_per_impression_excluded_from_request": percentiles(ingest),
        "candidates_per_request_mean": round(float(np.mean(pool_sizes)), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--method", default="popular", help="which trained re-ranker to time")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--k", type=int, default=config.CANDIDATE_K)
    parser.add_argument("--scaling-ks", default="50,100,200", help="K values for the stage-2 scaling curve")
    parser.add_argument("--sla-ms", type=float, default=100.0, help="target p99 (A2 Q4 suggests 100ms)")
    parser.add_argument("--instance-usd-per-hour", type=float, default=0.17,
                        help="priced per vCPU-hour; recorded in the JSON so the estimate is auditable")
    parser.add_argument("--out", default=None)
    parser.add_argument("--measure-component", default=None,
                        help="internal: build one component in this process and print its footprint")
    args = parser.parse_args()

    processed_dir = PROCESSED_DIRS[args.dataset]

    if args.measure_component:
        print(json.dumps(build_component(args.measure_component, processed_dir, args.dataset, args.split)))
        return

    import lightgbm as lgb

    report: dict = {
        "dataset": args.dataset,
        "split": args.split,
        "config": {
            "k": args.k,
            "requests": args.requests,
            "warmup": args.warmup,
            "sla_ms": args.sla_ms,
            "instance_usd_per_hour": args.instance_usd_per_hour,
            "article_stats_splits": list(config.ARTICLE_STATS_SPLITS),
            "fresh_window_hours": config.FRESH_WINDOW_HOURS,
            "click_lag_minutes": config.CLICK_REPORTING_LAG_MINUTES,
        },
        "notes": [],
    }

    # ---- memory: one isolated subprocess per component ----------------------
    report["memory"] = measure_components(args.dataset, args.split)
    report["memory"]["_method"] = (
        "Each component is built in its own process; rss_mib is the resident-set "
        "growth attributable to it once its dependencies are already loaded, and "
        "explicit_mib is the exact array/frame bytes where they are walkable. "
        "With history-content features on, the shipped `popular` path also holds "
        "bm25_inverted_index and semantic_ann_index (they score each request's "
        "candidates against the user's history), so its serving footprint is the "
        "fresh-pool window, the article statistics index and those two."
    )

    # ---- startup state for the latency pass --------------------------------
    behaviors = sessionize.add_session_context(
        pd.read_parquet(processed_dir / f"behaviors_{args.split}.parquet"), args.dataset
    )
    articles_lookup = load_articles_lookup(processed_dir)
    article_index = article_stats.TrainEventIndex(
        processed_dir, args.dataset, splits=config.ARTICLE_STATS_SPLITS,
        click_lag_minutes=config.CLICK_REPORTING_LAG_MINUTES,
    )
    log = fresh_pool.load_log(processed_dir, args.split)
    pool = fresh_pool.FreshPool(log)

    model_path = processed_dir / "reranker" / f"lambdamart_{args.method}.txt"
    if not model_path.exists():
        available = sorted(p.stem.replace("lambdamart_", "") for p in (processed_dir / "reranker").glob("lambdamart_*.txt"))
        if not available:
            raise SystemExit(f"no trained re-ranker under {processed_dir/'reranker'}")
        fallback = available[0]
        report["notes"].append(
            f"'{args.method}' model not on this machine; timed '{fallback}' instead. "
            "Tree count drives scoring latency, so re-run where the shipped model lives."
        )
        model_path = processed_dir / "reranker" / f"lambdamart_{fallback}.txt"

    booster = lgb.Booster(model_file=str(model_path))
    features = booster.feature_name()
    # Startup, not request work: built once, like the article index above.
    content_scorer = (
        history_content.get_scorer(processed_dir, args.dataset)
        if set(history_content.FEATURES) <= set(features) else None
    )
    report["model"] = {
        "path": str(model_path.relative_to(processed_dir.parent.parent)),
        "num_trees": booster.num_trees(),
        "num_features": len(features),
        "history_content_features": content_scorer is not None,
    }

    # ---- latency -------------------------------------------------------------
    requests = sample_requests(behaviors, args.requests + args.warmup)
    warm, timed = requests.iloc[: args.warmup], requests.iloc[args.warmup :]
    benchmark_latency(warm, pool, articles_lookup, article_index, booster, features, args.k, content_scorer)
    report["latency"] = benchmark_latency(
        timed, pool, articles_lookup, article_index, booster, features, args.k, content_scorer
    )
    report["latency"]["n_requests"] = int(len(timed))

    # ---- throughput and cost -------------------------------------------------
    mean_ms = report["latency"]["total_request"]["mean_ms"]
    p99_ms = report["latency"]["total_request"]["p99_ms"]
    qps = 1000.0 / mean_ms
    report["throughput_and_cost"] = {
        "single_core_qps": round(qps, 1),
        "p99_ms": p99_ms,
        "meets_sla": bool(p99_ms < args.sla_ms),
        "sla_ms": args.sla_ms,
        "usd_per_1000_queries": round((1000.0 / qps / 3600.0) * args.instance_usd_per_hour, 6),
        "_method": (
            "Single-threaded, one request at a time, so QPS is per core and scales "
            "with cores until memory or ingest becomes the limit. Cost assumes the "
            "core is saturated; real serving provisions for peak, so divide by the "
            "target utilisation to size a fleet."
        ),
    }

    # ---- measured scaling points --------------------------------------------
    scaling = {}
    for k in [int(x) for x in args.scaling_ks.split(",")]:
        subset = timed.iloc[: min(200, len(timed))]
        pool_k = fresh_pool.FreshPool(log)
        result = benchmark_latency(subset, pool_k, articles_lookup, article_index, booster, features, k, content_scorer)
        scaling[f"k={k}"] = {
            "total_ms_mean": result["total_request"]["mean_ms"],
            "stage2a_ms_mean": result["stage2a_feature_building"]["mean_ms"],
            "stage2b_ms_mean": result["stage2b_model_scoring"]["mean_ms"],
        }

    split_growth = {}
    for splits in [("train",), ("train", "val"), config.ARTICLE_STATS_SPLITS]:
        gc.collect()
        before = rss_bytes()
        index = article_stats.TrainEventIndex(processed_dir, args.dataset, splits=splits)
        split_growth["+".join(splits)] = {
            "rss_delta_mib": round((rss_bytes() - before) / 1024**2, 1),
            "explicit_mib": round(index_array_bytes(index) / 1024**2, 1),
            "distinct_articles": len(index._display_times),
        }
        del index

    report["scaling"] = {
        "latency_vs_k": scaling,
        "article_index_vs_event_volume": split_growth,
        "_method": (
            "latency_vs_k isolates what grows with candidate count (feature building "
            "and scoring); article_index_vs_event_volume grows the event log instead, "
            "which is the axis 10x traffic moves along."
        ),
    }

    out_path = Path(args.out) if args.out else Path("results") / f"serving_{args.dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    latency = report["latency"]
    print(f"{args.dataset} {args.split}: {report['latency']['n_requests']} timed requests, K={args.k}")
    for stage in ("stage1_candidate_generation", "stage2a_feature_building", "stage2b_model_scoring", "total_request"):
        s = latency[stage]
        print(f"  {stage:34s} p50={s['p50_ms']:8.3f}  p95={s['p95_ms']:8.3f}  p99={s['p99_ms']:8.3f} ms")
    cost = report["throughput_and_cost"]
    print(f"  single-core QPS {cost['single_core_qps']}  |  p99 {cost['p99_ms']}ms vs {args.sla_ms}ms SLA: "
          f"{'PASS' if cost['meets_sla'] else 'FAIL'}  |  ${cost['usd_per_1000_queries']}/1k queries")
    for note in report["notes"]:
        print(f"  NOTE: {note}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
