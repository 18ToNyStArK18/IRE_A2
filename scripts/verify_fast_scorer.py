#!/usr/bin/env python3
"""Gate for the submission scorer: does the cached-article-vector path reproduce
the numbers the per-impression path already committed?

    python scripts/verify_fast_scorer.py --dataset ebnerd

`src/nrms/serve.py` claims that encoding the catalogue once and gathering
vectors by row index is *identical* to re-encoding every candidate per
impression, because `NewsEncoder` has no context. That claim is the only thing
standing between a valid submission and 13.5M rows of quietly wrong ranking, so
it is checked numerically rather than argued: re-score our own labelled test
split through the fast path and require the metrics in
`data/processed/<ds>/nrms/<arm>/metrics_test.json` back (`--arm baseline` checks
the unflagged run at the artifact root).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import evaluate_impressions  # noqa: E402
from src.nrms import adapter, config as nrms_config, serve, train  # noqa: E402
from src.nrms.freshness import FreshnessLookup  # noqa: E402
from src.nrms.ids import load_or_build_codec  # noqa: E402

TOLERANCE = 5e-4  # GPU reductions are not bit-exact across batch shapes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--arm", default="freshness", help="run-tag subdirectory, or 'baseline' for the root run")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N impressions")
    args = parser.parse_args()

    processed_dir = nrms_config.PROCESSED_DIRS[args.dataset]
    artifact_dir = nrms_config.artifact_dir(args.dataset)
    device = train.resolve_device(args.device)

    codec = load_or_build_codec(processed_dir, artifact_dir)
    token_matrix = np.load(next(artifact_dir.glob("article_tokens_*.npy")))
    impressions = adapter.load_impressions(processed_dir, args.split, codec, nrms_config.HISTORY_SIZE)
    if args.limit:
        impressions = impressions.head(args.limit)

    # The baseline writes to the artifact root; every other arm to run_dir/<tag>.
    run_dir = artifact_dir if args.arm == "baseline" else artifact_dir / args.arm
    model = serve.load_trained_model(run_dir / "nrms.pt", device)  # arm read off the checkpoint

    # The processed token matrix already reserves code 0 for padding.
    article_vectors = serve.encode_catalogue(model, token_matrix, device, prepend_pad=False)
    print(f"{args.dataset}: encoded {article_vectors.shape[0]:,} articles -> {tuple(article_vectors.shape)}")

    history_rows = np.stack([np.asarray(h, dtype=np.int64) for h in impressions["history"]])
    widths = np.asarray([len(c) for c in impressions["candidates"]], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(widths)])
    cand_rows = np.concatenate([np.asarray(c, dtype=np.int64) for c in impressions["candidates"]])
    log_age = known = None
    if model.freshness_head is not None:
        lookup = FreshnessLookup.build(processed_dir, args.dataset, codec)
        times = impressions["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        log_age, known = lookup.ages(cand_rows, np.repeat(times, widths))

    scores = serve.score_chunk(
        model, article_vectors, history_rows, cand_rows, offsets,
        log_age=log_age, known=known, row_offset=0,
    )
    per_impression = [scores[offsets[i] : offsets[i + 1]] for i in range(len(widths))]
    labels = [np.asarray(l, dtype=np.float32) for l in impressions["labels"]]
    fast = evaluate_impressions(labels, per_impression, ndcg_ks=(5, 10))

    committed = json.loads((run_dir / f"metrics_{args.split}.json").read_text())
    print(f"{'metric':8s} {'committed':>10s} {'fast path':>10s} {'delta':>10s}")
    worst = 0.0
    for name in ("auc", "mrr", "ndcg@5", "ndcg@10"):
        delta = fast[name] - committed[name]
        worst = max(worst, abs(delta))
        print(f"{name:8s} {committed[name]:10.4f} {fast[name]:10.4f} {delta:+10.6f}")

    if args.limit:
        print(f"\n[skipped] --limit {args.limit} scores a subset, so the metrics are not comparable")
        return
    if worst > TOLERANCE:
        raise SystemExit(f"\nFAIL: worst deviation {worst:.6f} exceeds {TOLERANCE}; do not submit")
    print(f"\n[ok] cached-vector path matches the committed metrics (worst deviation {worst:.2e})")


if __name__ == "__main__":
    main()
