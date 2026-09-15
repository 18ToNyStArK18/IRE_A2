#!/usr/bin/env python3
"""Q5/Q7.3: generate Codabench submission files by ranking each impression's
in-view list with the trained NRMS freshness arm.

    python run_submit.py --dataset mind --limit 5000     # smoke test
    python run_submit.py --dataset mind                  # full file
    python run_submit.py --dataset ebnerd

Boards (both open, 10 submissions/day as of 2026-09-15):
    MIND    https://www.codabench.org/competitions/13967/
    EB-NeRD https://www.codabench.org/competitions/2469/

Why NRMS and not the two-stage re-ranker: the boards score a ranking of the
impression's *given* in-view list. The re-ranker ranks a retrieved fresh pool,
and its click-driven features are empty on an unlabelled test set, so it has
nothing to say about a list it did not choose. NRMS scores supplied candidates
natively -- that is the task it was trained on.

Everything streams (`src/submission.py`) and scoring uses the cached
article-vector path (`src/nrms/serve.py`), which `scripts/verify_fast_scorer.py`
shows reproduces the per-impression metrics exactly. Files land in
data/submissions/ -- gitignored, which Q8 requires.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from src import config, submission
from src.nrms import articles as article_utils, config as nrms_config, serve, train
from src.nrms.freshness import FreshnessLookup
from src.nrms.tracking import get_logger, setup_logging

log = get_logger("submit")

#: The official test sets ship separately from the demo/small bundles and are
#: large, so like data/raw/{mind,ebnerd} they are symlinked into place rather
#: than copied. Override with --test-root on a machine laid out differently.
TEST_ROOTS = {
    "mind": config.RAW_DIR / "testsets" / "MINDlarge_test",
    "ebnerd": config.RAW_DIR / "testsets" / "ebnerd_testset",
}
#: MIND's candidate lists are ~3x wider than EB-NeRD's, so it uses smaller chunks.
CHUNK = {"mind": 50_000, "ebnerd": 200_000}


def resolve_root(dataset: str, override: str | None) -> Path:
    root = Path(override) if override else TEST_ROOTS[dataset]
    if not root.exists():
        raise SystemExit(
            f"{root} not found -- the official test sets ship separately from the "
            "demo/small bundles this project trains on; pass --test-root if yours lives elsewhere"
        )
    return root


def catalogue_tokens(dataset: str, cat, title_size: int, cache_dir: Path, rebuild: bool) -> np.ndarray:
    """(n_articles, title_size) token ids in catalogue row order.

    Same tokenizer, `padding="max_length"` and truncation as
    `articles.build_token_matrix` used for training -- a different tokenisation
    here would feed the news encoder text it never saw in that shape. Cached
    because EB-NeRD's 125k articles carry full bodies.
    """
    model_name = nrms_config.TEXT_ENCODER[dataset]
    fingerprint = hashlib.sha1(
        "|".join([model_name, str(title_size), str(len(cat))]).encode()
    ).hexdigest()[:10]
    path = cache_dir / f"testset_tokens_{title_size}_{fingerprint}.npy"
    if path.exists() and not rebuild:
        log.info("  tokens: reusing %s", path.name)
        return np.load(path)

    tokenizer = article_utils.load_tokenizer(model_name)
    matrix = np.zeros((len(cat), title_size), dtype=np.int32)
    batch = 1024
    started = time.perf_counter()
    for start in range(0, len(cat.text), batch):
        chunk = cat.text[start : start + batch]
        matrix[start : start + len(chunk)] = tokenizer(
            chunk, padding="max_length", truncation=True, max_length=title_size, return_tensors="np"
        )["input_ids"].astype(np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    log.info("  tokens: %s articles in %.0fs -> %s", f"{len(cat):,}", time.perf_counter() - started, path.name)
    return matrix


def freshness_references(dataset: str, root: Path, cat, cache_dir: Path, rebuild: bool) -> np.ndarray:
    """Per-catalogue-row reference time in ns, row 0 reserved for padding.

    EB-NeRD carries a real `published_time`; MIND has none, so its reference is
    the earliest sighting in the test log -- the same proxy
    `article_stats.freshness_reference_times` uses, sourced from the period
    being scored. `FreshnessLookup.ages` still gates on the reference strictly
    preceding the impression, so nothing later in the file dates an earlier row.
    """
    if dataset == "ebnerd":
        reference = cat.reference_ns
    else:
        path = cache_dir / "testset_first_seen.npy"
        if path.exists() and not rebuild:
            log.info("  first-seen: reusing %s", path.name)
            reference = np.load(path)
        else:
            reference = submission.first_seen_reference_ns(root, cat, CHUNK[dataset])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, reference)
    # serve.py indexes vectors with row + 1, so the lookup needs the same shift.
    return np.concatenate([[np.nan], reference])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--arm", default="freshness", help="which trained NRMS run to score with")
    parser.add_argument("--limit", type=int, default=None, help="stop after N impressions (smoke test)")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--batch-pairs", type=int, default=200_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--test-root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--log-dir", default="log")
    args = parser.parse_args()

    started = time.perf_counter()
    log_path = setup_logging(f"submit_{args.dataset}", Path(args.log_dir))
    log.info("run log -> %s", log_path)

    dataset = args.dataset
    root = resolve_root(dataset, args.test_root)
    chunk_size = args.chunk_size or CHUNK[dataset]
    if args.limit:
        # Otherwise a smoke test rounds up to a whole chunk and scores 50,000
        # impressions when asked for 5,000.
        chunk_size = min(chunk_size, args.limit)
    device = train.resolve_device(args.device)
    cache_dir = nrms_config.PROCESSED_DIRS[dataset] / "submission"
    log.info("%s: test set %s, device %s", dataset, root, device)

    loader = submission.load_mind_catalogue if dataset == "mind" else submission.load_ebnerd_catalogue
    cat = loader(root)
    token_matrix = catalogue_tokens(dataset, cat, nrms_config.TITLE_SIZE, cache_dir, args.rebuild_cache)
    # Text is only needed for tokenisation; 125k EB-NeRD bodies are not small.
    cat.text = []

    checkpoint = nrms_config.artifact_dir(dataset) / args.arm / "nrms.pt"
    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint at {checkpoint} -- train the {args.arm} arm first")
    model = serve.load_trained_model(checkpoint, device, freshness=True)
    article_vectors = serve.encode_catalogue(model, token_matrix, device)
    log.info("  encoded %s articles -> %s", f"{len(cat):,}", tuple(article_vectors.shape))

    lookup = FreshnessLookup(freshness_references(dataset, root, cat, cache_dir, args.rebuild_cache))
    dated = int(np.isfinite(lookup.reference_ns).sum())
    log.info("  freshness: reference known for %s/%s articles (%.1f%%)",
             f"{dated:,}", f"{len(cat):,}", 100 * dated / max(len(cat), 1))

    total = (
        submission.count_mind_impressions(root) if dataset == "mind"
        else submission.count_ebnerd_impressions(root)
    )
    expected = submission.EXPECTED_IMPRESSIONS[dataset]
    if total != expected:
        log.warning("  %s impressions in this copy, official count is %s", f"{total:,}", f"{expected:,}")

    out = Path(args.out) if args.out else config.DATA_DIR / "submissions" / f"{dataset}_nrms_{args.arm}.zip"
    writer = submission.SubmissionWriter(out, dataset, resume=not args.no_resume)

    if dataset == "mind":
        chunks = submission.iter_mind_chunks(
            root, cat, nrms_config.HISTORY_SIZE, chunk_size, skip=writer.written
        )
    else:
        user_ids, user_history = submission.load_ebnerd_histories(root, cat, nrms_config.HISTORY_SIZE)
        chunks = submission.iter_ebnerd_chunks(
            root, cat, user_ids, user_history, chunk_size, skip=writer.written
        )

    target = min(args.limit, total) if args.limit else total
    for chunk in chunks:
        widths = np.diff(chunk.offsets)
        log_age, known = lookup.ages(
            chunk.cand_rows.astype(np.int64) + 1, np.repeat(chunk.impression_times, widths)
        )
        scores = serve.score_chunk(
            model, article_vectors, chunk.history_rows, chunk.cand_rows, chunk.offsets,
            log_age=log_age, known=known, batch_pairs=args.batch_pairs,
        )
        writer.write_chunk(chunk, scores)
        rate = writer.written / max(time.perf_counter() - started, 1e-9)
        log.info("  %s/%s impressions (%.0f/s, eta %.0f min)",
                 f"{writer.written:,}", f"{target:,}", rate, (target - writer.written) / rate / 60)
        if writer.written >= target:
            break

    if args.limit:
        log.info("smoke run: %s impressions scored, leaving %s unfinished (no zip written)",
                 f"{writer.written:,}", out.name)
        return

    writer.finalise(expected=total)
    submission.verify_submission(out, dataset, total)
    log.info("total wall time: %.1f min", (time.perf_counter() - started) / 60)


if __name__ == "__main__":
    main()
