#!/usr/bin/env python3
"""Train and evaluate the NRMS baseline (ebnerd-benchmark's starter model,
reimplemented in PyTorch -- see src/nrms/).

    python run_nrms.py --dataset ebnerd --stage all
    python run_nrms.py --dataset mind --stage all --epochs 3
    python run_nrms.py --dataset ebnerd --stage all --fraction 0.02 --epochs 1  # smoke run
    python run_nrms.py --dataset mind --stage eval                              # reuse checkpoint

Requires build_pipeline.py to have produced data/processed/<dataset>/ first.
Deliberately not a build_pipeline.py stage: that script builds the dataset,
and NRMS consumes its output.

`--fraction` subsamples training impressions, mirroring the benchmark's
TRAIN_FRACTION, for quick end-to-end checks on a new machine.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.nrms import adapter, articles as article_utils, config as nrms_config, evaluate, train
from src.nrms.ids import load_or_build_codec
from src.nrms.freshness import FreshnessLookup
from src.nrms.model import NRMS
from src.nrms.sampling import sampling_strategy_wu2019
from src.nrms.tracking import WandbTracker, get_logger, setup_logging

log = get_logger("run")


def build_everything(args):
    dataset = args.dataset
    processed_dir = nrms_config.PROCESSED_DIRS[dataset]
    artifact_dir = nrms_config.artifact_dir(dataset)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not processed_dir.exists():
        raise SystemExit(
            f"{processed_dir} not found -- run `python build_pipeline.py --dataset {dataset}` first"
        )

    log.info("%s: building article id codec", dataset)
    codec = load_or_build_codec(processed_dir, artifact_dir, rebuild=args.rebuild_cache)
    log.info("%s: %d articles (code 0 reserved for padding)", dataset, codec.n_articles)

    model_name = nrms_config.TEXT_ENCODER[dataset]
    text_columns = nrms_config.TEXT_COLUMNS[dataset]
    log.info("%s: tokenizing article text with %s", dataset, model_name)
    tokenizer = article_utils.load_tokenizer(model_name)
    article_text = adapter.load_article_text(processed_dir, text_columns)
    token_matrix = article_utils.load_or_build_token_matrix(
        artifact_dir,
        article_text,
        codec,
        tokenizer,
        args.title_size,
        model_name,
        text_columns,
        rebuild=args.rebuild_cache,
    )

    return dataset, processed_dir, artifact_dir, codec, tokenizer, token_matrix, model_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--stage", choices=["prepare", "train", "eval", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=nrms_config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=nrms_config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=nrms_config.LEARNING_RATE)
    parser.add_argument("--history-size", type=int, default=nrms_config.HISTORY_SIZE)
    parser.add_argument("--title-size", type=int, default=nrms_config.TITLE_SIZE)
    parser.add_argument("--npratio", type=int, default=nrms_config.NPRATIO)
    parser.add_argument("--fraction", type=float, default=1.0, help="subsample training impressions")
    parser.add_argument("--seed", type=int, default=nrms_config.SEED)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--log-dir", default="log", help="directory for run log files")
    parser.add_argument("--no-wandb", action="store_true", help="disable Weights & Biases tracking")
    parser.add_argument("--wandb-project", default="ire-a2-nrms")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--freshness",
        action="store_true",
        help="Q3 arm: give the scorer an article-age signal (see src/nrms/freshness.py)",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="subdirectory under the artifact dir for this run's checkpoint/metrics. "
             "Defaults to 'freshness' when --freshness is set, else the artifact dir "
             "root, so an ablation arm can never silently overwrite the baseline.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="regenerate the article id map and token matrix instead of reusing cached ones",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    log_path = setup_logging(args.dataset, Path(args.log_dir))
    log.info("run log -> %s", log_path)
    log.info("args: %s", vars(args))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    (
        dataset,
        processed_dir,
        artifact_dir,
        codec,
        tokenizer,
        token_matrix,
        model_name,
    ) = build_everything(args)

    if args.stage == "prepare":
        log.info("%s: prepared token matrix %s -> %s", dataset, token_matrix.shape, artifact_dir)
        return

    device = train.resolve_device(args.device)
    log.info("device: %s", device)

    log.info("%s: loading pretrained word embeddings from %s", dataset, model_name)
    embedding_weights = article_utils.load_word_embeddings(model_name)
    model = NRMS(
        embedding_weights,
        num_heads=nrms_config.HEAD_NUM,
        head_dim=nrms_config.HEAD_DIM,
        attention_hidden_dim=nrms_config.ATTENTION_HIDDEN_DIM,
        dropout=nrms_config.DROPOUT,
        # padding_idx deliberately unset -- the benchmark trains every embedding
        # row, and with no attention masking the pad vector does real work. See
        # NewsEncoder.__init__.
        freshness=args.freshness,
        freshness_hidden_dim=nrms_config.FRESHNESS_HIDDEN_DIM,
    )

    freshness_lookup = None
    if args.freshness:
        freshness_lookup = FreshnessLookup.build(processed_dir, dataset, codec)
        n_known = int(np.isfinite(freshness_lookup.reference_ns[1:]).sum())
        log.info(
            "freshness: reference time known for %s/%s catalogue articles (%.1f%%)",
            f"{n_known:,}", f"{codec.n_articles:,}", 100 * n_known / max(codec.n_articles, 1),
        )

    n_params = sum(p.numel() for p in model.parameters())
    n_embedding = model.news_encoder.embedding.weight.numel()
    log.info(
        "model: %s params total, %s in the embedding (%.1f%%), %s in attention layers",
        f"{n_params:,}", f"{n_embedding:,}", 100 * n_embedding / n_params,
        f"{n_params - n_embedding:,}",
    )

    tracker = WandbTracker.start(
        enabled=not args.no_wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name or f"{dataset}-{args.stage}",
        group=dataset,
        config={
            **vars(args),
            "text_encoder": model_name,
            "text_columns": list(nrms_config.TEXT_COLUMNS[dataset]),
            "n_articles": codec.n_articles,
            "vocab_size": embedding_weights.shape[0],
            "embedding_dim": embedding_weights.shape[1],
            "params_total": n_params,
            "params_embedding": n_embedding,
            "params_attention": n_params - n_embedding,
            "head_num": nrms_config.HEAD_NUM,
            "head_dim": nrms_config.HEAD_DIM,
            "attention_hidden_dim": nrms_config.ATTENTION_HIDDEN_DIM,
            "dropout": nrms_config.DROPOUT,
            "early_stopping_patience": nrms_config.EARLY_STOPPING_PATIENCE,
            "lr_plateau_factor": nrms_config.LR_PLATEAU_FACTOR,
            "lr_plateau_patience": nrms_config.LR_PLATEAU_PATIENCE,
        },
    )

    # Shared caches (codec, token matrix) stay in artifact_dir so arms reuse
    # them; per-arm outputs go under run_dir so a second arm cannot overwrite
    # the first's checkpoint and metrics.
    tag = args.run_tag or ("freshness" if args.freshness else None)
    run_dir = artifact_dir / tag if tag else artifact_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("run artifacts -> %s", run_dir)

    checkpoint_path = run_dir / "nrms.pt"
    try:
        _run_stages(args, dataset, processed_dir, run_dir, codec, token_matrix,
                    model, device, checkpoint_path, tracker, freshness_lookup)
    finally:
        elapsed = time.perf_counter() - started
        log.info("total wall time: %.1f min", elapsed / 60)
        tracker.set_summary({"wall_time_min": round(elapsed / 60, 2), "log_file": str(log_path)})
        tracker.finish()


def _run_stages(args, dataset, processed_dir, artifact_dir, codec, token_matrix,
                model, device, checkpoint_path, tracker, freshness_lookup) -> None:
    if args.stage in ("train", "all"):
        train_impressions = adapter.load_impressions(
            processed_dir, "train", codec, args.history_size
        )
        if args.fraction < 1.0:
            train_impressions = train_impressions.sample(
                frac=args.fraction, random_state=args.seed
            ).reset_index(drop=True)
        val_impressions = adapter.load_impressions(processed_dir, "val", codec, args.history_size)
        log.info(
            "%s: %d train / %d val impressions",
            dataset, len(train_impressions), len(val_impressions),
        )

        train_samples = sampling_strategy_wu2019(train_impressions, args.npratio, seed=args.seed)
        val_samples = sampling_strategy_wu2019(val_impressions, args.npratio, seed=args.seed)
        log.info(
            "%s: %d train / %d val sampled rows (1 positive + %d negatives each)",
            dataset, len(train_samples), len(val_samples), args.npratio,
        )

        train.train_model(
            model,
            train_samples,
            val_samples,
            token_matrix,
            artifact_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            num_workers=args.num_workers,
            tracker=tracker,
            freshness=freshness_lookup,
        )
    elif checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        log.info("loaded checkpoint %s", checkpoint_path)
    else:
        raise SystemExit(f"no checkpoint at {checkpoint_path} -- run --stage train first")

    if args.stage in ("eval", "all"):
        impressions = adapter.load_impressions(
            processed_dir, args.eval_split, codec, args.history_size
        )
        log.info("%s: scoring %d %s impressions", dataset, len(impressions), args.eval_split)
        evaluate.evaluate_split(
            model,
            impressions,
            token_matrix,
            codec,
            artifact_dir,
            split=args.eval_split,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
            tracker=tracker,
            freshness=freshness_lookup,
        )


if __name__ == "__main__":
    main()
