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

import numpy as np
import torch

from src.nrms import adapter, articles as article_utils, config as nrms_config, evaluate, train
from src.nrms.ids import load_or_build_codec
from src.nrms.model import NRMS
from src.nrms.sampling import sampling_strategy_wu2019


def build_everything(args):
    dataset = args.dataset
    processed_dir = nrms_config.PROCESSED_DIRS[dataset]
    artifact_dir = nrms_config.artifact_dir(dataset)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not processed_dir.exists():
        raise SystemExit(
            f"{processed_dir} not found -- run `python build_pipeline.py --dataset {dataset}` first"
        )

    print(f"[nrms] {dataset}: building article id codec")
    codec = load_or_build_codec(processed_dir, artifact_dir)
    print(f"[nrms] {dataset}: {codec.n_articles} articles (code 0 reserved for padding)")

    model_name = nrms_config.TEXT_ENCODER[dataset]
    print(f"[nrms] {dataset}: tokenizing article text with {model_name}")
    tokenizer = article_utils.load_tokenizer(model_name)
    article_text = adapter.load_article_text(processed_dir, nrms_config.TEXT_COLUMNS[dataset])
    token_matrix = article_utils.load_or_build_token_matrix(
        artifact_dir, article_text, codec, tokenizer, args.title_size
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
    args = parser.parse_args()

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
        print(f"[nrms] {dataset}: prepared token matrix {token_matrix.shape} -> {artifact_dir}")
        return

    device = train.resolve_device(args.device)
    print(f"[nrms] device: {device}")

    print(f"[nrms] {dataset}: loading pretrained word embeddings from {model_name}")
    embedding_weights = article_utils.load_word_embeddings(model_name)
    model = NRMS(
        embedding_weights,
        num_heads=nrms_config.HEAD_NUM,
        head_dim=nrms_config.HEAD_DIM,
        attention_hidden_dim=nrms_config.ATTENTION_HIDDEN_DIM,
        dropout=nrms_config.DROPOUT,
        padding_idx=tokenizer.pad_token_id,
    )

    checkpoint_path = artifact_dir / "nrms.pt"
    if args.stage in ("train", "all"):
        train_impressions = adapter.load_impressions(
            processed_dir, "train", codec, args.history_size
        )
        if args.fraction < 1.0:
            train_impressions = train_impressions.sample(
                frac=args.fraction, random_state=args.seed
            ).reset_index(drop=True)
        val_impressions = adapter.load_impressions(processed_dir, "val", codec, args.history_size)
        print(
            f"[nrms] {dataset}: {len(train_impressions)} train / "
            f"{len(val_impressions)} val impressions"
        )

        train_samples = sampling_strategy_wu2019(train_impressions, args.npratio, seed=args.seed)
        val_samples = sampling_strategy_wu2019(val_impressions, args.npratio, seed=args.seed)
        print(
            f"[nrms] {dataset}: {len(train_samples)} train / {len(val_samples)} val "
            f"sampled rows (1 positive + {args.npratio} negatives each)"
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
        )
    elif checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[nrms] loaded checkpoint {checkpoint_path}")
    else:
        raise SystemExit(f"no checkpoint at {checkpoint_path} -- run --stage train first")

    if args.stage in ("eval", "all"):
        impressions = adapter.load_impressions(
            processed_dir, args.eval_split, codec, args.history_size
        )
        print(f"[nrms] {dataset}: scoring {len(impressions)} {args.eval_split} impressions")
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
        )


if __name__ == "__main__":
    main()
