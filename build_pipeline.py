#!/usr/bin/env python3
"""One-command rebuild of the entire dataset pipeline, from raw URLs to a
ready-to-use feature store.

    python build_pipeline.py                       # both datasets, all stages
    python build_pipeline.py --dataset mind         # one dataset only
    python build_pipeline.py --stage download       # one stage only
    python build_pipeline.py --skip-download        # reuse already-downloaded raw files
    python build_pipeline.py --hf-token <token>      # or set HF_TOKEN env var (needed for MIND)

Stages run in order: download -> parse -> split -> feature_store -> popularity.
"""

from __future__ import annotations

import argparse

from src import config, download, embeddings_index, feature_store, parse, popularity, split

STAGES = ("download", "parse", "split", "feature_store", "popularity", "embeddings")


def run(dataset: str, stage: str, hf_token: str | None, ebnerd_bundle: str, force_download: bool) -> None:
    datasets = ("mind", "ebnerd") if dataset == "all" else (dataset,)
    stages = STAGES if stage == "all" else (stage,)

    if "download" in stages:
        if "mind" in datasets:
            download.download_mind(hf_token=hf_token, force=force_download)
        if "ebnerd" in datasets:
            download.download_ebnerd(bundle=ebnerd_bundle, force=force_download)
            # EB-NeRD's semantic retriever loads the provided article
            # embeddings rather than recomputing them, so the artifact is part
            # of the rebuild -- MIND has no such artifact and computes its own
            # in the embeddings stage.
            download.download_ebnerd_artifact(force=force_download)

    if "parse" in stages:
        if "mind" in datasets:
            parse.parse_mind()
        if "ebnerd" in datasets:
            parse.parse_ebnerd(bundle=ebnerd_bundle)

    if "split" in stages:
        if "mind" in datasets:
            split.split_dataset(config.MIND_INTERIM_DIR, config.MIND_PROCESSED_DIR)
        if "ebnerd" in datasets:
            split.split_dataset(config.EBNERD_INTERIM_DIR, config.EBNERD_PROCESSED_DIR)

    if "feature_store" in stages:
        if "mind" in datasets:
            feature_store.build_feature_store(config.MIND_PROCESSED_DIR)
        if "ebnerd" in datasets:
            feature_store.build_feature_store(config.EBNERD_PROCESSED_DIR)

    if "popularity" in stages:
        if "mind" in datasets:
            popularity.compute_popularity(config.MIND_PROCESSED_DIR)
        if "ebnerd" in datasets:
            popularity.compute_popularity(config.EBNERD_PROCESSED_DIR)

    if "embeddings" in stages:
        # Article embeddings for the semantic retriever (src/candidates.py).
        # MIND computes its own with MIND_SEMANTIC_MODEL; EB-NeRD loads the
        # provided artifact downloaded above.
        if "mind" in datasets:
            embeddings_index.build_mind_embeddings()
        if "ebnerd" in datasets:
            embeddings_index.load_ebnerd_provided_embeddings()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for gated MIND repo (or set HF_TOKEN)")
    parser.add_argument("--ebnerd-bundle", choices=["demo", "small"], default=config.EBNERD_BUNDLE)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true", help="Shortcut for --stage without download")
    args = parser.parse_args()

    if args.skip_download and args.stage == "all":
        for stage in ("parse", "split", "feature_store", "popularity", "embeddings"):
            run(args.dataset, stage, args.hf_token, args.ebnerd_bundle, args.force_download)
        return

    run(args.dataset, args.stage, args.hf_token, args.ebnerd_bundle, args.force_download)


if __name__ == "__main__":
    main()
