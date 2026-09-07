"""Lightweight, reusable feature store.

Article features: one row per article_id -- title, abstract, body, category,
subcategory, entities, published_time, plus an `embedding` placeholder column
(filled in by Phase 3's embeddings_index.py; left as None here).

User features: one row per user_id *per split* -- the user's click history
and its length, snapshotted from their most recent impression within that
split. Scoping by split (rather than one global table) means val/test user
features never see history that leaked in from a later impression, and stay
consistent with whichever behaviors_{split}.parquet they're paired with.
"""

from __future__ import annotations

import pandas as pd

from src import config


def build_article_features(processed_dir) -> pd.DataFrame:
    articles = pd.read_parquet(processed_dir / "articles.parquet")
    feats = articles.copy()
    feats["embedding"] = None  # placeholder, populated in Phase 3
    return feats


def build_user_features(behaviors: pd.DataFrame) -> pd.DataFrame:
    latest = behaviors.sort_values("time").groupby("user_id", as_index=False).last()
    out = pd.DataFrame({
        "user_id": latest["user_id"],
        "history": latest["history"],
        "history_length": latest["history"].apply(len),
        "as_of_time": latest["time"],
    })
    return out.reset_index(drop=True)


def build_feature_store(processed_dir) -> None:
    fs_dir = processed_dir / "feature_store"
    fs_dir.mkdir(parents=True, exist_ok=True)

    article_feats = build_article_features(processed_dir)
    article_feats.to_parquet(fs_dir / "articles.parquet", index=False)

    n_users = 0
    for split in ("train", "val", "test"):
        behaviors = pd.read_parquet(processed_dir / f"behaviors_{split}.parquet")
        user_feats = build_user_features(behaviors)
        user_feats.to_parquet(fs_dir / f"users_{split}.parquet", index=False)
        n_users += len(user_feats)

    print(
        f"[{processed_dir.name}] feature store: {len(article_feats)} articles, "
        f"{n_users} user-feature rows across train/val/test -> {fs_dir}"
    )


def build_all() -> None:
    build_feature_store(config.MIND_PROCESSED_DIR)
    build_feature_store(config.EBNERD_PROCESSED_DIR)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    args = parser.parse_args()

    if args.dataset in ("mind", "all"):
        build_feature_store(config.MIND_PROCESSED_DIR)
    if args.dataset in ("ebnerd", "all"):
        build_feature_store(config.EBNERD_PROCESSED_DIR)


if __name__ == "__main__":
    main()
