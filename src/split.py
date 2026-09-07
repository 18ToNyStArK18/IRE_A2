"""Temporal train/val/test split of behaviors (never random).

Strategy (see config.py for the rationale): the dataset's own later period
(raw_split == "dev") becomes TEST. Within the earlier period
(raw_split == "train"), rows are sorted by impression time and the last
VAL_FRACTION_OF_TRAIN_PERIOD of that period (by time, not row count) becomes
VAL; everything strictly before that cutoff is TRAIN. This keeps a strict
temporal ordering max(train.time) <= min(val.time) <= min(test.time) and
never shuffles rows.

articles.parquet (the global catalog used to build the search index) is
copied through unchanged -- it is NOT split, per the assignment.
"""

from __future__ import annotations

import pandas as pd

from src import config


def _split_one(behaviors: pd.DataFrame, val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_period = behaviors[behaviors["raw_split"] == "train"].sort_values("time")
    test = behaviors[behaviors["raw_split"] == "dev"].sort_values("time")

    t_min, t_max = train_period["time"].min(), train_period["time"].max()
    cutoff = t_min + (t_max - t_min) * (1 - val_fraction)

    train = train_period[train_period["time"] < cutoff]
    val = train_period[train_period["time"] >= cutoff]

    assert train["time"].max() <= val["time"].min(), "train/val temporal overlap"
    assert val["time"].max() <= test["time"].min(), "val/test temporal overlap"

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def split_dataset(
    interim_dir,
    processed_dir,
    val_fraction: float = config.VAL_FRACTION_OF_TRAIN_PERIOD,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)

    behaviors = pd.read_parquet(interim_dir / "behaviors.parquet")
    train, val, test = _split_one(behaviors, val_fraction)

    train.to_parquet(processed_dir / "behaviors_train.parquet", index=False)
    val.to_parquet(processed_dir / "behaviors_val.parquet", index=False)
    test.to_parquet(processed_dir / "behaviors_test.parquet", index=False)

    articles = pd.read_parquet(interim_dir / "articles.parquet")
    articles.to_parquet(processed_dir / "articles.parquet", index=False)

    print(
        f"[{processed_dir.name}] train={len(train)} "
        f"({train['time'].min()} .. {train['time'].max()}), "
        f"val={len(val)} ({val['time'].min()} .. {val['time'].max()}), "
        f"test={len(test)} ({test['time'].min()} .. {test['time'].max()})"
    )


def split_all() -> None:
    split_dataset(config.MIND_INTERIM_DIR, config.MIND_PROCESSED_DIR)
    split_dataset(config.EBNERD_INTERIM_DIR, config.EBNERD_PROCESSED_DIR)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    args = parser.parse_args()

    if args.dataset in ("mind", "all"):
        split_dataset(config.MIND_INTERIM_DIR, config.MIND_PROCESSED_DIR)
    if args.dataset in ("ebnerd", "all"):
        split_dataset(config.EBNERD_INTERIM_DIR, config.EBNERD_PROCESSED_DIR)


if __name__ == "__main__":
    main()
