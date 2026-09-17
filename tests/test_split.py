"""Temporal split (A1 Q1.3): never random, and train < val < test by time.

The split is what every leakage guarantee downstream rests on -- as-of features
assume no val/test impression precedes a train one -- so its ordering is pinned
on a hand-built log rather than trusted to the runtime asserts alone.
"""

from __future__ import annotations

import pandas as pd

from src import split

T0 = pd.Timestamp("2024-01-01")


def _behaviors():
    # Train period spans 0..100h and is deliberately stored out of order;
    # the dev period (later) becomes test.
    rows = [
        ("t3", "train", 95), ("t0", "train", 0), ("t2", "train", 50),
        ("t4", "train", 100), ("t1", "train", 10),
        ("d1", "dev", 130), ("d0", "dev", 120),
    ]
    return pd.DataFrame(
        {
            "impression_id": [r[0] for r in rows],
            "raw_split": [r[1] for r in rows],
            "time": [T0 + pd.Timedelta(hours=r[2]) for r in rows],
        }
    )


def test_split_is_strictly_ordered_train_val_test():
    train, val, test = split._split_one(_behaviors(), val_fraction=0.10)
    assert train["time"].max() <= val["time"].min()
    assert val["time"].max() <= test["time"].min()


def test_val_is_the_last_fraction_of_the_train_period_by_time_not_row_count():
    # 10% of a 100h period = the last 10h: only the 95h and 100h impressions,
    # even though that is 40% of the train-period rows.
    train, val, test = split._split_one(_behaviors(), val_fraction=0.10)
    assert set(val["impression_id"]) == {"t3", "t4"}
    assert set(train["impression_id"]) == {"t0", "t1", "t2"}
    assert set(test["impression_id"]) == {"d0", "d1"}


def test_split_neither_drops_nor_duplicates_rows_and_sorts_each_part():
    behaviors = _behaviors()
    parts = split._split_one(behaviors, val_fraction=0.10)
    ids = [i for part in parts for i in part["impression_id"]]
    assert sorted(ids) == sorted(behaviors["impression_id"])
    for part in parts:
        assert part["time"].is_monotonic_increasing


def test_split_dataset_writes_all_splits_and_copies_the_catalogue(tmp_path):
    interim, processed = tmp_path / "interim", tmp_path / "processed"
    interim.mkdir()
    _behaviors().to_parquet(interim / "behaviors.parquet", index=False)
    pd.DataFrame({"article_id": ["a", "b"]}).to_parquet(interim / "articles.parquet", index=False)

    split.split_dataset(interim, processed, val_fraction=0.10)

    for name, expected in (("train", 3), ("val", 2), ("test", 2)):
        assert len(pd.read_parquet(processed / f"behaviors_{name}.parquet")) == expected
    assert pd.read_parquet(processed / "articles.parquet")["article_id"].tolist() == ["a", "b"]
