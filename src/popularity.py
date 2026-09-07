"""Static "Top Popular Articles" fallback list, computed strictly from
behaviors_train (never val/test, to avoid leaking future popularity).

Popularity = raw click count (label == 1) per article_id in the train split.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src import config


def compute_popularity(processed_dir, top_n: int = config.POPULARITY_TOP_N) -> pd.DataFrame:
    train = pd.read_parquet(processed_dir / "behaviors_train.parquet")

    counts = Counter()
    for candidates, labels in zip(train["candidates"], train["labels"]):
        for article_id, label in zip(candidates, labels):
            if label == 1:
                counts[article_id] += 1

    pop = (
        pd.DataFrame(counts.items(), columns=["article_id", "click_count"])
        .sort_values("click_count", ascending=False)
        .reset_index(drop=True)
    )
    pop["rank"] = pop.index + 1

    pop.to_parquet(processed_dir / "popularity.parquet", index=False)
    top = pop.head(top_n)
    top.to_parquet(processed_dir / "popularity_top.parquet", index=False)

    print(
        f"[{processed_dir.name}] popularity: {len(pop)} distinct clicked articles, "
        f"top-{top_n} fallback list saved"
    )
    return top


def compute_all(top_n: int = config.POPULARITY_TOP_N) -> None:
    compute_popularity(config.MIND_PROCESSED_DIR, top_n)
    compute_popularity(config.EBNERD_PROCESSED_DIR, top_n)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--top-n", type=int, default=config.POPULARITY_TOP_N)
    args = parser.parse_args()

    if args.dataset in ("mind", "all"):
        compute_popularity(config.MIND_PROCESSED_DIR, args.top_n)
    if args.dataset in ("ebnerd", "all"):
        compute_popularity(config.EBNERD_PROCESSED_DIR, args.top_n)


if __name__ == "__main__":
    main()
