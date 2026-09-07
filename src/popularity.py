"""Static "Top Popular Articles" fallback list, computed strictly from
behaviors_train (never val/test, to avoid leaking future popularity).

Popularity = raw click count (label == 1) per article_id in the train split.
Also records `display_count` (how many times an article was shown as a
candidate at all, click or not, in train) so downstream feature engineering
can compute a smoothed CTR = (click_count + alpha) / (display_count + beta)
without recomputing this scan -- same train-only discipline as click_count,
since display_count is exactly as leakage-sensitive (Q9).
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src import config


def compute_popularity(processed_dir, top_n: int = config.POPULARITY_TOP_N) -> pd.DataFrame:
    train = pd.read_parquet(processed_dir / "behaviors_train.parquet")

    click_counts = Counter()
    display_counts = Counter()
    for candidates, labels in zip(train["candidates"], train["labels"]):
        for article_id, label in zip(candidates, labels):
            display_counts[article_id] += 1
            if label == 1:
                click_counts[article_id] += 1

    pop = (
        pd.DataFrame(display_counts.items(), columns=["article_id", "display_count"])
        .assign(click_count=lambda d: d["article_id"].map(click_counts).fillna(0).astype(int))
        .sort_values("click_count", ascending=False)
        .reset_index(drop=True)
    )
    pop["rank"] = pop.index + 1

    pop.to_parquet(processed_dir / "popularity.parquet", index=False)
    top = pop.head(top_n)
    top.to_parquet(processed_dir / "popularity_top.parquet", index=False)

    n_clicked = int((pop["click_count"] > 0).sum())
    print(
        f"[{processed_dir.name}] popularity: {len(pop)} distinct displayed articles "
        f"({n_clicked} with >=1 click), top-{top_n} fallback list saved"
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
