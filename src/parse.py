"""Parse raw MIND / EB-NeRD files into a unified schema.

Unified articles schema (one row per article):
    dataset, article_id, title, abstract, body, category, subcategory,
    published_time, entities

Unified behaviors schema (one row per impression):
    dataset, raw_split, impression_id, user_id, time,
    history (list[str] article_ids, most-recent-last),
    history_read_time (list[float|None] dwell-time seconds aligned 1:1 with
        `history`, or dataset-level None when unavailable -- MIND has no
        dwell-time signal at all),
    history_impression_time (list[timestamp|None] aligned 1:1 with `history`,
        or dataset-level None when unavailable),
    session_id (str|None, EB-NeRD only -- MIND has no session concept),
    candidates (list[str] article_ids shown in this impression),
    labels (list[int] 0/1, aligned with candidates)

`raw_split` records which originally-downloaded period the row came from,
normalized to "train" (earlier period) / "dev" (later period: MIND's dev.tsv,
EB-NeRD's validation/ folder) for both datasets, so split.py can turn it into
a genuine temporal train/val/test split without shuffling anything.

`history_read_time`/`history_impression_time` describe *past* clicks only
(dwell time the user already spent on articles before this impression) --
never confuse these with a candidate's own engagement, which isn't known
until after it's shown. EB-NeRD's raw behaviors.parquet also carries the
current impression's own `read_time`/`scroll_percentage` and the *next*
impression's `next_read_time`/`next_scroll_percentage`; none of those are
pulled into the unified schema because they describe an outcome not yet
known at serving time (the next-impression fields are literally future data).
"""

from __future__ import annotations

import json

import pandas as pd

from src import config


def _entities_from_json(raw: str | None) -> list[str]:
    if not raw or not isinstance(raw, str):
        return []
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [item.get("Label", "") for item in items if item.get("Label")]


# --------------------------------------------------------------------- MIND

def _parse_mind_news(path) -> pd.DataFrame:
    cols = [
        "article_id", "category", "subcategory", "title", "abstract",
        "url", "title_entities", "abstract_entities",
    ]
    df = pd.read_csv(path, sep="\t", header=None, names=cols, dtype=str)
    df["entities"] = df["title_entities"].apply(_entities_from_json) + df[
        "abstract_entities"
    ].apply(_entities_from_json)
    df["abstract"] = df["abstract"].fillna("")
    df["title"] = df["title"].fillna("")
    df["body"] = None
    df["published_time"] = pd.NaT
    return df[[
        "article_id", "title", "abstract", "body", "category", "subcategory",
        "published_time", "entities",
    ]]


def _parse_mind_behaviors(path, raw_split: str) -> pd.DataFrame:
    cols = ["impression_id", "user_id", "time", "history", "impressions"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols, dtype=str)
    df["time"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p")
    df["history"] = df["history"].fillna("").apply(
        lambda s: s.split() if s else []
    )

    def _split_impressions(s: str):
        pairs = [p.rsplit("-", 1) for p in s.split()]
        candidates = [p[0] for p in pairs]
        labels = [int(p[1]) for p in pairs]
        return candidates, labels

    parsed = df["impressions"].apply(_split_impressions)
    df["candidates"] = parsed.apply(lambda t: t[0])
    df["labels"] = parsed.apply(lambda t: t[1])
    df["raw_split"] = raw_split
    df["dataset"] = "mind"
    # MIND ships no dwell-time or session signal at all -- dataset-level None
    # (not a per-item list of Nones) to mark "unavailable", distinct from
    # EB-NeRD's per-history-item values.
    df["history_read_time"] = None
    df["history_impression_time"] = None
    df["session_id"] = None
    return df[[
        "dataset", "raw_split", "impression_id", "user_id", "time", "history",
        "history_read_time", "history_impression_time", "session_id",
        "candidates", "labels",
    ]]


def parse_mind() -> None:
    config.MIND_INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    news_frames = []
    behavior_frames = []
    for raw_split in ("train", "dev"):
        split_dir = config.MIND_RAW_DIR / raw_split
        news_frames.append(_parse_mind_news(split_dir / "news.tsv"))
        behavior_frames.append(
            _parse_mind_behaviors(split_dir / "behaviors.tsv", raw_split)
        )

    articles = pd.concat(news_frames, ignore_index=True).drop_duplicates("article_id")
    articles.insert(0, "dataset", "mind")
    behaviors = pd.concat(behavior_frames, ignore_index=True)

    articles.to_parquet(config.MIND_INTERIM_DIR / "articles.parquet", index=False)
    behaviors.to_parquet(config.MIND_INTERIM_DIR / "behaviors.parquet", index=False)
    print(f"[mind] {len(articles)} articles, {len(behaviors)} impressions -> {config.MIND_INTERIM_DIR}")


# ------------------------------------------------------------------ EB-NeRD

def _parse_ebnerd_articles(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    out = pd.DataFrame({
        "article_id": df["article_id"].astype(str),
        "title": df["title"].fillna(""),
        "abstract": df["subtitle"].fillna(""),
        "body": df["body"].fillna(""),
        "category": df["category_str"].fillna(""),
        "subcategory": df["subcategory"].apply(
            lambda x: list(x) if x is not None else []
        ),
        "published_time": df["published_time"],
        "entities": df["ner_clusters"].apply(
            lambda x: list(x) if x is not None else []
        ),
    })
    return out


def _parse_ebnerd_behaviors(behaviors_path, history_path, raw_split: str) -> pd.DataFrame:
    beh = pd.read_parquet(behaviors_path)
    hist = pd.read_parquet(history_path)

    def _as_list(x):
        return list(x) if x is not None else []

    hist_map = {
        int(row.user_id): {
            "article_ids": [str(a) for a in _as_list(row.article_id_fixed)],
            "read_times": _as_list(row.read_time_fixed),
            "impression_times": _as_list(row.impression_time_fixed),
        }
        for row in hist.itertuples(index=False)
    }
    empty_hist = {"article_ids": [], "read_times": [], "impression_times": []}

    df = pd.DataFrame({
        "dataset": "ebnerd",
        "raw_split": raw_split,
        "impression_id": beh["impression_id"].astype(str),
        "user_id": beh["user_id"].astype(str),
        "time": beh["impression_time"],
        "history": beh["user_id"].apply(lambda u: hist_map.get(int(u), empty_hist)["article_ids"]),
        "history_read_time": beh["user_id"].apply(lambda u: hist_map.get(int(u), empty_hist)["read_times"]),
        "history_impression_time": beh["user_id"].apply(
            lambda u: hist_map.get(int(u), empty_hist)["impression_times"]
        ),
        "session_id": beh["session_id"].astype(str),
        "candidates": beh["article_ids_inview"].apply(
            lambda x: [str(a) for a in x] if x is not None else []
        ),
    })
    clicked = beh["article_ids_clicked"].apply(
        lambda x: set(str(a) for a in x) if x is not None else set()
    )
    df["labels"] = [
        [1 if c in clicked_set else 0 for c in cands]
        for cands, clicked_set in zip(df["candidates"], clicked)
    ]
    return df


def parse_ebnerd(bundle: str = config.EBNERD_BUNDLE) -> None:
    config.EBNERD_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    bundle_dir = config.EBNERD_RAW_DIR / bundle

    articles = _parse_ebnerd_articles(bundle_dir / "articles.parquet")
    articles.insert(0, "dataset", "ebnerd")

    behavior_frames = []
    for raw_split, folder in (("train", "train"), ("dev", "validation")):
        behavior_frames.append(
            _parse_ebnerd_behaviors(
                bundle_dir / folder / "behaviors.parquet",
                bundle_dir / folder / "history.parquet",
                raw_split,
            )
        )
    behaviors = pd.concat(behavior_frames, ignore_index=True)

    articles.to_parquet(config.EBNERD_INTERIM_DIR / "articles.parquet", index=False)
    behaviors.to_parquet(config.EBNERD_INTERIM_DIR / "behaviors.parquet", index=False)
    print(f"[ebnerd] {len(articles)} articles, {len(behaviors)} impressions -> {config.EBNERD_INTERIM_DIR}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    parser.add_argument("--ebnerd-bundle", default=config.EBNERD_BUNDLE)
    args = parser.parse_args()

    if args.dataset in ("mind", "all"):
        parse_mind()
    if args.dataset in ("ebnerd", "all"):
        parse_ebnerd(bundle=args.ebnerd_bundle)


if __name__ == "__main__":
    main()
