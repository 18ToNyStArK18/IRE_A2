"""Article-id codec: our unified schema stores article ids as strings
("N55189" for MIND, stringified integers for EB-NeRD), but NRMS indexes
articles into a token matrix, so it needs dense integer codes.

Code 0 is reserved and never assigned to a real article. It is the padding
slot for short histories and the representation of an article we have no text
for -- exactly the roles ebnerd-benchmark gives it via `padding=0` and
`unknown_representation="zeros"`. Real articles get 1..N.

The map is persisted so a later Codabench submission can decode model scores
back to the original ids, and so training runs are reproducible without
re-deriving codes (which would renumber articles if the catalogue changed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PAD_CODE = 0
_SPLITS = ("train", "val", "test")


@dataclass
class ArticleCodec:
    """Bidirectional map between string article ids and dense int32 codes."""

    id_to_code: dict[str, int]

    @property
    def n_articles(self) -> int:
        """Number of real articles; the token matrix has n_articles + 1 rows
        because of the reserved padding row at index 0."""
        return len(self.id_to_code)

    @property
    def code_to_id(self) -> dict[int, str]:
        return {code: article_id for article_id, code in self.id_to_code.items()}

    def encode(self, article_ids) -> np.ndarray:
        """Unknown ids map to PAD_CODE rather than raising: A1's retriever and
        the raw behaviour logs both occasionally reference articles missing
        from the catalogue, and a zero vector is the honest representation."""
        return np.fromiter(
            (self.id_to_code.get(str(a), PAD_CODE) for a in article_ids),
            dtype=np.int32,
            count=len(article_ids),
        )

    def decode(self, codes) -> list[str]:
        lookup = self.code_to_id
        return [lookup.get(int(c), "") for c in codes]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "article_id": list(self.id_to_code.keys()),
                "code": np.fromiter(self.id_to_code.values(), dtype=np.int32),
            }
        ).to_parquet(path, index=False)

    @classmethod
    def load(cls, path: Path) -> "ArticleCodec":
        df = pd.read_parquet(path)
        return cls({str(a): int(c) for a, c in zip(df["article_id"], df["code"])})


def build_article_codec(processed_dir: Path) -> ArticleCodec:
    """Codes every article in the catalogue, plus any id that only ever appears
    in the behaviour logs (as a candidate or in a history) -- those still need a
    slot so the history/candidate tensors can reference them, even though their
    token row will be empty.

    Ids are sorted before numbering so the mapping is deterministic across runs
    and machines; an unordered set would renumber the token matrix each build.
    """
    article_ids: set[str] = set()

    articles_path = processed_dir / "articles.parquet"
    if articles_path.exists():
        catalog = pd.read_parquet(articles_path, columns=["article_id"])
        article_ids.update(str(a) for a in catalog["article_id"])

    for split in _SPLITS:
        behaviors_path = processed_dir / f"behaviors_{split}.parquet"
        if not behaviors_path.exists():
            continue
        behaviors = pd.read_parquet(behaviors_path, columns=["history", "candidates"])
        for column in ("history", "candidates"):
            for row in behaviors[column]:
                if row is None:
                    continue
                article_ids.update(str(a) for a in row)

    article_ids.discard("")
    return ArticleCodec({article_id: code for code, article_id in enumerate(sorted(article_ids), start=1)})


def load_or_build_codec(processed_dir: Path, artifact_dir: Path) -> ArticleCodec:
    path = artifact_dir / "article_id_map.parquet"
    if path.exists():
        return ArticleCodec.load(path)
    codec = build_article_codec(processed_dir)
    codec.save(path)
    return codec
