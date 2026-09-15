"""Codabench submission plumbing: stream an unlabelled competition test set,
resolve ids to catalogue rows, and write the ranked file each board expects.

Ported from A1's `src/newsrec/predict.py`, which produced submissions the MIND
leaderboard scored (bm25 0.5934, minilm 0.6425). The streaming design, the
id->row resolution and the file format are kept as they were; what changed is
who does the scoring (A2 ranks with NRMS, see src/nrms/serve.py) and that a
`Chunk` now carries each impression's own time, which the freshness arm needs.

Everything streams. The competition test sets are two orders of magnitude larger
than the splits the rest of the project uses -- 2,370,727 impressions for
MINDlarge_test and 13,536,710 for ebnerd_testset -- so nothing is materialised
whole. Peak memory is one chunk plus the catalogue, independent of test-set size.

The two boards want *different* inner filenames and integer ranks rather than
scores; both were read from the competition APIs in A1, not inferred.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.nrms.tracking import get_logger

log = get_logger("submit").info

#: Codabench wants a different inner filename for each competition.
INNER_NAME = {"mind": "prediction.txt", "ebnerd": "predictions.txt"}

#: Official impression counts, asserted before upload.
EXPECTED_IMPRESSIONS = {"mind": 2_370_727, "ebnerd": 13_536_710}


def rank_from_scores(scores: np.ndarray) -> np.ndarray:
    """Scores -> 1-indexed ranks, rank 1 = highest score.

    Matches the EB-NeRD benchmark's `rank_predictions_by_score`:
    `[0.2, 0.1, 0.3] -> [2, 3, 1]`. Ties get distinct adjacent ranks, because the
    format demands a permutation of 1..n rather than competition ranking.
    """
    return np.argsort(np.argsort(-scores, kind="stable"), kind="stable") + 1


def _split_impressions(field: str) -> list[str]:
    """Article ids from MIND's impressions column.

    Train/dev use `N123-0` / `N123-1`; **the test split has no label suffix at
    all**. Stripping two characters unconditionally would turn `N55689` into
    `N556` on every row here.
    """
    out = []
    for token in field.split():
        if len(token) > 2 and token[-2] == "-" and token[-1] in "01":
            out.append(token[:-2])
        else:
            out.append(token)
    return out


@dataclass
class Chunk:
    """One batch of impressions, already resolved to article row indices."""

    impression_ids: list[str]
    #: Flat candidate rows for every impression in order; -1 = not in catalogue.
    cand_rows: np.ndarray
    #: len(impression_ids) + 1 boundaries into cand_rows.
    offsets: np.ndarray
    #: (n, max_history) row indices, right-aligned and padded with -1.
    history_rows: np.ndarray
    #: (n,) impression time in ns. The freshness arm dates each candidate
    #: against the impression it appears in, so the time cannot be dropped the
    #: way A1's similarity scorer could drop it.
    impression_times: np.ndarray


@dataclass
class Catalogue:
    """The article set a test set ships with, plus the lookup into it.

    `text` is in ORIGINAL row order (rows_of returns original row indices), and
    is concatenated exactly as `nrms.adapter.load_article_text` does for this
    dataset, so the tokenisation matches what the model was trained on.
    """

    ids: np.ndarray  # sorted, for searchsorted
    order: np.ndarray  # ids[i] lives at original row order[i]
    text: list[str]
    #: Per-article reference time in ns for the freshness arm, original row
    #: order; NaN where unknown. EB-NeRD fills this from published_time, MIND
    #: from a first-seen pre-pass (see first_seen_reference_ns).
    reference_ns: np.ndarray = None

    @classmethod
    def build(cls, ids, text: list[str], reference_ns: np.ndarray = None) -> "Catalogue":
        arr = np.asarray(ids)
        order = np.argsort(arr, kind="stable")
        return cls(ids=arr[order], order=order, text=text, reference_ns=reference_ns)

    def rows_of(self, values: np.ndarray) -> np.ndarray:
        """Vectorised id -> row lookup. Unknown ids map to -1."""
        pos = np.searchsorted(self.ids, values)
        pos = np.clip(pos, 0, len(self.ids) - 1)
        hit = self.ids[pos] == values
        return np.where(hit, self.order[pos], -1).astype(np.int32)

    def __len__(self) -> int:
        return len(self.ids)


# --------------------------------------------------------------------------
# MIND
# --------------------------------------------------------------------------

MIND_NEWS_COLUMNS = [
    "article_id", "category", "subcategory", "title", "abstract",
    "url", "title_entities", "abstract_entities",
]


def load_mind_catalogue(root: Path) -> Catalogue:
    """title + abstract, matching TEXT_COLUMNS['mind'] (MIND ships no body)."""
    news = pl.read_csv(
        root / "news.tsv",
        separator="\t",
        has_header=False,
        new_columns=MIND_NEWS_COLUMNS,
        quote_char=None,
        infer_schema_length=0,
        missing_utf8_is_empty_string=True,
    )
    text = (
        news["title"].fill_null("") + " " + news["abstract"].fill_null("")
    ).str.strip_chars().to_list()
    log(f"  catalogue: {news.height:,} articles")
    return Catalogue.build(news["article_id"].to_list(), text)


def count_mind_impressions(root: Path) -> int:
    with (root / "behaviors.tsv").open("rb") as f:
        return sum(1 for _ in f)


def _mind_times_ns(stamps: list[str]) -> np.ndarray:
    """MIND stamps look like `11/19/2019 11:37:45 AM`."""
    return (
        pl.Series(stamps)
        .str.to_datetime("%m/%d/%Y %I:%M:%S %p", strict=False)
        .cast(pl.Int64)
        .fill_null(0)
        .to_numpy()
        * 1_000  # polars gives microseconds; the rest of the project uses ns
    )


def iter_mind_chunks(
    root: Path, cat: Catalogue, n_history: int, chunk_size: int, skip: int = 0
) -> Iterator[Chunk]:
    """Stream behaviors.tsv. MIND carries each impression's history inline.

    `n_history=0` means "no history needed" -- the first-seen pre-pass only wants
    candidates and times. Note `[-0:]` slices the WHOLE list rather than nothing,
    so that case is branched on explicitly in both places below.
    """
    ids: list[str] = []
    stamps: list[str] = []
    cand_tokens: list[str] = []
    cand_lengths: list[int] = []
    hist_tokens: list[str] = []
    hist_lengths: list[int] = []

    def flush() -> Chunk:
        # One searchsorted over the whole chunk rather than one per impression:
        # the per-impression version cost ~3.4 ms each, i.e. 2.3 h for the file.
        cand_rows = cat.rows_of(np.asarray(cand_tokens, dtype=object))
        hist_rows = cat.rows_of(np.asarray(hist_tokens, dtype=object))
        offsets = np.concatenate([[0], np.cumsum(np.asarray(cand_lengths, dtype=np.int64))])
        hist_offsets = np.concatenate([[0], np.cumsum(np.asarray(hist_lengths, dtype=np.int64))])

        padded = np.full((len(ids), max(n_history, 1)), -1, dtype=np.int32)
        for i in range(len(ids)):
            piece = hist_rows[hist_offsets[i] : hist_offsets[i + 1]]
            piece = piece[piece >= 0][-n_history:] if n_history else piece[:0]
            if piece.size:
                padded[i, -piece.size :] = piece

        chunk = Chunk(list(ids), cand_rows.astype(np.int32), offsets, padded, _mind_times_ns(stamps))
        ids.clear()
        stamps.clear()
        cand_tokens.clear()
        cand_lengths.clear()
        hist_tokens.clear()
        hist_lengths.clear()
        return chunk

    with (root / "behaviors.tsv").open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            if lineno < skip:
                continue
            parts = line.rstrip("\n").split("\t")
            ids.append(parts[0])
            stamps.append(parts[2])
            cands = _split_impressions(parts[4])
            cand_tokens.extend(cands)
            cand_lengths.append(len(cands))
            # Only the tail is ever used, so do not carry the whole history.
            history = parts[3].split()[-n_history:] if n_history else []
            hist_tokens.extend(history)
            hist_lengths.append(len(history))
            if len(ids) >= chunk_size:
                yield flush()
    if ids:
        yield flush()


def first_seen_reference_ns(root: Path, cat: Catalogue, chunk_size: int) -> np.ndarray:
    """MIND's freshness reference: each article's earliest sighting in the test
    log, in original catalogue row order.

    Same definition `article_stats.freshness_reference_times` uses for MIND,
    sourced from the test set because that is the period being scored. Leak-free
    on its own: `FreshnessLookup.ages` only counts a reference that strictly
    precedes the impression, so an article first seen later in the file cannot
    date an earlier impression.

    One extra streaming pass, accumulated with `np.minimum.at` over whole chunks
    rather than a Python loop over ~88M (impression, candidate) pairs.
    """
    reference = np.full(len(cat), np.inf, dtype=np.float64)
    seen = 0
    for chunk in iter_mind_chunks(root, cat, n_history=0, chunk_size=chunk_size):
        widths = np.diff(chunk.offsets)
        times = np.repeat(chunk.impression_times.astype(np.float64), widths)
        keep = chunk.cand_rows >= 0
        np.minimum.at(reference, chunk.cand_rows[keep].astype(np.int64), times[keep])
        seen += len(chunk.impression_ids)
    reference[~np.isfinite(reference)] = np.nan
    log(f"  first-seen references: {int(np.isfinite(reference).sum()):,}/{len(cat):,} articles "
        f"from {seen:,} impressions")
    return reference


# --------------------------------------------------------------------------
# EB-NeRD
# --------------------------------------------------------------------------


def load_ebnerd_catalogue(root: Path) -> Catalogue:
    """title + subtitle + body, matching TEXT_COLUMNS['ebnerd'] -- `src/parse.py`
    maps EB-NeRD's `subtitle` onto our `abstract`. `published_time` doubles as
    the freshness reference, the same definition the trained arm used."""
    articles = pl.read_parquet(
        root / "articles.parquet",
        columns=["article_id", "title", "subtitle", "body", "published_time"],
    )
    text = (
        articles["title"].fill_null("")
        + " " + articles["subtitle"].fill_null("")
        + " " + articles["body"].fill_null("")
    ).str.strip_chars().to_list()
    reference = articles["published_time"].cast(pl.Int64).to_numpy().astype(np.float64) * 1_000
    reference[articles["published_time"].is_null().to_numpy()] = np.nan
    log(f"  catalogue: {articles.height:,} articles, "
        f"{int(np.isfinite(reference).sum()):,} with a published_time")
    return Catalogue.build(articles["article_id"].to_numpy(), text, reference)


def load_ebnerd_histories(root: Path, cat: Catalogue, n_history: int):
    """user_id -> that user's last n_history article rows, as one padded array.

    Built once: 807k users at n_history=20 is ~65 MB, against 13.5M impressions
    that would otherwise each carry their own copy.
    """
    hist = pl.read_parquet(
        root / "test" / "history.parquet", columns=["user_id", "article_id_fixed"]
    ).with_columns(pl.col("article_id_fixed").list.tail(n_history))

    users = hist["user_id"].to_numpy()
    order = np.argsort(users, kind="stable")
    # int64 throughout: numpy promotes int64 + uint32 to float64, which is not
    # a valid slice index.
    raw_lengths = hist["article_id_fixed"].list.len().to_numpy().astype(np.int64)
    lengths = raw_lengths[order]

    # One vectorised lookup over every retained history entry, rather than a
    # Python loop over 807k users -- that loop cost 110s per run.
    flat = hist["article_id_fixed"].explode().to_numpy()
    starts = np.concatenate([[0], np.cumsum(raw_lengths)]).astype(np.int64)
    rows = cat.rows_of(flat)

    padded = np.full((len(users), max(n_history, 1)), -1, dtype=np.int32)
    for out_i, src_i in enumerate(order):
        piece = rows[starts[src_i] : starts[src_i] + lengths[out_i]]
        piece = piece[piece >= 0]
        if piece.size:
            padded[out_i, -piece.size :] = piece

    log(f"  histories: {len(users):,} users, capped at {n_history}")
    return users[order], padded


def count_ebnerd_impressions(root: Path) -> int:
    return pq.ParquetFile(root / "test" / "behaviors.parquet").metadata.num_rows


def iter_ebnerd_chunks(
    root: Path, cat: Catalogue, user_ids, user_history, chunk_size: int, skip: int = 0
) -> Iterator[Chunk]:
    """Stream behaviors.parquet in Arrow batches.

    `article_ids_inview` is a large_list<int32>, so candidates come out as a flat
    integer array with offsets -- no Python objects on this path at all.
    """
    reader = pq.ParquetFile(root / "test" / "behaviors.parquet")
    seen = 0
    for batch in reader.iter_batches(
        batch_size=chunk_size,
        columns=["impression_id", "user_id", "article_ids_inview", "impression_time"],
    ):
        n = batch.num_rows
        if seen + n <= skip:
            seen += n
            continue

        col = batch.column("article_ids_inview")
        offsets = col.offsets.to_numpy().astype(np.int64)
        values = col.values.to_numpy(zero_copy_only=False)
        cand_rows = cat.rows_of(values)
        impression_ids = [str(x) for x in batch.column("impression_id").to_pylist()]
        users = np.asarray(batch.column("user_id").to_pylist())
        times = batch.column("impression_time").to_numpy(zero_copy_only=False).astype("datetime64[ns]").astype(np.int64)

        # Where a batch straddles the resume point, drop the finished prefix.
        start = max(0, skip - seen)
        if start:
            impression_ids = impression_ids[start:]
            users = users[start:]
            times = times[start:]
            cand_rows = cand_rows[offsets[start] :]
            offsets = offsets[start:] - offsets[start]
        offsets = offsets - offsets[0]
        seen += n

        pos = np.clip(np.searchsorted(user_ids, users), 0, len(user_ids) - 1)
        known = user_ids[pos] == users
        history = np.where(
            known[:, None], user_history[pos], np.full(user_history.shape[1], -1)
        ).astype(np.int32)

        yield Chunk(impression_ids, cand_rows.astype(np.int32), offsets, history, times)


# --------------------------------------------------------------------------
# Writing, with resume
# --------------------------------------------------------------------------


class SubmissionWriter:
    """Appends ranked lines per chunk and records progress so a run can resume."""

    def __init__(self, path: Path, dataset: str, resume: bool = True):
        self.zip_path = path
        self.dataset = dataset
        self.text_path = path.with_suffix(".txt")
        self.progress_path = path.with_suffix(".progress")
        path.parent.mkdir(parents=True, exist_ok=True)

        self.written = 0
        if resume and self.text_path.exists() and self.progress_path.exists():
            self.written = int(self.progress_path.read_text().strip())
            # Truncate any partial line left by a kill mid-write.
            with self.text_path.open("r+", encoding="utf-8") as f:
                seen = 0
                while seen < self.written and f.readline():
                    seen += 1
                f.truncate(f.tell())
            self.written = seen
            log(f"  resuming after {self.written:,} impressions")
        else:
            self.text_path.unlink(missing_ok=True)
            self.progress_path.unlink(missing_ok=True)

        self.handle = self.text_path.open("a", encoding="utf-8")

    def write_chunk(self, chunk: Chunk, scores: np.ndarray) -> None:
        lines = []
        for i, impression in enumerate(chunk.impression_ids):
            piece = scores[chunk.offsets[i] : chunk.offsets[i + 1]]
            if piece.size == 0:
                lines.append(f"{impression} []")
                continue
            ranks = rank_from_scores(piece)
            lines.append(f"{impression} [{','.join(map(str, ranks.tolist()))}]")
        self.handle.write("\n".join(lines) + "\n")
        self.handle.flush()
        self.written += len(chunk.impression_ids)
        self.progress_path.write_text(str(self.written))

    def finalise(self, expected: int | None = None) -> Path:
        self.handle.close()
        if expected is not None and self.written != expected:
            raise AssertionError(f"wrote {self.written:,} lines for {expected:,} impressions")
        inner = INNER_NAME[self.dataset]
        with zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(self.text_path, arcname=inner)
        self.text_path.unlink()
        self.progress_path.unlink(missing_ok=True)
        log(f"  wrote {self.zip_path} "
            f"({self.zip_path.stat().st_size / 1e6:.1f} MB, {self.written:,} lines)")
        return self.zip_path


def verify_submission(path: Path, dataset: str, expected_lines: int) -> None:
    """Stream-check the finished zip. Fails here rather than at upload."""
    inner = INNER_NAME[dataset]
    with zipfile.ZipFile(path) as z:
        assert z.namelist() == [inner], f"zip must contain exactly [{inner}], got {z.namelist()}"
        n = 0
        with z.open(inner) as f:
            for raw in f:
                n += 1
                if n <= 3 or n == expected_lines:
                    body = raw.decode().split(" ", 1)[1].strip()[1:-1]
                    ranks = [int(x) for x in body.split(",")] if body else []
                    assert sorted(ranks) == list(range(1, len(ranks) + 1)), (
                        f"line {n}: ranks are not a permutation of 1..n"
                    )
    assert n == expected_lines, f"{n:,} lines for {expected_lines:,} impressions"
    log(f"  [ok] {n:,} lines, ranks verified")
