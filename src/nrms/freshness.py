"""Article-age signal for the Q3 freshness-weighting arm.

NRMS's only input is title tokens, so it has no channel at all for how old an
article is at the moment it is shown. That is the largest measured gap in this
project: the Q2 re-ranker put 80-87% of its gain on `freshness_log_hours` for
EB-NeRD, and the fresh-pool work measured a median clicked-article age at click
time of 3.1 hours. This module supplies the missing channel.

Reference times come from `article_stats.freshness_reference_times`, so the
definition is the one the re-ranker already uses rather than a second,
divergent notion of "age" (EB-NeRD's real `published_time`; MIND's earliest
sighting, since MIND ships no publish date).

The as-of gate lives in `ages()`: a reference only counts when it strictly
precedes the impression being scored. That is what keeps the feature honest on
train-split rows, where an article's first recorded sighting can otherwise fall
*after* the impression being scored (Q9). Anything ungated is reported as
unknown rather than silently clamped to age zero, which would tell the model
"brand new" about an article we simply had not seen yet.

PAD_CODE (0) has no reference by construction, so padded candidate slots are
always `known=0` and contribute nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import article_stats
from src.nrms.ids import ArticleCodec

_NS_PER_HOUR = 3.6e12


class FreshnessLookup:
    """Code-indexed reference times -> per-candidate (log1p(age_hours), known)."""

    def __init__(self, reference_ns: np.ndarray):
        self.reference_ns = np.asarray(reference_ns, dtype=np.float64)

    @classmethod
    def build(cls, processed_dir, dataset: str, codec: ArticleCodec, splits=("train",)) -> "FreshnessLookup":
        references = article_stats.freshness_reference_times(processed_dir, dataset, splits)
        reference_ns = np.full(codec.n_articles + 1, np.nan, dtype=np.float64)
        for article_id, timestamp in references.items():
            code = codec.id_to_code.get(str(article_id))
            if code is not None:
                reference_ns[code] = pd.Timestamp(timestamp).value
        return cls(reference_ns)

    def ages(self, codes, impression_time_ns) -> tuple[np.ndarray, np.ndarray]:
        """(log1p(age in hours), known) as float32 arrays shaped like `codes`.

        `known` is 0 wherever the article has no reference or its reference does
        not strictly precede the impression; `log_age` is 0 there too, so the
        head sees a consistent (0, 0) for "no information" rather than an
        arbitrary age paired with a zero flag.
        """
        reference = self.reference_ns[np.asarray(codes, dtype=np.int64)]
        age_hours = (float(impression_time_ns) - reference) / _NS_PER_HOUR
        known = np.isfinite(age_hours) & (age_hours > 0)
        log_age = np.where(known, np.log1p(np.where(known, age_hours, 0.0)), 0.0)
        return log_age.astype(np.float32), known.astype(np.float32)


def zeros_like_codes(codes) -> tuple[np.ndarray, np.ndarray]:
    """The no-freshness path: baseline runs still emit the tensors so the data
    pipeline has one shape, and the model ignores them when it has no head."""
    shape = np.shape(codes)
    return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
