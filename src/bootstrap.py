"""Bootstrap 95% confidence intervals over per-impression metric values (A2 Q5).

Ported from A1's module of the same name.

Resamples *impressions* with replacement, not individual candidate rows: each
draw is a full resample of the evaluation set, matching how the metric was
aggregated in the first place (a mean over impressions). Resampling candidates
would treat the 200 rows of one impression as 200 independent observations and
report an interval several times too narrow.

Single-system intervals. For "is arm B better than arm A", use
`scripts/paired_bootstrap.py` instead: two overlapping single-system intervals
are not evidence of no difference, because they ignore the per-impression
pairing that cancels impression-to-impression variance.

Deliberately loops rather than materialising an (iterations, n) index matrix.
The matrix form is faster but costs iterations x n x 8 bytes -- at 73k MIND test
impressions and 1000 iterations that is ~585 MB per metric, and it grows with
any request for more resamples.
"""

from __future__ import annotations

import numpy as np

DEFAULT_ITERATIONS = 1000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 42


def _interval(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(samples, [alpha, 1 - alpha])
    return float(lower), float(upper)


def bootstrap_ci(
    values,
    n_iterations: int = DEFAULT_ITERATIONS,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Mean and CI of a per-impression metric. NaNs are dropped first, which is
    how an undefined AUC (an impression with no negative, or no positive) leaves
    the AUC mean without affecting the others."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "n": 0}

    rng = np.random.default_rng(seed)
    means = np.empty(n_iterations)
    for i in range(n_iterations):
        means[i] = values[rng.integers(0, n, size=n)].mean()

    lower, upper = _interval(means, confidence)
    return {"mean": float(values.mean()), "ci_lower": lower, "ci_upper": upper, "n": n}


def coverage_ci(
    recommended_ids_by_impression,
    catalogue_size: int,
    n_iterations: int = DEFAULT_ITERATIONS,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Coverage is a set union over impressions, not a mean, so it needs its own
    resampler: each draw bootstraps impressions and re-unions their lists.

    The naive interval here is biased LOW and does not contain its own point
    estimate. A bootstrap resample repeats roughly 1/e of the impressions and
    omits the rest, so it unions fewer distinct articles than the real set --
    the bias is a property of union statistics, not of the data. Reporting
    `[0.1535, 0.1588]` beside a point estimate of 0.1736, as an earlier version
    did, is indefensible in a report.

    So the interval is shifted by (point - mean of draws): a first-order bias
    correction that keeps the resampled *spread* while re-centring on the
    statistic actually being reported. Both the raw draw mean and the shift are
    returned so the correction is auditable rather than hidden.
    """
    lists = [list(ids) for ids in recommended_ids_by_impression]
    n = len(lists)
    if n == 0 or catalogue_size == 0:
        return {"value": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "n": n}

    point = len({a for ids in lists for a in ids}) / catalogue_size

    rng = np.random.default_rng(seed)
    draws = np.empty(n_iterations)
    for i in range(n_iterations):
        seen: set[str] = set()
        for index in rng.integers(0, n, size=n):
            seen.update(lists[index])
        draws[i] = len(seen) / catalogue_size

    lower, upper = _interval(draws, confidence)
    shift = point - float(draws.mean())
    return {
        "value": float(point),
        "ci_lower": lower + shift,
        "ci_upper": upper + shift,
        "n": n,
        "resample_mean_uncorrected": float(draws.mean()),
        "bias_shift_applied": shift,
    }
