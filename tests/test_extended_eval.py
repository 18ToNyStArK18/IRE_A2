"""Q5 extended-evaluation modules: beyond-accuracy, slicing, bootstrap.

Two of these tests are regressions for bugs the first run surfaced, and they are
the reason the module is worth testing at all:

  * novelty returned `inf` because popularity.parquet lists every *displayed*
    article, so most rows have click_count 0 and -log2(0) propagated through the
    mean and its whole CI;
  * the coverage CI did not contain its own point estimate, because a bootstrap
    resample omits ~1/e of the impressions and so unions fewer distinct articles.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import beyond_accuracy, bootstrap, slicing


# ------------------------------------------------------------ beyond-accuracy

def test_diversity_is_none_below_two_embedded_articles():
    """A pair is the smallest thing a diversity score can describe; 0.0 would
    read as 'maximally similar' rather than 'not measurable'."""
    lookup = {"a": np.array([1.0, 0.0])}
    assert beyond_accuracy.intra_list_diversity(["a"], lookup) is None
    assert beyond_accuracy.intra_list_diversity(["a", "missing"], lookup) is None


def test_diversity_is_zero_for_identical_articles_and_one_for_orthogonal():
    lookup = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0]), "c": np.array([0.0, 1.0])}
    assert beyond_accuracy.intra_list_diversity(["a", "b"], lookup) == pytest.approx(0.0, abs=1e-9)
    assert beyond_accuracy.intra_list_diversity(["a", "c"], lookup) == pytest.approx(1.0, abs=1e-9)


def test_novelty_is_finite_when_popularity_contains_zero_click_articles():
    """Regression: popularity.parquet lists displayed-but-never-clicked articles,
    and mapping those to p=0 made novelty -log2(0) = inf."""
    counts = {"clicked": 10, "shown_never_clicked": 0}
    probabilities, default = beyond_accuracy.click_probabilities(counts)
    assert "shown_never_clicked" not in probabilities
    value = beyond_accuracy.novelty(["clicked", "shown_never_clicked"], probabilities, default)
    assert np.isfinite(value)


def test_novelty_rewards_rarer_articles():
    counts = {"popular": 100, "rare": 1}
    probabilities, default = beyond_accuracy.click_probabilities(counts)
    assert beyond_accuracy.novelty(["rare"], probabilities, default) > beyond_accuracy.novelty(
        ["popular"], probabilities, default
    )


def test_coverage_counts_distinct_articles_over_the_catalogue():
    assert beyond_accuracy.coverage([["a", "b"], ["b", "c"]], catalogue_size=10) == pytest.approx(0.3)
    assert beyond_accuracy.coverage([], catalogue_size=10) == 0.0


# -------------------------------------------------------------------- slicing

def test_cold_start_threshold_is_inclusive():
    threshold = slicing.COLD_START_THRESHOLD
    assert slicing.cold_start_or_warm(threshold) == "cold_start"
    assert slicing.cold_start_or_warm(threshold + 1) == "warm"


def test_head_set_takes_the_top_fraction_and_excludes_unclicked():
    counts = {f"a{i}": 10 - i for i in range(10)}
    head = slicing.build_head_article_set(counts, head_fraction=0.2)
    assert head == {"a0", "a1"}
    assert slicing.head_or_tail("a9", head) == "tail"
    assert slicing.head_or_tail("never_clicked", head) == "tail"


# ------------------------------------------------------------------ bootstrap

def test_bootstrap_ci_brackets_the_mean_and_drops_nans():
    values = [1.0, 2.0, 3.0, np.nan, 5.0]
    out = bootstrap.bootstrap_ci(values, n_iterations=200, seed=0)
    assert out["n"] == 4
    assert out["mean"] == pytest.approx(2.75)
    assert out["ci_lower"] <= out["mean"] <= out["ci_upper"]


def test_bootstrap_ci_handles_empty_input():
    out = bootstrap.bootstrap_ci([], n_iterations=10)
    assert out["n"] == 0 and np.isnan(out["mean"])


def test_coverage_ci_contains_its_point_estimate():
    """Regression: a bootstrap resample repeats ~1/e of the impressions and so
    unions fewer distinct articles, biasing the naive interval low enough that it
    excluded the very number being reported."""
    rng = np.random.default_rng(0)
    lists = [[f"a{int(x)}" for x in rng.integers(0, 500, size=10)] for _ in range(300)]
    out = bootstrap.coverage_ci(lists, catalogue_size=500, n_iterations=200, seed=0)
    assert out["ci_lower"] <= out["value"] <= out["ci_upper"]
    assert out["resample_mean_uncorrected"] < out["value"]  # the bias it corrects
    assert out["bias_shift_applied"] > 0


def test_coverage_ci_point_estimate_uses_the_real_set():
    lists = [["a", "b"], ["c"]]
    out = bootstrap.coverage_ci(lists, catalogue_size=10, n_iterations=50, seed=0)
    assert out["value"] == pytest.approx(0.3)
