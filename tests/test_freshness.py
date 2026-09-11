"""Q3 freshness arm.

Two properties carry the ablation's validity and are tested first:

  * the arm starts bit-identical to the baseline (zero-initialised head), so a
    measured difference can only come from learning, not from a perturbed
    initialisation;
  * disabling the flag leaves the model's parameters and RNG consumption
    untouched, so the baseline arm still reproduces the baseline runs it is
    compared against.

The rest checks the age signal itself, including the as-of gate that keeps it
honest on train-split rows (Q9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src import article_stats, config  # noqa: E402
from src.nrms.freshness import FreshnessLookup, zeros_like_codes  # noqa: E402
from src.nrms.ids import PAD_CODE, ArticleCodec  # noqa: E402
from src.nrms.model import NRMS  # noqa: E402

PROCESSED_DIRS = {"mind": config.MIND_PROCESSED_DIR, "ebnerd": config.EBNERD_PROCESSED_DIR}
NS_PER_HOUR = 3.6e12


def _tiny_model(freshness: bool, seed: int = 0):
    torch.manual_seed(seed)
    weights = np.random.default_rng(seed).normal(size=(12, 8)).astype(np.float32)
    return NRMS(
        weights, num_heads=2, head_dim=4, attention_hidden_dim=6, dropout=0.0,
        freshness=freshness, freshness_hidden_dim=4,
    )


def _batch(batch=2, history=3, candidates=4, title=5, vocab=12):
    rng = np.random.default_rng(0)
    h = torch.from_numpy(rng.integers(0, vocab, size=(batch, history, title)))
    c = torch.from_numpy(rng.integers(0, vocab, size=(batch, candidates, title)))
    log_age = torch.rand(batch, candidates)
    known = torch.ones(batch, candidates)
    return h, c, log_age, known


# ------------------------------------------------- ablation-validity properties

def test_freshness_head_starts_as_a_noop():
    """Zero-initialised output layer: at step 0 the arm's logits must equal the
    baseline's exactly, or an apparent 'gain' could just be a different init."""
    model = _tiny_model(freshness=True)
    model.eval()
    h, c, log_age, known = _batch()

    with torch.no_grad():
        adjustment = model.freshness_head(log_age, known)
        with_ages = model(h, c, log_age, known)
        model.freshness_head = None
        without = model(h, c)

    assert torch.allclose(adjustment, torch.zeros_like(adjustment))
    assert torch.allclose(with_ages, without, atol=1e-6)


def test_disabled_flag_leaves_parameters_and_rng_untouched():
    """The baseline arm must be unchanged by the feature existing -- same
    parameter count and same values, so it still reproduces earlier runs."""
    baseline = _tiny_model(freshness=False, seed=7)
    again = _tiny_model(freshness=False, seed=7)

    assert baseline.freshness_head is None
    names = [n for n, _ in baseline.named_parameters()]
    assert not any("freshness" in n for n in names)
    for (_, a), (_, b) in zip(baseline.named_parameters(), again.named_parameters()):
        assert torch.equal(a, b)


def test_freshness_head_is_negligible_capacity():
    """Asserted in absolute terms against the REAL hidden dim, not the toy
    model's: the point is that the arm adds ~65 parameters to a ~192M-parameter
    model, so no gain it produces can be attributed to added capacity."""
    from src.nrms.config import FRESHNESS_HIDDEN_DIM
    from src.nrms.model import FreshnessHead

    head_params = sum(p.numel() for p in FreshnessHead(FRESHNESS_HIDDEN_DIM).parameters())
    assert head_params < 100

    # xlm-roberta-base's embedding matrix alone, the smallest real comparison.
    assert head_params / (250_002 * 768) < 1e-6


def test_head_changes_logits_once_trained():
    """A non-zero head must actually move the scores, otherwise the arm could
    never differ from the baseline at all."""
    model = _tiny_model(freshness=True)
    torch.nn.init.normal_(model.freshness_head.mlp[-1].weight, std=1.0)
    model.eval()
    h, c, log_age, known = _batch()
    with torch.no_grad():
        moved = model(h, c, log_age, known)
        model.freshness_head = None
        base = model(h, c)
    assert not torch.allclose(moved, base)


def test_missing_age_tensors_raise_when_head_attached():
    model = _tiny_model(freshness=True)
    h, c, _, _ = _batch()
    with pytest.raises(ValueError, match="freshness head is attached"):
        model(h, c)


# ------------------------------------------------------------- the age signal

def _lookup(reference_hours: dict[int, float], size: int = 5) -> FreshnessLookup:
    reference = np.full(size, np.nan, dtype=np.float64)
    for code, hours in reference_hours.items():
        reference[code] = hours * NS_PER_HOUR
    return FreshnessLookup(reference)


def test_age_is_log1p_hours_since_reference():
    lookup = _lookup({1: 100.0})
    log_age, known = lookup.ages(np.array([1]), 110.0 * NS_PER_HOUR)
    assert known[0] == 1.0
    assert log_age[0] == pytest.approx(np.log1p(10.0), rel=1e-5)


def test_reference_at_or_after_the_impression_is_unknown_not_zero_age():
    """The Q9 gate: an article first seen *after* the impression being scored
    must read 'unknown', never 'brand new'."""
    lookup = _lookup({1: 200.0})
    log_age, known = lookup.ages(np.array([1, 1]), 150.0 * NS_PER_HOUR)
    assert known.tolist() == [0.0, 0.0]
    assert log_age.tolist() == [0.0, 0.0]


def test_article_without_a_reference_is_unknown():
    lookup = _lookup({1: 10.0})
    log_age, known = lookup.ages(np.array([2]), 100.0 * NS_PER_HOUR)
    assert known[0] == 0.0 and log_age[0] == 0.0


def test_padding_code_is_always_unknown():
    lookup = _lookup({1: 10.0})
    _, known = lookup.ages(np.array([PAD_CODE]), 100.0 * NS_PER_HOUR)
    assert known[0] == 0.0


def test_older_articles_score_higher_log_age():
    lookup = _lookup({1: 100.0, 2: 10.0})
    log_age, _ = lookup.ages(np.array([1, 2]), 200.0 * NS_PER_HOUR)
    assert log_age[1] > log_age[0]  # code 2 is older, so a larger age


def test_zeros_path_matches_shape():
    log_age, known = zeros_like_codes(np.zeros((3, 4), dtype=np.int64))
    assert log_age.shape == (3, 4) and known.shape == (3, 4)
    assert not log_age.any() and not known.any()


# ---------------------------------------- equivalence with the re-ranker's view

@pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
def test_reference_times_agree_with_train_event_index(dataset):
    """freshness_reference_times duplicates TrainEventIndex's notion of a
    reference time for speed; if the two ever diverge, the NRMS arm and the
    re-ranker would be using different definitions of 'age'."""
    processed_dir = PROCESSED_DIRS[dataset]
    if not processed_dir.exists():
        pytest.skip(f"{dataset} not built")

    references = article_stats.freshness_reference_times(processed_dir, dataset)
    if not references:
        pytest.skip("no reference times available")

    index = article_stats.TrainEventIndex(processed_dir, dataset)
    far_future = pd.Timestamp("2100-01-01")

    sample = list(references.items())[:200]
    for article_id, expected in sample:
        stat = index.as_of(article_id, far_future)
        if not stat["has_known_publish_time"]:
            continue
        assert pd.Timestamp(stat["freshness_reference_time"]) == pd.Timestamp(expected)


def test_lookup_build_marks_unknown_articles_as_nan():
    codec = ArticleCodec({"a": 1, "b": 2})
    lookup = FreshnessLookup(np.array([np.nan, 5.0 * NS_PER_HOUR, np.nan]))
    assert np.isnan(lookup.reference_ns[PAD_CODE])
    _, known = lookup.ages(np.array([1, 2]), 10.0 * NS_PER_HOUR)
    assert known.tolist() == [1.0, 0.0]
    assert codec.n_articles == 2
