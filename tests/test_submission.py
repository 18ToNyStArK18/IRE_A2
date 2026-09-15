"""Codabench submission plumbing (A2 Q5/Q7.3).

The two properties worth pinning here are the ones a bad submission fails
silently on: the boards require a permutation of 1..n rather than scores, and
the fast scorer's padding row has to be the same vector the training path used
for padded history slots -- otherwise 13.5M rows rank subtly wrong and nothing
in the run says so.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import submission

torch = pytest.importorskip("torch")

from src.nrms import serve  # noqa: E402
from src.nrms.freshness import FreshnessLookup  # noqa: E402
from src.nrms.model import NRMS  # noqa: E402

NS_PER_HOUR = 3.6e12


# --------------------------------------------------------------- file format

def test_rank_from_scores_is_a_permutation_with_one_highest():
    ranks = submission.rank_from_scores(np.array([0.2, 0.1, 0.3]))
    assert ranks.tolist() == [2, 3, 1]  # matches rank_predictions_by_score
    assert sorted(ranks) == [1, 2, 3]


def test_rank_from_scores_breaks_ties_into_distinct_adjacent_ranks():
    """The format demands 1..n, so tied scores cannot share a rank."""
    ranks = submission.rank_from_scores(np.array([0.5, 0.5, 0.5]))
    assert sorted(ranks.tolist()) == [1, 2, 3]


def test_inner_filenames_differ_per_board():
    """MIND wants prediction.txt, EB-NeRD predictions.txt -- read from the
    competition APIs, and a swap is rejected at upload."""
    assert submission.INNER_NAME == {"mind": "prediction.txt", "ebnerd": "predictions.txt"}


def test_mind_test_ids_have_no_label_suffix_to_strip():
    """Train/dev carry `N123-0`; the test split carries bare ids. Stripping two
    characters unconditionally would turn N55689 into N556 on every row."""
    assert submission._split_impressions("N55689 N1234") == ["N55689", "N1234"]
    assert submission._split_impressions("N55689-0 N1234-1") == ["N55689", "N1234"]


def test_catalogue_rows_map_back_to_original_order_and_flag_unknowns():
    cat = submission.Catalogue.build(["c", "a", "b"], ["C", "A", "B"])
    rows = cat.rows_of(np.array(["a", "b", "c", "zzz"], dtype=object))
    assert rows.tolist() == [1, 2, 0, -1]
    assert [cat.text[r] for r in rows[:3]] == ["A", "B", "C"]


# ------------------------------------------------------------- fast scorer

def _tiny_model(freshness: bool = True):
    torch.manual_seed(0)
    weights = np.random.default_rng(0).normal(size=(12, 8)).astype(np.float32)
    model = NRMS(weights, num_heads=2, head_dim=4, attention_hidden_dim=6, dropout=0.0,
                 freshness=freshness, freshness_hidden_dim=4)
    return model.eval()


def test_pad_row_is_the_encoded_all_zero_title():
    """Padded history slots were all-zero token rows during training, and
    layers.py deliberately does no masking, so that vector is load-bearing."""
    model = _tiny_model(freshness=False)
    tokens = np.random.default_rng(1).integers(0, 12, size=(4, 5))
    vectors = serve.encode_catalogue(model, tokens, torch.device("cpu"))

    with torch.no_grad():
        expected = model.news_encoder(torch.zeros(1, 5, dtype=torch.long))
    assert vectors.shape[0] == 5  # 4 articles + pad
    assert torch.allclose(vectors[0], expected[0], atol=1e-6)


def test_unknown_candidate_scores_as_the_pad_article():
    """`rows_of` returns -1 for an id missing from the catalogue; row_offset=1
    must land those on the pad row rather than wrapping to the last article."""
    model = _tiny_model(freshness=False)
    tokens = np.random.default_rng(2).integers(0, 12, size=(3, 5))
    vectors = serve.encode_catalogue(model, tokens, torch.device("cpu"))
    history = np.array([[0, 1]], dtype=np.int64)

    missing = serve.score_chunk(model, vectors, history, np.array([-1]), np.array([0, 1]))
    pad_direct = serve.score_chunk(model, vectors, history, np.array([-1]), np.array([0, 1]), row_offset=1)
    assert missing == pytest.approx(pad_direct)

    users = serve.user_vectors(model, vectors, history)
    assert missing[0] == pytest.approx(float((vectors[0] * users[0]).sum()), abs=1e-5)


def test_score_chunk_batching_does_not_change_scores():
    """Batching is by candidate pairs; a smaller budget must only change how the
    work is split, never the result."""
    model = _tiny_model(freshness=False)
    tokens = np.random.default_rng(3).integers(0, 12, size=(20, 5))
    vectors = serve.encode_catalogue(model, tokens, torch.device("cpu"))
    history = np.random.default_rng(4).integers(0, 20, size=(6, 3))
    offsets = np.array([0, 3, 7, 10, 14, 17, 20])
    cands = np.random.default_rng(5).integers(0, 20, size=20)

    whole = serve.score_chunk(model, vectors, history, cands, offsets, batch_pairs=10_000)
    split = serve.score_chunk(model, vectors, history, cands, offsets, batch_pairs=4)
    assert np.allclose(whole, split, atol=1e-6)


# ----------------------------------------------------------------- ages

def test_ages_accept_one_time_per_candidate():
    """The submission scorer dates a flat run of candidates from many
    impressions in one call; the per-impression callers still pass a scalar."""
    reference = np.array([np.nan, 10.0 * NS_PER_HOUR, 20.0 * NS_PER_HOUR])
    lookup = FreshnessLookup(reference)

    codes = np.array([1, 2])
    per_candidate, known = lookup.ages(codes, np.array([30.0 * NS_PER_HOUR, 30.0 * NS_PER_HOUR]))
    scalar, scalar_known = lookup.ages(codes, 30.0 * NS_PER_HOUR)
    assert np.allclose(per_candidate, scalar)
    assert np.allclose(known, scalar_known)

    # Different times for the same article give different ages.
    mixed, _ = lookup.ages(np.array([1, 1]), np.array([11.0 * NS_PER_HOUR, 110.0 * NS_PER_HOUR]))
    assert mixed[1] > mixed[0]
