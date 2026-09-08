"""NRMS baseline tests that need no dataset, no network and no GPU.

Covers the parts most likely to be silently wrong: the reserved padding code,
history truncation direction, the Wu-2019 sampling invariants, and the metric
formulas. The model itself is shape-checked in test_nrms_model.py, which is
skipped when torch is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import evaluate_impressions, mrr_score, ndcg_score
from src.nrms.adapter import truncate_history
from src.nrms.ids import PAD_CODE, ArticleCodec
from src.nrms.sampling import sampling_strategy_wu2019


# ------------------------------------------------------------------ id codec

def _codec() -> ArticleCodec:
    return ArticleCodec({"N1": 1, "N2": 2, "N3": 3})


def test_codec_never_assigns_the_reserved_padding_code():
    codec = _codec()
    assert PAD_CODE == 0
    assert 0 not in codec.id_to_code.values()
    assert min(codec.id_to_code.values()) == 1


def test_codec_maps_unknown_ids_to_padding():
    codec = _codec()
    assert list(codec.encode(["N1", "NOPE", "N3"])) == [1, 0, 3]


def test_codec_round_trips():
    codec = _codec()
    assert codec.decode(codec.encode(["N3", "N1"])) == ["N3", "N1"]


# ------------------------------------------------------- history truncation

def test_truncate_history_keeps_the_most_recent_tail():
    # history is most-recent-LAST, so truncation must drop from the front
    assert list(truncate_history(np.array([1, 2, 3, 4, 5]), 3)) == [3, 4, 5]


def test_truncate_history_left_pads_short_histories():
    # padding on the left keeps the newest click in the final slot for everyone
    assert list(truncate_history(np.array([7, 8]), 5)) == [0, 0, 0, 7, 8]


def test_truncate_history_handles_empty_history():
    assert list(truncate_history(np.array([], dtype=np.int32), 3)) == [0, 0, 0]


# -------------------------------------------------------- wu2019 sampling

def _impressions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "impression_id": ["i1", "i2"],
            "history": [np.array([1, 2], dtype=np.int32), np.array([3, 0], dtype=np.int32)],
            "candidates": [
                np.array([10, 11, 12, 13], dtype=np.int32),
                np.array([20, 21, 22], dtype=np.int32),
            ],
            "labels": [
                np.array([1, 0, 0, 0], dtype=np.int8),
                np.array([1, 1, 0], dtype=np.int8),
            ],
        }
    )


def test_wu2019_emits_one_row_per_positive_click():
    # i1 has 1 click, i2 has 2 -> 3 training rows
    samples = sampling_strategy_wu2019(_impressions(), npratio=2, seed=0)
    assert len(samples) == 3


def test_wu2019_row_width_is_npratio_plus_one():
    npratio = 3
    samples = sampling_strategy_wu2019(_impressions(), npratio=npratio, seed=0)
    assert all(len(c) == npratio + 1 for c in samples["candidates"])


def test_wu2019_label_indexes_the_actual_positive():
    """The whole objective depends on this: `label` must point at the positive
    after the shuffle, or training silently optimises for a negative."""
    impressions = _impressions()
    positives = {10, 20, 21}
    samples = sampling_strategy_wu2019(impressions, npratio=2, seed=7)
    for row in samples.itertuples(index=False):
        assert row.candidates[row.label] in positives


def test_wu2019_negatives_are_drawn_only_from_negatives():
    samples = sampling_strategy_wu2019(_impressions(), npratio=2, seed=7)
    negatives_by_impression = {"i1": {11, 12, 13}, "i2": {22}}
    for row in samples.itertuples(index=False):
        drawn = [c for i, c in enumerate(row.candidates) if i != row.label]
        assert set(drawn) <= negatives_by_impression[row.impression_id]


def test_wu2019_skips_impressions_with_no_negatives():
    all_clicked = pd.DataFrame(
        {
            "impression_id": ["i1"],
            "history": [np.array([1], dtype=np.int32)],
            "candidates": [np.array([10, 11], dtype=np.int32)],
            "labels": [np.array([1, 1], dtype=np.int8)],
        }
    )
    assert len(sampling_strategy_wu2019(all_clicked, npratio=2, seed=0)) == 0


def test_wu2019_is_deterministic_given_a_seed():
    a = sampling_strategy_wu2019(_impressions(), npratio=2, seed=123)
    b = sampling_strategy_wu2019(_impressions(), npratio=2, seed=123)
    assert [list(c) for c in a["candidates"]] == [list(c) for c in b["candidates"]]
    assert list(a["label"]) == list(b["label"])


# ------------------------------------------------------------------ metrics

def test_mrr_perfect_ranking():
    assert mrr_score([1, 0, 0], [0.9, 0.5, 0.1]) == pytest.approx(1.0)


def test_mrr_positive_ranked_third():
    assert mrr_score([0, 0, 1], [0.9, 0.5, 0.1]) == pytest.approx(1 / 3)


def test_mrr_averages_over_multiple_positives():
    # positives land at ranks 1 and 3 -> (1/1 + 1/3) / 2
    assert mrr_score([1, 0, 1], [0.9, 0.5, 0.1]) == pytest.approx((1 + 1 / 3) / 2)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_score([1, 0, 0], [0.9, 0.5, 0.1], k=3) == pytest.approx(1.0)


def test_ndcg_matches_manual_computation():
    # single positive at rank 2 -> DCG = 1/log2(3), ideal = 1/log2(2) = 1
    assert ndcg_score([0, 1, 0], [0.9, 0.5, 0.1], k=3) == pytest.approx(1 / np.log2(3))


def test_ndcg_is_zero_when_no_positive_exists():
    assert ndcg_score([0, 0, 0], [0.9, 0.5, 0.1], k=3) == 0.0


def test_evaluate_impressions_skips_degenerate_groups_for_auc_only():
    """An impression with no negative has undefined AUC but perfectly good
    MRR/nDCG; it must not drag AUC down or be dropped entirely."""
    labels = [[1, 0], [1, 1]]
    scores = [[0.9, 0.1], [0.9, 0.1]]
    results = evaluate_impressions(labels, scores, ndcg_ks=(2,))
    assert results["n_impressions"] == 2
    assert results["n_impressions_scored_for_auc"] == 1
    assert results["auc"] == pytest.approx(1.0)


def test_evaluate_impressions_reports_requested_ndcg_cutoffs():
    results = evaluate_impressions([[1, 0, 0]], [[0.9, 0.5, 0.1]], ndcg_ks=(5, 10))
    assert "ndcg@5" in results and "ndcg@10" in results


# --------------------------------------------------- article token matrix

class _FakeTokenizer:
    """Stands in for a HF tokenizer: encodes each character as its ordinal,
    padded/truncated to max_length. Lets us test the matrix assembly without a
    model download."""

    def __call__(self, texts, padding=None, truncation=None, max_length=None, return_tensors=None):
        rows = []
        for text in texts:
            ids = [ord(c) for c in text][:max_length]
            rows.append(ids + [0] * (max_length - len(ids)))
        return {"input_ids": np.asarray(rows, dtype=np.int64)}


def test_token_matrix_row_zero_is_reserved_and_empty():
    from src.nrms.articles import build_token_matrix

    codec = ArticleCodec({"N1": 1, "N2": 2})
    text = pd.DataFrame({"article_id": ["N1", "N2"], "text": ["ab", "cd"]})
    matrix = build_token_matrix(text, codec, _FakeTokenizer(), title_size=4)

    assert matrix.shape == (3, 4)  # n_articles + 1
    assert not matrix[0].any()  # padding/unknown article is all zeros


def test_token_matrix_rows_align_with_article_codes():
    from src.nrms.articles import build_token_matrix

    codec = ArticleCodec({"N1": 1, "N2": 2})
    text = pd.DataFrame({"article_id": ["N2", "N1"], "text": ["cd", "ab"]})  # deliberately unordered
    matrix = build_token_matrix(text, codec, _FakeTokenizer(), title_size=4)

    assert list(matrix[1]) == [ord("a"), ord("b"), 0, 0]
    assert list(matrix[2]) == [ord("c"), ord("d"), 0, 0]


def test_token_matrix_batching_matches_single_pass():
    """Batching is a memory optimisation and must not change the result."""
    from src.nrms.articles import build_token_matrix

    codec = ArticleCodec({f"N{i}": i for i in range(1, 11)})
    text = pd.DataFrame(
        {"article_id": [f"N{i}" for i in range(1, 11)], "text": [chr(96 + i) * 3 for i in range(1, 11)]}
    )
    one_pass = build_token_matrix(text, codec, _FakeTokenizer(), title_size=4, batch_size=1000)
    batched = build_token_matrix(text, codec, _FakeTokenizer(), title_size=4, batch_size=3)
    assert np.array_equal(one_pass, batched)


def test_token_matrix_ignores_articles_missing_from_codec():
    from src.nrms.articles import build_token_matrix

    codec = ArticleCodec({"N1": 1})
    text = pd.DataFrame({"article_id": ["N1", "GHOST"], "text": ["ab", "zz"]})
    matrix = build_token_matrix(text, codec, _FakeTokenizer(), title_size=4)

    assert matrix.shape == (2, 4)
    assert list(matrix[1]) == [ord("a"), ord("b"), 0, 0]
