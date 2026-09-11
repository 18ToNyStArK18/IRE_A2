"""Q2 re-ranker tests.

The important one is `test_population_shortcut_matches_brute_force`: the
re-ranker only scores impressions that contain a positive, and claims the
resulting full-population metrics are *exact* rather than approximate. That
claim is load-bearing (it is what makes MIND train tractable at 27.7M candidate
rows) so it is checked against a brute-force evaluation of every impression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# reranker imports lightgbm at module level; without it these tests are skipped
# instead of one collection error aborting the entire suite.
pytest.importorskip("lightgbm")

from src import config, reranker  # noqa: E402
from src.metrics import evaluate_impressions  # noqa: E402

PROCESSED_DIRS = {"mind": config.MIND_PROCESSED_DIR, "ebnerd": config.EBNERD_PROCESSED_DIR}


def _require_candidates(dataset: str, method: str, split: str):
    d = PROCESSED_DIRS[dataset]
    if not (d / f"candidates_{method}_{split}.parquet").exists():
        pytest.skip(f"{dataset}/{method}/{split} candidates not generated yet")
    return d


# ------------------------------------------------------------------ labelling

def test_label_candidates_marks_only_true_clicks():
    cands = pd.DataFrame(
        {
            "impression_id": ["i1", "i1", "i2", "i2"],
            "article_id": ["a", "b", "a", "c"],
            "rank": [1, 2, 1, 2],
            "retrieval_score": [9.0, 8.0, 7.0, 6.0],
        }
    )
    clicks = {"i1": {"b"}}  # i2 has no retrieved positive
    out = reranker.label_candidates(cands, clicks)
    assert out["label"].tolist() == [0, 1, 0, 0]


def test_impressions_with_positive_selects_only_those_with_a_click():
    labelled = pd.DataFrame(
        {"impression_id": ["i1", "i1", "i2", "i2", "i3"], "label": [0, 1, 0, 0, 1]}
    )
    assert sorted(reranker.impressions_with_positive(labelled).tolist()) == ["i1", "i3"]


# --------------------------------------------------- the exactness claim (key)

def test_population_shortcut_matches_brute_force():
    """Scoring only positive-bearing impressions must give byte-identical
    full-population metrics to scoring every impression."""
    rng = np.random.default_rng(0)
    n_total = 50

    labels_all, scores_all = [], []
    labels_sub, scores_sub = [], []
    for i in range(n_total):
        size = 20
        scores = rng.normal(size=size)
        labels = np.zeros(size)
        if i % 10 == 0:  # only every 10th impression contains a positive
            labels[rng.integers(size)] = 1.0
            labels_sub.append(labels)
            scores_sub.append(scores)
        labels_all.append(labels)
        scores_all.append(scores)

    brute = evaluate_impressions(labels_all, scores_all, ndcg_ks=(5, 10))
    short = reranker.evaluate_over_population(labels_sub, scores_sub, n_total, ndcg_ks=(5, 10))

    assert short["n_impressions"] == brute["n_impressions"] == n_total
    assert short["n_impressions_scored_for_auc"] == brute["n_impressions_scored_for_auc"]
    for key in ("auc", "mrr", "ndcg@5", "ndcg@10"):
        assert short[key] == pytest.approx(brute[key], rel=1e-12, abs=1e-12), key


def test_population_shortcut_matches_brute_force_on_real_data():
    d = _require_candidates("ebnerd", "bm25", "val")
    from src import article_stats

    index = article_stats.TrainEventIndex(d, "ebnerd")
    full = reranker.build_matrix(d, "ebnerd", "val", "bm25", index, positives_only=False)
    subset = full[full.groupby("impression_id")["label"].transform("max") > 0]

    n_total = full["impression_id"].nunique()
    rng = np.random.default_rng(0)
    full = full.assign(s=rng.normal(size=len(full)))
    subset = subset.assign(s=full.loc[subset.index, "s"])

    labels_all, scores_all = reranker.group_arrays(full, "s")
    labels_sub, scores_sub = reranker.group_arrays(subset, "s")

    brute = evaluate_impressions(labels_all, scores_all, ndcg_ks=(5, 10))
    short = reranker.evaluate_over_population(labels_sub, scores_sub, n_total, ndcg_ks=(5, 10))
    for key in ("auc", "mrr", "ndcg@5", "ndcg@10"):
        assert short[key] == pytest.approx(brute[key], rel=1e-9, abs=1e-12), key


# ------------------------------------------------------------ lgb constraints

def test_build_matrix_keeps_each_impressions_rows_contiguous():
    """LightGBM's group array assumes it; a non-contiguous frame would silently
    mis-assign candidates to the wrong impression."""
    d = _require_candidates("ebnerd", "bm25", "val")
    from src import article_stats

    matrix = reranker.build_matrix(
        d, "ebnerd", "val", "bm25", article_stats.TrainEventIndex(d, "ebnerd"), positives_only=True
    )
    ids = matrix["impression_id"].to_numpy()
    boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    blocks = np.split(ids, boundaries)
    assert len({b[0] for b in blocks}) == len(blocks), "an impression's rows are split across blocks"


def test_feature_columns_excludes_ids_and_label():
    matrix = pd.DataFrame(
        {"impression_id": ["i"], "article_id": ["a"], "label": [1], "ctr_article": [0.1]}
    )
    assert reranker.feature_columns(matrix) == ["ctr_article"]


def test_training_filter_never_applies_to_evaluation_population():
    """The reported denominator must be every impression in the split, not just
    the ones that survived the training filter."""
    d = _require_candidates("ebnerd", "bm25", "val")
    n_total = len(reranker.true_clicks(d, "val"))
    behaviors = pd.read_parquet(d / "behaviors_val.parquet", columns=["impression_id"])
    assert n_total == len(behaviors)


# ---------------------------------------------- scale: sampling and chunking

def _labelled(n_impressions=3, size=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_impressions):
        positive = int(rng.integers(size))
        for r in range(1, size + 1):
            rows.append({
                "impression_id": f"i{i}",
                "article_id": f"a{i}_{r}",
                "rank": r,
                "retrieval_score": float(size - r),
                "label": int(r - 1 == positive),
            })
    return pd.DataFrame(rows)


def test_sample_negatives_keeps_every_positive_and_caps_negatives():
    labelled = _labelled()
    out = reranker.sample_negatives(labelled, n_hard=5, n_random=7, seed=0)
    assert out["label"].sum() == labelled["label"].sum()
    per_group = out[out["label"] == 0].groupby("impression_id").size()
    assert (per_group == 12).all()


def test_sample_negatives_keeps_the_best_ranked_negatives():
    """The negatives just below the positive decide MRR and nDCG@5, so they must
    survive sampling deterministically."""
    labelled = _labelled()
    out = reranker.sample_negatives(labelled, n_hard=5, n_random=7, seed=0)
    for impression, block in labelled[labelled["label"] == 0].groupby("impression_id"):
        best = set(block.nsmallest(5, "rank")["article_id"])
        assert best <= set(out.loc[out["impression_id"] == impression, "article_id"])


def test_sample_negatives_is_seeded():
    labelled = _labelled()
    a = reranker.sample_negatives(labelled, 5, 7, seed=3)
    b = reranker.sample_negatives(labelled, 5, 7, seed=3)
    assert a["article_id"].tolist() == b["article_id"].tolist()


def test_before_follows_stage1_rank_not_tied_scores():
    """Popularity scores are integer counts with ties everywhere. 'Before' must be
    the stage-1 order -- rank -- or ties get scrambled: sorting four tied zeros
    would put the rank-1 positive last."""
    matrix = pd.DataFrame({
        "impression_id": ["i"] * 4,
        "retrieval_rank": [1, 2, 3, 4],
        "retrieval_score": [0.0, 0.0, 0.0, 0.0],
        "label": [1, 0, 0, 0],
    })
    report = reranker.evaluate_before_after(matrix, np.zeros(4), n_impressions_total=1)
    assert report["before"]["mrr"] == pytest.approx(1.0)


def test_chunked_groups_equal_a_single_pass():
    rng = np.random.default_rng(1)
    frames = []
    for i in range(6):
        labels = np.zeros(30, dtype=int)
        labels[rng.integers(30)] = 1
        frames.append(pd.DataFrame({"impression_id": f"i{i}", "retrieval_rank": np.arange(1, 31), "label": labels}))
    matrix = pd.concat(frames, ignore_index=True)
    scores = rng.normal(size=len(matrix))
    single = reranker.evaluate_before_after(matrix, scores, n_impressions_total=10)

    labels, before, after = [], [], []
    for ids in (["i0", "i1"], ["i2", "i3", "i4"], ["i5"]):
        mask = matrix["impression_id"].isin(ids).to_numpy()
        chunk_labels, chunk_before, chunk_after = reranker.score_groups(matrix[mask], scores[mask])
        labels += chunk_labels
        before += chunk_before
        after += chunk_after
    chunked = {
        "before": reranker.evaluate_over_population(labels, before, 10),
        "after": reranker.evaluate_over_population(labels, after, 10),
    }
    for side in ("before", "after"):
        for key in ("auc", "mrr", "ndcg@5", "ndcg@10"):
            assert chunked[side][key] == pytest.approx(single[side][key], rel=1e-12), (side, key)
