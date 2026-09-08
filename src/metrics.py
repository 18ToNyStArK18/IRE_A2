"""Ranking metrics, scored per impression and averaged -- the protocol
ebnerd-benchmark's `MetricEvaluator` uses (AUC, MRR, nDCG@k), and the metric
set A2 asks for in Q2 and Q5.

Deliberately at src/metrics.py rather than inside src/nrms/: the same harness
has to score the Q1 re-ranker and the full two-stage pipeline later, so it must
not be owned by one model.

"Per impression" is the important part. A single global AUC over pooled
(candidate, label) pairs answers a different and easier question than "within
this impression, did the model rank the clicked article above the others",
which is what a news recommender is judged on.
"""

from __future__ import annotations

import numpy as np

from sklearn.metrics import roc_auc_score


def _as_arrays(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels, dtype=np.float64).ravel()
    y_score = np.asarray(scores, dtype=np.float64).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError(f"labels {y_true.shape} and scores {y_score.shape} misaligned")
    return y_true, y_score


def mrr_score(labels, scores) -> float:
    """Mean reciprocal rank over the positives in one impression, matching
    their implementation: every positive contributes 1/rank, normalised by the
    number of positives (so a multi-click impression is not counted twice)."""
    y_true, y_score = _as_arrays(labels, scores)
    order = np.argsort(-y_score)
    ranked = y_true[order]
    reciprocal = ranked / (np.arange(len(ranked)) + 1)
    total = ranked.sum()
    return float(reciprocal.sum() / total) if total > 0 else 0.0


def dcg_score(labels, scores, k: int) -> float:
    y_true, y_score = _as_arrays(labels, scores)
    order = np.argsort(-y_score)[:k]
    gains = 2 ** y_true[order] - 1
    discounts = np.log2(np.arange(len(order)) + 2)
    return float((gains / discounts).sum())


def ndcg_score(labels, scores, k: int) -> float:
    """DCG@k normalised by the best achievable DCG@k for this impression."""
    ideal = dcg_score(labels, labels, k)
    if ideal == 0:
        return 0.0
    return dcg_score(labels, scores, k) / ideal


def evaluate_impressions(
    labels_per_impression,
    scores_per_impression,
    ndcg_ks=(5, 10),
) -> dict[str, float]:
    """Average AUC/MRR/nDCG@k across impressions.

    Impressions whose labels are all-0 or all-1 are skipped for AUC only --
    AUC is undefined there -- while MRR/nDCG still accept them. The count of
    usable impressions is reported so a degenerate split cannot masquerade as
    a good score.
    """
    auc_values: list[float] = []
    mrr_values: list[float] = []
    ndcg_values: dict[int, list[float]] = {k: [] for k in ndcg_ks}

    n_impressions = 0
    for labels, scores in zip(labels_per_impression, scores_per_impression):
        y_true, y_score = _as_arrays(labels, scores)
        if len(y_true) == 0:
            continue
        n_impressions += 1

        if 0 < y_true.sum() < len(y_true):
            auc_values.append(float(roc_auc_score(y_true, y_score)))
        mrr_values.append(mrr_score(y_true, y_score))
        for k in ndcg_ks:
            ndcg_values[k].append(ndcg_score(y_true, y_score, k))

    results = {
        "auc": float(np.mean(auc_values)) if auc_values else float("nan"),
        "mrr": float(np.mean(mrr_values)) if mrr_values else float("nan"),
        "n_impressions": n_impressions,
        "n_impressions_scored_for_auc": len(auc_values),
    }
    for k in ndcg_ks:
        results[f"ndcg@{k}"] = float(np.mean(ndcg_values[k])) if ndcg_values[k] else float("nan")
    return results
