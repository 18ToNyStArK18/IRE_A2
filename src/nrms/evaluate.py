"""Score a held-out split with a trained NRMS and report the metric set A2 Q2
asks for (AUC, MRR, nDCG@5, nDCG@10), using the per-impression protocol in
src/metrics.py -- the same one ebnerd-benchmark's `MetricEvaluator` applies.

Evaluation runs at full in-view width: no negative sampling, every article the
platform actually showed is scored. Batches are padded on the candidate axis
(see dataset.eval_collate) and the mask trims each row back to its real
candidates before any metric sees it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.metrics import evaluate_impressions
from src.nrms import config as nrms_config
from src.nrms.dataset import NrmsEvalDataset, eval_collate
from src.nrms.ids import ArticleCodec
from src.nrms.tracking import WandbTracker, get_logger

log = get_logger("evaluate")


@torch.no_grad()
def score_split(
    model,
    impressions: pd.DataFrame,
    token_matrix: np.ndarray,
    *,
    batch_size: int = nrms_config.BATCH_SIZE,
    device: torch.device | None = None,
    num_workers: int = 0,
    freshness=None,
) -> list[np.ndarray]:
    """Per-impression score arrays, aligned with `impressions` row order."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    loader = DataLoader(
        NrmsEvalDataset(impressions, token_matrix, freshness),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=eval_collate,
    )

    scores: list[np.ndarray | None] = [None] * len(impressions)
    for history, candidates, _labels, mask, indices, log_age, known in loader:
        logits = (
            model(history.to(device), candidates.to(device), log_age.to(device), known.to(device))
            .float()
            .cpu()
            .numpy()
        )
        mask = mask.numpy()
        for row, index in enumerate(indices.numpy()):
            scores[int(index)] = logits[row][mask[row]]
    return scores


def evaluate_split(
    model,
    impressions: pd.DataFrame,
    token_matrix: np.ndarray,
    codec: ArticleCodec,
    artifact_dir: Path,
    *,
    split: str = "test",
    batch_size: int = nrms_config.BATCH_SIZE,
    device: torch.device | None = None,
    num_workers: int = 0,
    tracker: WandbTracker | None = None,
    freshness=None,
) -> dict:
    tracker = tracker or WandbTracker(enabled=False)
    if device is not None and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    scores = score_split(
        model,
        impressions,
        token_matrix,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        freshness=freshness,
    )
    labels = [np.asarray(l, dtype=np.float32) for l in impressions["labels"]]

    metrics = evaluate_impressions(labels, scores, ndcg_ks=(5, 10))
    metrics["split"] = split
    if device is not None and device.type == "cuda":
        # Eval is the memory-peaky phase: batches are padded to the widest
        # candidate list they contain, so this is measured separately.
        metrics["peak_gpu_mib_eval"] = round(
            torch.cuda.max_memory_allocated(device) / 1024**2, 1
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Article ids are decoded back to the originals so predictions are usable
    # for a Codabench submission without carrying the codec around.
    pd.DataFrame(
        {
            "impression_id": impressions["impression_id"].to_numpy(),
            "user_id": impressions["user_id"].to_numpy(),
            "article_ids": [codec.decode(c) for c in impressions["candidates"]],
            "scores": [s.astype(np.float32) for s in scores],
            "labels": labels,
        }
    ).to_parquet(artifact_dir / f"predictions_{split}.parquet", index=False)

    log.info("%s: %s", split, json.dumps(metrics))
    tracker.set_summary({f"{split}/{k}": v for k, v in metrics.items() if k != "split"})
    return metrics
