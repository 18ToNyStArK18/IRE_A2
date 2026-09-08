"""torch Datasets for NRMS, corresponding to ebnerd-benchmark's
`NRMSDataLoader` in its two modes.

Training (their eval_mode=False) is rectangular: every row has exactly
npratio+1 candidates, so it batches with the default collate.

Evaluation (their eval_mode=True) is ragged -- an impression shows however
many articles it showed. We pad the candidate axis per batch and return a
boolean mask, so evaluation still batches instead of degenerating to one
impression at a time. Padded slots must be dropped before scoring metrics;
`evaluate.py` uses the mask for exactly that.

Both datasets index a shared (n_articles+1, title_size) token matrix by article
code, so article text is stored once rather than per occurrence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NrmsTrainDataset(Dataset):
    """Rows from sampling.sampling_strategy_wu2019."""

    def __init__(self, samples: pd.DataFrame, token_matrix: np.ndarray):
        self.histories = list(samples["history"])
        self.candidates = list(samples["candidates"])
        self.labels = samples["label"].to_numpy(dtype=np.int64)
        self.token_matrix = token_matrix

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        history = self.token_matrix[np.asarray(self.histories[index], dtype=np.int64)]
        candidates = self.token_matrix[np.asarray(self.candidates[index], dtype=np.int64)]
        return (
            torch.from_numpy(history.astype(np.int64)),
            torch.from_numpy(candidates.astype(np.int64)),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


class NrmsEvalDataset(Dataset):
    """Rows from adapter.load_impressions, at full in-view width."""

    def __init__(self, impressions: pd.DataFrame, token_matrix: np.ndarray):
        self.impression_ids = list(impressions["impression_id"])
        self.histories = list(impressions["history"])
        self.candidates = list(impressions["candidates"])
        self.labels = list(impressions["labels"])
        self.token_matrix = token_matrix

    def __len__(self) -> int:
        return len(self.impression_ids)

    def __getitem__(self, index: int):
        history = self.token_matrix[np.asarray(self.histories[index], dtype=np.int64)]
        candidates = self.token_matrix[np.asarray(self.candidates[index], dtype=np.int64)]
        return (
            torch.from_numpy(history.astype(np.int64)),
            torch.from_numpy(candidates.astype(np.int64)),
            torch.from_numpy(np.asarray(self.labels[index], dtype=np.float32)),
            index,
        )


def eval_collate(batch):
    """Pad the ragged candidate axis and report which slots are real.

    Returns (history, candidates, labels, mask, indices) where mask is
    (B, max_candidates) True on real candidates. Padded candidate slots point
    at token row 0, which is the all-zero "unknown article" row, so they are
    harmless to score -- but they are still excluded from metrics via the mask.
    """
    histories, candidates, labels, indices = zip(*batch)
    max_candidates = max(c.shape[0] for c in candidates)
    title_size = candidates[0].shape[1]

    padded = torch.zeros(len(batch), max_candidates, title_size, dtype=torch.long)
    padded_labels = torch.zeros(len(batch), max_candidates, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_candidates, dtype=torch.bool)

    for i, (cand, lab) in enumerate(zip(candidates, labels)):
        n = cand.shape[0]
        padded[i, :n] = cand
        padded_labels[i, :n] = lab
        mask[i, :n] = True

    return (
        torch.stack(histories),
        padded,
        padded_labels,
        mask,
        torch.tensor(indices, dtype=torch.long),
    )
