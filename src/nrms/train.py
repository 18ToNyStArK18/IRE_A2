"""NRMS training loop, following ebnerd-benchmark's reproducibility script
(`examples/reproducibility_scripts/ebnerd_nrms.py`).

Their Keras `fit` is configured with Adam, early stopping on validation AUC
(patience 4, restore best weights), ReduceLROnPlateau and a checkpoint. The
same protocol is expressed here as an explicit loop.

Objective: softmax cross-entropy over each row's npratio+1 candidates. Wu-2019
sampling puts exactly one positive in every row and records its index, so
`nn.CrossEntropyLoss` over the candidate axis is identical to their
categorical cross-entropy on a one-hot target.

Validation is negatively sampled too, matching their script -- the monitored
val AUC is therefore a 1-positive-vs-npratio-negatives figure, not the
full-inview test metric that evaluate.py reports.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import evaluate_impressions
from src.nrms import config as nrms_config
from src.nrms.dataset import NrmsTrainDataset


def resolve_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _grouped_auc(logits: np.ndarray, labels: np.ndarray) -> float:
    """Validation AUC over sampled groups: each row is one positive against
    npratio negatives, so every group is scorable."""
    one_hot = np.zeros_like(logits)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return evaluate_impressions(list(one_hot), list(logits), ndcg_ks=())["auc"]


@torch.no_grad()
def _validate(model, loader, device, criterion) -> tuple[float, float]:
    model.eval()
    losses, all_logits, all_labels = [], [], []
    for history, candidates, labels in loader:
        history = history.to(device)
        candidates = candidates.to(device)
        labels = labels.to(device)

        logits = model(history, candidates)
        losses.append(criterion(logits, labels).item())
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    if not all_logits:
        return float("nan"), float("nan")
    return float(np.mean(losses)), _grouped_auc(
        np.concatenate(all_logits), np.concatenate(all_labels)
    )


def train_model(
    model,
    train_samples,
    val_samples,
    token_matrix,
    artifact_dir: Path,
    *,
    epochs: int = nrms_config.EPOCHS,
    batch_size: int = nrms_config.BATCH_SIZE,
    learning_rate: float = nrms_config.LEARNING_RATE,
    device: torch.device | None = None,
    num_workers: int = 0,
) -> dict:
    device = device or resolve_device()
    model = model.to(device)

    train_loader = DataLoader(
        NrmsTrainDataset(train_samples, token_matrix),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        NrmsTrainDataset(val_samples, token_matrix),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=nrms_config.LR_PLATEAU_FACTOR,
        patience=nrms_config.LR_PLATEAU_PATIENCE,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "nrms.pt"

    history: list[dict] = []
    best_auc = -np.inf
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch_history, batch_candidates, batch_labels in train_loader:
            batch_history = batch_history.to(device)
            batch_candidates = batch_candidates.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_history, batch_candidates)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss, val_auc = _validate(model, val_loader, device, criterion)
        scheduler.step(val_auc)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"[nrms] epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            # Keep the best weights on CPU so a long run cannot be lost to an
            # OOM while a second copy sits on the GPU.
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
            torch.save(best_state, checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= nrms_config.EARLY_STOPPING_PATIENCE:
                print(f"[nrms] early stopping at epoch {epoch} (best val_auc={best_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # restore_best_weights=True

    summary = {"best_val_auc": float(best_auc), "epochs_run": len(history), "history": history}
    (artifact_dir / "train_history.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
