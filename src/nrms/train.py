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

One protocol difference worth stating in the report: our monitored val AUC is a
per-group mean (AUC within each sampled row, averaged), whereas Keras'
`metrics=["AUC"]` pools every flattened (candidate, label) pair into a single
AUC. Ours is the same shape as the evaluation protocol -- their `AucScore` also
scores per impression and averages -- so it is the more meaningful number, but
the per-epoch values will NOT line up with the curves in their training logs.
Do not present it as a reproduction of their curve.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import evaluate_impressions
from src.nrms import config as nrms_config
from src.nrms.dataset import NrmsTrainDataset
from src.nrms.tracking import WandbTracker, get_logger

log = get_logger("train")


def peak_memory_mib(device) -> float:
    """Exact peak allocation since the last reset, in MiB.

    torch's own counter rather than sampled `nvidia-smi`: it is precise, it is
    attributable to a specific phase, and it needs no background poller.
    """
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**2


def _state_dict_to_cpu(model) -> dict:
    """CPU copy of the state dict that preserves storage sharing.

    `UserEncoder` holds a reference to the *same* `NewsEncoder` the candidate
    path uses, so the embedding appears twice in `state_dict()` --
    `news_encoder.embedding.weight` and
    `user_encoder.news_encoder.embedding.weight` -- as one shared tensor. Calling
    `.cpu()` per entry would materialise it twice, doubling both the host-RAM
    copy and the file on disk (1.5 GB for a ~774 MB model). Copying once per
    distinct storage keeps the alias an alias.
    """
    copied: dict[int, torch.Tensor] = {}
    out = {}
    for key, tensor in model.state_dict().items():
        pointer = tensor.data_ptr()
        if pointer not in copied:
            copied[pointer] = tensor.detach().cpu().clone()
        out[key] = copied[pointer]
    return out


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
    n_rows = 0
    for history, candidates, labels in loader:
        history = history.to(device)
        candidates = candidates.to(device)
        labels = labels.to(device)

        logits = model(history, candidates)
        # Weighted by batch size: a short final batch must not count as much as
        # a full one in the reported loss.
        losses.append(criterion(logits, labels).item() * len(labels))
        n_rows += len(labels)
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    if not all_logits:
        return float("nan"), float("nan")
    return float(sum(losses) / n_rows), _grouped_auc(
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
    tracker: WandbTracker | None = None,
) -> dict:
    device = device or resolve_device()
    tracker = tracker or WandbTracker(enabled=False)
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
        min_lr=nrms_config.LR_PLATEAU_MIN_LR,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "nrms.pt"

    history: list[dict] = []
    best_auc = -np.inf
    best_state = None
    epochs_without_improvement = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
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

        epoch_seconds = time.perf_counter() - epoch_started
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(epoch_seconds, 1),
            "peak_gpu_mib": round(peak_memory_mib(device), 1),
        }
        history.append(record)
        log.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f val_auc=%.4f (%.0fs, peak %.0f MiB)",
            epoch, epochs, train_loss, val_loss, val_auc,
            epoch_seconds, record["peak_gpu_mib"],
        )
        tracker.log_metrics({f"train/{k}": v for k, v in record.items() if k != "epoch"}, step=epoch)

        if val_auc > best_auc:
            best_auc = val_auc
            # Keep the best weights on CPU so a long run cannot be lost to an
            # OOM while a second copy sits on the GPU.
            best_state = _state_dict_to_cpu(model)
            torch.save(best_state, checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= nrms_config.EARLY_STOPPING_PATIENCE:
                log.info("early stopping at epoch %d (best val_auc=%.4f)", epoch, best_auc)
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # restore_best_weights=True

    summary = {
        "best_val_auc": float(best_auc),
        "epochs_run": len(history),
        "epochs_cap": epochs,
        "early_stopped": len(history) < epochs,
        "peak_gpu_mib_train": round(peak_memory_mib(device), 1),
        "history": history,
    }
    if not summary["early_stopped"]:
        # A run that ends because it ran out of epochs, rather than because it
        # stopped improving, has not converged -- and an unconverged baseline
        # makes any later "improvement" partly an artifact of training longer.
        log.warning(
            "hit the %d-epoch cap without early stopping; baseline may not be converged",
            epochs,
        )
    tracker.set_summary({
        "best_val_auc": summary["best_val_auc"],
        "epochs_run": summary["epochs_run"],
        "early_stopped": summary["early_stopped"],
        "peak_gpu_mib_train": summary["peak_gpu_mib_train"],
    })
    (artifact_dir / "train_history.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
