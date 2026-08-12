"""Training loop for FireSatNet.

Trains on a chronological train/validation split (validation = trailing
months, per region) with per-horizon cross-entropy loss summed across the
three heads. Designed to run in minutes on CPU against the synthetic
dataset for demo/CI purposes; swap in real acquired data (see
``firesat.data.pipeline``) for a production run -- the training code itself
does not change.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from firesat.config import (
    CHECKPOINTS_DIR,
    HORIZONS_MONTHS,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    WEATHER_FEATURE_COLUMNS,
)
from firesat.data.synthetic import SyntheticRegionDataset
from firesat.features.build_features import fit_normalization_stats
from firesat.models.firesat_net import FireSatNet
from firesat.training.dataset import build_concat_dataset, collate_fire_risk_batch

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 6
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    sequence_length: int = SEQUENCE_LENGTH
    val_fraction: float = 0.2
    device: str = "cpu"
    seed: int = RANDOM_SEED
    class_weights: bool = True


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: dict[str, list[float]] = field(default_factory=dict)


def _compute_class_weights(datasets: dict[str, SyntheticRegionDataset]) -> torch.Tensor:
    """Inverse-frequency class weights (across all horizons/regions) to counter
    the natural No-Risk >> Moderate > High class imbalance in wildfire data."""
    from firesat.config import N_CLASSES

    counts = np.zeros(N_CLASSES)
    for ds in datasets.values():
        valid = ds.labels[ds.labels >= 0]
        for c in range(N_CLASSES):
            counts[c] += (valid == c).sum()
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (N_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_firesat_model(
    datasets: dict[str, SyntheticRegionDataset],
    config: TrainConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[FireSatNet, TrainHistory, dict]:
    """Train FireSatNet on the given per-region datasets.

    Returns ``(model, history, metadata)`` where ``metadata`` bundles the
    fitted normalization stats and config needed to reproduce inference.
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    all_spatial = np.concatenate([d.spatial for d in datasets.values()], axis=0)
    all_weather = np.concatenate(
        [d.weather[WEATHER_FEATURE_COLUMNS].to_numpy(dtype=np.float32) for d in datasets.values()],
        axis=0,
    )
    stats = fit_normalization_stats(all_spatial, all_weather)

    train_ds = build_concat_dataset(
        datasets, stats, config.sequence_length, split="train", val_fraction=config.val_fraction
    )
    val_ds = build_concat_dataset(
        datasets, stats, config.sequence_length, split="val", val_fraction=config.val_fraction
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fire_risk_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fire_risk_batch,
    )

    device = torch.device(config.device)
    model = FireSatNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    class_weights = _compute_class_weights(datasets).to(device) if config.class_weights else None
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    history = TrainHistory(val_accuracy={f"horizon_{h}m": [] for h in HORIZONS_MONTHS})

    for epoch in range(config.epochs):
        t0 = time.time()
        model.train()
        train_losses = []
        for batch in train_loader:
            spatial = batch["spatial"].to(device)
            weather = batch["weather"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(spatial, weather)
            loss = torch.zeros((), device=device)
            for h_idx, h in enumerate(HORIZONS_MONTHS):
                loss = loss + criterion(out["logits"][f"horizon_{h}m"], labels[:, h_idx])
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        correct = {f"horizon_{h}m": 0 for h in HORIZONS_MONTHS}
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                spatial = batch["spatial"].to(device)
                weather = batch["weather"].to(device)
                labels = batch["labels"].to(device)
                out = model(spatial, weather)
                loss = torch.zeros((), device=device)
                for h_idx, h in enumerate(HORIZONS_MONTHS):
                    key = f"horizon_{h}m"
                    loss = loss + criterion(out["logits"][key], labels[:, h_idx])
                    preds = out["logits"][key].argmax(dim=-1)
                    correct[key] += (preds == labels[:, h_idx]).sum().item()
                val_losses.append(loss.item())
                total += labels.shape[0]

        epoch_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        epoch_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.train_loss.append(epoch_train_loss)
        history.val_loss.append(epoch_val_loss)
        for h in HORIZONS_MONTHS:
            key = f"horizon_{h}m"
            acc = correct[key] / total if total else float("nan")
            history.val_accuracy[key].append(acc)

        logger.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f val_acc=%s (%.1fs)",
            epoch + 1,
            config.epochs,
            epoch_train_loss,
            epoch_val_loss,
            {k: round(v[-1], 3) for k, v in history.val_accuracy.items()},
            time.time() - t0,
        )

    metadata = {
        "normalization_stats": stats.to_dict(),
        "config": config.__dict__,
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
        "regions": list(datasets.keys()),
    }

    if checkpoint_path is not None:
        save_checkpoint(model, metadata, checkpoint_path)

    return model, history, metadata


def save_checkpoint(model: FireSatNet, metadata: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[FireSatNet, dict]:
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = FireSatNet()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt["metadata"]


DEFAULT_CHECKPOINT_PATH = CHECKPOINTS_DIR / "firesat_demo.pt"
