"""Evaluation + backtesting against realized fire ignitions (a stand-in for
NASA FIRMS detections / Alaska Fire Service perimeters -- the ground truth
labels are themselves derived from the synthetic ignition process in offline
mode, and from FIRMS/perimeter data in the live pipeline; this module treats
that ground truth identically either way).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from firesat.config import HORIZONS_MONTHS, RISK_CLASSES
from firesat.models.firesat_net import FireSatNet
from firesat.training.dataset import collate_fire_risk_batch


@dataclass
class HorizonMetrics:
    horizon_months: int
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    confusion: list[list[int]]
    fire_recall: float  # of months a fire actually occurred/escalated, fraction flagged >= Moderate
    false_alarm_rate: float  # of months predicted High, fraction where no elevated risk realized


def evaluate_model(
    model: FireSatNet, dataset, device: str = "cpu", batch_size: int = 32
) -> dict[str, HorizonMetrics]:
    """Run the model over ``dataset`` and compute per-horizon classification
    metrics plus a wildfire-specific backtest (fire recall / false-alarm rate).
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fire_risk_batch
    )
    model.eval()
    device_t = torch.device(device)

    all_preds: dict[str, list[int]] = {f"horizon_{h}m": [] for h in HORIZONS_MONTHS}
    all_labels: dict[str, list[int]] = {f"horizon_{h}m": [] for h in HORIZONS_MONTHS}

    with torch.no_grad():
        for batch in loader:
            spatial = batch["spatial"].to(device_t)
            weather = batch["weather"].to(device_t)
            labels = batch["labels"].to(device_t)
            out = model(spatial, weather)
            for h_idx, h in enumerate(HORIZONS_MONTHS):
                key = f"horizon_{h}m"
                preds = out["logits"][key].argmax(dim=-1).cpu().numpy().tolist()
                all_preds[key].extend(preds)
                all_labels[key].extend(labels[:, h_idx].cpu().numpy().tolist())

    results = {}
    for h in HORIZONS_MONTHS:
        key = f"horizon_{h}m"
        y_true = np.array(all_labels[key])
        y_pred = np.array(all_preds[key])
        if len(y_true) == 0:
            continue
        labels_range = list(range(len(RISK_CLASSES)))
        acc = float((y_true == y_pred).mean())
        f1 = float(f1_score(y_true, y_pred, labels=labels_range, average="macro", zero_division=0))
        prec = float(
            precision_score(y_true, y_pred, labels=labels_range, average="macro", zero_division=0)
        )
        rec = float(
            recall_score(y_true, y_pred, labels=labels_range, average="macro", zero_division=0)
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels_range).tolist()

        fire_months = y_true > 0
        fire_recall = (
            float((y_pred[fire_months] > 0).mean()) if fire_months.any() else float("nan")
        )
        predicted_high = y_pred == 2
        false_alarm_rate = (
            float((y_true[predicted_high] == 0).mean()) if predicted_high.any() else 0.0
        )

        results[key] = HorizonMetrics(
            horizon_months=h,
            accuracy=acc,
            macro_f1=f1,
            macro_precision=prec,
            macro_recall=rec,
            confusion=cm,
            fire_recall=fire_recall,
            false_alarm_rate=false_alarm_rate,
        )
    return results


def metrics_to_dict(metrics: dict[str, HorizonMetrics]) -> dict:
    return {
        key: {
            "horizon_months": m.horizon_months,
            "accuracy": round(m.accuracy, 4),
            "macro_f1": round(m.macro_f1, 4),
            "macro_precision": round(m.macro_precision, 4),
            "macro_recall": round(m.macro_recall, 4),
            "confusion_matrix": m.confusion,
            "fire_recall": None if np.isnan(m.fire_recall) else round(m.fire_recall, 4),
            "false_alarm_rate": round(m.false_alarm_rate, 4),
            "risk_classes": RISK_CLASSES,
        }
        for key, m in metrics.items()
    }
