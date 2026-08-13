#!/usr/bin/env python
"""Train FireSatNet on the processed (synthetic, by default) dataset and
save a checkpoint the API/dashboard load automatically.

Usage:
    python scripts/generate_demo_data.py   # once, if not already run
    python scripts/train_demo.py
    python scripts/train_demo.py --epochs 12 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesat.config import PROCESSED_DATA_DIR  # noqa: E402
from firesat.data.pipeline import load_all_processed_regions  # noqa: E402
from firesat.features.build_features import NormalizationStats  # noqa: E402
from firesat.training.dataset import build_concat_dataset  # noqa: E402
from firesat.training.evaluate import evaluate_model, metrics_to_dict  # noqa: E402
from firesat.training.train import DEFAULT_CHECKPOINT_PATH, TrainConfig, train_firesat_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--metrics-out", type=str, default="docs/eval_metrics.json")
    args = parser.parse_args()

    logger.info("Loading processed data from %s", args.data_dir)
    datasets = load_all_processed_regions(args.data_dir)

    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    model, history, metadata = train_firesat_model(
        datasets, config=config, checkpoint_path=args.checkpoint
    )

    stats = NormalizationStats.from_dict(metadata["normalization_stats"])
    val_ds = build_concat_dataset(
        datasets, stats, config.sequence_length, split="val", val_fraction=config.val_fraction
    )
    metrics = evaluate_model(model, val_ds)
    metrics_dict = metrics_to_dict(metrics)

    logger.info("Final validation metrics:")
    for key, m in metrics_dict.items():
        logger.info(
            "  %s: acc=%.3f macro_f1=%.3f fire_recall=%s false_alarm_rate=%.3f",
            key,
            m["accuracy"],
            m["macro_f1"],
            m["fire_recall"],
            m["false_alarm_rate"],
        )

    out_path = Path(args.metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "history": {
                    "train_loss": history.train_loss,
                    "val_loss": history.val_loss,
                    "val_accuracy": history.val_accuracy,
                },
                "val_metrics": metrics_dict,
                "n_train_samples": metadata["n_train_samples"],
                "n_val_samples": metadata["n_val_samples"],
            },
            fh,
            indent=2,
        )
    logger.info("Wrote metrics to %s", out_path)
    logger.info("Checkpoint saved to %s", args.checkpoint)


if __name__ == "__main__":
    main()
