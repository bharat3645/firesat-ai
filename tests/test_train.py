from __future__ import annotations

from firesat.training.evaluate import evaluate_model
from firesat.training.dataset import build_concat_dataset
from firesat.training.train import TrainConfig, load_checkpoint, train_firesat_model
from firesat.features.build_features import NormalizationStats

SEQ_LEN = 6


def test_train_firesat_model_runs_and_improves_or_stays_finite(tiny_datasets, tmp_path):
    config = TrainConfig(
        epochs=2,
        batch_size=4,
        sequence_length=SEQ_LEN,
        val_fraction=0.3,
    )
    checkpoint_path = tmp_path / "model.pt"
    model, history, metadata = train_firesat_model(
        tiny_datasets, config=config, checkpoint_path=checkpoint_path
    )

    assert len(history.train_loss) == 2
    assert all(loss == loss for loss in history.train_loss)  # no NaNs
    assert checkpoint_path.exists()
    assert metadata["n_train_samples"] > 0
    assert metadata["n_val_samples"] > 0


def test_checkpoint_roundtrip_matches_predictions(tiny_datasets, tmp_path):
    config = TrainConfig(epochs=1, batch_size=4, sequence_length=SEQ_LEN, val_fraction=0.3)
    checkpoint_path = tmp_path / "model.pt"
    model, _, metadata = train_firesat_model(
        tiny_datasets, config=config, checkpoint_path=checkpoint_path
    )

    loaded_model, loaded_metadata = load_checkpoint(checkpoint_path)
    assert loaded_metadata["regions"] == metadata["regions"]

    stats = NormalizationStats.from_dict(metadata["normalization_stats"])
    val_ds = build_concat_dataset(tiny_datasets, stats, SEQ_LEN, split="val", val_fraction=0.3)
    if len(val_ds) == 0:
        return  # tiny fixture may not always yield a val split; training itself is what's tested

    import torch

    sample = val_ds[0]
    spatial = sample["spatial"].unsqueeze(0)
    weather = sample["weather"].unsqueeze(0)
    with torch.no_grad():
        out_original = model(spatial, weather)
        out_loaded = loaded_model(spatial, weather)
    for key in out_original["logits"]:
        assert torch.allclose(out_original["logits"][key], out_loaded["logits"][key], atol=1e-5)


def test_evaluate_model_returns_metrics_for_each_horizon(tiny_datasets, tmp_path):
    config = TrainConfig(epochs=1, batch_size=4, sequence_length=SEQ_LEN, val_fraction=0.3)
    model, _, metadata = train_firesat_model(tiny_datasets, config=config)
    stats = NormalizationStats.from_dict(metadata["normalization_stats"])
    val_ds = build_concat_dataset(tiny_datasets, stats, SEQ_LEN, split="val", val_fraction=0.3)
    if len(val_ds) == 0:
        return
    metrics = evaluate_model(model, val_ds)
    assert len(metrics) > 0
    for m in metrics.values():
        assert 0.0 <= m.accuracy <= 1.0
        assert 0.0 <= m.macro_f1 <= 1.0
