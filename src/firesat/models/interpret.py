"""Interpretability utilities: turn the model's attention weights and
gradients into human-readable summaries and plots.

Two complementary views, matching the proposal's "interpretability features
(attention/feature-importance visualization)" deliverable:

1. **Attention weights** (free, exact, already computed in ``forward``):
   which channels (NDVI vs. SAR vs. thermal...) and which months in the
   lookback window mattered most for a given prediction.
2. **Gradient x input saliency** (a standard, cheap approximation -- *not*
   full Integrated Gradients/SHAP, which would need a proper baseline and
   path integral; documented as an approximation rather than oversold).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from firesat.config import SPATIAL_FEATURE_CHANNELS, WEATHER_FEATURE_COLUMNS


def summarize_channel_attention(channel_weights: np.ndarray) -> dict[str, float]:
    """``channel_weights``: ``(T, C)`` -> mean weight per named spatial channel."""
    mean_weights = channel_weights.mean(axis=0)
    return {
        name: float(mean_weights[i]) for i, name in enumerate(SPATIAL_FEATURE_CHANNELS)
    }


def summarize_temporal_attention(
    temporal_weights: np.ndarray, time_labels: Sequence[str]
) -> list[dict]:
    """``temporal_weights``: ``(T,)`` -> ranked list of ``{label, weight}``."""
    order = np.argsort(-temporal_weights)
    return [
        {"time": time_labels[i], "weight": float(temporal_weights[i])} for i in order
    ]


def gradient_input_saliency(
    model: torch.nn.Module,
    spatial: torch.Tensor,
    weather: torch.Tensor,
    horizon_key: str,
    target_class: int,
) -> dict[str, np.ndarray]:
    """Gradient x input saliency for one (sample, horizon, class) triple.

    Returns per-spatial-channel and per-weather-feature importance scores
    (mean absolute gradient*input, averaged over the batch/spatial/time
    dims as appropriate). This is a lightweight, single-backward-pass
    approximation -- fine for a "which inputs mattered" sanity check, not a
    rigorous attribution method.
    """
    model.eval()
    spatial = spatial.clone().detach().requires_grad_(True)
    weather = weather.clone().detach().requires_grad_(True)

    out = model(spatial, weather)
    logits = out["logits"][horizon_key]
    score = logits[:, target_class].sum()
    grads = torch.autograd.grad(score, [spatial, weather], retain_graph=False)
    spatial_grad, weather_grad = grads

    spatial_saliency = (spatial_grad * spatial).abs().mean(dim=(0, 1, 3, 4)).detach().cpu().numpy()
    weather_saliency = (weather_grad * weather).abs().mean(dim=(0, 1)).detach().cpu().numpy()

    return {
        "spatial_channel_importance": {
            name: float(spatial_saliency[i]) for i, name in enumerate(SPATIAL_FEATURE_CHANNELS)
        },
        "weather_feature_importance": {
            name: float(weather_saliency[i]) for i, name in enumerate(WEATHER_FEATURE_COLUMNS)
        },
    }


def plot_temporal_attention(time_labels: Sequence[str], weights: np.ndarray):
    """Return a matplotlib Figure showing the attention weight per lookback month."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(range(len(weights)), weights, color="#d1495b")
    ax.set_xticks(range(len(time_labels)))
    ax.set_xticklabels(time_labels, rotation=90, fontsize=6)
    ax.set_ylabel("attention weight")
    ax.set_title("Temporal attention over lookback window")
    fig.tight_layout()
    return fig


def plot_channel_importance(importances: dict[str, float]):
    """Return a matplotlib Figure: horizontal bar chart of channel attention/importance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(importances.keys())
    values = [importances[n] for n in names]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(names, values, color="#2e6f95")
    ax.set_xlabel("mean attention weight")
    ax.set_title("Spatial channel attention")
    fig.tight_layout()
    return fig
