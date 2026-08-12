"""FireSatNet: the hybrid CNN + BiLSTM/GRU + attention wildfire-risk model.

Architecture (matches the proposal's design):

    spatial stack (B,T,C,H,W) --[ResNet-style CNN + channel attention]--> per-timestep embedding
    concat with weather (B,T,F) --> (B,T,D)
    --[Bidirectional LSTM/GRU]--> (B,T,2H)
    --[additive temporal attention]--> context (B,2H) + per-month attention weights
    --[per-horizon linear heads]--> 3 sets of (No Risk / Moderate / High) logits,
      one head per forecast horizon (1, 3, 6 months)

``forward`` always returns attention weights alongside the logits -- there is
no separate "interpretability mode" to remember to enable, since the whole
point of the attention module is to be inspectable by default.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from firesat.config import HORIZONS_MONTHS, N_CLASSES, N_SPATIAL_CHANNELS, N_WEATHER_FEATURES
from firesat.models.attention import TemporalAttention
from firesat.models.cnn_encoder import SpatialEncoder
from firesat.models.temporal import TemporalEncoder


class FireSatNet(nn.Module):
    def __init__(
        self,
        n_spatial_channels: int = N_SPATIAL_CHANNELS,
        n_weather_features: int = N_WEATHER_FEATURES,
        spatial_embedding_dim: int = 64,
        temporal_hidden_dim: int = 64,
        temporal_cell: str = "lstm",
        temporal_layers: int = 1,
        n_classes: int = N_CLASSES,
        horizons: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizons = horizons or HORIZONS_MONTHS
        self.spatial_encoder = SpatialEncoder(
            in_channels=n_spatial_channels, embedding_dim=spatial_embedding_dim
        )
        temporal_input_dim = spatial_embedding_dim + n_weather_features
        self.temporal_encoder = TemporalEncoder(
            input_dim=temporal_input_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=temporal_layers,
            cell=temporal_cell,
        )
        self.temporal_attention = TemporalAttention(self.temporal_encoder.output_dim)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {
                f"horizon_{h}m": nn.Linear(self.temporal_encoder.output_dim, n_classes)
                for h in self.horizons
            }
        )

    def forward(
        self, spatial: torch.Tensor, weather: torch.Tensor
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """``spatial``: ``(B, T, C, H, W)``. ``weather``: ``(B, T, F)``.

        Returns a dict with:
          - ``logits``: ``{f"horizon_{h}m": (B, n_classes)}`` for each horizon
          - ``temporal_attention``: ``(B, T)`` weights, sum to 1 over T
          - ``channel_attention``: ``(B, T, C)`` weights in (0, 1)
          - ``embedding``: ``(B, 2*hidden)`` pooled context vector (for probing/eval)
        """
        b, t, c, h, w = spatial.shape
        flat_spatial = spatial.reshape(b * t, c, h, w)
        spatial_embedding, channel_weights = self.spatial_encoder(flat_spatial)
        spatial_embedding = spatial_embedding.reshape(b, t, -1)
        channel_weights = channel_weights.reshape(b, t, c)

        combined = torch.cat([spatial_embedding, weather], dim=-1)  # (B, T, D)
        temporal_out = self.temporal_encoder(combined)  # (B, T, 2H)
        context, temporal_weights = self.temporal_attention(temporal_out)  # (B,2H), (B,T)
        context = self.dropout(context)

        logits = {name: head(context) for name, head in self.heads.items()}
        return {
            "logits": logits,
            "temporal_attention": temporal_weights,
            "channel_attention": channel_weights,
            "embedding": context,
        }

    def predict_proba(
        self, spatial: torch.Tensor, weather: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Convenience wrapper: softmax probabilities per horizon, no grad."""
        self.eval()
        with torch.no_grad():
            out = self.forward(spatial, weather)
        return {name: torch.softmax(logit, dim=-1) for name, logit in out["logits"].items()}
