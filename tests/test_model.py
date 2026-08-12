from __future__ import annotations

import torch

from firesat.config import HORIZONS_MONTHS, N_CLASSES, N_SPATIAL_CHANNELS, N_WEATHER_FEATURES
from firesat.models.attention import TemporalAttention
from firesat.models.cnn_encoder import ChannelAttention, SpatialEncoder
from firesat.models.firesat_net import FireSatNet
from firesat.models.temporal import TemporalEncoder


def test_channel_attention_weights_in_unit_interval():
    module = ChannelAttention(n_channels=6)
    x = torch.randn(4, 6, 8, 8)
    out, weights = module(x)
    assert out.shape == x.shape
    assert weights.shape == (4, 6)
    assert torch.all(weights >= 0) and torch.all(weights <= 1)


def test_spatial_encoder_output_shape():
    encoder = SpatialEncoder(in_channels=6, embedding_dim=32)
    x = torch.randn(5, 6, 8, 8)
    embedding, channel_weights = encoder(x)
    assert embedding.shape == (5, 32)
    assert channel_weights.shape == (5, 6)


def test_temporal_encoder_bidirectional_output_dim():
    encoder = TemporalEncoder(input_dim=10, hidden_dim=16, cell="lstm")
    x = torch.randn(3, 5, 10)
    out = encoder(x)
    assert out.shape == (3, 5, 32)  # bidirectional -> 2*hidden_dim
    assert encoder.output_dim == 32


def test_temporal_encoder_gru_variant():
    encoder = TemporalEncoder(input_dim=10, hidden_dim=8, cell="gru")
    out = encoder(torch.randn(2, 4, 10))
    assert out.shape == (2, 4, 16)


def test_temporal_encoder_rejects_unknown_cell():
    import pytest

    with pytest.raises(ValueError):
        TemporalEncoder(input_dim=10, hidden_dim=8, cell="not-a-cell")


def test_temporal_attention_weights_sum_to_one():
    attn = TemporalAttention(input_dim=16)
    seq = torch.randn(4, 7, 16)
    context, weights = attn(seq)
    assert context.shape == (4, 16)
    assert weights.shape == (4, 7)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)
    assert torch.all(weights >= 0)


def test_firesat_net_forward_shapes():
    model = FireSatNet(
        n_spatial_channels=N_SPATIAL_CHANNELS,
        n_weather_features=N_WEATHER_FEATURES,
        spatial_embedding_dim=16,
        temporal_hidden_dim=16,
    )
    batch, timesteps, h, w = 3, 6, 8, 8
    spatial = torch.randn(batch, timesteps, N_SPATIAL_CHANNELS, h, w)
    weather = torch.randn(batch, timesteps, N_WEATHER_FEATURES)

    out = model(spatial, weather)

    assert set(out["logits"].keys()) == {f"horizon_{h}m" for h in HORIZONS_MONTHS}
    for logits in out["logits"].values():
        assert logits.shape == (batch, N_CLASSES)
    assert out["temporal_attention"].shape == (batch, timesteps)
    assert torch.allclose(out["temporal_attention"].sum(dim=-1), torch.ones(batch), atol=1e-5)
    assert out["channel_attention"].shape == (batch, timesteps, N_SPATIAL_CHANNELS)
    assert torch.all(out["channel_attention"] >= 0) and torch.all(out["channel_attention"] <= 1)


def test_firesat_net_predict_proba_sums_to_one():
    model = FireSatNet(spatial_embedding_dim=8, temporal_hidden_dim=8)
    spatial = torch.randn(2, 4, N_SPATIAL_CHANNELS, 6, 6)
    weather = torch.randn(2, 4, N_WEATHER_FEATURES)
    probs = model.predict_proba(spatial, weather)
    for p in probs.values():
        assert torch.allclose(p.sum(dim=-1), torch.ones(2), atol=1e-5)
        assert torch.all(p >= 0)


def test_firesat_net_gradients_flow():
    model = FireSatNet(spatial_embedding_dim=8, temporal_hidden_dim=8)
    spatial = torch.randn(2, 4, N_SPATIAL_CHANNELS, 6, 6, requires_grad=True)
    weather = torch.randn(2, 4, N_WEATHER_FEATURES, requires_grad=True)
    out = model(spatial, weather)
    loss = sum(logit.sum() for logit in out["logits"].values())
    loss.backward()
    assert spatial.grad is not None
    assert torch.any(spatial.grad != 0)
    assert weather.grad is not None
