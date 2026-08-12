"""ResNet-style spatial encoder with a squeeze-and-excitation channel-attention
block, used to summarize each monthly (C, H, W) feature stack into an
embedding vector before it enters the temporal branch.

Kept intentionally small (a handful of residual blocks, no pretrained
weights) since inputs here are 6-channel geophysical feature stacks, not
3-channel natural images -- ImageNet-pretrained ResNet weights would not
transfer meaningfully to NDVI/NBR/SAR channels anyway.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style attention over input feature channels.

    Produces a per-channel weight in (0, 1) used to rescale the input before
    the convolutional trunk -- this is the "attention-based feature
    weighting" the proposal calls out, and its weights are directly
    interpretable as "how much did NDVI vs. SAR vs. thermal anomaly matter
    for this prediction" (see ``firesat.models.interpret``).
    """

    def __init__(self, n_channels: int, reduction: int = 2) -> None:
        super().__init__()
        hidden = max(1, n_channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        b, c, _, _ = x.shape
        pooled = self.pool(x).view(b, c)
        weights = self.fc(pooled)  # (B, C) in (0, 1)
        out = x * weights.view(b, c, 1, 1)
        return out, weights


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class SpatialEncoder(nn.Module):
    """Small ResNet-style CNN: stem -> N residual blocks -> global pool -> embedding.

    Input: ``(B, C_in, H, W)``. Output: ``(B, embedding_dim)`` plus the
    channel-attention weights ``(B, C_in)`` for interpretability.
    """

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int = 64,
        stem_channels: int = 32,
        n_residual_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(stem_channels) for _ in range(n_residual_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(stem_channels, embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, channel_weights = self.channel_attention(x)
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.pool(x).flatten(1)
        embedding = self.project(x)
        return embedding, channel_weights
