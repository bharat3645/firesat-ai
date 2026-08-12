"""Additive (Bahdanau-style) temporal attention over the BiLSTM/GRU outputs.

Produces a fixed-size context vector as a learned weighted sum over the
sequence, plus per-timestep attention weights that sum to 1 -- these are
the second interpretability artifact ("which recent months drove this risk
score") alongside the CNN encoder's per-channel attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int = 32) -> None:
        super().__init__()
        self.score_proj = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``sequence``: ``(B, T, D)``.

        Returns ``(context, weights)`` where ``context`` is ``(B, D)`` and
        ``weights`` is ``(B, T)`` and sums to 1 along ``T``.
        """
        scores = self.score_proj(sequence).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=-1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)  # (B, D)
        return context, weights
