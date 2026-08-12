"""Temporal encoder: Bidirectional LSTM or GRU over the sequence of monthly
spatial embeddings + weather features.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """BiLSTM/BiGRU wrapper. Input ``(B, T, D_in)`` -> output ``(B, T, 2*hidden)``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        cell: str = "lstm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        cell = cell.lower()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}.get(cell)
        if rnn_cls is None:
            raise ValueError(f"Unknown cell type '{cell}', expected 'lstm' or 'gru'.")
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out  # (B, T, 2*hidden_dim)
