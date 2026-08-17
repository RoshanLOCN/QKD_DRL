"""Shared network construction for the learning agents (PyTorch)."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def build_mlp(input_size: int, hidden_sizes: Sequence[int], output_size: int) -> nn.Sequential:
    """A ReLU MLP (Section X: ReLU activations)."""
    layers = []
    prev = input_size
    for hidden in hidden_sizes:
        layers.append(nn.Linear(prev, hidden))
        layers.append(nn.ReLU())
        prev = hidden
    layers.append(nn.Linear(prev, output_size))
    return nn.Sequential(*layers)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set logits of infeasible actions to -inf so they get zero probability."""
    return logits.masked_fill(~mask, float("-inf"))
