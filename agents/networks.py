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


class DuelingMLP(nn.Module):
    """Dueling Q-network (Wang et al., 2016): a shared ReLU trunk feeding a scalar
    state-value stream and an advantage stream, combined as
    ``Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')``.

    Motivation here: with a gamma=0.95 horizon the state value is ~20x larger than the
    differences between actions, so a single-head regression spends its capacity on
    the common value and leaves the action ranking to noise. The dueling split learns
    the value once and the advantages separately.
    """

    def __init__(self, input_size: int, hidden_sizes: Sequence[int], output_size: int) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("DuelingMLP needs at least one hidden layer")
        trunk = []
        prev = input_size
        for hidden in hidden_sizes[:-1]:
            trunk.append(nn.Linear(prev, hidden))
            trunk.append(nn.ReLU())
            prev = hidden
        self.trunk = nn.Sequential(*trunk) if trunk else nn.Identity()
        last = hidden_sizes[-1]
        self.value = nn.Sequential(nn.Linear(prev, last), nn.ReLU(), nn.Linear(last, 1))
        self.advantage = nn.Sequential(nn.Linear(prev, last), nn.ReLU(), nn.Linear(last, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        advantage = self.advantage(h)
        return self.value(h) + advantage - advantage.mean(dim=-1, keepdim=True)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set logits of infeasible actions to -inf so they get zero probability."""
    return logits.masked_fill(~mask, float("-inf"))
