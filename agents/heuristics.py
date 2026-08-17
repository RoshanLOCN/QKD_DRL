"""Non-learning baselines: First-Fit (FF) and Random-Fit (RF).

Both operate over the same masked joint action space as the learning agents. FF takes the
first feasible joint combination; RF takes a uniformly random feasible one.
"""

from __future__ import annotations

import numpy as np

from agents.base import ActionSelection, Agent


class FirstFitAgent(Agent):
    trainable = False

    def select(self, state: np.ndarray, mask: np.ndarray, explore: bool) -> ActionSelection:
        feasible = np.flatnonzero(mask)
        if feasible.size == 0:
            raise ValueError("FirstFitAgent.select called with an empty mask")
        return ActionSelection(action=int(feasible[0]))


class RandomFitAgent(Agent):
    trainable = False

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def select(self, state: np.ndarray, mask: np.ndarray, explore: bool) -> ActionSelection:
        feasible = np.flatnonzero(mask)
        if feasible.size == 0:
            raise ValueError("RandomFitAgent.select called with an empty mask")
        return ActionSelection(action=int(self._rng.choice(feasible)))
