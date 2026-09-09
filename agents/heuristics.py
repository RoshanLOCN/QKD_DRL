"""Non-learning baselines: First-Fit (FF) and Random-Fit (RF).

Both operate over the same masked joint action space as the learning agents. FF takes the
first feasible joint combination; RF takes a uniformly random feasible one.

Physical-layer constraints are NOT re-implemented here, and do not need to be: every
action either agent can pick indexes a candidate produced by ``env.candidates.
build_request_candidates``, which already enforces spectrum contiguity (contiguous FSU
runs only, ``core.blocks._free_runs``), spectral continuity (slots free on every link of
the route, ``core.resource_grid.SlotTable.free_mask``), core continuity (one core fixed
for the whole route per candidate), XT-avoidance (adjacent-core exclusion when
``network.xt_avoided``) and guard bands (folded into the required block size). FF/RF
therefore face exactly the same feasible set as PPO and DQN -- the methods differ only
in the selection rule; see tests/test_heuristics_constraints.py for the proof.
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
