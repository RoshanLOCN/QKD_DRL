"""Non-learning baselines: First-Fit (FF), Best-Fit (BF) and Random-Fit (RF).

Physical-layer constraints are NOT re-implemented here, and do not need to be: every
placement any of these heuristics can return comes from the environment's own block
search (``core.blocks``), which already enforces spectrum contiguity, spectral
continuity, core continuity, XT-avoidance and guard bands. The heuristics differ from
PPO/DQN only in the selection rule; see tests/test_heuristics_constraints.py.

* **FF** is the literature first-fit (as in Sharma et al. and the EON literature):
  the first candidate route on which all three channels fit, and on it the free block
  with the LOWEST starting slot for each channel, on the lowest-index data core for the
  DC. It searches the full free-run list, not just the closest-fit candidates offered
  to the learners, so it is the genuine first-fit policy.
* **BF** is best-fit: the first feasible joint action of the masked action space, i.e.
  the shortest route with the CLOSEST-FIT block per channel (the candidate ranking is
  closest-fit first). This is a stronger heuristic than FF; it was previously
  (mis)labelled FF.
* **RF** draws a feasible joint action uniformly at random.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from agents.base import ActionSelection, Agent
from core.blocks import MultiCoreBlock, SingleCoreBlock
from env.candidates import RequestCandidates

#: env-provided search: (route index k1, DC route index k2 or None, channel) -> all
#: feasible blocks for that channel on that route, or () if none.
BlockSearch = Callable[[int, Optional[int], str], Sequence]

Placement = Tuple[int, int, SingleCoreBlock, SingleCoreBlock, MultiCoreBlock]


class BestFitAgent(Agent):
    """Shortest route + closest-fit block per channel (first feasible masked action)."""

    trainable = False

    def select(self, state: np.ndarray, mask: np.ndarray, explore: bool) -> ActionSelection:
        feasible = np.flatnonzero(mask)
        if feasible.size == 0:
            raise ValueError("BestFitAgent.select called with an empty mask")
        return ActionSelection(action=int(feasible[0]))


class FirstFitAgent(Agent):
    """Literature first-fit: first route where everything fits, lowest-start blocks."""

    trainable = False

    def select(self, state: np.ndarray, mask: np.ndarray, explore: bool) -> ActionSelection:
        # Only reached if the environment does not use ``choose_placement``; fall back to
        # the first feasible masked action so the interface stays complete.
        feasible = np.flatnonzero(mask)
        if feasible.size == 0:
            raise ValueError("FirstFitAgent.select called with an empty mask")
        return ActionSelection(action=int(feasible[0]))

    def choose_placement(self, candidates: RequestCandidates, search: BlockSearch) -> Optional[Placement]:
        for k1 in range(len(candidates.k1_routes)):
            qc = _lowest_start(search(k1, None, "QC"))
            cc = _lowest_start(search(k1, None, "CC"))
            if qc is None or cc is None:
                continue
            for k2 in range(len(candidates.eligible_dc.get(k1, ()))):
                dc = _lowest_core_then_start(search(k1, k2, "DC"))
                if dc is not None:
                    return (k1, k2, qc, cc, dc)
        return None


def _lowest_start(blocks: Sequence[SingleCoreBlock]) -> Optional[SingleCoreBlock]:
    return min(blocks, key=lambda b: b.start) if blocks else None


def _lowest_core_then_start(blocks: Sequence[MultiCoreBlock]) -> Optional[MultiCoreBlock]:
    return min(blocks, key=lambda b: (b.core, b.start)) if blocks else None


class RandomFitAgent(Agent):
    trainable = False

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def select(self, state: np.ndarray, mask: np.ndarray, explore: bool) -> ActionSelection:
        feasible = np.flatnonzero(mask)
        if feasible.size == 0:
            raise ValueError("RandomFitAgent.select called with an empty mask")
        return ActionSelection(action=int(self._rng.choice(feasible)))
