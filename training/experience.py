"""Experience buffer -- one entry per served/blocked QLR arrival.

The buffer is an on-policy batch: it fills to size ``N``, drives a single update, then is
cleared (PPO is on-policy and the traffic-driven transition space is effectively
unbounded, so no persistent replay memory is used).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class Experience:
    """A single whole-QLR interaction.

    ``has_action`` is False for a blocked arrival: the environment offered no feasible
    joint action, so ``action``/``mask``/``log_prob`` are meaningless, but ``state`` and
    ``value`` (the critic's estimate at that state) are still real -- they let a return
    computed over the *full* arrival trajectory carry the blocking penalty back into the
    placement decisions that caused it, per the credit-assignment fix in the diagnosis.
    """

    state: np.ndarray
    action: Optional[int]
    reward: float
    mask: Optional[np.ndarray]
    log_prob: Optional[float] = None
    value: Optional[float] = None
    has_action: bool = True


class ExperienceBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity (N) must be >= 1")
        self._capacity = capacity
        self._items: List[Experience] = []

    def __len__(self) -> int:
        return len(self._items)

    def append(self, experience: Experience) -> None:
        self._items.append(experience)

    def is_full(self) -> bool:
        return len(self._items) >= self._capacity

    def items(self) -> List[Experience]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()
