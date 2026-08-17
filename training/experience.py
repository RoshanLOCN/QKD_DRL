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
    """A single whole-QLR interaction."""

    state: np.ndarray
    action: int
    reward: float
    mask: np.ndarray
    log_prob: Optional[float] = None
    value: Optional[float] = None


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
