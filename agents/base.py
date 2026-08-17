"""Agent interface shared by the learning agents (PPO, DQN) and the heuristics (FF, RF).

This module is intentionally free of any heavy dependency (no torch), so the environment
and tests can import :class:`ActionSelection` without pulling in a deep-learning stack.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ActionSelection:
    """A chosen joint-action index, with optional PPO bookkeeping.

    ``log_prob`` and ``value`` are populated only by policy-gradient agents that need
    them at collection time; heuristics leave them ``None``.
    """

    action: int
    log_prob: Optional[float] = None
    value: Optional[float] = None


class Agent(ABC):
    """Selects one joint action from the masked joint action space."""

    #: Whether this agent learns from stored experience (drives buffering/updates).
    trainable: bool = False

    @abstractmethod
    def select(self, state: np.ndarray, mask: np.ndarray, explore: bool) -> ActionSelection:
        """Return a selection whose action index is guaranteed feasible under ``mask``.

        Precondition: ``mask`` has at least one True entry (callers block empty-mask QLRs
        before reaching here).
        """

    def update(self, experiences) -> Optional[float]:  # pragma: no cover - default no-op
        """Learn from a batch of experiences. Returns an optional scalar loss."""
        return None

    def decay_exploration(self) -> None:  # pragma: no cover - default no-op
        """Advance any exploration schedule (called after each update)."""
        return None
