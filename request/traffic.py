"""Traffic generator -- Poisson arrivals, exponential holding, uniform node pairs.

The arrival rate is held on the generator (not baked into config) so evaluation can
sweep it (``configure_arrival_rate``). All randomness flows from a single seeded NumPy
Generator, so a given seed reproduces the same request stream exactly.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from configs.config import TrafficConfig
from core.topology import NodeId, Topology
from request.qlr import QLR


class TrafficGenerator:
    """Event-driven QLR stream over simulated time."""

    def __init__(self, config: TrafficConfig, topology: Topology) -> None:
        if topology.num_nodes < 2:
            raise ValueError("topology must have at least two nodes to form requests")
        self._config = config
        self._nodes: Sequence[NodeId] = topology.nodes
        self._arrival_rate = config.arrival_rate
        self._rng = np.random.default_rng(config.seed)
        self._clock = 0.0
        self._next_id = 0

    def configure_arrival_rate(self, arrival_rate: float) -> None:
        if arrival_rate <= 0:
            raise ValueError("arrival_rate must be > 0")
        self._arrival_rate = arrival_rate

    def reset(self, seed: int | None = None) -> None:
        """Reset the clock and (optionally) reseed, for reproducible episodes/runs."""
        self._rng = np.random.default_rng(self._config.seed if seed is None else seed)
        self._clock = 0.0
        self._next_id = 0

    def advance_time(self) -> float:
        """Advance the arrival clock by one exponential inter-arrival time."""
        self._clock += float(self._rng.exponential(1.0 / self._arrival_rate))
        return self._clock

    def generate_request(self) -> QLR:
        """Produce the next QLR at the current arrival clock."""
        source_idx, dest_idx = self._rng.choice(len(self._nodes), size=2, replace=False)
        holding = float(self._rng.exponential(self._config.mean_holding_time))
        b_cc = float(self._rng.choice(self._config.b_cc_classes))
        b_dc = float(self._rng.choice(self._config.b_dc_classes))
        request = QLR(
            request_id=self._next_id,
            source=self._nodes[int(source_idx)],
            destination=self._nodes[int(dest_idx)],
            arrival_time=self._clock,
            update_time=self._clock + self._config.key_update_period,
            departure_time=self._clock + holding,
            key_update_period=self._config.key_update_period,
            b_qc=self._config.b_qc,
            b_cc=b_cc,
            b_dc=b_dc,
        )
        self._next_id += 1
        return request
