"""Quantum Lightpath Request (QLR) model -- Eq. (3).

Each request carries three independent per-channel demand figures, since QC, CC and DC
are each sized separately in SS-EON.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.topology import NodeId


@dataclass(frozen=True)
class QLR:
    """A single Quantum Lightpath Request."""

    request_id: int
    source: NodeId          # o_t
    destination: NodeId     # d_t
    arrival_time: float     # t_arr
    update_time: float      # t_upd
    departure_time: float   # t_dep
    key_update_period: float  # T
    b_qc: float             # protocol-fixed quantum demand
    b_cc: float             # classical control-channel demand (Gbps)
    b_dc: float             # classical data-channel demand (Gbps)
