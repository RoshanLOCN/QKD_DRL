"""Joint state encoding (Eq. 10) and joint-action decoding (Eq. 11).

The state is the nested structure of Eq. 10 flattened to a **fixed-length** vector by
padding every missing candidate (route or block) with a sentinel value, so the policy
network's input size is constant regardless of how many real candidates a given request
has. The nesting (DC candidates listed per ``k1``) is preserved by the layout order.

The joint action is a mixed-radix encoding of ``(k1, i_qc, i_cc, k2, i_dc)`` with radices
``(K1, I_QC, I_CC, K2, I_DC)``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from configs.config import CandidateConfig, RoutingConfig, SimulationConfig
from core.resource_grid import CorePartition, SlotTable
from core.topology import NodeId, Topology
from env.candidates import RequestCandidates

JointAction = Tuple[int, int, int, int, int]


#: Global utilisation features: occupied fraction of the QC core, the CC core and the
#: data-core pool over the whole network. Per QC/CC route: free fraction common to all
#: its links on the QC core and on the CC core. Per DC route: mean free fraction over the
#: data cores. Without these the policy cannot tell what load regime it is in, and one
#: policy trained over a pool of loads has to act identically at 300 and 750 Erlangs.
GLOBAL_UTIL_FEATURES = 3
ROUTE_UTIL_FEATURES_K1 = 2
ROUTE_UTIL_FEATURES_K2 = 1


def state_size(routing: RoutingConfig, candidates: CandidateConfig) -> int:
    """Fixed length of the encoded state vector."""
    per_k2 = ROUTE_UTIL_FEATURES_K2 + 1 + 3 * candidates.i_dc   # free frac, F_DC, (core, start, size) * I_DC
    per_k1 = (
        ROUTE_UTIL_FEATURES_K1                                  # free frac on QC core, on CC core
        + 1 + 2 * candidates.i_qc                               # F_QC + (start, size) * I_QC
        + 1 + 2 * candidates.i_cc                               # F_CC + (start, size) * I_CC
        + routing.k2 * per_k2
    )
    return 2 + GLOBAL_UTIL_FEATURES + routing.k1 * per_k1     # o_t, d_t, utilisation, per-k1 blocks


def encode_action(action: JointAction, routing: RoutingConfig, candidates: CandidateConfig) -> int:
    k1, i_qc, i_cc, k2, i_dc = action
    idx = k1
    idx = idx * candidates.i_qc + i_qc
    idx = idx * candidates.i_cc + i_cc
    idx = idx * routing.k2 + k2
    idx = idx * candidates.i_dc + i_dc
    return idx


def decode_action(action: int, routing: RoutingConfig, candidates: CandidateConfig) -> JointAction:
    i_dc = action % candidates.i_dc
    action //= candidates.i_dc
    k2 = action % routing.k2
    action //= routing.k2
    i_cc = action % candidates.i_cc
    action //= candidates.i_cc
    i_qc = action % candidates.i_qc
    action //= candidates.i_qc
    k1 = action
    return (k1, i_qc, i_cc, k2, i_dc)


class StateEncoder:
    """Encodes a :class:`RequestCandidates` into a fixed-length state vector."""

    def __init__(self, config: SimulationConfig, topology: Topology) -> None:
        self._routing = config.routing
        self._candidates = config.candidates
        self._sentinel = config.sentinel_value
        self._node_index: Dict[NodeId, int] = {node: i for i, node in enumerate(topology.nodes)}
        self._size = state_size(config.routing, config.candidates)
        self._topology = topology
        self._partition = CorePartition.from_config(config.network)

    @property
    def size(self) -> int:
        return self._size

    def build(self, request: RequestCandidates, slot_table: SlotTable) -> np.ndarray:
        vec = np.full(self._size, self._sentinel, dtype=np.float32)
        cursor = 0
        vec[cursor] = self._node_index[request.qlr.source]
        vec[cursor + 1] = self._node_index[request.qlr.destination]
        cursor += 2

        part = self._partition
        per_core = self._topology.num_links * slot_table.fsus_per_core
        vec[cursor] = slot_table.occupied_fsus_on_core(part.core_qc) / per_core
        vec[cursor + 1] = slot_table.occupied_fsus_on_core(part.core_cc) / per_core
        vec[cursor + 2] = slot_table.occupied_fsus_on_cores(part.data_cores) / (per_core * len(part.data_cores))
        cursor += GLOBAL_UTIL_FEATURES

        for k1 in range(self._routing.k1):
            has_k1 = k1 < len(request.k1_routes)

            if has_k1:
                links = self._topology.route_link_indices(request.k1_routes[k1])
                vec[cursor] = slot_table.free_mask(links, part.core_qc).mean()
                vec[cursor + 1] = slot_table.free_mask(links, part.core_cc).mean()
            cursor += ROUTE_UTIL_FEATURES_K1

            if has_k1:
                vec[cursor] = request.f_qc[k1]
            cursor += 1
            cursor = self._write_single_blocks(
                vec, cursor, request.blocks_qc.get(k1, ()) if has_k1 else (), self._candidates.i_qc
            )

            if has_k1 and request.f_cc.get(k1) is not None:
                vec[cursor] = request.f_cc[k1]
            cursor += 1
            cursor = self._write_single_blocks(
                vec, cursor, request.blocks_cc.get(k1, ()) if has_k1 else (), self._candidates.i_cc
            )

            for k2 in range(self._routing.k2):
                has_k2 = has_k1 and k2 < len(request.eligible_dc.get(k1, ()))
                if has_k2:
                    links = self._topology.route_link_indices(request.eligible_dc[k1][k2])
                    vec[cursor] = float(np.mean([slot_table.free_mask(links, c).mean() for c in part.data_cores]))
                cursor += ROUTE_UTIL_FEATURES_K2
                if has_k2 and request.f_dc.get((k1, k2)) is not None:
                    vec[cursor] = request.f_dc[(k1, k2)]
                cursor += 1
                cursor = self._write_multi_blocks(
                    vec,
                    cursor,
                    request.blocks_dc.get((k1, k2), ()) if has_k2 else (),
                    self._candidates.i_dc,
                )

        assert cursor == self._size, (cursor, self._size)
        return vec

    def _write_single_blocks(self, vec, cursor, blocks, cap):
        for i in range(cap):
            if i < len(blocks):
                vec[cursor] = blocks[i].start
                vec[cursor + 1] = blocks[i].size
            cursor += 2
        return cursor

    def _write_multi_blocks(self, vec, cursor, blocks, cap):
        for i in range(cap):
            if i < len(blocks):
                vec[cursor] = blocks[i].core
                vec[cursor + 1] = blocks[i].start
                vec[cursor + 2] = blocks[i].size
            cursor += 3
        return cursor
