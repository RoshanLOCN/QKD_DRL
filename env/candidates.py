"""Per-request candidate structure shared by state, mask, and commit.

For one QLR this assembles everything Steps 8.3-8.5 (Algorithm 1 lines 1-11) produce:
the R_QKD-filtered QC/CC routes, their eligible link-disjoint DC routes, the per-channel
FSU requirements, and the ranked closest-fit candidate blocks per channel. Building it
once and passing it to the state builder, the mask builder, and the commit step keeps
the three views perfectly consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from configs.config import CandidateConfig, QuantumConfig, RoutingConfig
from core.blocks import (
    MultiCoreBlock,
    SingleCoreBlock,
    feasible_blocks_multi_core,
    feasible_blocks_single_core,
)
from core.modulation import ModulationTable
from core.resource_grid import CorePartition, SlotTable
from core.routing import RouteCaches
from core.topology import Route, Topology
from request.qlr import QLR


@dataclass(frozen=True)
class RequestCandidates:
    """Everything needed to build the state, mask, and commit for one QLR."""

    qlr: QLR
    k1_routes: Tuple[Route, ...]
    eligible_dc: Dict[int, Tuple[Route, ...]]
    f_qc: Dict[int, int]
    f_cc: Dict[int, Optional[int]]
    f_dc: Dict[Tuple[int, int], Optional[int]]
    blocks_qc: Dict[int, Tuple[SingleCoreBlock, ...]]
    blocks_cc: Dict[int, Tuple[SingleCoreBlock, ...]]
    blocks_dc: Dict[Tuple[int, int], Tuple[MultiCoreBlock, ...]]


def build_request_candidates(
    qlr: QLR,
    topology: Topology,
    caches: RouteCaches,
    partition: CorePartition,
    slot_table: SlotTable,
    modulation: ModulationTable,
    quantum: QuantumConfig,
    routing: RoutingConfig,
    candidates: CandidateConfig,
) -> RequestCandidates:
    """Assemble the candidate structure for one QLR (Algorithm 1, lines 1-11)."""
    k1_routes = caches.candidate_routes(qlr.source, qlr.destination)[: routing.k1]

    eligible_dc: Dict[int, Tuple[Route, ...]] = {}
    f_qc: Dict[int, int] = {}
    f_cc: Dict[int, Optional[int]] = {}
    f_dc: Dict[Tuple[int, int], Optional[int]] = {}
    blocks_qc: Dict[int, Tuple[SingleCoreBlock, ...]] = {}
    blocks_cc: Dict[int, Tuple[SingleCoreBlock, ...]] = {}
    blocks_dc: Dict[Tuple[int, int], Tuple[MultiCoreBlock, ...]] = {}

    for k1_idx, k1_route in enumerate(k1_routes):
        k1_links = topology.route_link_indices(k1_route)

        f_qc[k1_idx] = quantum.f_qc_min
        blocks_qc[k1_idx] = feasible_blocks_single_core(
            k1_links, partition.core_qc, quantum.f_qc_min, slot_table, candidates.i_qc
        )

        cc_fs = modulation.required_fs(topology.route_length_km(k1_route), qlr.b_cc)
        f_cc[k1_idx] = cc_fs
        blocks_cc[k1_idx] = (
            feasible_blocks_single_core(
                k1_links, partition.core_cc, cc_fs, slot_table, candidates.i_cc
            )
            if cc_fs is not None
            else ()
        )

        dc_routes = caches.eligible_dc_routes(qlr.source, qlr.destination, k1_route)[: routing.k2]
        eligible_dc[k1_idx] = dc_routes
        for k2_idx, k2_route in enumerate(dc_routes):
            k2_links = topology.route_link_indices(k2_route)
            dc_fs = modulation.required_fs(topology.route_length_km(k2_route), qlr.b_dc)
            f_dc[(k1_idx, k2_idx)] = dc_fs
            blocks_dc[(k1_idx, k2_idx)] = (
                feasible_blocks_multi_core(
                    k2_links, partition.data_cores, dc_fs, slot_table, candidates.i_dc
                )
                if dc_fs is not None
                else ()
            )

    return RequestCandidates(
        qlr=qlr,
        k1_routes=k1_routes,
        eligible_dc=eligible_dc,
        f_qc=f_qc,
        f_cc=f_cc,
        f_dc=f_dc,
        blocks_qc=blocks_qc,
        blocks_cc=blocks_cc,
        blocks_dc=blocks_dc,
    )
