"""Joint RRCSA environment -- Algorithm 1 orchestration and release/reassignment.

Binds together candidate construction, state/mask building, agent action selection, and
the atomic all-or-nothing commit, and implements the periodic release + key-update
reassignment that runs before each new arrival (Step 8.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from agents.base import ActionSelection, Agent
from configs.config import SimulationConfig
from core.modulation import ModulationTable
from core.resource_grid import CorePartition, SlotTable
from core.routing import RouteCaches
from core.topology import Topology
from env.candidates import RequestCandidates, build_request_candidates
from env.mask import build_joint_action_mask
from env.registry import ActiveQLRRegistry, ActiveRecord
from env.results import CommitResult
from env.state import StateEncoder, decode_action
from request.qlr import QLR


@dataclass(frozen=True)
class ProvisionOutcome:
    """Result of attempting one QLR, plus what the learner needs to store experience."""

    result: CommitResult
    action_taken: bool
    state: Optional[np.ndarray] = None
    selection: Optional[ActionSelection] = None
    mask: Optional[np.ndarray] = None


class JointRRCSAEnvironment:
    def __init__(
        self,
        config: SimulationConfig,
        topology: Topology,
        caches: RouteCaches,
        slot_table: SlotTable,
    ) -> None:
        self._config = config
        self._topology = topology
        self._caches = caches
        self._slot_table = slot_table
        self._partition = CorePartition.from_config(config.network)
        self._modulation = ModulationTable(config.modulation, config.network.fsu_width_ghz)
        self._encoder = StateEncoder(config, topology)
        self._registry = ActiveQLRRegistry()

    @property
    def slot_table(self) -> SlotTable:
        return self._slot_table

    @property
    def registry(self) -> ActiveQLRRegistry:
        return self._registry

    @property
    def partition(self) -> CorePartition:
        return self._partition

    @property
    def state_size(self) -> int:
        return self._encoder.size

    def reset(self) -> None:
        self._slot_table.reset()
        self._registry.clear()

    def build_candidates(self, qlr: QLR) -> RequestCandidates:
        return build_request_candidates(
            qlr,
            self._topology,
            self._caches,
            self._partition,
            self._slot_table,
            self._modulation,
            self._config.quantum,
            self._config.routing,
            self._config.candidates,
        )

    def provision(self, qlr: QLR, agent: Agent, explore: bool) -> ProvisionOutcome:
        """Algorithm 1: build state+mask, select, atomically commit (fresh arrival)."""
        return self._attempt(
            qlr,
            agent,
            explore,
            release_time=qlr.departure_time,
            next_update_time=qlr.update_time,
        )

    def release_and_reassign(self, current_time: float, agent: Agent, explore: bool) -> None:
        """Step 8.1: free expired QLRs, then re-solve placement for those due for update."""
        for record in self._registry.due_for_release(current_time):
            self._free_record(record)
            self._registry.remove(record.qlr.request_id)

        for record in self._registry.due_for_update(current_time):
            self._free_record(record)
            self._registry.remove(record.qlr.request_id)
            next_update = current_time + record.qlr.key_update_period
            # Reassignment re-runs the full joint placement (Steps 8.2-8.9) with the
            # request's original demands. It is a refresh, not a new arrival, so -- to
            # keep the pseudocode's "one experience per arrival" density -- it neither
            # stores experience nor counts toward blocking metrics. If no feasible
            # placement exists at refresh time, the QLR cannot be refreshed and is
            # dropped (its resources are already freed); the documents do not fix this
            # edge case, so the most consistent choice is made and flagged here.
            self._attempt(
                record.qlr,
                agent,
                explore,
                release_time=record.release_time,
                next_update_time=next_update,
            )

    def _attempt(
        self,
        qlr: QLR,
        agent: Agent,
        explore: bool,
        release_time: float,
        next_update_time: float,
    ) -> ProvisionOutcome:
        candidates = self.build_candidates(qlr)
        mask = build_joint_action_mask(candidates, self._config.routing, self._config.candidates)
        # Built unconditionally (not just on the feasible path) so a blocked arrival still
        # carries a state: the learner needs it to value-bootstrap through the blocking
        # event, even though there is no action to attribute a policy gradient to.
        state = self._encoder.build(candidates)

        if not mask.any():
            return ProvisionOutcome(result=CommitResult.blocked(qlr.request_id), action_taken=False, state=state)

        selection = agent.select(state, mask, explore)
        action_tuple = decode_action(selection.action, self._config.routing, self._config.candidates)
        result = self._commit(qlr, candidates, action_tuple, release_time, next_update_time)
        return ProvisionOutcome(
            result=result,
            action_taken=True,
            state=state,
            selection=selection,
            mask=mask,
        )

    def _commit(
        self,
        qlr: QLR,
        candidates: RequestCandidates,
        action_tuple,
        release_time: float,
        next_update_time: float,
    ) -> CommitResult:
        k1, i_qc, i_cc, k2, i_dc = action_tuple
        blocks_qc = candidates.blocks_qc.get(k1, ())
        blocks_cc = candidates.blocks_cc.get(k1, ())
        blocks_dc = candidates.blocks_dc.get((k1, k2), ())
        if i_qc >= len(blocks_qc) or i_cc >= len(blocks_cc) or i_dc >= len(blocks_dc):
            return CommitResult.blocked(qlr.request_id)

        k1_route = candidates.k1_routes[k1]
        k2_route = candidates.eligible_dc[k1][k2]
        block_qc = blocks_qc[i_qc]
        block_cc = blocks_cc[i_cc]
        block_dc = blocks_dc[i_dc]

        k1_links = self._topology.route_link_indices(k1_route)
        k2_links = self._topology.route_link_indices(k2_route)

        # Safety re-check (Algorithm 1, line 20): confirm all three are still free,
        # including XT-avoided adjacency (a slot must also still be free on every
        # core physically adjacent to the one being committed).
        qc_ok = self._slot_table.is_block_free(
            k1_links, self._partition.core_qc, block_qc.start, block_qc.size
        ) and self._adjacent_free(k1_links, self._partition.core_qc, block_qc.start, block_qc.size)
        cc_ok = self._slot_table.is_block_free(
            k1_links, self._partition.core_cc, block_cc.start, block_cc.size
        ) and self._adjacent_free(k1_links, self._partition.core_cc, block_cc.start, block_cc.size)
        dc_ok = self._slot_table.is_block_free(
            k2_links, block_dc.core, block_dc.start, block_dc.size
        ) and self._adjacent_free(k2_links, block_dc.core, block_dc.start, block_dc.size)
        if not (qc_ok and cc_ok and dc_ok):
            return CommitResult.blocked(qlr.request_id)

        # Atomic commit: all three or none (no partial-commit/rollback).
        self._slot_table.occupy(k1_links, self._partition.core_qc, block_qc.start, block_qc.size)
        self._slot_table.occupy(k1_links, self._partition.core_cc, block_cc.start, block_cc.size)
        self._slot_table.occupy(k2_links, block_dc.core, block_dc.start, block_dc.size)

        self._registry.replace_record(
            ActiveRecord(
                qlr=qlr,
                k1_route=k1_route,
                k2_route=k2_route,
                block_qc=block_qc,
                block_cc=block_cc,
                block_dc=block_dc,
                release_time=release_time,
                next_update_time=next_update_time,
            )
        )

        return CommitResult(
            served=True,
            request_id=qlr.request_id,
            k1_route=k1_route,
            k2_route=k2_route,
            block_qc=block_qc,
            block_cc=block_cc,
            block_dc=block_dc,
            hops_qc_cc=self._topology.hop_count(k1_route),
            hops_dc=self._topology.hop_count(k2_route),
            eta_cc=self._modulation.efficiency(self._topology.route_length_km(k1_route)),
            eta_dc=self._modulation.efficiency(self._topology.route_length_km(k2_route)),
        )

    def _adjacent_free(self, link_indices, core: int, start: int, size: int) -> bool:
        """XT-avoided check: is ``start:start+size`` also free on every core adjacent
        to ``core``? Empty adjacency (xt_avoided off) trivially passes."""
        return all(
            self._slot_table.is_block_free(link_indices, adjacent, start, size)
            for adjacent in self._partition.adjacency.get(core, ())
        )

    def _free_record(self, record: ActiveRecord) -> None:
        k1_links = self._topology.route_link_indices(record.k1_route)
        k2_links = self._topology.route_link_indices(record.k2_route)
        self._slot_table.release(k1_links, self._partition.core_qc, record.block_qc.start, record.block_qc.size)
        self._slot_table.release(k1_links, self._partition.core_cc, record.block_cc.start, record.block_cc.size)
        self._slot_table.release(k2_links, record.block_dc.core, record.block_dc.start, record.block_dc.size)
