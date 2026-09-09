"""FF and RF obey the same physical-layer constraints as the learning agents.

The heuristics never re-implement spectrum rules -- they pick from the shared
candidate set -- so these tests verify the CONSEQUENCE: every allocation an FF or
RF run commits satisfies spectrum contiguity, spectral continuity, core continuity
and (when enabled) the XT-avoided adjacency constraint. This is the executable
form of the paper's fairness claim: the compared policies differ only in the
selection rule over an identical feasible set.
"""

from __future__ import annotations

import itertools

import pytest

from agents.heuristics import FirstFitAgent, RandomFitAgent
from core.resource_grid import CorePartition, SlotTable
from core.routing import precompute_routes
from core.topology import load_topology
from env.environment import JointRRCSAEnvironment
from request.qlr import QLR


def _env(base_config, topology):
    caches = precompute_routes(
        topology, base_config.routing.k1, base_config.routing.k2, base_config.quantum.r_qkd_km
    )
    slot = SlotTable(
        topology.num_links, base_config.network.cores_per_link, base_config.network.fsus_per_core
    )
    return JointRRCSAEnvironment(base_config, topology, caches, slot), slot


def _provision_stream(env, agent, topology, n=12):
    """Provision a stream of QLRs (some will block once the grid fills -- fine)."""
    pairs = itertools.cycle([("A", "C"), ("B", "E"), ("A", "D"), ("C", "E"), ("B", "D")])
    served = 0
    for i in range(n):
        src, dst = next(pairs)
        qlr = QLR(i, src, dst, 0.0, 3.0, 1000.0, 3.0, 1.0, 10.0, 40.0)
        outcome = env.provision(qlr, agent, explore=False)
        served += int(outcome.result.served)
    return served


def _placements(env, topology, record):
    """The three (link_indices, core, block) placements of one active record."""
    k1_links = topology.route_link_indices(record.k1_route)
    k2_links = topology.route_link_indices(record.k2_route)
    return (
        (k1_links, env.partition.core_qc, record.block_qc),
        (k1_links, env.partition.core_cc, record.block_cc),
        (k2_links, record.block_dc.core, record.block_dc),
    )


def _assert_constraints(env, slot, topology):
    records = list(env.registry.due_for_release(float("inf")))
    assert records, "test needs at least one served request"

    for record in records:
        for links, core, block in _placements(env, topology, record):
            # Contiguity: the block is one consecutive FSU range of positive size.
            assert block.size >= 1
            start, end = block.start, block.start + block.size
            # Continuity + core continuity: that same range is occupied on the SAME
            # core on EVERY link of the route (per-link check, not just the AND).
            for link in links:
                free = slot.free_mask([link], core)
                assert not free[start:end].any(), (
                    f"request {record.qlr.request_id}: slots [{start},{end}) not "
                    f"occupied on link {link} core {core}"
                )

    # XT-avoided invariant: two allocations on physically adjacent cores sharing a
    # link never overlap in spectrum.
    adjacency = env.partition.adjacency
    if adjacency:
        flat = [
            (record.qlr.request_id, links, core, block)
            for record in records
            for links, core, block in _placements(env, topology, record)
        ]
        for (id_a, links_a, core_a, blk_a), (id_b, links_b, core_b, blk_b) in (
            itertools.combinations(flat, 2)
        ):
            if id_a == id_b:
                continue
            if core_b not in adjacency.get(core_a, ()):
                continue
            if not set(links_a) & set(links_b):
                continue
            overlap = min(blk_a.start + blk_a.size, blk_b.start + blk_b.size) - max(
                blk_a.start, blk_b.start
            )
            assert overlap <= 0, (
                f"XT violation: requests {id_a}/{id_b} overlap on adjacent cores "
                f"{core_a}/{core_b}"
            )


@pytest.mark.parametrize("make_agent", [lambda: FirstFitAgent(), lambda: RandomFitAgent(7)])
def test_heuristic_allocations_satisfy_physical_constraints(base_config, make_agent):
    topology = load_topology(base_config.topology_path)
    env, slot = _env(base_config, topology)
    served = _provision_stream(env, make_agent(), topology)
    assert served > 0
    _assert_constraints(env, slot, topology)
