"""Extended Eq. 12 reward: term definitions, ranges, and the original-form ablation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from agents.heuristics import FirstFitAgent
from configs.config import RewardConfig
from core.blocks import free_run_length, xt_new_footprint
from core.resource_grid import SlotTable
from core.routing import precompute_routes
from env.environment import JointRRCSAEnvironment
from env.results import CommitResult
from env.reward import compute_reward
from request.qlr import QLR


def _served(**overrides) -> CommitResult:
    base = dict(
        served=True, request_id=0, hops_qc_cc=2, hops_dc=3, eta_cc=2.0, eta_dc=4.0,
        hops_min_qc_cc=1, hops_min_dc=3, eta_max=4.0, fit=0.5, xt=0.25, compactness=0.8,
    )
    base.update(overrides)
    return CommitResult(**base)


def _config(**overrides) -> RewardConfig:
    base = dict(
        served_base=1.0, beta_qc_cc_hops=0.2, beta_dc_hops=0.2, beta_efficiency=0.2,
        relative_terms=True, beta_fit=0.2, beta_xt=0.2, beta_compact=0.05,
    )
    base.update(overrides)
    return RewardConfig(**base)


def test_original_eq12_is_reproduced_by_ablation_config():
    cfg = _config(relative_terms=False, beta_efficiency=0.05, beta_fit=0.0, beta_xt=0.0, beta_compact=0.0)
    r = compute_reward(_served(), cfg)
    assert r == pytest.approx(1.0 + 0.2 / 2 + 0.2 / 3 + 0.05 * (2.0 + 4.0))


def test_extended_terms_are_decision_relative():
    r = compute_reward(_served(), _config())
    expected = (
        1.0
        + 0.2 * (1 / 2)            # chose a 2-hop route when a 1-hop candidate existed
        + 0.2 * (3 / 3)            # shortest DC candidate chosen
        + 0.2 * (2.0 + 4.0) / 8.0  # (eta_cc + eta_dc) / (2 eta_max)
        + 0.2 * 0.5 - 0.2 * 0.25 + 0.05 * 0.8
    )
    assert r == pytest.approx(expected)
    # The same request served with the shortest route and a perfect fit earns strictly more.
    better = compute_reward(_served(hops_qc_cc=1, fit=1.0, xt=0.0, compactness=1.0), _config())
    assert better > r


def test_blocked_reward_and_serving_dominates():
    assert compute_reward(CommitResult.blocked(7), _config()) == -1.0
    worst_served = compute_reward(
        _served(hops_qc_cc=10, hops_dc=10, hops_min_qc_cc=1, hops_min_dc=1, eta_cc=1.0, eta_dc=1.0,
                fit=0.01, xt=1.0, compactness=0.0),
        _config(),
    )
    assert worst_served > 0.0 > -1.0


def test_placement_betas_require_placement_fields():
    with pytest.raises(ValueError):
        compute_reward(_served(fit=None), _config())
    assert compute_reward(_served(fit=None, xt=None, compactness=None),
                          _config(beta_fit=0.0, beta_xt=0.0, beta_compact=0.0)) > 0


def test_free_run_length_and_xt_footprint_on_ring_cores():
    # 1 link, 4 cores in a ring (0-1-2-3-0), 10 FSUs.
    adjacency = {0: (1, 3), 1: (0, 2), 2: (1, 3), 3: (2, 0)}
    slot = SlotTable(num_links=1, cores_per_link=4, fsus_per_core=10)
    slot.occupy([0], 0, 6, 2)                      # core 0 busy at slots 6-7
    assert free_run_length([0], 0, 0, slot, adjacency) == 6   # run [0,6)
    assert free_run_length([0], 0, 8, slot, adjacency) == 2   # run [8,10)
    assert free_run_length([0], 0, 6, slot, adjacency) == 0   # occupied
    # XT: core 1 slots 6-7 are now forbidden (neighbour 0 occupied) -> run on core 1 splits.
    assert free_run_length([0], 1, 0, slot, adjacency) == 6

    # Footprint of placing 2 slots at [0,2) on core 1: neighbours 0 and 2, both usable there.
    assert xt_new_footprint([0], 1, 0, 2, slot, adjacency) == 4
    # Placing at [6,8) on core 1 is impossible (forbidden), but at [8,10): neighbour 0 is
    # usable at 8-9 (its own neighbours 1 and 3 are free) and neighbour 2 too -> 4.
    assert xt_new_footprint([0], 1, 8, 2, slot, adjacency) == 4
    # Aligning with existing occupancy costs less: occupy core 2 at [8,10); then placing
    # core 1 at [8,10) only newly sterilises core 0 (core 2 is already occupied).
    slot.occupy([0], 2, 8, 2)
    assert xt_new_footprint([0], 1, 8, 2, slot, adjacency) == 2
    assert xt_new_footprint([0], 1, 8, 2, slot, adjacency=None) == 0


def test_environment_populates_placement_terms(base_config, topology):
    caches = precompute_routes(topology, base_config.routing.k1, base_config.routing.k2, base_config.quantum.r_qkd_km)
    slot = SlotTable(topology.num_links, base_config.network.cores_per_link, base_config.network.fsus_per_core)
    env = JointRRCSAEnvironment(base_config, topology, caches, slot)
    outcome = env.provision(QLR(0, "A", "C", 0.0, 3.0, 100.0, 3.0, 1.0, 10.0, 40.0), FirstFitAgent(), False)
    res = outcome.result
    assert res.served
    assert 0.0 < res.fit <= 1.0 and 0.0 <= res.xt <= 1.0 and 0.0 <= res.compactness <= 1.0
    assert res.hops_min_qc_cc <= res.hops_qc_cc and res.hops_min_dc <= res.hops_dc
    assert res.eta_max == 4.0
    # Empty network, first-fit: every block starts at slot 0 -> maximal compactness.
    assert res.compactness == 1.0
    assert compute_reward(res, base_config.reward) > 1.0
