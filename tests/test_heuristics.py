import numpy as np
import pytest

from agents.heuristics import BestFitAgent, FirstFitAgent, RandomFitAgent
from core.blocks import MultiCoreBlock, SingleCoreBlock
from core.resource_grid import SlotTable
from core.routing import precompute_routes
from env.environment import JointRRCSAEnvironment
from request.qlr import QLR


def test_best_fit_picks_lowest_feasible_index():
    mask = np.array([False, False, True, False, True])
    assert BestFitAgent().select(np.zeros(1), mask, explore=False).action == 2


def test_random_fit_stays_within_feasible():
    mask = np.array([False, True, False, True, True])
    agent = RandomFitAgent(seed=7)
    feasible = set(np.flatnonzero(mask).tolist())
    for _ in range(20):
        assert agent.select(np.zeros(1), mask, explore=True).action in feasible


def test_empty_mask_raises():
    empty = np.zeros(4, dtype=bool)
    with pytest.raises(ValueError):
        BestFitAgent().select(np.zeros(1), empty, explore=False)
    with pytest.raises(ValueError):
        FirstFitAgent().select(np.zeros(1), empty, explore=False)


def test_first_fit_takes_lowest_start_and_first_route_where_all_fit():
    class Cands:  # minimal stand-in for RequestCandidates
        k1_routes = ("r0", "r1")
        eligible_dc = {0: ("d0", "d1"), 1: ("d0",)}

    # Route 0: QC has a lower-start *and* a closer-fit block; CC has nothing -> skip route 0.
    table = {
        (0, None, "QC"): (SingleCoreBlock(20, 2), SingleCoreBlock(4, 2)),
        (0, None, "CC"): (),
        (1, None, "QC"): (SingleCoreBlock(9, 2), SingleCoreBlock(3, 2)),
        (1, None, "CC"): (SingleCoreBlock(7, 3),),
        (1, 0, "DC"): (MultiCoreBlock(4, 0, 5), MultiCoreBlock(2, 30, 5), MultiCoreBlock(2, 12, 5)),
    }
    placement = FirstFitAgent().choose_placement(Cands(), lambda k1, k2, ch: table.get((k1, k2, ch), ()))
    k1, k2, qc, cc, dc = placement
    assert (k1, k2) == (1, 0)
    assert qc.start == 3 and cc.start == 7
    # DC: lowest core first (core 2 before core 4), then lowest start on that core.
    assert (dc.core, dc.start) == (2, 12)


def test_first_fit_blocks_when_no_route_fits():
    class Cands:
        k1_routes = ("r0",)
        eligible_dc = {0: ("d0",)}

    assert FirstFitAgent().choose_placement(Cands(), lambda k1, k2, ch: ()) is None


def test_first_fit_in_environment_starts_at_slot_zero(base_config, topology):
    caches = precompute_routes(topology, base_config.routing.k1, base_config.routing.k2, base_config.quantum.r_qkd_km)
    slot = SlotTable(topology.num_links, base_config.network.cores_per_link, base_config.network.fsus_per_core)
    env = JointRRCSAEnvironment(base_config, topology, caches, slot)
    outcome = env.provision(QLR(0, "A", "C", 0.0, 3.0, 100.0, 3.0, 1.0, 10.0, 40.0), FirstFitAgent(), False)
    assert outcome.result.served and outcome.action_taken
    assert outcome.result.block_qc.start == 0 and outcome.result.block_cc.start == 0
    assert outcome.result.block_dc.start == 0 and outcome.result.block_dc.core == env.partition.data_cores[0]
    assert len(env.registry) == 1
