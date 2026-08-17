from agents.heuristics import FirstFitAgent
from core.resource_grid import SlotTable
from core.routing import precompute_routes
from env.environment import JointRRCSAEnvironment
from env.reward import compute_reward
from request.qlr import QLR


def _env(base_config, topology, slot=None):
    caches = precompute_routes(
        topology, base_config.routing.k1, base_config.routing.k2, base_config.quantum.r_qkd_km
    )
    slot = slot or SlotTable(
        topology.num_links, base_config.network.cores_per_link, base_config.network.fsus_per_core
    )
    return JointRRCSAEnvironment(base_config, topology, caches, slot), slot


def test_provision_commits_all_three_channels(base_config, topology):
    env, slot = _env(base_config, topology)
    agent = FirstFitAgent()
    qlr = QLR(0, "A", "C", 0.0, 3.0, 100.0, 3.0, 1.0, 10.0, 40.0)
    outcome = env.provision(qlr, agent, explore=False)
    assert outcome.result.served
    assert len(env.registry) == 1
    # All three roles now hold occupied FSUs.
    assert slot.occupied_fsus_on_core(base_config.network.core_qc_index) > 0
    assert slot.occupied_fsus_on_core(base_config.network.core_cc_index) > 0
    assert slot.occupied_fsus_on_cores(env.partition.data_cores) > 0
    assert compute_reward(outcome.result, base_config.reward) > 0


def test_block_when_no_capacity(base_config, topology):
    slot = SlotTable(topology.num_links, base_config.network.cores_per_link, base_config.network.fsus_per_core)
    for link in range(topology.num_links):
        for core in range(base_config.network.cores_per_link):
            slot.occupy([link], core, 0, base_config.network.fsus_per_core)
    env, _ = _env(base_config, topology, slot)
    outcome = env.provision(QLR(1, "A", "C", 0.0, 3.0, 100.0, 3.0, 1.0, 10.0, 40.0), FirstFitAgent(), False)
    assert not outcome.result.served
    assert not outcome.action_taken
    assert len(env.registry) == 0


def test_release_frees_all_resources(base_config, topology):
    env, slot = _env(base_config, topology)
    env.provision(QLR(0, "A", "C", 0.0, 3.0, 10.0, 3.0, 1.0, 10.0, 40.0), FirstFitAgent(), False)
    env.release_and_reassign(current_time=20.0, agent=FirstFitAgent(), explore=False)
    assert len(env.registry) == 0
    assert slot.occupied_fsus_on_core(base_config.network.core_qc_index) == 0


def test_key_update_reassignment_keeps_qlr_and_advances_update(base_config, topology):
    env, _ = _env(base_config, topology)
    # departure far in the future, update due at t=3.
    env.provision(QLR(0, "A", "C", 0.0, 3.0, 1000.0, 3.0, 1.0, 10.0, 40.0), FirstFitAgent(), False)
    env.release_and_reassign(current_time=5.0, agent=FirstFitAgent(), explore=False)
    assert len(env.registry) == 1
    record = env.registry.due_for_release(2000.0)[0]
    assert record.next_update_time == 5.0 + record.qlr.key_update_period
