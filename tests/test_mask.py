import numpy as np

from core.resource_grid import SlotTable
from core.routing import precompute_routes
from env.environment import JointRRCSAEnvironment
from env.mask import build_joint_action_mask
from env.state import decode_action
from request.qlr import QLR


def _env(base_config, topology, slot=None):
    caches = precompute_routes(
        topology, base_config.routing.k1, base_config.routing.k2, base_config.quantum.r_qkd_km
    )
    slot = slot or SlotTable(
        topology.num_links, base_config.network.cores_per_link, base_config.network.fsus_per_core
    )
    return JointRRCSAEnvironment(base_config, topology, caches, slot), slot


def test_mask_true_only_for_available_blocks(base_config, topology):
    env, _ = _env(base_config, topology)
    qlr = QLR(0, "A", "C", 0.0, 3.0, 5.0, 3.0, 1.0, 10.0, 40.0)
    cand = env.build_candidates(qlr)
    mask = build_joint_action_mask(cand, base_config.routing, base_config.candidates)

    for idx in np.flatnonzero(mask):
        k1, i_qc, i_cc, k2, i_dc = decode_action(int(idx), base_config.routing, base_config.candidates)
        assert i_qc < len(cand.blocks_qc[k1])
        assert i_cc < len(cand.blocks_cc[k1])
        assert i_dc < len(cand.blocks_dc[(k1, k2)])


def test_mask_empty_when_cc_core_full(base_config, topology):
    slot = SlotTable(topology.num_links, base_config.network.cores_per_link, base_config.network.fsus_per_core)
    # Fill the dedicated CC core on every link: no CC block can ever be placed.
    for link in range(topology.num_links):
        slot.occupy([link], base_config.network.core_cc_index, 0, base_config.network.fsus_per_core)
    env, _ = _env(base_config, topology, slot)
    qlr = QLR(0, "A", "C", 0.0, 3.0, 5.0, 3.0, 1.0, 10.0, 40.0)
    cand = env.build_candidates(qlr)
    mask = build_joint_action_mask(cand, base_config.routing, base_config.candidates)
    assert not mask.any()
