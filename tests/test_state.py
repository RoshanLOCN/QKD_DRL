import numpy as np

from env.state import StateEncoder, decode_action, encode_action, state_size


def test_encode_decode_roundtrip(base_config):
    routing, candidates = base_config.routing, base_config.candidates
    size = (
        routing.k1 * candidates.i_qc * candidates.i_cc * routing.k2 * candidates.i_dc
    )
    for a in range(size):
        assert encode_action(decode_action(a, routing, candidates), routing, candidates) == a


def test_decode_component_ranges(base_config):
    routing, candidates = base_config.routing, base_config.candidates
    k1, i_qc, i_cc, k2, i_dc = decode_action(0, routing, candidates)
    assert (k1, i_qc, i_cc, k2, i_dc) == (0, 0, 0, 0, 0)
    last = routing.k1 * candidates.i_qc * candidates.i_cc * routing.k2 * candidates.i_dc - 1
    k1, i_qc, i_cc, k2, i_dc = decode_action(last, routing, candidates)
    assert k1 == routing.k1 - 1 and i_dc == candidates.i_dc - 1


def test_state_size_matches_encoder(base_config, topology):
    encoder = StateEncoder(base_config, topology)
    assert encoder.size == state_size(base_config.routing, base_config.candidates)


def test_state_vector_length_is_fixed(base_config, topology):
    from core.routing import precompute_routes
    from core.resource_grid import SlotTable
    from env.environment import JointRRCSAEnvironment
    from request.qlr import QLR

    caches = precompute_routes(topology, base_config.routing.k1, base_config.routing.k2, base_config.quantum.r_qkd_km)
    slot = SlotTable(topology.num_links, base_config.network.cores_per_link, base_config.network.fsus_per_core)
    env = JointRRCSAEnvironment(base_config, topology, caches, slot)
    encoder = StateEncoder(base_config, topology)

    qlr = QLR(0, "A", "C", 0.0, 3.0, 5.0, 3.0, 1.0, 10.0, 40.0)
    vec = encoder.build(env.build_candidates(qlr))
    assert vec.shape == (encoder.size,)
    assert vec.dtype == np.float32
