from core.routing import (
    find_disjoint_routes,
    k_shortest_paths,
    precompute_routes,
)


def test_k_shortest_paths_ordered_by_distance(topology):
    paths = k_shortest_paths(topology, "A", "C", 3)
    lengths = [topology.route_length_km(p) for p in paths]
    assert lengths == sorted(lengths)
    assert ("A", "B", "C") in paths


def test_disjoint_routes_share_no_link(topology):
    reference = ("A", "B", "C")
    disjoint = find_disjoint_routes(topology, "A", "C", reference, k2=3)
    ref_edges = topology.route_edges(reference)
    assert disjoint
    assert all(topology.route_edges(r).isdisjoint(ref_edges) for r in disjoint)


def test_precompute_filters_by_r_qkd(topology):
    caches = precompute_routes(topology, k1=5, k2=3, r_qkd_km=100.0)
    for routes in caches.qc_cc_routes.values():
        assert all(topology.route_length_km(r) <= 100.0 for r in routes)


def test_precompute_both_directions(topology):
    caches = precompute_routes(topology, k1=3, k2=3, r_qkd_km=1000.0)
    assert caches.candidate_routes("A", "C")
    assert caches.candidate_routes("C", "A")
