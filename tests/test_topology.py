import pytest

from core.topology import build_topology, load_topology


def test_load_topology_and_geometry(topology):
    assert topology.num_nodes == 5
    assert topology.num_links == 6
    route = ("A", "B", "C")
    assert topology.route_length_km(route) == 100.0
    assert topology.hop_count(route) == 2
    assert len(topology.route_link_indices(route)) == 2


def test_link_index_is_symmetric(topology):
    assert topology.link_index("A", "B") == topology.link_index("B", "A")


def test_route_edges_disjointness(topology):
    top = topology.route_edges(("A", "B", "C"))
    bottom = topology.route_edges(("A", "D", "C"))
    assert top.isdisjoint(bottom)


def test_build_topology_rejects_unknown_node():
    with pytest.raises(ValueError):
        build_topology("x", ["A", "B"], [{"source": "A", "target": "Z", "distance_km": 1.0}])


def test_build_topology_rejects_nonpositive_distance():
    with pytest.raises(ValueError):
        build_topology("x", ["A", "B"], [{"source": "A", "target": "B", "distance_km": 0.0}])
