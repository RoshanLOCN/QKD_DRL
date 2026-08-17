"""Routing precomputation -- K-shortest paths, R_QKD filtering, and disjoint routes.

All routing is static and precomputed once (per Phase 3): the ``K1`` shortest QC/CC
candidate routes per node pair (Yen's algorithm [13], filtered to ``length <= R_QKD``),
and, for each surviving QC/CC route, up to ``K2`` routes link-disjoint from it (the
eligible DC routes). Disjointness depends only on the topology, so it is cached, never
searched per request.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import networkx as nx

from core.topology import NodeId, Route, Topology


def k_shortest_paths(topology: Topology, source: NodeId, target: NodeId, k: int) -> List[Route]:
    """First ``k`` loopless shortest paths by physical distance (Yen's algorithm)."""
    if source == target:
        return []
    paths: List[Route] = []
    generator = nx.shortest_simple_paths(topology.graph, source, target, weight="distance_km")
    for path in generator:
        paths.append(tuple(path))
        if len(paths) >= k:
            break
    return paths


# def find_disjoint_routes(
#     topology: Topology, source: NodeId, target: NodeId, reference: Route, k2: int
# ) -> List[Route]:
#     """Up to ``k2`` routes link-disjoint from ``reference`` (eligible DC routes).

#     DC routes are classical and are **not** subject to the ``R_QKD`` reach filter.
#     """
#     reference_edges = topology.route_edges(reference)
#     disjoint: List[Route] = []
#     generator = nx.shortest_simple_paths(topology.graph, source, target, weight="distance_km")
#     for path in generator:
#         route = tuple(path)
#         if topology.route_edges(route).isdisjoint(reference_edges):
#             disjoint.append(route)
#             if len(disjoint) >= k2:
#                 break
#     return disjoint

def find_disjoint_routes(topology, source, target, reference, k2, max_candidates_examined=2000):
    reference_edges = topology.route_edges(reference)
    disjoint = []
    
    generator = nx.shortest_simple_paths(topology.graph, source, target, weight="distance_km")
    
    # Enumerate allows us to track exactly how many paths we've checked
    for examined, path in enumerate(generator):
        if examined >= max_candidates_examined:
            break
            
        route = tuple(path)
        if topology.route_edges(route).isdisjoint(reference_edges):
            disjoint.append(route)
            if len(disjoint) >= k2:
                break
                
    return disjoint




@dataclass(frozen=True)
class RouteCaches:
    """Precomputed static routing caches keyed by node pair / (node pair, k1 route)."""

    qc_cc_routes: Dict[Tuple[NodeId, NodeId], Tuple[Route, ...]]
    disjoint_routes: Dict[Tuple[NodeId, NodeId, Route], Tuple[Route, ...]]

    def candidate_routes(self, source: NodeId, target: NodeId) -> Tuple[Route, ...]:
        return self.qc_cc_routes.get((source, target), ())

    def eligible_dc_routes(self, source: NodeId, target: NodeId, k1_route: Route) -> Tuple[Route, ...]:
        return self.disjoint_routes.get((source, target, k1_route), ())


def _ordered_pair(u: NodeId, v: NodeId) -> Tuple[NodeId, NodeId]:
    return (u, v)


def precompute_routes(topology: Topology, k1: int, k2: int, r_qkd_km: float) -> RouteCaches:
    """Build the QC/CC route cache (R_QKD-filtered) and the disjoint DC route cache.

    Caches are built for both directions of every node pair so lookups by an arbitrary
    (source, destination) order always hit.
    """
    qc_cc: Dict[Tuple[NodeId, NodeId], Tuple[Route, ...]] = {}
    disjoint: Dict[Tuple[NodeId, NodeId, Route], Tuple[Route, ...]] = {}

    for u, v in combinations(topology.nodes, 2):
        for source, target in ((u, v), (v, u)):
            candidates = [
                route
                for route in k_shortest_paths(topology, source, target, k1)
                if topology.route_length_km(route) <= r_qkd_km
            ]
            qc_cc[_ordered_pair(source, target)] = tuple(candidates)
            for k1_route in candidates:
                disjoint[(source, target, k1_route)] = tuple(
                    find_disjoint_routes(topology, source, target, k1_route, k2)
                )

    return RouteCaches(qc_cc_routes=qc_cc, disjoint_routes=disjoint)
