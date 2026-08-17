
"""Topology module -- the static SS-EON graph loaded from an external file.

No topology data is embedded in the code. The Tokyo12 graph (or any other) is read
from a JSON file with the following shape::

    {
      "name": "Tokyo12",
      "nodes": ["n0", "n1", ...],
      "links": [
        {"source": "n0", "target": "n1", "distance_km": 50.0},
        ...
      ]
    }

Links are undirected. A ``Route`` is an ordered tuple of node ids; its physical
length is the sum of its links' ``distance_km`` and its hop count is the number of
links it traverses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, FrozenSet, Hashable, List, Mapping, Sequence, Tuple

import networkx as nx

NodeId = Hashable
Route = Tuple[NodeId, ...]


@dataclass(frozen=True)
class Topology:
    """Static network graph with a stable link indexing for the slot table."""

    name: str
    graph: nx.Graph
    _link_index: Mapping[FrozenSet[NodeId], int]

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_links(self) -> int:
        return self.graph.number_of_edges()

    @property
    def nodes(self) -> Tuple[NodeId, ...]:
        return tuple(self.graph.nodes)

    def link_index(self, u: NodeId, v: NodeId) -> int:
        """Stable index of the undirected link ``{u, v}`` for slot-table addressing."""
        return self._link_index[frozenset((u, v))]

    def route_link_indices(self, route: Route) -> Tuple[int, ...]:
        return tuple(self.link_index(route[i], route[i + 1]) for i in range(len(route) - 1))

    def route_length_km(self, route: Route) -> float:
        return sum(
            self.graph[route[i]][route[i + 1]]["distance_km"] for i in range(len(route) - 1)
        )

    @staticmethod
    def hop_count(route: Route) -> int:
        return len(route) - 1

    def route_edges(self, route: Route) -> FrozenSet[FrozenSet[NodeId]]:
        return frozenset(
            frozenset((route[i], route[i + 1])) for i in range(len(route) - 1)
        )

    @staticmethod
    def inter_core_adjacency(num_cores) -> List[List[int]]:
        """Returns a num_cores x num_cores 0/1 adjacency matrix for crosstalk tracking."""
        adj = [[0] * num_cores for _ in range(num_cores)]

        if num_cores == 5:
            neighbours = {
                0: [1, 3, 4],
                1: [0, 2, 4],
                2: [1, 3, 4],
                3: [0, 2, 4],
                4: [0, 1, 2, 3],
            }
            for c, nbrs in neighbours.items():
                for n in nbrs:
                    adj[c][n] = 1
        elif num_cores == 4:
            # 4-core adjacency: each core is connected to its immediate neighbours.
            for c in range(num_cores - 1):
                adj[c][c + 1] = 1
                adj[c + 1][c] = 1
        elif num_cores == 7:
            # 7-core adjacency: each core is connected to its immediate neighbours.
            neighbours = {
                0: [1, 5, 6],
                1: [0, 2, 6],
                2: [1, 3, 6],
                3: [2, 4, 6],
                4: [3, 5, 6],
                5: [0, 4, 6],
                6: [0, 1, 2, 3, 4, 5],
            }
            for c, nbrs in neighbours.items():
                for n in nbrs:
                    adj[c][n] = 1
        else:
            # Fallback: linear nearest-neighbour adjacency.
            for c in range(num_cores - 1):
                adj[c][c + 1] = 1
                adj[c + 1][c] = 1

        return adj


def build_topology(name: str, nodes: Sequence[NodeId], links: Sequence[Mapping]) -> Topology:
    """Construct a validated :class:`Topology` from primitive node/link data."""
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    node_set = set(nodes)
    if len(node_set) != len(nodes):
        raise ValueError("duplicate node ids in topology")

    link_index: Dict[FrozenSet[NodeId], int] = {}
    for link in links:
        u, v, distance = link["source"], link["target"], link["distance_km"]
        if u not in node_set or v not in node_set:
            raise ValueError(f"link references unknown node(s): {u!r}, {v!r}")
        if u == v:
            raise ValueError(f"self-loop link on node {u!r} is not allowed")
        if distance <= 0:
            raise ValueError(f"link {u!r}-{v!r} must have distance_km > 0")
        key = frozenset((u, v))
        if key in link_index:
            raise ValueError(f"duplicate link {u!r}-{v!r}")
        link_index[key] = len(link_index)
        graph.add_edge(u, v, distance_km=float(distance))

    if graph.number_of_edges() == 0:
        raise ValueError("topology must contain at least one link")

    return Topology(name=name, graph=graph, _link_index=link_index)


def load_topology(path: str) -> Topology:
    """Load a :class:`Topology` from a JSON file (see module docstring for the schema)."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("nodes", "links"):
        if key not in data:
            raise ValueError(f"topology file {path!r} is missing required key {key!r}")
    return build_topology(
        name=data.get("name", path),
        nodes=data["nodes"],
        links=data["links"],
    )

