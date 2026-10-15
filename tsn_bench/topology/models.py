"""Core topology data model shared by every generator, solver, and serializer.

Kept deliberately simple (plain dataclasses + a NetworkX conversion) so every
method under evaluation — CP-SAT baselines today, GNN/SLM methods later —
consumes the exact same representation of "what is the network."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import networkx as nx


class NodeType(str, Enum):
    TALKER = "talker"
    LISTENER = "listener"
    SWITCH = "switch"


@dataclass
class Node:
    node_id: str
    node_type: NodeType
    # switches have multiple queues (priority levels) per egress port; end
    # stations are modeled as single-port devices.
    num_queues: int = 8
    added_at_step: int = 0  # 0 = present in the original topology; >0 = added later

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "num_queues": self.num_queues,
            "added_at_step": self.added_at_step,
        }

    @staticmethod
    def from_dict(d: dict) -> "Node":
        return Node(
            node_id=d["node_id"],
            node_type=NodeType(d["node_type"]),
            num_queues=d.get("num_queues", 8),
            added_at_step=d.get("added_at_step", 0),
        )


@dataclass
class Link:
    link_id: str
    src: str
    dst: str
    rate_mbps: float = 1000.0
    propagation_delay_us: float = 1.0

    def to_dict(self) -> dict:
        return {
            "link_id": self.link_id,
            "src": self.src,
            "dst": self.dst,
            "rate_mbps": self.rate_mbps,
            "propagation_delay_us": self.propagation_delay_us,
        }

    @staticmethod
    def from_dict(d: dict) -> "Link":
        return Link(
            link_id=d["link_id"],
            src=d["src"],
            dst=d["dst"],
            rate_mbps=d.get("rate_mbps", 1000.0),
            propagation_delay_us=d.get("propagation_delay_us", 1.0),
        )


@dataclass
class Topology:
    topology_id: str
    nodes: dict[str, Node] = field(default_factory=dict)
    links: dict[str, Link] = field(default_factory=dict)
    # generator metadata, useful for stratifying dataset splits later
    switch_count: int = 0
    generator_family: str = "craciunas_config3"

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node

    def add_link(self, link: Link) -> None:
        self.links[link.link_id] = link
        # links are bidirectional at the physical layer (full-duplex);
        # add the reverse edge with its own link id so egress-port
        # scheduling constraints can be attached per direction.
        rev_id = f"{link.link_id}__rev"
        if rev_id not in self.links and f"{link.src}->{link.dst}" != f"{link.dst}->{link.src}":
            self.links[rev_id] = Link(
                link_id=rev_id,
                src=link.dst,
                dst=link.src,
                rate_mbps=link.rate_mbps,
                propagation_delay_us=link.propagation_delay_us,
            )

    def switches(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.SWITCH]

    def talkers(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.TALKER]

    def listeners(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.LISTENER]

    def to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for n in self.nodes.values():
            g.add_node(n.node_id, node_type=n.node_type.value, num_queues=n.num_queues,
                       added_at_step=n.added_at_step)
        for l in self.links.values():
            g.add_edge(l.src, l.dst, link_id=l.link_id, rate_mbps=l.rate_mbps,
                       propagation_delay_us=l.propagation_delay_us)
        return g

    def shortest_path(self, src: str, dst: str) -> list[str]:
        g = self.to_networkx()
        return nx.shortest_path(g, src, dst)

    def link_between(self, a: str, b: str) -> Optional[Link]:
        for l in self.links.values():
            if l.src == a and l.dst == b:
                return l
        return None

    def _node_subview(self, keep_ids: set[str], topology_id: str) -> "Topology":
        """Shared filtering logic behind `pre_addition_subview` and
        `k_hop_subview`: a new Topology containing only `keep_ids` and the
        links between them."""
        sub = Topology(topology_id=topology_id,
                        switch_count=len([n for n in keep_ids if self.nodes[n].node_type == NodeType.SWITCH]),
                        generator_family=self.generator_family)
        for nid in keep_ids:
            sub.nodes[nid] = self.nodes[nid]
        for lid, link in self.links.items():
            if link.src in keep_ids and link.dst in keep_ids:
                sub.links[lid] = link
        return sub

    def pre_addition_subview(self, max_step: int = 0) -> "Topology":
        """Return a new Topology with only nodes added at `max_step` or
        earlier (default 0 = the original topology before any node
        additions) and the links between them. Used to reconstruct "the
        network as it was before this node addition" for incremental
        baselines, without needing to persist a separate before-snapshot."""
        keep_ids = {nid for nid, n in self.nodes.items() if n.added_at_step <= max_step}
        return self._node_subview(keep_ids, f"{self.topology_id}_before{max_step}")

    def k_hop_subview(self, center_node_ids: set[str], k_hop: int = 2) -> "Topology":
        """Return a new Topology restricted to the k-hop neighborhood of
        `center_node_ids` (e.g. the newly added node(s)) -- the
        "retrieve only the relevant subtopology" piece for S3/GraphRAG
        (slm_future/s3_graphrag/), same `nx.ego_graph` pattern
        `b2_incremental_ilp.k_hop_affected_flows` already uses for flows,
        applied to nodes directly instead. Falls back to the full
        topology if a center node isn't actually in the graph (shouldn't
        happen for a real new-node-addition instance, but keeps this
        total rather than raising)."""
        undirected = self.to_networkx().to_undirected()
        keep_ids: set[str] = set()
        for nid in center_node_ids:
            if nid not in undirected:
                continue
            keep_ids |= set(nx.ego_graph(undirected, nid, radius=k_hop).nodes())
        if not keep_ids:
            keep_ids = set(self.nodes.keys())
        return self._node_subview(keep_ids, f"{self.topology_id}_khop{k_hop}")

    def to_dict(self) -> dict:
        return {
            "topology_id": self.topology_id,
            "switch_count": self.switch_count,
            "generator_family": self.generator_family,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "links": [l.to_dict() for l in self.links.values()],
        }

    @staticmethod
    def from_dict(d: dict) -> "Topology":
        t = Topology(
            topology_id=d["topology_id"],
            switch_count=d.get("switch_count", 0),
            generator_family=d.get("generator_family", "craciunas_config3"),
        )
        for nd in d["nodes"]:
            t.nodes[nd["node_id"]] = Node.from_dict(nd)
        for ld in d["links"]:
            t.links[ld["link_id"]] = Link.from_dict(ld)
        return t
