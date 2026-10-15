"""Topology family generator, in the style of Craciunas et al.'s "configuration 3"
backbone (a switch mesh with a handful of redundant cross-links, and talkers/
listeners hanging off the edge switches) — the same style GCN-TD3 builds on.

Parameterized purely by switch_count so we can generate matched families at
{2,4,6,8,12,16} switches and hold some sizes out of training entirely to test
generalization to unseen node counts (the axis GCN-TD3 could not handle).
"""
from __future__ import annotations

import random

from tsn_bench.topology.models import Link, Node, NodeType, Topology


def _switch_id(i: int) -> str:
    return f"sw{i}"


def _end_station_id(kind: str, i: int) -> str:
    return f"{kind}{i}"


def generate_topology(
    switch_count: int,
    talkers_range: tuple[int, int],
    listeners_range: tuple[int, int],
    link_rate_mbps: float,
    rng: random.Random,
    topology_id: str,
    extra_cross_links: int | None = None,
) -> Topology:
    """Build one topology instance with `switch_count` switches.

    Backbone: a random spanning tree over the switches (guarantees connectivity
    with the minimum number of links) plus a small number of extra cross-links
    for path redundancy, matching the "mesh with a few extra links" shape of
    Craciunas et al.'s configuration 3. Each switch gets a random number of
    talkers and listeners attached as edge (leaf) end-stations.
    """
    topo = Topology(topology_id=topology_id, switch_count=switch_count)

    switch_ids = [_switch_id(i) for i in range(switch_count)]
    for sid in switch_ids:
        topo.add_node(Node(node_id=sid, node_type=NodeType.SWITCH, num_queues=8))

    # random spanning tree over switches
    remaining = switch_ids[1:]
    connected = [switch_ids[0]]
    rng.shuffle(remaining)
    link_counter = 0
    for sid in remaining:
        parent = rng.choice(connected)
        topo.add_link(Link(
            link_id=f"L{link_counter}",
            src=parent, dst=sid,
            rate_mbps=link_rate_mbps,
        ))
        link_counter += 1
        connected.append(sid)

    # extra cross-links for redundancy/mesh shape (skip for trivially small topologies)
    if extra_cross_links is None:
        extra_cross_links = max(0, switch_count // 4)
    attempts = 0
    added = 0
    while added < extra_cross_links and attempts < extra_cross_links * 10:
        attempts += 1
        a, b = rng.sample(switch_ids, 2)
        if topo.link_between(a, b) is not None:
            continue
        topo.add_link(Link(link_id=f"L{link_counter}", src=a, dst=b, rate_mbps=link_rate_mbps))
        link_counter += 1
        added += 1

    # end stations: talkers and listeners attached to random switches
    talker_idx = 0
    listener_idx = 0
    for sid in switch_ids:
        n_talkers = rng.randint(*talkers_range)
        n_listeners = rng.randint(*listeners_range)
        for _ in range(n_talkers):
            tid = _end_station_id("tk", talker_idx)
            talker_idx += 1
            topo.add_node(Node(node_id=tid, node_type=NodeType.TALKER, num_queues=1))
            topo.add_link(Link(link_id=f"L{link_counter}", src=tid, dst=sid, rate_mbps=link_rate_mbps))
            link_counter += 1
        for _ in range(n_listeners):
            lid = _end_station_id("ls", listener_idx)
            listener_idx += 1
            topo.add_node(Node(node_id=lid, node_type=NodeType.LISTENER, num_queues=1))
            topo.add_link(Link(link_id=f"L{link_counter}", src=sid, dst=lid, rate_mbps=link_rate_mbps))
            link_counter += 1

    return topo


def add_node_to_topology(
    topo: Topology,
    node_type: NodeType,
    rng: random.Random,
    step: int,
    link_rate_mbps: float = 1000.0,
) -> list[str]:
    """Mutate `topo` in place by attaching one new node, returning the ids of
    every node created (more than one for SWITCH, see below).

    - TALKER/LISTENER: attached to a random existing switch.
    - SWITCH: attached to a random existing switch, and given one talker and
      one listener of its own so it has traffic to schedule.
    Node ids are namespaced by `step` so repeated additions (e.g. within a
    batch node-addition scenario) never collide — callers must pass a
    distinct `step` per node added in the same batch.
    """
    existing_switches = topo.switches()
    if not existing_switches:
        raise ValueError("cannot add a node to a topology with no switches")
    anchor = rng.choice(existing_switches).node_id
    link_counter = len(topo.links)

    if node_type == NodeType.SWITCH:
        new_id = f"sw_add{step}"
        topo.add_node(Node(node_id=new_id, node_type=NodeType.SWITCH, num_queues=8, added_at_step=step))
        topo.add_link(Link(link_id=f"Ladd{link_counter}", src=anchor, dst=new_id, rate_mbps=link_rate_mbps))
        link_counter += 1
        created = [new_id]
        for kind, ntype in (("tk", NodeType.TALKER), ("ls", NodeType.LISTENER)):
            leaf_id = f"{kind}_add{step}"
            topo.add_node(Node(node_id=leaf_id, node_type=ntype, num_queues=1, added_at_step=step))
            if ntype == NodeType.TALKER:
                topo.add_link(Link(link_id=f"Ladd{link_counter}", src=leaf_id, dst=new_id, rate_mbps=link_rate_mbps))
            else:
                topo.add_link(Link(link_id=f"Ladd{link_counter}", src=new_id, dst=leaf_id, rate_mbps=link_rate_mbps))
            link_counter += 1
            created.append(leaf_id)
        return created

    prefix = "tk" if node_type == NodeType.TALKER else "ls"
    new_id = f"{prefix}_add{step}"
    topo.add_node(Node(node_id=new_id, node_type=node_type, num_queues=1, added_at_step=step))
    if node_type == NodeType.TALKER:
        topo.add_link(Link(link_id=f"Ladd{link_counter}", src=new_id, dst=anchor, rate_mbps=link_rate_mbps))
    else:
        topo.add_link(Link(link_id=f"Ladd{link_counter}", src=anchor, dst=new_id, rate_mbps=link_rate_mbps))
    return [new_id]
