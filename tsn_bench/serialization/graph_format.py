"""GNN-ready serialization: adjacency + per-node feature table.

Two schemas live here:
  - `node_features`/`to_graph_dict`: the original generic placeholder
    (switch/talker/listener nodes, fixed-width features) -- kept for any
    future GNN-as-retriever use (e.g. an S3 GNN reranker) that just needs
    "a reasonable graph view," not a specific paper's exact input.
  - `gcn_td3_node_features`/`to_gcn_td3_graph`: GCN-TD3's Table II schema
    (arXiv:2405.05019), transcribed exactly for B4
    (`baselines_future/b4_gcn_td3/`). See that function's docstring for the
    transcription and the one documented deviation from the paper's node
    model.
"""
from __future__ import annotations

import numpy as np

from tsn_bench.flows.models import Flow
from tsn_bench.topology.models import NodeType, Topology

NODE_TYPE_ONE_HOT = {NodeType.SWITCH: 0, NodeType.TALKER: 1, NodeType.LISTENER: 2}


def node_features(topology: Topology, flows: list[Flow]) -> tuple[list[str], np.ndarray]:
    """Returns (node_ids in row order, feature matrix [N, F]).

    Features per node: [is_switch, is_talker, is_listener, degree, num_queues,
    is_newly_added, incident_flow_count, incident_bandwidth_mbps].
    """
    g = topology.to_networkx()
    undirected = g.to_undirected()
    node_ids = list(topology.nodes.keys())

    incident_flow_count = {nid: 0 for nid in node_ids}
    incident_bw = {nid: 0.0 for nid in node_ids}
    for f in flows:
        for nid in (f.src, f.dst):
            if nid in incident_flow_count:
                incident_flow_count[nid] += 1
                incident_bw[nid] += f.bandwidth_mbps()

    rows = []
    for nid in node_ids:
        node = topology.nodes[nid]
        one_hot = [0, 0, 0]
        one_hot[NODE_TYPE_ONE_HOT[node.node_type]] = 1
        rows.append([
            *one_hot,
            undirected.degree[nid],
            node.num_queues,
            1 if node.added_at_step > 0 else 0,
            incident_flow_count[nid],
            incident_bw[nid],
        ])
    return node_ids, np.array(rows, dtype=np.float32)


def adjacency_edge_list(topology: Topology, node_ids: list[str]) -> np.ndarray:
    """Returns a [2, E] array of (src_idx, dst_idx) directed edges, indices
    into `node_ids`, for use as a PyG-style edge_index."""
    index = {nid: i for i, nid in enumerate(node_ids)}
    edges = [[index[l.src], index[l.dst]] for l in topology.links.values()
              if l.src in index and l.dst in index]
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(edges, dtype=np.int64).T


def to_graph_dict(topology: Topology, flows: list[Flow]) -> dict:
    node_ids, features = node_features(topology, flows)
    edge_index = adjacency_edge_list(topology, node_ids)
    return {
        "node_ids": node_ids,
        "node_features": features.tolist(),
        "edge_index": edge_index.tolist(),
        "feature_names": [
            "is_switch", "is_talker", "is_listener", "degree", "num_queues",
            "is_newly_added", "incident_flow_count", "incident_bandwidth_mbps",
        ],
    }


# --- GCN-TD3 (arXiv:2405.05019) Table II schema, for B4 -------------------
#
# Table II, transcribed verbatim (features considered "(bullet)" per node
# type, "-" not considered):
#   ID:               talker, listener, scheduled_queue, be_queue
#   No. of Flows:                       scheduled_queue, be_queue
#   PCP:               talker
#   Flow ID:           talker, listener
#   Size:               talker
#   Period:             talker
#   Latency Bound:      talker
#   Destination:        talker
#   Source:                     listener
#   Start Time:         talker,          scheduled_queue
#   End Time:           talker,          scheduled_queue
#   Bridge Delay:       talker
#   Observed Latency:   talker
#   Flows Contained:                    scheduled_queue, be_queue
#   Flows Dropped:                      scheduled_queue, be_queue
#   Switch ID:                          scheduled_queue, be_queue
#   Queue ID:                           scheduled_queue, be_queue
#   Preemptivity:                       scheduled_queue, be_queue
#   Gate Event (Open/Close Time) pairs: scheduled_queue, be_queue
#
# Fig. 3's summary formula (Talker 2+10n, Listener 2+2n, Scheduled
# 7+2(n+s), BE 6+2b) is the paper's own derived shorthand for these same
# marked columns (n/s/b = per-node flow count / scheduled gate-event pairs
# / BE gate-event pairs) -- it doesn't fully reconcile digit-for-digit
# against the marked table for the BE row (their own summary implies BE
# has one fewer scalar than the marked columns give it), which we can't
# resolve without their code. We implement directly from the marked
# columns (the actual data in the table) rather than force-fit the
# shorthand formula.
#
# One documented deviation from the paper's node model: the paper puts
# scheduled/BE queue nodes at EGRESS-PORT granularity (Fig. 1b references
# BR1.Q2, BR1.Q3 as distinct nodes per link direction). `tsn_bench`'s
# `Topology`/`Node` model tracks queues per-SWITCH (`Node.num_queues`),
# not per-port -- the same granularity every other baseline (B1-B3) and
# the shared scheduling simulator already use. Reproducing per-port
# queue nodes would mean forking the topology model just for B4, breaking
# "every method sees the same instance" (see MILESTONES.md). We instead
# emit ONE scheduled_queue node and ONE be_queue node per switch,
# aggregating across that switch's egress ports -- a defensible
# simplification in the same spirit as B1/B2's documented ILP-formulation
# simplifications, not a silent shortcut.
QUEUE_NODE_TYPES = ("scheduled_queue", "be_queue")


def _flow_index_feature(flow_id: str, all_flow_ids: list[str]) -> float:
    """Flow ID as a stable numeric feature: its index in the instance's
    flow list. A GNN treats this as a magnitude-free categorical id (the
    embedding layers learn what to do with it), same role Table II's "Flow
    ID" column plays -- an identifier, not a physical quantity."""
    try:
        return float(all_flow_ids.index(flow_id))
    except ValueError:
        return -1.0


def gcn_td3_node_features(
    topology: Topology,
    flows: list[Flow],
    schedule=None,       # ScheduleResult | None -- already-decided offsets, for Start/End Time
    flow_paths=None,      # dict[str, FlowPath] | None -- resolved paths, for Observed Latency/Bridge Delay
    gcl_events: dict[str, list[tuple[int, int]]] | None = None,  # queue_node_id -> [(open_ns, close_ns), ...]
) -> dict:
    """Table II node features, per node type, as RAGGED (variable-length)
    per-node vectors -- deliberately not padded to one matrix, since the
    whole point of GCN-TD3's per-node-type input layers (Sec. IV-B) is to
    consume exactly this variable width before the GCN homogenizes it.

    `schedule`/`flow_paths` are optional: for a flow that hasn't been
    decided yet (the one the RL agent is currently choosing an action
    for), Start Time/End Time/Observed Latency/Bridge Delay are zero --
    there's nothing to report yet, which is the correct "no signal" value
    for a not-yet-scheduled quantity, and matches how the paper's own
    dynamic-arrival setting has no such history for the arriving flow
    either.

    Returns {"node_ids": {type: [ids]}, "features": {node_id: np.ndarray},
    "edges": [(src_id, dst_id), ...]} -- edges are queue-to-queue,
    talker-to-queue, queue-to-listener, following the physical topology
    with each switch replaced by its two queue nodes (see module docstring
    deviation note). A TT/AVB flow's path only ever touches scheduled_queue
    nodes hop-to-hop (this dataset has no explicit BE flows -- BE queue
    nodes exist structurally, per Table II, but carry no flow-level
    features here, a direct consequence of B1-B3/S1-S4's shared flow
    generator never producing BE-class flows)."""
    flow_paths = flow_paths or {}
    gcl_events = gcl_events or {}
    all_flow_ids = [f.flow_id for f in flows]
    flows_by_id = {f.flow_id: f for f in flows}

    switches = topology.switches()
    switch_index = {n.node_id: i for i, n in enumerate(switches)}
    talkers = topology.talkers()
    listeners = topology.listeners()

    def sq_id(switch_id: str) -> str:
        return f"{switch_id}__sq"

    def be_id(switch_id: str) -> str:
        return f"{switch_id}__be"

    def resolved_timing(flow_id: str) -> tuple[float, float, float, float]:
        """(start_ns, end_ns, bridge_delay_ns, observed_latency_ns), all 0
        if unresolved."""
        fp = flow_paths.get(flow_id)
        if fp is None or schedule is None or flow_id not in schedule.offsets_ns:
            return 0.0, 0.0, 0.0, 0.0
        start = float(schedule.offsets_ns[flow_id])
        observed = float(fp.total_delay_ns)
        end = start + observed
        # Bridge Delay: time spent inside switches (everything after the
        # first hop's transmission off the talker) -- not defined
        # explicitly in the paper text; this is the natural reading given
        # "Bridge Delay" is listed only for Talker, alongside per-hop
        # HopTiming already separating trans/prop time per hop.
        bridge_delay = sum(h.trans_time_ns + h.prop_delay_ns for h in fp.hops[1:])
        return start, end, bridge_delay, observed

    node_ids: dict[str, list[str]] = {"talker": [], "listener": [], "scheduled_queue": [], "be_queue": []}
    features: dict[str, np.ndarray] = {}
    edges: list[tuple[str, str]] = []

    for tk in talkers:
        node_ids["talker"].append(tk.node_id)
        my_flows = [f for f in flows if f.src == tk.node_id]
        row = [float(len(node_ids["talker"]) - 1)]  # ID
        for f in my_flows:
            start, end, bridge_delay, observed = resolved_timing(f.flow_id)
            dst_idx = float(next((i for i, l in enumerate(listeners) if l.node_id == f.dst), -1))
            row.extend([
                float(f.pcp), _flow_index_feature(f.flow_id, all_flow_ids),
                float(f.size_bytes), float(f.period_us), float(f.deadline_us),
                dst_idx, start, end, bridge_delay, observed,
            ])
        features[tk.node_id] = np.array(row, dtype=np.float32)

    for lr in listeners:
        node_ids["listener"].append(lr.node_id)
        my_flows = [f for f in flows if f.dst == lr.node_id]
        row = [float(len(node_ids["listener"]) - 1)]  # ID
        for f in my_flows:
            src_idx = float(next((i for i, t in enumerate(talkers) if t.node_id == f.src), -1))
            row.extend([_flow_index_feature(f.flow_id, all_flow_ids), src_idx])
        features[lr.node_id] = np.array(row, dtype=np.float32)

    def crosses(flow: Flow, switch_id: str) -> bool:
        fp = flow_paths.get(flow.flow_id)
        path = fp.node_path if fp is not None else topology.shortest_path(flow.src, flow.dst)
        return switch_id in path

    for sw in switches:
        crossing = [f for f in flows if crosses(f, sw.node_id)]
        admitted_here = [f for f in crossing if schedule is not None and f.flow_id in schedule.admitted]
        dropped_here = [f for f in crossing if schedule is not None and f.flow_id in schedule.rejected]

        sqid = sq_id(sw.node_id)
        beid = be_id(sw.node_id)
        node_ids["scheduled_queue"].append(sqid)
        node_ids["be_queue"].append(beid)

        sq_events = gcl_events.get(sqid, [])
        be_events = gcl_events.get(beid, [])

        sq_row = [
            float(len(node_ids["scheduled_queue"]) - 1),  # ID
            float(len(crossing)),                          # No. of Flows
            float(len(admitted_here)),                      # Flows Contained
            float(len(dropped_here)),                       # Flows Dropped
            float(switch_index[sw.node_id]),                # Switch ID
            0.0,                                            # Queue ID (0 = scheduled)
            0.0,                                            # Preemptivity (not preemptable)
        ]
        for f in crossing:
            start, end, _bd, _ol = resolved_timing(f.flow_id)
            sq_row.extend([start, end])
        for open_ns, close_ns in sq_events:
            sq_row.extend([float(open_ns), float(close_ns)])
        features[sqid] = np.array(sq_row, dtype=np.float32)

        be_row = [
            float(len(node_ids["be_queue"]) - 1),  # ID
            0.0,   # No. of Flows -- this dataset generates no BE-class flows
            0.0,   # Flows Contained
            0.0,   # Flows Dropped
            float(switch_index[sw.node_id]),        # Switch ID
            1.0,                                    # Queue ID (1 = BE)
            1.0,                                    # Preemptivity (preemptable)
        ]
        for open_ns, close_ns in be_events:
            be_row.extend([float(open_ns), float(close_ns)])
        features[beid] = np.array(be_row, dtype=np.float32)

    # Edges: talker -> first switch's scheduled_queue; scheduled_queue ->
    # scheduled_queue along the path; last switch's scheduled_queue ->
    # listener. BE queues get switch-to-switch structural edges too (no
    # flow ever crosses them in this dataset, but the graph still needs
    # them reachable for the GCN's message passing to see them at all).
    for link in topology.links.values():
        a_is_switch = link.src in switch_index
        b_is_switch = link.dst in switch_index
        if a_is_switch and b_is_switch:
            edges.append((sq_id(link.src), sq_id(link.dst)))
            edges.append((be_id(link.src), be_id(link.dst)))
        elif a_is_switch and not b_is_switch:
            edges.append((sq_id(link.src), link.dst))
        elif not a_is_switch and b_is_switch:
            edges.append((link.src, sq_id(link.dst)))
        # talker/listener directly adjacent with no switch between them
        # doesn't occur in this dataset's generator, so no case for it.

    return {"node_ids": node_ids, "features": features, "edges": edges}
