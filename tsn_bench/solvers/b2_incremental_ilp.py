"""B2: incremental ILP scheduling. On a node addition, only the sub-schedule
affected by the new node (its k-hop neighborhood) is re-solved; everything
else from the previous schedule stays fixed. This is the classic "topology
pruning" incremental-ILP baseline (Nayak et al. / TSSDN style) that S2-S4's
solver-tool calls are meant to narrow down to an even smaller region.
"""
from __future__ import annotations

import time

import networkx as nx

from tsn_bench.flows.models import Flow, FlowClass
from tsn_bench.scheduling.simulator import (
    FlowPath,
    ScheduleResult,
    compute_hyperperiod_ns,
    resolve_flow_path,
)
from tsn_bench.solvers.cp_sat_common import admit_avb_flows, schedule_tt_flows_cpsat
from tsn_bench.topology.models import Topology


def k_hop_affected_flows(
    topology: Topology,
    flow_paths: dict[str, FlowPath],
    new_node_ids: set[str],
    k_hop: int = 1,
) -> set[str]:
    """Flows whose path passes through the k-hop neighborhood of the newly
    added node(s) — the "affected NDPs" the incremental solve is restricted to."""
    undirected = topology.to_networkx().to_undirected()
    affected_nodes: set[str] = set()
    for nid in new_node_ids:
        if nid not in undirected:
            continue
        affected_nodes |= set(nx.ego_graph(undirected, nid, radius=k_hop).nodes())

    affected: set[str] = set()
    for flow_id, fp in flow_paths.items():
        if affected_nodes.intersection(fp.node_path):
            affected.add(flow_id)
    return affected


def solve_b2(
    topology: Topology,
    previous_flows: list[Flow],
    previous_result: ScheduleResult,
    new_flows: list[Flow],
    new_node_ids: set[str],
    k_hop: int = 1,
    time_limit_s: float = 30.0,
) -> tuple[ScheduleResult, dict[str, FlowPath], int]:
    t0 = time.perf_counter()

    all_flows = previous_flows + new_flows
    flow_paths = {f.flow_id: resolve_flow_path(topology, f) for f in all_flows}

    affected = k_hop_affected_flows(topology, flow_paths, new_node_ids, k_hop)
    affected |= {f.flow_id for f in new_flows}  # new flows are always "affected" by definition

    prev_tt_ids = {f.flow_id for f in previous_flows if f.flow_class == FlowClass.TT}
    prev_avb_ids = {f.flow_id for f in previous_flows if f.flow_class == FlowClass.AVB}
    new_tt_ids = {f.flow_id for f in new_flows if f.flow_class == FlowClass.TT}
    new_avb_ids = {f.flow_id for f in new_flows if f.flow_class == FlowClass.AVB}

    fixed_tt_admitted = (prev_tt_ids & previous_result.admitted) - affected
    free_tt_ids = list((prev_tt_ids & previous_result.admitted & affected) | new_tt_ids)
    fixed_offsets_ns = {fid: previous_result.offsets_ns[fid] for fid in fixed_tt_admitted}

    tt_paths = {fid: flow_paths[fid] for fid in (prev_tt_ids | new_tt_ids)}
    hyperperiod_ns = compute_hyperperiod_ns(
        [flow_paths[fid] for fid in fixed_tt_admitted | set(free_tt_ids)]
    ) if (fixed_tt_admitted or free_tt_ids) else 1

    tt_admitted, tt_rejected, offsets, status = schedule_tt_flows_cpsat(
        tt_paths, hyperperiod_ns, fixed_offsets_ns=fixed_offsets_ns, free_ids=free_tt_ids,
        time_limit_s=time_limit_s,
    )

    # AVB: previously admitted + unaffected keep their reservation implicitly by
    # being folded into prior_avb_reserved_mbps; affected/new AVB flows are re-decided.
    fixed_avb_admitted = (prev_avb_ids & previous_result.admitted) - affected
    free_avb_ids = (prev_avb_ids & previous_result.admitted & affected) | new_avb_ids
    prior_reserved: dict[str, float] = {}
    for fid in fixed_avb_admitted:
        fp = flow_paths[fid]
        for hop in fp.hops:
            prior_reserved[hop.link.link_id] = prior_reserved.get(hop.link.link_id, 0.0) + fp.flow.bandwidth_mbps()

    avb_free_paths = {fid: flow_paths[fid] for fid in free_avb_ids}
    avb_admitted, avb_rejected, _ = admit_avb_flows(
        topology, tt_paths, tt_admitted, avb_free_paths, prior_avb_reserved_mbps=prior_reserved,
    )

    decision_latency_s = time.perf_counter() - t0
    result = ScheduleResult(
        admitted=tt_admitted | fixed_avb_admitted | avb_admitted,
        rejected=tt_rejected | avb_rejected | (prev_tt_ids - previous_result.admitted - affected)
        | (prev_avb_ids - previous_result.admitted - affected),
        offsets_ns=offsets,
        method="B2_incremental_ilp",
        decision_latency_s=decision_latency_s,
        solver_status=status,
    )
    return result, flow_paths, hyperperiod_ns
