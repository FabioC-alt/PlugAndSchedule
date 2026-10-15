"""B1: full offline ILP rescheduling — recompute the entire schedule from
scratch for the whole topology + flow set every time. This is the optimality
upper bound and the "cost of recomputing everything" baseline that B2/B3 and
later methods are compared against.
"""
from __future__ import annotations

import time

from tsn_bench.flows.models import Flow, FlowClass
from tsn_bench.scheduling.simulator import FlowPath, ScheduleResult, compute_hyperperiod_ns, resolve_flow_path
from tsn_bench.solvers.cp_sat_common import admit_avb_flows, schedule_tt_flows_cpsat
from tsn_bench.topology.models import Topology


def solve_b1(
    topology: Topology,
    flows: list[Flow],
    time_limit_s: float = 60.0,
) -> tuple[ScheduleResult, dict[str, FlowPath], int]:
    t0 = time.perf_counter()

    flow_paths = {f.flow_id: resolve_flow_path(topology, f) for f in flows}
    tt_ids = [f.flow_id for f in flows if f.flow_class == FlowClass.TT]
    avb_ids = [f.flow_id for f in flows if f.flow_class == FlowClass.AVB]
    tt_paths = {fid: flow_paths[fid] for fid in tt_ids}
    avb_paths = {fid: flow_paths[fid] for fid in avb_ids}

    hyperperiod_ns = compute_hyperperiod_ns(list(tt_paths.values())) if tt_paths else 1

    tt_admitted, tt_rejected, offsets, status = schedule_tt_flows_cpsat(
        tt_paths, hyperperiod_ns, fixed_offsets_ns={}, free_ids=tt_ids, time_limit_s=time_limit_s,
    )
    avb_admitted, avb_rejected, _ = admit_avb_flows(topology, tt_paths, tt_admitted, avb_paths)

    decision_latency_s = time.perf_counter() - t0
    result = ScheduleResult(
        admitted=tt_admitted | avb_admitted,
        rejected=tt_rejected | avb_rejected,
        offsets_ns=offsets,
        method="B1_full_ilp",
        decision_latency_s=decision_latency_s,
        solver_status=status,
    )
    return result, flow_paths, hyperperiod_ns
