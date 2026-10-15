"""B3: fast heuristic incremental reconfiguration — a greedy "flexibility
along path" admissibility check, no solver involved. Existing (fixed) flows'
scheduled intervals are treated as obstacles; each candidate flow is tried at
a bounded number of random offsets within its period and admitted at the
first offset whose entire path is collision-free, else rejected. No
optimality guarantee — this is the fast/no-learning baseline B1/B2 are
compared against on decision latency.
"""
from __future__ import annotations

import random
import time

from tsn_bench.flows.models import Flow, FlowClass
from tsn_bench.scheduling.simulator import (
    FlowPath,
    LinkOccupancy,
    ScheduleResult,
    compute_hyperperiod_ns,
    flow_intervals,
    resolve_flow_path,
)
from tsn_bench.solvers.cp_sat_common import admit_avb_flows
from tsn_bench.topology.models import Topology


def solve_b3(
    topology: Topology,
    flows_to_schedule: list[Flow],
    fixed_result: ScheduleResult | None = None,
    fixed_flows: list[Flow] | None = None,
    max_offset_trials: int = 64,
    rng: random.Random | None = None,
) -> tuple[ScheduleResult, dict[str, FlowPath], int]:
    t0 = time.perf_counter()
    rng = rng or random.Random(0)
    fixed_flows = fixed_flows or []
    fixed_result = fixed_result or ScheduleResult()

    all_flows = fixed_flows + flows_to_schedule
    flow_paths = {f.flow_id: resolve_flow_path(topology, f) for f in all_flows}
    tt_paths_all = {fid: fp for fid, fp in flow_paths.items()
                     if fp.flow.flow_class == FlowClass.TT}
    hyperperiod_ns = compute_hyperperiod_ns(list(tt_paths_all.values())) if tt_paths_all else 1

    occupancy: dict[str, LinkOccupancy] = {}
    offsets_ns: dict[str, int] = {}
    avb_reserved: dict[str, float] = {}

    for fid in fixed_result.admitted:
        fp = flow_paths.get(fid)
        if fp is None:
            continue
        if fp.flow.flow_class == FlowClass.TT:
            offset = fixed_result.offsets_ns.get(fid, 0)
            offsets_ns[fid] = offset
            for link_id, s, e in flow_intervals(fp, offset, hyperperiod_ns):
                occupancy.setdefault(link_id, LinkOccupancy()).insert(s, e)
        else:
            for hop in fp.hops:
                avb_reserved[hop.link.link_id] = avb_reserved.get(hop.link.link_id, 0.0) + fp.flow.bandwidth_mbps()

    admitted: set[str] = set(fixed_result.admitted)
    rejected: set[str] = set(fixed_result.rejected)

    tt_candidates = sorted(
        [f for f in flows_to_schedule if f.flow_class == FlowClass.TT],
        key=lambda f: f.arrival_order,
    )
    for f in tt_candidates:
        fp = flow_paths[f.flow_id]
        period = int(fp.period_ns)
        placed = False
        trial_offsets = [0] + [rng.randrange(0, max(1, period)) for _ in range(max_offset_trials)]
        for offset in trial_offsets:
            candidate_intervals = flow_intervals(fp, offset, hyperperiod_ns)
            if offset + fp.total_delay_ns > fp.deadline_ns:
                continue
            collision = any(
                occupancy.get(link_id, LinkOccupancy()).overlaps(s, e)
                for link_id, s, e in candidate_intervals
            )
            if not collision:
                for link_id, s, e in candidate_intervals:
                    occupancy.setdefault(link_id, LinkOccupancy()).insert(s, e)
                offsets_ns[f.flow_id] = offset
                admitted.add(f.flow_id)
                placed = True
                break
        if not placed:
            rejected.add(f.flow_id)

    avb_candidates = {f.flow_id: flow_paths[f.flow_id] for f in flows_to_schedule if f.flow_class == FlowClass.AVB}
    avb_admitted, avb_rejected, _ = admit_avb_flows(
        topology, tt_paths_all, admitted, avb_candidates, prior_avb_reserved_mbps=avb_reserved,
    )
    admitted |= avb_admitted
    rejected |= avb_rejected

    decision_latency_s = time.perf_counter() - t0
    result = ScheduleResult(
        admitted=admitted,
        rejected=rejected,
        offsets_ns=offsets_ns,
        method="B3_heuristic",
        decision_latency_s=decision_latency_s,
        solver_status="heuristic",
    )
    return result, flow_paths, hyperperiod_ns
