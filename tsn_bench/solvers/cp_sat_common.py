"""Shared CP-SAT model builder used by both B1 (full ILP) and B2 (incremental
ILP). Only TT flows go through the exact solver — AVB flows are admitted via
a reserved-bandwidth budget check (see module docstring in scheduling/simulator.py
for why this split is a documented simplification, not a full CBS simulation).
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from tsn_bench.flows.models import Flow
from tsn_bench.scheduling.simulator import FlowPath
from tsn_bench.topology.models import Topology


def schedule_tt_flows_cpsat(
    flow_paths: dict[str, FlowPath],
    hyperperiod_ns: int,
    fixed_offsets_ns: dict[str, int],
    free_ids: list[str],
    time_limit_s: float = 30.0,
    num_workers: int = 8,
) -> tuple[set[str], set[str], dict[str, int], str]:
    """Solve for offsets of `free_ids` TT flows, treating `fixed_offsets_ns`
    (already-scheduled flows, e.g. from before a node addition) as immovable
    obstacles. Maximizes the number of admitted free flows subject to
    per-link no-overlap and per-flow deadline constraints.

    Returns (admitted_ids, rejected_ids, offsets_ns, solver_status). The
    returned admitted/offsets include the fixed flows unchanged.
    """
    model = cp_model.CpModel()
    link_intervals: dict[str, list] = {}

    for flow_id, offset_ns in fixed_offsets_ns.items():
        fp = flow_paths[flow_id]
        repeats = max(1, hyperperiod_ns // fp.period_ns)
        for hop in fp.hops:
            for k in range(repeats):
                start = offset_ns + k * fp.period_ns + hop.pre_delay_ns
                iv = model.NewIntervalVar(
                    start, hop.trans_time_ns, start + hop.trans_time_ns,
                    f"fix_{flow_id}_{hop.link.link_id}_{k}",
                )
                link_intervals.setdefault(hop.link.link_id, []).append(iv)

    offset_vars: dict[str, cp_model.IntVar] = {}
    admitted_vars: dict[str, cp_model.IntVar] = {}
    for flow_id in free_ids:
        fp = flow_paths[flow_id]
        period = fp.period_ns
        offset = model.NewIntVar(0, max(0, period - 1), f"off_{flow_id}")
        admitted = model.NewBoolVar(f"adm_{flow_id}")
        offset_vars[flow_id] = offset
        admitted_vars[flow_id] = admitted

        model.Add(offset + fp.total_delay_ns <= int(fp.deadline_ns)).OnlyEnforceIf(admitted)

        repeats = max(1, hyperperiod_ns // period)
        for hop in fp.hops:
            for k in range(repeats):
                start_expr = offset + k * period + hop.pre_delay_ns
                iv = model.NewOptionalIntervalVar(
                    start_expr, hop.trans_time_ns, start_expr + hop.trans_time_ns,
                    admitted, f"opt_{flow_id}_{hop.link.link_id}_{k}",
                )
                link_intervals.setdefault(hop.link.link_id, []).append(iv)

    for intervals in link_intervals.values():
        model.AddNoOverlap(intervals)

    if admitted_vars:
        model.Maximize(sum(admitted_vars.values()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = num_workers
    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    admitted_set = set(fixed_offsets_ns.keys())
    rejected_set: set[str] = set()
    offsets = dict(fixed_offsets_ns)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for flow_id in free_ids:
            if solver.Value(admitted_vars[flow_id]):
                admitted_set.add(flow_id)
                offsets[flow_id] = solver.Value(offset_vars[flow_id])
            else:
                rejected_set.add(flow_id)
    else:
        rejected_set.update(free_ids)

    return admitted_set, rejected_set, offsets, status_name


def admit_avb_flows(
    topology: Topology,
    tt_flow_paths: dict[str, FlowPath],
    tt_admitted: set[str],
    avb_flow_paths: dict[str, FlowPath],
    prior_avb_reserved_mbps: dict[str, float] | None = None,
) -> tuple[set[str], set[str], dict[str, float]]:
    """Greedy reserved-bandwidth admission for AVB flows: a flow is admitted
    if, on every link along its path, (TT reserved bandwidth + already
    admitted AVB bandwidth + this flow's bandwidth) stays within the link
    rate. Processed in arrival_order to match "flows arriving after a node
    is added" ordering.
    """
    reserved: dict[str, float] = dict(prior_avb_reserved_mbps or {})
    tt_reserved: dict[str, float] = {}
    for flow_id in tt_admitted:
        fp = tt_flow_paths.get(flow_id)
        if fp is None:
            continue
        bw = fp.flow.bandwidth_mbps()
        for hop in fp.hops:
            tt_reserved[hop.link.link_id] = tt_reserved.get(hop.link.link_id, 0.0) + bw

    admitted: set[str] = set()
    rejected: set[str] = set()
    ordered = sorted(avb_flow_paths.values(), key=lambda fp: fp.flow.arrival_order)
    for fp in ordered:
        bw = fp.flow.bandwidth_mbps()
        fits = True
        for hop in fp.hops:
            link_id = hop.link.link_id
            used = tt_reserved.get(link_id, 0.0) + reserved.get(link_id, 0.0)
            if used + bw > hop.link.rate_mbps:
                fits = False
                break
        if fits:
            for hop in fp.hops:
                reserved[hop.link.link_id] = reserved.get(hop.link.link_id, 0.0) + bw
            admitted.add(fp.flow.flow_id)
        else:
            rejected.add(fp.flow.flow_id)
    return admitted, rejected, reserved
