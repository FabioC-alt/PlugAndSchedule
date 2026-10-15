"""The one tool S2 exposes to the model: `solve_subschedule`. It is a thin
wrapper around the already-existing B2 incremental ILP solver
(tsn_bench.solvers.b2_incremental_ilp.solve_b2) — no new scheduling logic.
The model's job narrows to deciding *which* of the candidate new flows to
hand the solver; whatever it requests gets a real, collision-checked,
deadline-checked offset (or a real rejection), not a hallucinated one.
"""
from __future__ import annotations

from tsn_bench.flows.models import Flow
from tsn_bench.scheduling.simulator import ScheduleResult
from tsn_bench.solvers.b2_incremental_ilp import solve_b2
from tsn_bench.topology.models import Topology

SOLVE_SUBSCHEDULE_TOOL = {
    "type": "function",
    "function": {
        "name": "solve_subschedule",
        "description": (
            "Compute a real, collision-free, deadline-checked schedule for the given "
            "requested flow IDs, incrementally re-solving only the sub-schedule affected "
            "by the newly added node. Returns, for each flow ID, whether it was admitted "
            "and (if admitted and time-triggered) its transmission offset in microseconds. "
            "Flows not included in flow_ids are not considered and will not be scheduled."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "flow_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs of the requested flows to admit and schedule, from the list given in the prompt.",
                },
            },
            "required": ["flow_ids"],
        },
    },
}


def solve_subschedule_tool(
    topology: Topology,
    admitted_base_flows: list[Flow],
    previous_result: ScheduleResult,
    new_flows_by_id: dict[str, Flow],
    requested_flow_ids: list[str],
    new_node_ids: set[str],
    time_limit_s: float = 20.0,
) -> tuple[dict[str, dict], ScheduleResult]:
    """Execute the tool for real: solve_b2 restricted to the requested flow
    IDs (silently dropping any ID that isn't an actual candidate flow —
    hallucinated IDs are a metric to track at the eval layer, not something
    to crash on here).

    Returns (decisions_for_requested_flows, full_result). solve_b2 can
    legitimately reposition previously-admitted base flows within the new
    node's k-hop neighborhood (that's the point of incremental
    re-optimization) — the full ScheduleResult is what actually reflects
    the network's schedule after this call, not just the requested flows'
    decisions. Callers making further tool calls (or checking feasibility
    afterwards) must use `full_result` as the new reference schedule, not
    the `previous_result` this call was given — using the stale one would
    miss any repositioning solve_b2 just did.
    """
    requested_flows = [new_flows_by_id[fid] for fid in requested_flow_ids if fid in new_flows_by_id]
    if not requested_flows:
        return {}, previous_result

    result, _flow_paths, _hyperperiod_ns = solve_b2(
        topology, admitted_base_flows, previous_result, requested_flows, new_node_ids,
        k_hop=1, time_limit_s=time_limit_s,
    )

    out: dict[str, dict] = {}
    for f in requested_flows:
        admitted = f.flow_id in result.admitted
        offset_ns = result.offsets_ns.get(f.flow_id)
        out[f.flow_id] = {
            "admitted": admitted,
            "offset_us": (offset_ns / 1000.0) if offset_ns is not None else None,
        }
    return out, result
