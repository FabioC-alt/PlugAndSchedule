"""SLM prompt/completion serialization, following the (prompt = topology +
flow request) -> (completion = admission decision + offset + queue
assignment) recipe from "LLMs can Schedule" (arXiv:2408.06993), adapted from
JSSP vocabulary (job/machine/operation) to TSN vocabulary (talker/listener/
bridge/queue/deadline/period). Not consumed by anything in the foundation
phase — this is the artifact future S1-S4 sessions fine-tune/prompt against,
built from the exact same Topology/Flow objects as graph_format.py so every
method compares on identical instances.
"""
from __future__ import annotations

from tsn_bench.flows.models import Flow
from tsn_bench.scheduling.simulator import FlowPath, ScheduleResult
from tsn_bench.topology.models import NodeType, Topology


def describe_topology(topology: Topology) -> str:
    """Rendered-text topology description, shared by every prompt variant
    (S1's direct-decision prompt, S2's tool-calling prompt, ...) so they all
    describe the same instance identically."""
    lines = ["Network topology:"]
    for sw in topology.switches():
        neighbors = sorted({l.dst for l in topology.links.values() if l.src == sw.node_id}
                            | {l.src for l in topology.links.values() if l.dst == sw.node_id})
        tag = " (newly added)" if sw.added_at_step > 0 else ""
        lines.append(f"- bridge {sw.node_id}{tag}: connects to {', '.join(neighbors) if neighbors else 'none'}")
    for kind, nodes in (("talker", topology.talkers()), ("listener", topology.listeners())):
        for n in nodes:
            tag = " (newly added)" if n.added_at_step > 0 else ""
            lines.append(f"- {kind} {n.node_id}{tag}")
    return "\n".join(lines)


def describe_flow_request(flow: Flow) -> str:
    return (
        f"flow {flow.flow_id}: talker={flow.src}, listener={flow.dst}, "
        f"class={flow.flow_class.value.upper()}, size={flow.size_bytes}B, "
        f"period={flow.period_us}us, deadline={flow.deadline_us}us, pcp={flow.pcp}"
    )


def build_prompt(topology: Topology, new_flows: list[Flow]) -> dict:
    """Structured + rendered-text prompt for one node-addition scheduling
    instance: describe the (possibly just-mutated) topology and ask for an
    admission + offset/queue decision for each newly requested flow."""
    instruction = (
        "You are a TSN (Time-Sensitive Networking) scheduler. Given the network "
        "topology below and a set of newly requested flows, decide for each flow "
        "whether to admit it, and if admitted, assign a transmission offset "
        "(microseconds, within the flow's own period) and confirm its priority "
        "queue, such that no two admitted flows collide on any shared link and "
        "every admitted flow's end-to-end latency stays within its deadline."
    )
    topo_text = describe_topology(topology)
    flows_text = "\n".join(describe_flow_request(f) for f in new_flows)
    prompt_text = f"{instruction}\n\n{topo_text}\n\nRequested flows:\n{flows_text}\n\nDecision:"
    return {
        "instruction": instruction,
        "topology": topology.to_dict(),
        "requested_flows": [f.to_dict() for f in new_flows],
        "prompt_text": prompt_text,
    }


def build_completion(
    new_flows: list[Flow],
    result: ScheduleResult,
    flow_paths: dict[str, FlowPath],
) -> dict:
    decisions = []
    for f in new_flows:
        admitted = f.flow_id in result.admitted
        offset_ns = result.offsets_ns.get(f.flow_id)
        decisions.append({
            "flow_id": f.flow_id,
            "admitted": admitted,
            "offset_us": (offset_ns / 1000.0) if offset_ns is not None else None,
            "queue": f.pcp,
            "path": flow_paths[f.flow_id].node_path if f.flow_id in flow_paths else None,
        })
    lines = []
    for d in decisions:
        if d["admitted"] and d["offset_us"] is not None:
            lines.append(f"flow {d['flow_id']}: ADMIT, offset={d['offset_us']:.2f}us, queue={d['queue']}")
        elif d["admitted"]:
            lines.append(f"flow {d['flow_id']}: ADMIT (best-effort AVB, no fixed offset), queue={d['queue']}")
        else:
            lines.append(f"flow {d['flow_id']}: REJECT")
    return {
        "decisions": decisions,
        "completion_text": "\n".join(lines),
    }
