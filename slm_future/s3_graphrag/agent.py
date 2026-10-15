"""S3/GraphRAG: same architecture as S2 (base model + solve_subschedule
tool), but the model's prompt describes only a k-hop-retrieved subtopology
instead of the full network -- "retrieve only the relevant subtopology"
(option (a) in the original research plan: plain k-hop graph traversal,
no GNN reranker). Isolates the retrieval question specifically: does a
shorter, focused context change quality or latency for the tool-calling
architecture, holding the base model and the tool itself constant?

Reuses S2's run_agent() unchanged (slm_future.s2_solver_tool.agent) via
its `prompt_topology` parameter -- the retrieval only narrows what the
model *sees*; solve_subschedule always gets the real, full topology, so
scheduling correctness is never affected by what the model was shown.
"""
from __future__ import annotations

from slm_future.s2_solver_tool.agent import run_agent
from tsn_bench.flows.models import Flow
from tsn_bench.scheduling.simulator import ScheduleResult
from tsn_bench.topology.models import Topology


def retrieval_node_ids(topology: Topology, new_flows: list[Flow], new_node_ids: set[str]) -> set[str]:
    """The nodes that MUST be in the retrieved subview for it to be a
    complete, correct description: the newly added node(s) themselves,
    plus every new flow's actual shortest-path route (guarantees each
    flow's talker/listener endpoints are always described, even when one
    endpoint sits far from the new node -- the generator only requires
    ONE endpoint of a new flow to touch new_node_ids, see
    tsn_bench/flows/generator.py, so the other endpoint can be anywhere).
    k_hop_subview then expands a k-hop buffer around this set."""
    ids = set(new_node_ids)
    for f in new_flows:
        ids.update(topology.shortest_path(f.src, f.dst))
    return ids


def run_agent_with_retrieval(
    model,
    tokenizer,
    topology: Topology,
    new_flows: list[Flow],
    admitted_base_flows: list[Flow],
    previous_result: ScheduleResult,
    new_node_ids: set[str],
    k_hop: int = 0,
    **run_agent_kwargs,
) -> dict:
    center_ids = retrieval_node_ids(topology, new_flows, new_node_ids)
    retrieved = topology.k_hop_subview(center_ids, k_hop=k_hop)
    result = run_agent(
        model, tokenizer, topology, new_flows, admitted_base_flows, previous_result,
        new_node_ids, prompt_topology=retrieved, **run_agent_kwargs,
    )
    result["retrieved_node_count"] = len(retrieved.nodes)
    result["full_node_count"] = len(topology.nodes)
    return result
