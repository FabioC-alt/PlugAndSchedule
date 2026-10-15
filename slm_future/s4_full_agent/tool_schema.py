"""S4's second tool: `retrieve_subgraph`, letting the model choose WHETHER
and WHEN to narrow its own view of the topology, unlike S3 where retrieval
always happens automatically before the model ever sees the prompt. Reuses
S3's exact retrieval mechanism (`retrieval_node_ids` + `k_hop_subview`,
k_hop=0 -- see slm_future/s3_graphrag/README.md for why any k-hop buffer
beyond the flow routes themselves washes out most of the reduction) --
this tool is a genuine choice point, not a new retrieval algorithm.
"""
from __future__ import annotations

from slm_future.s3_graphrag.agent import retrieval_node_ids
from tsn_bench.flows.models import Flow
from tsn_bench.serialization.text_format import describe_topology
from tsn_bench.topology.models import Topology

RETRIEVE_SUBGRAPH_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_subgraph",
        "description": (
            "Get a focused, shorter description of the network topology, restricted to the "
            "newly added node(s) and every requested flow's actual route through the network -- "
            "everything relevant to this decision, nothing else. Optional: only call this if the "
            "full topology above is large and a narrower view would help you reason about it. "
            "Calling this does not submit anything for scheduling -- you must still call "
            "solve_subschedule afterwards with every candidate flow ID to get real decisions."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def retrieve_subgraph_tool(topology: Topology, new_flows: list[Flow], new_node_ids: set[str]) -> str:
    """Executes the tool for real: same retrieval set S3 uses
    (new node(s) + every new flow's shortest-path route), k_hop=0, rendered
    as the same topology-description text every other prompt variant uses
    so the model sees a consistent format regardless of which tool built it."""
    center_ids = retrieval_node_ids(topology, new_flows, new_node_ids)
    retrieved = topology.k_hop_subview(center_ids, k_hop=0)
    return describe_topology(retrieved)
