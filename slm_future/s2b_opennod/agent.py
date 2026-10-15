"""S2b agent: same flow-selection task as S2 (which candidate flows to hand
solve_subschedule), but decided via OpenNod's typed-logits scoring instead
of generating a tool-call JSON. The solver tool itself is untouched --
reused directly from S2 (slm_future.s2_solver_tool.tool_schema).

Needs `semif_phase1` (OpenNod, github.com/TheoLeeCJ/OpenNod) importable --
see README.md for why that means a separate venv from S1/S2's `.venv-gpu`
(OpenNod pins torch==2.10.0/transformers==5.17.0, a different stack).
"""
from __future__ import annotations

from semif_phase1.shared import score_shared

from tsn_bench.flows.models import Flow
from tsn_bench.serialization.text_format import describe_flow_request, describe_topology
from tsn_bench.topology.models import Topology

QUESTION_TEMPLATE = (
    "New flow request: {flow_text}\n"
    "Should this flow be submitted to the solver for scheduling? "
    "The solver alone decides admission and computes exact, "
    "collision-free, deadline-checked offsets -- submitting a flow "
    "just means letting the solver evaluate it, not admitting it "
    "outright. Omitting it means it is never even considered."
)

OPTIONS = [
    {"id": "yes", "description": "Submit this flow to the solver."},
    {"id": "no", "description": "Do not submit this flow to the solver."},
]


def build_rows(topology: Topology, new_flows: list[Flow]) -> tuple[str, list[dict]]:
    """One shared 'state' (the topology, same text S2's prompt uses) plus
    one row per candidate flow -- state must be byte-identical across rows
    for score_shared's shared-prefix cache to apply."""
    state = describe_topology(topology)
    rows = [
        {
            "id": f.flow_id,
            "state": state,
            "question": QUESTION_TEMPLATE.format(flow_text=describe_flow_request(f)),
            "options": OPTIONS,
        }
        for f in new_flows
    ]
    return state, rows


def select_flow_ids(model, tokenizer, topology: Topology, new_flows: list[Flow], metadata: dict) -> tuple[list[str], dict]:
    """Scores every candidate flow in one score_shared() call, returns the
    IDs that scored yes > no plus timing info (mirrors run_agent's
    llm_time_s/tool_calls_made-style return shape so evaluate.py can log
    it the same way S2's does)."""
    _state, rows = build_rows(topology, new_flows)
    scored, timing = score_shared(model, tokenizer, rows, metadata)
    requested_ids = [
        r["id"] for r, s in zip(rows, scored)
        if s["probabilities"][s["option_ids"].index("yes")] > s["probabilities"][s["option_ids"].index("no")]
    ]
    return requested_ids, timing
