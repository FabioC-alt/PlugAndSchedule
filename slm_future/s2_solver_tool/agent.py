"""S2 agent loop: give the model one tool (`solve_subschedule`, see
tool_schema.py) and let it decide which candidate flows to hand the solver.
Up to `max_tool_calls` rounds are allowed (a flow set can be split across
multiple calls, or the model can call it once for everything); if the model
never calls the tool at all, its direct text answer is parsed the same way
as S1's (`slm_future.s1_lora.evaluate.parse_completion`) as a fallback, so a
"didn't use the tool" instance is still scored rather than discarded.

Any candidate flow that never gets included in any tool call and never
appears in a fallback direct answer is scored as an implicit reject (the
model chose not to request it, so it was never admitted).
"""
from __future__ import annotations

import json
import re
import time

import torch

from slm_future.s1_lora.evaluate import parse_completion
from slm_future.s2_solver_tool.tool_schema import SOLVE_SUBSCHEDULE_TOOL, solve_subschedule_tool
from tsn_bench.flows.models import Flow
from tsn_bench.scheduling.simulator import ScheduleResult
from tsn_bench.serialization.text_format import describe_flow_request, describe_topology
from tsn_bench.topology.models import Topology

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

AGENT_INSTRUCTION = (
    "You are a TSN (Time-Sensitive Networking) scheduler agent. Below is the network "
    "topology and a list of newly requested flows. Your only job is to submit ALL of "
    "them to the solve_subschedule tool, in a single call, listing every flow ID from "
    "the list below in the flow_ids argument — do not leave any out, and do not try to "
    "guess or pre-filter which ones are likely to fit. The tool is the one that decides "
    "admission and computes exact, collision-free, deadline-checked offsets — that is "
    "not your job, and you have no way to know in advance which flows will fit. Omitting "
    "a flow ID only means that flow is never even considered for scheduling, which is "
    "strictly worse than letting the solver evaluate and reject it. Call the tool "
    "exactly once, with every listed flow ID included, and nothing else."
)

# Terser variant for the decision-latency experiment (see PROGRESS.md /
# TEST_PLAN.md): same requirement (call the tool once, every flow ID, no
# pre-filtering), ~1/3 the words, no justification/reasoning content --
# testing whether a shorter instruction cuts down on preamble reasoning
# before the tool call, not just prompt-token count.
AGENT_INSTRUCTION_TERSE = (
    "TSN scheduler agent. Call solve_subschedule exactly once with every flow ID "
    "listed below in flow_ids -- do not omit or pre-filter any. The tool alone "
    "decides admission and computes offsets; that is not your job."
)


def build_agent_prompt(topology: Topology, new_flows: list[Flow], terse: bool = False) -> str:
    instruction = AGENT_INSTRUCTION_TERSE if terse else AGENT_INSTRUCTION
    topo_text = describe_topology(topology)
    flows_text = "\n".join(describe_flow_request(f) for f in new_flows)
    n = len(new_flows)
    reminder = f"There are {n} flows listed above — flow_ids must include all {n} of their IDs."
    return f"{instruction}\n\n{topo_text}\n\nRequested flows:\n{flows_text}\n\n{reminder}"


def run_agent(
    model,
    tokenizer,
    topology: Topology,
    new_flows: list[Flow],
    admitted_base_flows: list[Flow],
    previous_result: ScheduleResult,
    new_node_ids: set[str],
    max_tool_calls: int = 3,
    # near_saturation instances have 100+ flow IDs to list (~7 tokens/ID in
    # the tool-call JSON); the old default of 512 truncated the JSON before
    # its closing tag past ~70 flows, so the regex never matched at all --
    # total failure, not partial (confirmed at full scale, see PROGRESS.md).
    # 3072 covers the largest observed instances (~120 flows) with headroom.
    max_new_tokens: int = 3072,
    solver_time_limit_s: float = 20.0,
    terse_prompt: bool = False,
    prompt_topology: Topology | None = None,
) -> dict:
    """`prompt_topology`: if given, used to build the model's prompt
    instead of `topology` -- for S3/GraphRAG's retrieval-narrowed context
    (see slm_future/s3_graphrag/). `topology` itself is always what's
    actually passed to the solver tool, regardless of `prompt_topology`;
    narrowing what the LLM *sees* must never narrow what the solver
    *schedules against* -- the tool needs the real, full topology to
    route and check collisions correctly."""
    new_flows_by_id = {f.flow_id: f for f in new_flows}
    candidate_ids = set(new_flows_by_id.keys())

    prompt_text = build_agent_prompt(prompt_topology or topology, new_flows, terse=terse_prompt)
    messages = [{"role": "user", "content": prompt_text}]

    decisions: dict[str, dict] = {}
    requested_ids: set[str] = set()
    context_flows = list(admitted_base_flows)  # grows as tool calls admit new flows, see below
    hallucinated_id_count = 0
    tool_calls_made = 0
    llm_time_s = 0.0
    solver_time_s = 0.0
    used_tool = False
    last_generated_text = ""

    for _round in range(max_tool_calls):
        chat_text = tokenizer.apply_chat_template(
            messages, tools=[SOLVE_SUBSCHEDULE_TOOL], tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        llm_time_s += time.perf_counter() - t0
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        last_generated_text = generated

        match = TOOL_CALL_RE.search(generated)
        if not match:
            break  # model gave a direct answer instead of calling the tool

        try:
            call = json.loads(match.group(1))
            flow_ids = list(call.get("arguments", {}).get("flow_ids", []))
        except (json.JSONDecodeError, AttributeError, TypeError):
            break  # malformed tool call, treat like "no tool call" and fall through

        valid_ids = [fid for fid in flow_ids if fid in candidate_ids]
        hallucinated_id_count += len(flow_ids) - len(valid_ids)
        requested_ids.update(valid_ids)

        t0 = time.perf_counter()
        tool_result, previous_result = solve_subschedule_tool(
            topology, context_flows, previous_result, new_flows_by_id, valid_ids,
            new_node_ids, time_limit_s=solver_time_limit_s,
        )
        # previous_result is reassigned to the just-solved full schedule (not
        # just the requested flows' decisions) so: (a) a second tool call in
        # this same loop solves against what the first call actually placed,
        # including any base-flow repositioning, instead of stale pre-call
        # state; (b) the caller can use the final previous_result to check
        # feasibility against what was truly scheduled.
        #
        # context_flows must grow too, not just previous_result: solve_b2
        # only treats a flow as "existing traffic to potentially re-optimize
        # near the new node" if it's in the `previous_flows` list passed in
        # (it intersects that list's ids against previous_result.admitted to
        # find what's fixed) -- a flow this call just admitted is present in
        # previous_result.admitted but NOT in context_flows yet, so without
        # this update a later call in the same session would solve as if
        # this round's new admissions don't exist, silently allowing
        # collisions with them.
        context_flows = context_flows + [
            new_flows_by_id[fid] for fid in valid_ids if tool_result.get(fid, {}).get("admitted")
        ]
        solver_time_s += time.perf_counter() - t0
        tool_calls_made += 1
        used_tool = True
        decisions.update(tool_result)

        messages.append({"role": "assistant", "content": generated})
        messages.append({"role": "tool", "name": "solve_subschedule", "content": json.dumps(tool_result)})

    if not used_tool:
        # fallback: score whatever direct text the model gave, same parser as S1
        decisions = parse_completion(last_generated_text)

    for fid in candidate_ids - set(decisions.keys()):
        decisions[fid] = {"admitted": False, "offset_us": None}

    return {
        "decisions": decisions,
        "used_tool": used_tool,
        "tool_calls_made": tool_calls_made,
        "requested_flow_count": len(requested_ids),
        "candidate_flow_count": len(candidate_ids),
        "hallucinated_flow_id_count": hallucinated_id_count,
        "llm_time_s": llm_time_s,
        "solver_time_s": solver_time_s,
        "last_generated_text": last_generated_text,
        # the true final schedule after all tool calls (== the original
        # previous_result unchanged if the tool was never called) -- use
        # this, not the previous_result passed into run_agent, to check
        # feasibility or to seed any further solving.
        "final_schedule": previous_result,
    }
