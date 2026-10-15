"""S4 agent loop: the model gets TWO tools -- `retrieve_subgraph`
(optional, S3's retrieval mechanism) and `solve_subschedule` (S2's solver
tool) -- and decides for itself whether/when to retrieve before
submitting flows, rather than S3's design where retrieval always happens
automatically before the model ever sees the prompt. This is the actual
research question S4 asks: does giving the model that choice change
anything, versus S3's "always retrieve" and S2's "never retrieve"?

Reuses S2's exact solve_subschedule dispatch logic and S1's fallback
parser -- the only new piece is the retrieve_subgraph branch and the
two-tool dispatch by name.
"""
from __future__ import annotations

import json
import re
import time

import torch

from slm_future.s1_lora.evaluate import parse_completion
from slm_future.s2_solver_tool.tool_schema import SOLVE_SUBSCHEDULE_TOOL, solve_subschedule_tool
from slm_future.s4_full_agent.tool_schema import RETRIEVE_SUBGRAPH_TOOL, retrieve_subgraph_tool
from tsn_bench.flows.models import Flow
from tsn_bench.scheduling.simulator import ScheduleResult
from tsn_bench.serialization.text_format import describe_flow_request, describe_topology
from tsn_bench.topology.models import Topology

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

AGENT_INSTRUCTION = (
    "You are a TSN (Time-Sensitive Networking) scheduler agent. Below is the network "
    "topology and a list of newly requested flows. You have two tools available. "
    "retrieve_subgraph is OPTIONAL -- it gives you a shorter, focused view of just the "
    "relevant part of the topology, useful if the full topology below is large. "
    "solve_subschedule is REQUIRED -- call it exactly once, listing every flow ID from the "
    "list below in flow_ids, to get real admission and offset decisions. Do not leave any "
    "flow ID out and do not try to guess or pre-filter which ones are likely to fit -- the "
    "tool is the one that decides admission and computes exact, collision-free, "
    "deadline-checked offsets, not you. Omitting a flow ID only means that flow is never "
    "even considered for scheduling, which is strictly worse than letting the solver "
    "evaluate and reject it. You may call retrieve_subgraph first if it would help, then "
    "you must call solve_subschedule with every listed flow ID included."
)


def build_agent_prompt(topology: Topology, new_flows: list[Flow]) -> str:
    topo_text = describe_topology(topology)
    flows_text = "\n".join(describe_flow_request(f) for f in new_flows)
    n = len(new_flows)
    reminder = f"There are {n} flows listed above — flow_ids must include all {n} of their IDs."
    return f"{AGENT_INSTRUCTION}\n\n{topo_text}\n\nRequested flows:\n{flows_text}\n\n{reminder}"


def run_agent(
    model,
    tokenizer,
    topology: Topology,
    new_flows: list[Flow],
    admitted_base_flows: list[Flow],
    previous_result: ScheduleResult,
    new_node_ids: set[str],
    max_tool_calls: int = 4,  # one more than S2's 3, to allow an optional retrieve_subgraph round
    max_new_tokens: int = 3072,  # see s2_solver_tool/agent.py for why not the old 512
    solver_time_limit_s: float = 20.0,
) -> dict:
    new_flows_by_id = {f.flow_id: f for f in new_flows}
    candidate_ids = set(new_flows_by_id.keys())

    # near_saturation instances (100+ flows) must list every flow_id in the
    # solve_subschedule call. Post-mortem on job 53504 (held_out_test)
    # found generation stopping (apparent early EOS) well short of that --
    # anywhere from ~20 tokens (barely starting the arguments object) to
    # ~150-300 tokens (partway through the flow_ids list), always far
    # under max_new_tokens, and the resulting text never even reaches a
    # closed </tool_call> tag. min_new_tokens forces past that early stop;
    # scaled by candidate count so small instances (which already complete
    # correctly in ~150-250 tokens) aren't forced through needless extra
    # generation.
    min_new_tokens = min(max_new_tokens - 200, 100 + 8 * len(candidate_ids))

    prompt_text = build_agent_prompt(topology, new_flows)
    messages = [{"role": "user", "content": prompt_text}]
    tools = [RETRIEVE_SUBGRAPH_TOOL, SOLVE_SUBSCHEDULE_TOOL]

    decisions: dict[str, dict] = {}
    requested_ids: set[str] = set()
    context_flows = list(admitted_base_flows)
    hallucinated_id_count = 0
    tool_calls_made = 0
    retrieval_calls_made = 0
    llm_time_s = 0.0
    solver_time_s = 0.0
    used_solver_tool = False
    last_generated_text = ""

    for _round in range(max_tool_calls):
        chat_text = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                do_sample=False,
                # REMOVED no_repeat_ngram_size=6: smoke test 53558 showed it
                # backfires badly -- it blocks repeating ANY 6-token n-gram
                # seen anywhere earlier in the sequence, including the
                # PROMPT itself. AGENT_INSTRUCTION already says
                # "solve_subschedule" multiple times, so the model got
                # blocked from writing that exact identifier again and
                # mangled it into "solve_subscheduler" -- a new failure
                # mode that didn't exist before this "fix". Wrong tool for
                # a task that needs to copy fixed identifiers (tool names,
                # flow IDs already listed in the prompt) verbatim. The
                # repetition-loop failure mode it was meant to address
                # needs a different mitigation, not yet implemented.
                pad_token_id=tokenizer.pad_token_id,
            )
        llm_time_s += time.perf_counter() - t0
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        last_generated_text = generated

        # Try every <tool_call> occurrence in the text, in order, not just
        # the first. Post-mortem on job 53559 (min_new_tokens forcing
        # generation past the model's natural stop): the model sometimes
        # writes a malformed first attempt (e.g. one stray extra closing
        # brace), then -- forced to keep going -- immediately RETRIES with
        # a second, well-formed attempt repeating the same correct flow
        # list. Only ever looking at the first occurrence (the old
        # TOOL_CALL_RE.search behavior) threw away that usable retry.
        tool_name = None
        args: dict = {}
        for match in TOOL_CALL_RE.finditer(generated):
            candidate_raw = match.group(1)
            # Extract just the tool name by regex first, rather than
            # requiring the whole call's JSON to parse -- retrieve_subgraph
            # takes no arguments, so a malformed "arguments" object
            # shouldn't sink an otherwise-identifiable call.
            name_match = re.search(r'"name"\s*:\s*"(\w+)"', candidate_raw)
            candidate_name = name_match.group(1) if name_match else None

            if candidate_name == "retrieve_subgraph":
                tool_name = candidate_name
                break

            if candidate_name == "solve_subschedule":
                # solve_subschedule's arguments (flow_ids) are load-bearing,
                # unlike retrieve_subgraph's -- these still need to parse
                # cleanly, or this occurrence is skipped in favor of a
                # later one (if any).
                try:
                    candidate_args = json.loads(candidate_raw).get("arguments", {})
                except (json.JSONDecodeError, AttributeError, TypeError):
                    continue
                if not isinstance(candidate_args, dict):
                    # model emitted e.g. "arguments": [...] instead of an
                    # object -- crashed evaluate.py mid-run on instance
                    # 149/720 of job 53497 (args.get("flow_ids", ...) on a
                    # list). Same bucket as the JSONDecodeError case above.
                    continue
                tool_name, args = candidate_name, candidate_args
                break
            # unrecognized name in this occurrence -- try the next one

        if tool_name is None:
            break  # no occurrence in this generation yielded a usable call

        if tool_name == "retrieve_subgraph":
            messages.append({"role": "assistant", "content": generated})
            retrieved_text = retrieve_subgraph_tool(topology, new_flows, new_node_ids)
            retrieval_calls_made += 1
            messages.append({"role": "tool", "name": "retrieve_subgraph", "content": retrieved_text})
            # Without this, the model can lose the thread of its remaining
            # obligation after the tool round-trip -- confirmed on a smoke
            # test (2/5 instances): one re-asked for flow details already
            # given in the original prompt, another narrated "I will now
            # call solve_subschedule" and asked for confirmation instead of
            # actually emitting the call. A small model's instruction
            # following is weaker across a multi-turn tool exchange than
            # within a single turn, so the requirement needs restating,
            # not just having been said once at the very start.
            messages.append({"role": "user", "content": (
                "Now call solve_subschedule with every one of the flow IDs from the original "
                "request listed above -- do not ask for confirmation or additional information, "
                "and do not leave any flow ID out. Call the tool directly."
            )})
            continue

        messages.append({"role": "assistant", "content": generated})

        flow_ids = list(args.get("flow_ids", []))
        valid_ids = [fid for fid in flow_ids if fid in candidate_ids]
        hallucinated_id_count += len(flow_ids) - len(valid_ids)
        requested_ids.update(valid_ids)

        t0 = time.perf_counter()
        tool_result, previous_result = solve_subschedule_tool(
            topology, context_flows, previous_result, new_flows_by_id, valid_ids,
            new_node_ids, time_limit_s=solver_time_limit_s,
        )
        # see s2_solver_tool/agent.py's run_agent for why both previous_result
        # and context_flows must be updated with this call's real outcome
        # before any further tool call in the same session.
        context_flows = context_flows + [
            new_flows_by_id[fid] for fid in valid_ids if tool_result.get(fid, {}).get("admitted")
        ]
        solver_time_s += time.perf_counter() - t0
        tool_calls_made += 1
        used_solver_tool = True
        decisions.update(tool_result)

        messages.append({"role": "tool", "name": "solve_subschedule", "content": json.dumps(tool_result)})

        # Stop here rather than looping for an optional wrap-up round.
        # AGENT_INSTRUCTION already says to call solve_subschedule exactly
        # once with every flow ID -- decisions are fully captured the
        # moment this call succeeds, so a further round buys nothing for
        # scoring. Smoke test 53677 found the model sometimes rambles at
        # length in that round on its own (e.g. prose-summarizing results),
        # not because of any floor we set -- zeroing min_new_tokens for it
        # (tried first) doesn't cap that, it only removes a floor. Not
        # looping at all removes the cost entirely instead of bounding it.
        break

    if not used_solver_tool:
        # fallback: score whatever direct text the model gave, same parser as S1/S2
        decisions = parse_completion(last_generated_text)

    for fid in candidate_ids - set(decisions.keys()):
        decisions[fid] = {"admitted": False, "offset_us": None}

    return {
        "decisions": decisions,
        "used_tool": used_solver_tool,
        "tool_calls_made": tool_calls_made,
        "retrieval_calls_made": retrieval_calls_made,
        "retrieved": retrieval_calls_made > 0,
        "requested_flow_count": len(requested_ids),
        "candidate_flow_count": len(candidate_ids),
        "hallucinated_flow_id_count": hallucinated_id_count,
        "llm_time_s": llm_time_s,
        "solver_time_s": solver_time_s,
        "last_generated_text": last_generated_text,
        "final_schedule": previous_result,
    }
