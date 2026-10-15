"""S2 evaluation: run the tool-calling agent over a data split and score it
the same way as S1 (slm_future.s1_lora.evaluate) — admission accuracy
against the B1 ground truth, and feasibility rate (collision/deadline
checked, reusing tsn_bench.scheduling.simulator's interval logic) — plus
S2-specific tool-use metrics: whether the tool was used at all, how many
times, how many candidate flows actually got submitted vs. silently
implicit-rejected, and hallucinated flow IDs.

Uses the **base** Qwen2.5-Instruct model (no S1 LoRA adapter) by default:
S1's adapter was fine-tuned to imitate direct-decision completions, not to
call tools, so reusing it here would confound "does tool access help" with
"does this specific fine-tune generalize to a different task." The base
instruct model's native tool-calling ability is the cleaner variable to
isolate for a first S2 result — see README for the follow-up (fine-tuning
specifically for tool-calling) once this baseline exists.

Needs a GPU. Run via scripts/cluster/submit_s2_eval.sbatch on disi.

Usage:
    python -m slm_future.s2_solver_tool.evaluate \
        --data data --split held_out_test \
        --out results/s2_eval_held_out.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from slm_future.s1_lora.evaluate import check_feasibility, parse_completion
from slm_future.s2_solver_tool.agent import run_agent
from tsn_bench.flows.models import Flow
from tsn_bench.logging_utils import JsonlLogger
from tsn_bench.solvers.b1_full_ilp import solve_b1
from tsn_bench.topology.models import Topology


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the S2 tool-calling agent")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="held_out_test",
                     choices=["train", "val", "test", "held_out_test"])
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", default="results/s2_eval.jsonl")
    ap.add_argument("--max-tool-calls", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=3072,
                     help="see agent.py::run_agent's default for why 3072, not the old 512")
    ap.add_argument("--solver-time-limit-s", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=None, help="evaluate at most N instances")
    ap.add_argument("--load-factor", default=None, choices=["low", "medium", "near_saturation"],
                     help="restrict to instances whose ID contains this load factor -- for "
                          "targeted re-runs (e.g. confirming a fix on just near_saturation) "
                          "without a full-split run")
    ap.add_argument("--no-tool", action="store_true",
                     help="control condition: run the base model with the solve_subschedule "
                          "tool never offered at all, using S1's exact direct-decision prompt "
                          "and parser instead. Distinct from the tool-available path's "
                          "used_tool=False rows, where the model had the tool but chose not "
                          "to call it -- this isolates whether tool ACCESS itself changes "
                          "behavior, not just whether it gets used on a given instance.")
    ap.add_argument("--terse-prompt", action="store_true",
                     help="decision-latency experiment: use agent.py's AGENT_INSTRUCTION_TERSE "
                          "(~1/3 the words, no justification content) instead of the default "
                          "instruction, to test whether a shorter instruction cuts down on "
                          "preamble reasoning before the tool call, not just prompt-token count.")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    instance_files = sorted((Path(args.data) / args.split).glob("*/instance.json"))
    if args.load_factor:
        instance_files = [f for f in instance_files if f"_{args.load_factor}_" in f.parent.name]
    if args.limit:
        instance_files = instance_files[:args.limit]
    print(f"Evaluating {len(instance_files)} instances from {args.split}"
          f"{f' (load_factor={args.load_factor})' if args.load_factor else ''}")

    logger = JsonlLogger(args.out, mode="w")  # see slm_future/s1_lora/evaluate.py for why "w" not "a"
    for instance_file in instance_files:
        with open(instance_file, encoding="utf-8") as fh:
            instance = json.load(fh)

        topology = Topology.from_dict(instance["topology_after"])
        topology_before = topology.pre_addition_subview(max_step=0)
        base_flows_all = [Flow.from_dict(d) for d in instance["base_flows"]]
        admitted_ids = set(instance["base_admitted_flow_ids"])
        admitted_base_flows = [f for f in base_flows_all if f.flow_id in admitted_ids]
        new_flows = [Flow.from_dict(d) for d in instance["new_flows"]]
        new_node_ids = set(instance["new_node_ids"])
        gt = instance["ground_truth"]

        if args.no_tool:
            # Control condition: no solver call at all (nothing to seed it
            # with), so use S1's exact prompt/parser and the dataset's
            # stored ground-truth offsets for the fixed base schedule --
            # same reference S1's own evaluate.py checks feasibility
            # against, since neither method ever calls a solver.
            prompt_text = instance["prompt"]["prompt_text"]
            chat_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            total_time_s = time.perf_counter() - t0
            generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            predicted = parse_completion(generated)
            agent_result = {
                "used_tool": False, "tool_calls_made": 0, "requested_flow_count": None,
                "candidate_flow_count": len(new_flows), "hallucinated_flow_id_count": 0,
                "llm_time_s": total_time_s, "solver_time_s": 0.0, "last_generated_text": generated,
            }
            fixed_offsets_ns = {fid: off for fid, off in gt["offsets_ns"].items() if fid in admitted_ids}
            feasibility = check_feasibility(topology, admitted_base_flows, fixed_offsets_ns, new_flows, predicted)
        else:
            # previous (pre-addition) schedule, needed both by the tool itself
            # (see tool_schema.solve_subschedule_tool -> solve_b2) and by the
            # feasibility checker below, same reconstruction eval/run_baselines
            # and S1's evaluate.py use.
            previous_result, _paths, _hp = solve_b1(topology_before, admitted_base_flows, time_limit_s=20.0)

            t0 = time.perf_counter()
            agent_result = run_agent(
                model, tokenizer, topology, new_flows, admitted_base_flows, previous_result,
                new_node_ids, max_tool_calls=args.max_tool_calls, max_new_tokens=args.max_new_tokens,
                solver_time_limit_s=args.solver_time_limit_s, terse_prompt=args.terse_prompt,
            )
            total_time_s = time.perf_counter() - t0

            predicted = agent_result["decisions"]
            # Feasibility must be checked against the TRUE final schedule the
            # tool actually produced (agent_result["final_schedule"]), not the
            # pre-call previous_result and not the dataset's stored
            # ground_truth offsets. Two reasons: (1) CP-SAT can legitimately
            # pick different offsets for the same admitted base-flow set across
            # separate solves (multiple optima), so gt's offsets aren't the
            # schedule this call was ever placed alongside; (2) solve_b2 can
            # reposition (or even un-admit) previously-admitted flows within the
            # new node's k-hop neighborhood as part of incremental
            # re-optimization -- using the stale pre-call previous_result would
            # miss that and check against offsets the base flows no longer sit at.
            final_schedule = agent_result["final_schedule"]
            fixed_offsets_ns = {fid: off for fid, off in final_schedule.offsets_ns.items() if fid in admitted_ids}
            feasibility = check_feasibility(topology, admitted_base_flows, fixed_offsets_ns, new_flows, predicted)

        n_flows = len(new_flows)
        n_admission_correct = sum(
            1 for f in new_flows
            if f.flow_id in predicted and predicted[f.flow_id]["admitted"] == (f.flow_id in gt["admitted"])
        )
        # Restrict to admitted flows on both sides of the ratio -- a
        # rejected flow is trivially "feasible" per check_feasibility, so
        # counting it in n_feasible while dividing by n_predicted_admitted
        # would credit rejections as valid schedules (this exact bug
        # produced nonsensical >100% ratios like "22 feasible / 5 admitted"
        # on the first real run).
        n_predicted_admitted = sum(1 for d in predicted.values() if d["admitted"])
        n_feasible = sum(1 for fid, v in feasibility.items() if v and predicted.get(fid, {}).get("admitted"))

        logger.log({
            "instance_id": instance["instance_id"],
            "split": args.split,
            "switch_count": instance["switch_count"],
            "n_flows": n_flows,
            "admission_accuracy": n_admission_correct / n_flows if n_flows else None,
            "feasibility_rate": n_feasible / n_predicted_admitted if n_predicted_admitted else None,
            "used_tool": agent_result["used_tool"],
            "tool_calls_made": agent_result["tool_calls_made"],
            "requested_flow_count": agent_result["requested_flow_count"],
            "candidate_flow_count": agent_result["candidate_flow_count"],
            "hallucinated_flow_id_count": agent_result["hallucinated_flow_id_count"],
            "raw_generated_text": agent_result["last_generated_text"] if not agent_result["used_tool"] else None,
            "llm_time_s": agent_result["llm_time_s"],
            "solver_time_s": agent_result["solver_time_s"],
            "generation_time_s": total_time_s,
        })
        feasible_str = f"{n_feasible}/{n_predicted_admitted}" if n_predicted_admitted else "n/a (nothing admitted)"
        print(f"{instance['instance_id']}: used_tool={agent_result['used_tool']} "
              f"calls={agent_result['tool_calls_made']} "
              f"requested={agent_result['requested_flow_count']}/{n_flows} "
              f"admission_acc={n_admission_correct/n_flows:.2f} "
              f"feasible={feasible_str} "
              f"time={total_time_s:.1f}s")

    logger.close()
    print(f"Done. Wrote results to {args.out}")


if __name__ == "__main__":
    main()
