"""S4 evaluation: run the two-tool agent (retrieve_subgraph optional +
solve_subschedule required) over a data split, scored the same way as
S1-S3 -- admission accuracy against B1 ground truth, and feasibility rate
via `slm_future.s1_lora.evaluate.check_feasibility`. Also logs whether the
model chose to retrieve on each instance, the actual research question
here: does the model's own choice about when to narrow its context change
quality or latency, versus S2 (never) and S3 (always)?

Needs a GPU. Run via scripts/cluster/submit_s4_eval.sbatch on disi.

Usage:
    python -m slm_future.s4_full_agent.evaluate \
        --data data --split held_out_test \
        --out results/s4_eval_held_out_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from slm_future.s1_lora.evaluate import check_feasibility
from slm_future.s4_full_agent.agent import run_agent
from tsn_bench.flows.models import Flow
from tsn_bench.logging_utils import JsonlLogger
from tsn_bench.solvers.b1_full_ilp import solve_b1
from tsn_bench.topology.models import Topology


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the S4 two-tool (retrieval + solver) agent")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="held_out_test", choices=["train", "val", "test", "held_out_test"])
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", default="results/s4_eval.jsonl")
    ap.add_argument("--max-tool-calls", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--solver-time-limit-s", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=None, help="evaluate at most N instances")
    ap.add_argument("--load-factor", default=None, choices=["low", "medium", "near_saturation"])
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

    logger = JsonlLogger(args.out, mode="w")
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

        previous_result, _paths, _hp = solve_b1(topology_before, admitted_base_flows, time_limit_s=20.0)

        t0 = time.perf_counter()
        agent_result = run_agent(
            model, tokenizer, topology, new_flows, admitted_base_flows, previous_result,
            new_node_ids, max_tool_calls=args.max_tool_calls, max_new_tokens=args.max_new_tokens,
            solver_time_limit_s=args.solver_time_limit_s,
        )
        total_time_s = time.perf_counter() - t0

        predicted = agent_result["decisions"]
        # same reasoning as S2/S3: check against the TRUE final schedule
        # the tool loop actually produced, not the dataset's stored
        # ground_truth offsets.
        final_schedule = agent_result["final_schedule"]
        fixed_offsets_ns = {fid: off for fid, off in final_schedule.offsets_ns.items() if fid in admitted_ids}
        feasibility = check_feasibility(topology, admitted_base_flows, fixed_offsets_ns, new_flows, predicted)

        n_flows = len(new_flows)
        n_admission_correct = sum(
            1 for f in new_flows
            if f.flow_id in predicted and predicted[f.flow_id]["admitted"] == (f.flow_id in gt["admitted"])
        )
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
            "retrieved": agent_result["retrieved"],
            "retrieval_calls_made": agent_result["retrieval_calls_made"],
            "requested_flow_count": agent_result["requested_flow_count"],
            "candidate_flow_count": agent_result["candidate_flow_count"],
            "hallucinated_flow_id_count": agent_result["hallucinated_flow_id_count"],
            "raw_generated_text": agent_result["last_generated_text"] if not agent_result["used_tool"] else None,
            "llm_time_s": agent_result["llm_time_s"],
            "solver_time_s": agent_result["solver_time_s"],
            "generation_time_s": total_time_s,
        })
        feasible_str = f"{n_feasible}/{n_predicted_admitted}" if n_predicted_admitted else "n/a (nothing admitted)"
        print(f"{instance['instance_id']}: retrieved={agent_result['retrieved']} "
              f"used_tool={agent_result['used_tool']} calls={agent_result['tool_calls_made']} "
              f"requested={agent_result['requested_flow_count']}/{n_flows} "
              f"admission_acc={n_admission_correct/n_flows:.2f} "
              f"feasible={feasible_str} "
              f"time={total_time_s:.1f}s")

    logger.close()
    print(f"Done. Wrote results to {args.out}")


if __name__ == "__main__":
    main()
