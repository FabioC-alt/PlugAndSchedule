"""S3/GraphRAG evaluation: identical to S2's evaluate.py (same base model,
same tool, same scoring) except the agent call goes through
`run_agent_with_retrieval` instead of `run_agent`, so the model's prompt
describes only the k-hop-retrieved subtopology. Every metric S2 logs is
logged here too, plus the retrieval-specific ones (`k_hop`,
`retrieved_node_count`, `full_node_count`, `context_reduction_pct`) needed
to answer the README's question: does a shorter, focused context change
admission accuracy / feasibility / decision latency versus S2's full-topology
prompt, holding the base model and the tool itself constant?

Needs a GPU. Run via scripts/cluster/submit_s3_eval.sbatch on disi.

Usage:
    python -m slm_future.s3_graphrag.evaluate \
        --data data --split held_out_test --k-hop 2 \
        --out results/s3_eval_held_out.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from slm_future.s1_lora.evaluate import check_feasibility
from slm_future.s3_graphrag.agent import run_agent_with_retrieval
from tsn_bench.flows.models import Flow
from tsn_bench.logging_utils import JsonlLogger
from tsn_bench.solvers.b1_full_ilp import solve_b1
from tsn_bench.topology.models import Topology


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the S3 GraphRAG (retrieval-narrowed) tool-calling agent")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="held_out_test",
                     choices=["train", "val", "test", "held_out_test"])
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", default="results/s3_eval.jsonl")
    ap.add_argument("--k-hop", type=int, default=0,
                     help="retrieval radius around each new flow's route + the new node(s) "
                          "-- see agent.py::retrieval_node_ids for the center-node set. "
                          "Default 0 (no buffer beyond the routes themselves): local "
                          "measurement on held_out_test found k_hop>=1 keeps 91-100% of "
                          "nodes once flow routes are counted correctly (a route can span "
                          "most of the graph even for small flow counts), so only k_hop=0 "
                          "gives a real context reduction (39-75% of nodes kept, avg across "
                          "load factors) -- see PROGRESS.md for the full measurement.")
    ap.add_argument("--max-tool-calls", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--solver-time-limit-s", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=None, help="evaluate at most N instances")
    ap.add_argument("--load-factor", default=None, choices=["low", "medium", "near_saturation"],
                     help="restrict to instances whose ID contains this load factor")
    ap.add_argument("--terse-prompt", action="store_true")
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
    print(f"Evaluating {len(instance_files)} instances from {args.split} (k_hop={args.k_hop})"
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

        # previous (pre-addition) schedule, needed both by the tool itself
        # and by the feasibility checker below -- same reconstruction S2's
        # evaluate.py uses.
        previous_result, _paths, _hp = solve_b1(topology_before, admitted_base_flows, time_limit_s=20.0)

        t0 = time.perf_counter()
        agent_result = run_agent_with_retrieval(
            model, tokenizer, topology, new_flows, admitted_base_flows, previous_result,
            new_node_ids, k_hop=args.k_hop, max_tool_calls=args.max_tool_calls,
            max_new_tokens=args.max_new_tokens, solver_time_limit_s=args.solver_time_limit_s,
            terse_prompt=args.terse_prompt,
        )
        total_time_s = time.perf_counter() - t0

        predicted = agent_result["decisions"]
        # same reasoning as S2: check against the TRUE final schedule the
        # tool actually produced, not the dataset's stored ground_truth
        # offsets or the stale pre-call previous_result.
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

        retrieved_n = agent_result["retrieved_node_count"]
        full_n = agent_result["full_node_count"]

        logger.log({
            "instance_id": instance["instance_id"],
            "split": args.split,
            "switch_count": instance["switch_count"],
            "n_flows": n_flows,
            "k_hop": args.k_hop,
            "retrieved_node_count": retrieved_n,
            "full_node_count": full_n,
            "context_reduction_pct": 100.0 * (1 - retrieved_n / full_n) if full_n else None,
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
        print(f"{instance['instance_id']}: nodes={retrieved_n}/{full_n} used_tool={agent_result['used_tool']} "
              f"calls={agent_result['tool_calls_made']} "
              f"requested={agent_result['requested_flow_count']}/{n_flows} "
              f"admission_acc={n_admission_correct/n_flows:.2f} "
              f"feasible={feasible_str} "
              f"time={total_time_s:.1f}s")

    logger.close()
    print(f"Done. Wrote results to {args.out}")


if __name__ == "__main__":
    main()
