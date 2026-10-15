"""S2b evaluation: same two-metric scoring as S1/S2 (admission accuracy vs.
B1 ground truth, feasibility rate via the same collision/deadline check),
plus S2b-specific timing (OpenNod scoring time vs. solver time) and
coverage (requested_flow_count / candidate_flow_count, same definition as
S2's).

Prototyped 2026-09-19 on 10 instances (5 low-load, 5 near_saturation):
100% coverage/admission accuracy/feasibility on every instance, ~10x
faster than S2's tool-calling baseline (0.93s vs. 9.3s low-load, 3.35s vs.
~36s near_saturation). See ../s2_solver_tool/README.md's "OpenNod
typed-logits prototype" section and PROGRESS.md's 2026-09-19 entries for
the full writeup. This is the productionized version of that prototype
script -- same logic, S1/S2-style CLI and JSONL schema.

Needs a GPU and `semif_phase1` (OpenNod) importable -- see README.md for
the (separate-from-S1/S2) environment setup.

Usage:
    python -m slm_future.s2b_opennod.evaluate \
        --data data --split held_out_test \
        --out results/s2b_eval_held_out.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from slm_future.s1_lora.evaluate import check_feasibility
from slm_future.s2_solver_tool.tool_schema import solve_subschedule_tool
from slm_future.s2b_opennod.agent import select_flow_ids
from tsn_bench.flows.models import Flow
from tsn_bench.logging_utils import JsonlLogger
from tsn_bench.solvers.b1_full_ilp import solve_b1
from tsn_bench.topology.models import Topology

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the S2b OpenNod-typed-logits agent")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="held_out_test",
                     choices=["train", "val", "test", "held_out_test"])
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=DEFAULT_REVISION,
                     help="pinned per OpenNod's own README for reproducibility")
    ap.add_argument("--out", default="results/s2b_eval.jsonl")
    ap.add_argument("--solver-time-limit-s", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=None, help="evaluate at most N instances")
    ap.add_argument("--load-factor", default=None, choices=["low", "medium", "near_saturation"],
                     help="restrict to instances whose ID contains this load factor -- for "
                          "targeted re-runs without a full-split run, same as S2's evaluate.py")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, revision=args.revision, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    metadata = {"source": args.model_name, "revision": args.revision, "dtype": "bfloat16"}

    instance_files = sorted((Path(args.data) / args.split).glob("*/instance.json"))
    if args.load_factor:
        instance_files = [f for f in instance_files if f"_{args.load_factor}_" in f.parent.name]
    if args.limit:
        instance_files = instance_files[:args.limit]
    print(f"Evaluating {len(instance_files)} instances from {args.split}"
          f"{f' (load_factor={args.load_factor})' if args.load_factor else ''}")

    logger = JsonlLogger(args.out, mode="w")  # see s1_lora/evaluate.py for why "w" not "a"
    for instance_file in instance_files:
        with open(instance_file, encoding="utf-8") as fh:
            instance = json.load(fh)

        topology = Topology.from_dict(instance["topology_after"])
        topology_before = topology.pre_addition_subview(max_step=0)
        base_flows_all = [Flow.from_dict(d) for d in instance["base_flows"]]
        admitted_ids = set(instance["base_admitted_flow_ids"])
        admitted_base_flows = [f for f in base_flows_all if f.flow_id in admitted_ids]
        new_flows = [Flow.from_dict(d) for d in instance["new_flows"]]
        new_flows_by_id = {f.flow_id: f for f in new_flows}
        new_node_ids = set(instance["new_node_ids"])
        gt = instance["ground_truth"]

        previous_result, _paths, _hp = solve_b1(topology_before, admitted_base_flows, time_limit_s=20.0)

        t0 = time.perf_counter()
        requested_ids, opennod_timing = select_flow_ids(model, tokenizer, topology, new_flows, metadata)
        opennod_time_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        decisions, final_schedule = solve_subschedule_tool(
            topology, admitted_base_flows, previous_result, new_flows_by_id,
            requested_ids, new_node_ids, time_limit_s=args.solver_time_limit_s,
        )
        solver_time_s = time.perf_counter() - t0
        for fid in set(new_flows_by_id) - set(decisions.keys()):
            decisions[fid] = {"admitted": False, "offset_us": None}

        # same reasoning as S2's evaluate.py: check against the schedule the
        # solver actually just produced, not the pre-call one or the
        # dataset's stored ground truth (CP-SAT can pick different offsets
        # for the same admitted set across separate solves).
        fixed_offsets_ns = {fid: off for fid, off in final_schedule.offsets_ns.items() if fid in admitted_ids}
        feasibility = check_feasibility(topology, admitted_base_flows, fixed_offsets_ns, new_flows, decisions)

        n_flows = len(new_flows)
        n_admission_correct = sum(
            1 for f in new_flows
            if f.flow_id in decisions and decisions[f.flow_id]["admitted"] == (f.flow_id in gt["admitted"])
        )
        n_predicted_admitted = sum(1 for d in decisions.values() if d["admitted"])
        n_feasible = sum(1 for fid, v in feasibility.items() if v and decisions.get(fid, {}).get("admitted"))

        logger.log({
            "instance_id": instance["instance_id"],
            "split": args.split,
            "switch_count": instance["switch_count"],
            "n_flows": n_flows,
            "admission_accuracy": n_admission_correct / n_flows if n_flows else None,
            "feasibility_rate": n_feasible / n_predicted_admitted if n_predicted_admitted else None,
            "requested_flow_count": len(requested_ids),
            "candidate_flow_count": n_flows,
            "opennod_time_s": opennod_time_s,
            "solver_time_s": solver_time_s,
            "generation_time_s": opennod_time_s + solver_time_s,
        })
        feasible_str = f"{n_feasible}/{n_predicted_admitted}" if n_predicted_admitted else "n/a (nothing admitted)"
        print(f"{instance['instance_id']}: requested={len(requested_ids)}/{n_flows} "
              f"admission_acc={n_admission_correct/n_flows:.2f} "
              f"feasible={feasible_str} "
              f"time={opennod_time_s + solver_time_s:.2f}s "
              f"(opennod={opennod_time_s:.2f}s solver={solver_time_s:.2f}s)")

    logger.close()
    print(f"Done. Wrote results to {args.out}")


if __name__ == "__main__":
    main()
