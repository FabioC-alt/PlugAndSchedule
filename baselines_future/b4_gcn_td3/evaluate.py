"""B4 evaluation: run a trained GCN-TD3 agent greedily (no exploration
noise) over a data split, scored the same way as every SLM method --
admission accuracy against the B1 ground truth, and feasibility rate
(collision/deadline checked via `slm_future.s1_lora.evaluate.check_feasibility`,
the same feasibility checker S1/S2/S2b/S3 all reuse) -- plus B4-specific
metrics: how many ILP-fallback resets (epochs) the instance took, and the
final GCL length reached.

Needs a GPU. Run via scripts/cluster/submit_b4_eval.sbatch on disi.

Usage:
    python -m baselines_future.b4_gcn_td3.evaluate \
        --data data --split held_out_test \
        --checkpoint baselines_future/b4_gcn_td3/checkpoints/b4.pt \
        --out results/b4_eval_held_out_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from baselines_future.b4_gcn_td3.env import GCL_THRESHOLD_OMEGA, B4Environment
from baselines_future.b4_gcn_td3.train import _to_action
from baselines_future.b4_gcn_td3.td3 import TD3Agent
from slm_future.s1_lora.evaluate import check_feasibility
from tsn_bench.flows.models import Flow
from tsn_bench.logging_utils import JsonlLogger
from tsn_bench.solvers.b1_full_ilp import solve_b1
from tsn_bench.topology.models import Topology


def run_instance(instance: dict, agent: TD3Agent, gcl_threshold: int) -> dict:
    topology = Topology.from_dict(instance["topology_after"])
    topology_before = topology.pre_addition_subview(max_step=0)
    base_flows_all = [Flow.from_dict(d) for d in instance["base_flows"]]
    admitted_ids = set(instance["base_admitted_flow_ids"])
    admitted_base_flows = [f for f in base_flows_all if f.flow_id in admitted_ids]
    new_flows = sorted([Flow.from_dict(d) for d in instance["new_flows"]], key=lambda f: f.arrival_order)

    previous_result, _paths, _hp = solve_b1(topology_before, admitted_base_flows, time_limit_s=20.0)
    env = B4Environment(topology, admitted_base_flows, previous_result, gcl_threshold=gcl_threshold)

    decisions: dict[str, dict] = {}
    fallback_count = 0
    for flow in new_flows:
        state_graph = env.state_graph()
        raw_action = agent.select_action(state_graph, explore=False)
        low, high = env.valid_offset_range(flow)

        if flow.flow_class.value == "avb":
            _reward, done, info = env.step(flow, accept=False)
        else:
            accept, offset_ns = _to_action(raw_action, (low, high))
            _reward, done, info = env.step(flow, accept=accept, offset_ns=offset_ns)
            decisions[flow.flow_id] = {
                "admitted": info["admitted"],
                "offset_us": (offset_ns / 1000.0) if info["admitted"] else None,
            }

        if done:
            fallback_count += 1
            fallback_result, _fp, _hp2 = solve_b1(topology, env.all_flows_seen, time_limit_s=30.0)
            admitted_flows_for_seed = [f for f in env.all_flows_seen if f.flow_id in fallback_result.admitted]
            env = B4Environment(topology, admitted_flows_for_seed, fallback_result, gcl_threshold=gcl_threshold)
            # A flow that triggered the fallback keeps whatever decision it
            # was just given above (the fallback recomputes going forward,
            # same semantics as the paper's "reschedule and reintegrate,"
            # not a retroactive rewrite of this flow's own decision) -- but
            # see the reconciliation pass below for why the decision's
            # RECORDED VALUES can't be trusted as-is once a later fallback
            # has happened.

    # Reconcile every decision against the TRUE final schedule. A decision
    # is recorded the moment it's made, against whatever schedule state
    # existed then -- but if a fallback follows later in the SAME instance
    # (recomputing the whole schedule from scratch via solve_b1), an
    # earlier flow's recorded offset can go stale: the final schedule may
    # have moved that flow to a different offset, or dropped it from the
    # admitted set entirely. Without this reconciliation, feasibility
    # checking compares a stale offset against a schedule that's since
    # moved on, producing spurious collisions that have nothing to do with
    # policy quality -- confirmed via 8/226 real instances (all 2-switch
    # near_saturation, 100+ fallbacks each) scoring as low as 5% feasible
    # purely from this effect (see PROGRESS.md 2026-09-25).
    final_schedule = env.schedule
    for flow_id, decision in decisions.items():
        if not decision["admitted"]:
            continue
        if flow_id not in final_schedule.admitted:
            decision["admitted"] = False
            decision["offset_us"] = None
        elif flow_id in final_schedule.offsets_ns:
            decision["offset_us"] = final_schedule.offsets_ns[flow_id] / 1000.0

    return {
        "decisions": decisions,
        "final_schedule": final_schedule,
        "fallback_count": fallback_count,
        "final_gcl_max_entries": env.gcl_max_entries(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained B4 GCN-TD3 agent")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="held_out_test", choices=["train", "val", "test", "held_out_test"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gcl-threshold", type=int, default=GCL_THRESHOLD_OMEGA)
    ap.add_argument("--graph-embedding-dim", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--load-factor", default=None, choices=["low", "medium", "near_saturation"])
    ap.add_argument("--out", default="results/b4_eval.jsonl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    agent = TD3Agent(graph_embedding_dim=args.graph_embedding_dim, device=args.device)
    agent.load(args.checkpoint)

    instance_files = sorted((Path(args.data) / args.split).glob("*/instance.json"))
    if args.load_factor:
        instance_files = [f for f in instance_files if f"_{args.load_factor}_" in f.parent.name]
    if args.limit:
        instance_files = instance_files[: args.limit]
    print(f"Evaluating {len(instance_files)} instances from {args.split}")

    logger = JsonlLogger(args.out, mode="w")
    for instance_file in instance_files:
        with open(instance_file, encoding="utf-8") as fh:
            instance = json.load(fh)
        gt = instance["ground_truth"]
        new_flows = [Flow.from_dict(d) for d in instance["new_flows"]]
        base_flows_all = [Flow.from_dict(d) for d in instance["base_flows"]]
        admitted_ids = set(instance["base_admitted_flow_ids"])
        admitted_base_flows = [f for f in base_flows_all if f.flow_id in admitted_ids]
        topology = Topology.from_dict(instance["topology_after"])

        t0 = time.perf_counter()
        out = run_instance(instance, agent, args.gcl_threshold)
        decision_time_s = time.perf_counter() - t0

        predicted = out["decisions"]
        final_schedule = out["final_schedule"]
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
            "fallback_count": out["fallback_count"],
            "final_gcl_max_entries": out["final_gcl_max_entries"],
            "decision_latency_s": decision_time_s,
            # raw counts, not scored against B1 ground truth -- added to
            # compute a RAW admission rate (admitted/offered) comparable to
            # the source paper's own reported metric (arXiv:2405.05019
            # reports "~90% admission rate" as scheduled/requested flows
            # during training, not accuracy against an independent solver).
            "n_admitted": n_predicted_admitted,
            "n_gt_admitted": len(gt["admitted"]),
        })
        feasible_str = f"{n_feasible}/{n_predicted_admitted}" if n_predicted_admitted else "n/a (nothing admitted)"
        print(f"{instance['instance_id']}: admission_acc={n_admission_correct/n_flows:.2f} "
              f"feasible={feasible_str} fallbacks={out['fallback_count']} time={decision_time_s:.1f}s")

    logger.close()
    print(f"Done. Wrote results to {args.out}")


if __name__ == "__main__":
    main()
