"""Eval CLI (research plan Phase 5): runs B1 (full ILP), B2 (incremental ILP),
and B3 (heuristic) on every generated instance and logs one JSONL record per
(instance, method) with the metrics from research-plan section 5.1.

For B2/B3 (incremental methods), the "previous schedule" is recomputed by
solving the pre-addition topology (`Topology.pre_addition_subview`) with the
admitted base flows — the dataset only persists which base flows were
admitted, not their offsets, so this is regenerated deterministically at
eval time rather than duplicated in every instance file.

Usage:
    python -m tsn_bench.eval.run_baselines --data data --out results/baselines.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from tsn_bench.flows.models import Flow
from tsn_bench.logging_utils import JsonlLogger
from tsn_bench.scheduling.metrics import Metrics, compute_metrics
from tsn_bench.solvers.b1_full_ilp import solve_b1
from tsn_bench.solvers.b2_incremental_ilp import solve_b2
from tsn_bench.solvers.b3_heuristic import solve_b3
from tsn_bench.topology.models import Topology


def load_instances(data_dir: Path, splits: list[str] | None = None) -> list[dict]:
    instances = []
    for instance_file in sorted(data_dir.glob("*/*/instance.json")):
        split = instance_file.parent.parent.name
        if splits and split not in splits:
            continue
        with open(instance_file, encoding="utf-8") as fh:
            instance = json.load(fh)
        instance["_split"] = split
        instances.append(instance)
    return instances


def make_record(instance: dict, method: str, metrics: Metrics) -> dict:
    return {
        "instance_id": instance["instance_id"],
        "split": instance["_split"],
        "switch_count": instance["switch_count"],
        "load_factor_name": instance["load_factor_name"],
        "scenario": instance["scenario"],
        "seed": instance["seed"],
        "method": method,
        **metrics.to_dict(),
    }


def evaluate_instance(instance: dict, time_limit_s: float) -> list[dict]:
    topo_after = Topology.from_dict(instance["topology_after"])
    topo_before = topo_after.pre_addition_subview(max_step=0)

    base_flows_all = [Flow.from_dict(d) for d in instance["base_flows"]]
    admitted_ids = set(instance["base_admitted_flow_ids"])
    admitted_base_flows = [f for f in base_flows_all if f.flow_id in admitted_ids]
    new_flows = [Flow.from_dict(d) for d in instance["new_flows"]]
    new_node_ids = set(instance["new_node_ids"])

    records = []

    # B1: recompute-everything on the post-addition network
    b1_result, b1_paths, b1_hp = solve_b1(
        topo_after, admitted_base_flows + new_flows, time_limit_s=time_limit_s,
    )
    records.append(make_record(instance, "B1_full_ilp",
                                compute_metrics("B1_full_ilp", b1_result, b1_paths, b1_hp)))

    # shared "previous schedule" (pre-addition) for the incremental methods
    prev_ilp_result, _, _ = solve_b1(topo_before, admitted_base_flows, time_limit_s=time_limit_s)

    # B2: incremental ILP, only re-solves the k-hop neighborhood of the new node(s)
    b2_result, b2_paths, b2_hp = solve_b2(
        topo_after, admitted_base_flows, prev_ilp_result, new_flows, new_node_ids,
        k_hop=1, time_limit_s=time_limit_s,
    )
    records.append(make_record(instance, "B2_incremental_ilp",
                                compute_metrics("B2_incremental_ilp", b2_result, b2_paths, b2_hp)))

    # B3: heuristic, incremental in the same spirit as B2 but with no solver
    heuristic_rng = random.Random(instance["seed"])
    prev_heuristic_result, _, _ = solve_b3(topo_before, admitted_base_flows, rng=heuristic_rng)
    b3_result, b3_paths, b3_hp = solve_b3(
        topo_after, new_flows, fixed_result=prev_heuristic_result,
        fixed_flows=admitted_base_flows, rng=heuristic_rng,
    )
    records.append(make_record(instance, "B3_heuristic",
                                compute_metrics("B3_heuristic", b3_result, b3_paths, b3_hp)))

    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Run B1-B3 baselines across the generated dataset")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/baselines.jsonl")
    ap.add_argument("--splits", nargs="*", default=None,
                     help="restrict to these splits (default: all)")
    ap.add_argument("--time-limit-s", type=float, default=30.0)
    args = ap.parse_args()

    data_dir = Path(args.data)
    instances = load_instances(data_dir, args.splits)
    print(f"Loaded {len(instances)} instances from {data_dir}")

    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()
    logger = JsonlLogger(out_path)

    t0 = time.perf_counter()
    for i, instance in enumerate(instances, start=1):
        records = evaluate_instance(instance, args.time_limit_s)
        for r in records:
            logger.log(r)
        elapsed = time.perf_counter() - t0
        print(f"[{i}/{len(instances)}] evaluated {instance['instance_id']} "
              f"({len(records)} methods, {elapsed:.1f}s elapsed)")
    logger.close()
    print(f"Done. Wrote {len(instances) * 3} records to {out_path}")


if __name__ == "__main__":
    main()
