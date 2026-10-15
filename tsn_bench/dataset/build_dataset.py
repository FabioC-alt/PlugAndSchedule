"""Dataset generator CLI (research plan Phase 3).

Generates node-addition scheduling instances stratified by topology size,
with an explicit held-out-node-count test split for zero-shot generalization
evaluation. Ground truth for each instance is the B1 full-ILP schedule of
the post-addition topology with the union of (previously admitted) base
flows and newly requested flows — the same "recompute everything" solve
that also serves as baseline B1 in the evaluation matrix (eval/run_baselines.py).

Usage:
    python -m tsn_bench.dataset.build_dataset --config configs/default.yaml --out data
    python -m tsn_bench.dataset.build_dataset --config configs/default.yaml --out data --quick
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import yaml

from tsn_bench.dataset.splits import assign_split
from tsn_bench.flows.generator import generate_flows
from tsn_bench.flows.models import FlowClass
from tsn_bench.logging_utils import JsonlLogger
from tsn_bench.scheduling.metrics import compute_metrics
from tsn_bench.serialization.graph_format import to_graph_dict
from tsn_bench.serialization.text_format import build_completion, build_prompt
from tsn_bench.solvers.b1_full_ilp import solve_b1
from tsn_bench.topology.generator import add_node_to_topology, generate_topology
from tsn_bench.topology.models import NodeType


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def stable_seed(*parts) -> int:
    """Deterministic seed derived from `parts`, stable across processes/runs
    (unlike builtin hash(), which is salted per-process for strings)."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def build_instance(
    cfg: dict,
    switch_count: int,
    topo_idx: int,
    load_factor_name: str,
    load_factor_value: float,
    scenario: str,
    seed: int,
    time_limit_s: float,
) -> dict:
    rng = random.Random(seed)
    tt_profiles = cfg["flows"]["tt_profiles"]
    avb_profile = {
        "size_bytes_range": tuple(cfg["flows"]["avb_profile"]["size_bytes_range"]),
        "period_us": cfg["flows"]["avb_profile"]["period_us"],
        "deadline_us": cfg["flows"]["avb_profile"]["deadline_us"],
    }
    avb_fraction = cfg["flows"]["avb_fraction"]
    link_rate = cfg["topology"]["link_rate_mbps"]
    # only the scaling probe (configs/scaling_probe.yaml) sets this above the
    # generator's default -- its load factors deliberately exceed 1.0 to
    # stress-test the ILP solvers, which needs more than 2000 flow-generation
    # attempts to reach; the main dataset never approaches that cap.
    max_flow_attempts = cfg["flows"].get("max_attempts", 2000)
    # opt-in aggregate-capacity scaling, see generator.py's module docstring
    # and configs/aggregate_scaling.yaml -- None preserves the original
    # single-link-relative behavior byte-for-byte.
    capacity_scale_divisor = cfg["flows"].get("capacity_scale_divisor")

    topology_id = f"topo_{switch_count}sw_{topo_idx}_{load_factor_name}_{scenario}_seed{seed}"
    topo = generate_topology(
        switch_count,
        tuple(cfg["topology"]["talkers_per_switch"]),
        tuple(cfg["topology"]["listeners_per_switch"]),
        link_rate, rng, topology_id,
    )

    def capacity_scale_links(t) -> int | None:
        if not capacity_scale_divisor:
            return None
        n_links = sum(1 for lid in t.links if not lid.endswith("__rev"))
        return max(1, round(n_links / capacity_scale_divisor))

    base_flows = generate_flows(
        topo, tt_profiles, avb_profile, avb_fraction, link_rate, load_factor_value, rng,
        id_prefix="base", max_attempts=max_flow_attempts,
        capacity_scale_links=capacity_scale_links(topo),
    )
    base_result, base_paths, base_hp = solve_b1(topo, base_flows, time_limit_s=time_limit_s)
    admitted_base_flows = [f for f in base_flows if f.flow_id in base_result.admitted]

    if scenario == "single":
        node_types = [rng.choice(list(NodeType))]
    else:
        batch_size = rng.randint(*cfg["node_addition"]["batch_size_range"])
        node_types = [rng.choice(list(NodeType)) for _ in range(batch_size)]

    new_node_ids: set[str] = set()
    for i, nt in enumerate(node_types, start=1):
        created = add_node_to_topology(topo, nt, rng, step=i, link_rate_mbps=link_rate)
        new_node_ids.update(created)

    new_flow_load_factor = load_factor_value * 0.3  # smaller request volume than the base network
    new_flows = generate_flows(
        topo, tt_profiles, avb_profile, avb_fraction, link_rate, new_flow_load_factor, rng,
        restrict_src_dst_to=new_node_ids, id_prefix="new", added_at_step=1,
        start_arrival_order=len(base_flows), max_attempts=max_flow_attempts,
        capacity_scale_links=capacity_scale_links(topo),  # topo now includes the added node(s)
    )

    all_flows = admitted_base_flows + new_flows
    t0 = time.perf_counter()
    ground_truth, gt_paths, gt_hp = solve_b1(topo, all_flows, time_limit_s=time_limit_s)
    gt_metrics = compute_metrics("ground_truth_b1", ground_truth, gt_paths, gt_hp)

    prompt = build_prompt(topo, new_flows)
    completion = build_completion(new_flows, ground_truth, gt_paths)
    graph_before = to_graph_dict(
        # reconstruct pre-addition topology snapshot isn't stored separately;
        # graph_after (current `topo`) already flags newly-added nodes via
        # added_at_step, which is sufficient for the GNN feature schema.
        topo, all_flows,
    )

    return {
        "instance_id": topology_id,
        "switch_count": switch_count,
        "load_factor_name": load_factor_name,
        "load_factor_value": load_factor_value,
        "scenario": scenario,
        "seed": seed,
        "topology_after": topo.to_dict(),
        "new_node_ids": sorted(new_node_ids),
        "base_flows": [f.to_dict() for f in base_flows],
        "base_admitted_flow_ids": sorted(base_result.admitted),
        "new_flows": [f.to_dict() for f in new_flows],
        "prompt": prompt,
        "completion": completion,
        "graph": graph_before,
        "ground_truth": {
            "admitted": sorted(ground_truth.admitted),
            "rejected": sorted(ground_truth.rejected),
            "offsets_ns": ground_truth.offsets_ns,
            "hyperperiod_ns": gt_hp,
            "solver_status": ground_truth.solver_status,
            "metrics": gt_metrics.to_dict(),
            "generation_time_s": time.perf_counter() - t0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the TSN node-addition scheduling dataset")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="data")
    ap.add_argument("--quick", action="store_true",
                     help="tiny override for smoke-testing: 1 instance/size, 1 seed, 1 load factor")
    ap.add_argument("--time-limit-s", type=float, default=30.0,
                     help="CP-SAT time limit per solve")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # "w": each full run of this CLI produces a complete manifest of what it
    # generated. Instance files themselves are overwritten idempotently on
    # matching instance_id, but the manifest must not accumulate stale
    # append-only lines from earlier (e.g. --quick smoke-test) runs into the
    # same --out directory.
    manifest = JsonlLogger(out_dir / "manifest.jsonl", mode="w")

    switch_counts = (cfg["topology"]["in_distribution_switch_counts"]
                      + cfg["topology"]["held_out_switch_counts"])
    load_factors = cfg["flows"]["load_factors"]
    scenarios = cfg["node_addition"]["scenarios"]
    instances_per_size = cfg["topology"]["instances_per_size"]
    seeds_per_config = cfg["seeds_per_config"]

    if args.quick:
        load_factors = {"medium": load_factors["medium"]}
        scenarios = scenarios[:1]
        instances_per_size = 1
        seeds_per_config = 1

    counters = {"train": 0, "val": 0, "test": 0, "held_out_test": 0}
    total_start = time.perf_counter()
    n_built = 0

    for switch_count in switch_counts:
        for topo_idx in range(instances_per_size):
            for lf_name, lf_value in load_factors.items():
                for scenario in scenarios:
                    for seed_offset in range(seeds_per_config):
                        seed = stable_seed(switch_count, topo_idx, lf_name, scenario, seed_offset,
                                            cfg["seed"])
                        split = assign_split(switch_count, seed, cfg)
                        instance = build_instance(
                            cfg, switch_count, topo_idx, lf_name, lf_value, scenario, seed,
                            time_limit_s=args.time_limit_s,
                        )
                        split_dir = out_dir / split / instance["instance_id"]
                        split_dir.mkdir(parents=True, exist_ok=True)
                        with open(split_dir / "instance.json", "w", encoding="utf-8") as fh:
                            json.dump(instance, fh)
                        manifest.log({
                            "instance_id": instance["instance_id"],
                            "split": split,
                            "switch_count": switch_count,
                            "load_factor_name": lf_name,
                            "scenario": scenario,
                            "seed": seed,
                            "num_base_flows": len(instance["base_flows"]),
                            "num_new_flows": len(instance["new_flows"]),
                            "ground_truth_admission_rate": instance["ground_truth"]["metrics"]["admission_rate"],
                        })
                        counters[split] += 1
                        n_built += 1
                        elapsed = time.perf_counter() - total_start
                        print(f"[{n_built}] built {instance['instance_id']} -> {split} "
                              f"({elapsed:.1f}s elapsed)")

    manifest.close()
    print(f"Done. {n_built} instances written to {out_dir}. Split counts: {counters}")


if __name__ == "__main__":
    main()
