"""S1 evaluation: generate scheduling decisions with the fine-tuned model and
score them two ways —

1. **Admission accuracy**: does the model's ADMIT/REJECT per flow match the
   ground-truth (B1-optimal) decision? Cheap proxy metric.
2. **Feasibility rate**: for flows the model says ADMIT with an offset, is
   that offset actually collision-free (against every other flow the model
   also admitted, and against the already-scheduled base flows) and within
   deadline? This is the metric that actually matters — it catches
   hallucinated/invalid schedules (research plan's stated failure mode for
   SLM-only scheduling) rather than just rewarding copying the one
   particular optimal solution the model was trained on (there can be many
   equally-valid offsets for the same instance).

Reuses tsn_bench's own interval/overlap logic (the same code B3's heuristic
uses) for the feasibility check, rather than re-implementing scheduling
validation — this is the "does it produce a schedule that would actually
work" check, not a second opinion from a different model of the network.

Needs a GPU. Run via scripts/cluster/submit_s1_train.sbatch (eval mode) on
disi.

Usage:
    python -m slm_future.s1_lora.evaluate \
        --data data --split held_out_test \
        --adapter-dir slm_future/s1_lora/checkpoints/run1/adapter \
        --out results/s1_eval_held_out.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tsn_bench.flows.models import Flow
from tsn_bench.logging_utils import JsonlLogger
from tsn_bench.scheduling.simulator import (
    LinkOccupancy,
    compute_hyperperiod_ns,
    flow_intervals,
    resolve_flow_path,
)
from tsn_bench.topology.models import Topology

DECISION_RE = re.compile(
    r"flow (?P<flow_id>\S+):\s*"
    r"(?:(?P<reject>REJECT)"
    r"|ADMIT(?:\s*\(best-effort AVB[^)]*\))?,?\s*"
    r"(?:offset=(?P<offset_us>[\d.]+)us,\s*)?queue=(?P<queue>\d+))"
)


def parse_completion(text: str) -> dict[str, dict]:
    """Parse the model's generated decisions into {flow_id: {admitted, offset_us}}."""
    decisions = {}
    for m in DECISION_RE.finditer(text):
        flow_id = m.group("flow_id")
        if m.group("reject"):
            decisions[flow_id] = {"admitted": False, "offset_us": None}
        else:
            offset = m.group("offset_us")
            decisions[flow_id] = {"admitted": True, "offset_us": float(offset) if offset else None}
    return decisions


def check_feasibility(
    topology: Topology,
    fixed_admitted_flows: list[Flow],
    fixed_offsets_ns: dict[str, int],
    new_flows: list[Flow],
    predicted: dict[str, dict],
) -> dict:
    """Returns {flow_id: bool} — True if that flow's predicted admission is
    collision-free and deadline-compliant against everything else admitted."""
    occupancy: dict[str, LinkOccupancy] = {}
    # Hyperperiod must be computed from only the flows that actually occupy
    # the schedule (fixed + genuinely admitted), not every candidate flow.
    # A rejected/never-requested candidate's period doesn't need to have
    # been considered by whatever produced `predicted` — but if one has a
    # larger period than anything actually admitted, including it here
    # would make this check verify MORE periodic repetitions than the
    # solver (e.g. S2's solve_b2 call, scoped only to what was requested)
    # ever guaranteed collision-freedom over, producing phantom "collisions"
    # at repetition indices nothing ever validated. Since all periods here
    # are harmonic, a smaller hyperperiod is always a safe sub-check of a
    # larger one, so under-including is fine; over-including is not.
    admitted_new_flows = [f for f in new_flows if predicted.get(f.flow_id, {}).get("admitted")]
    all_tt_paths = [resolve_flow_path(topology, f) for f in fixed_admitted_flows + admitted_new_flows
                     if f.flow_class.value == "tt"]
    hyperperiod_ns = compute_hyperperiod_ns(all_tt_paths) if all_tt_paths else 1

    for f in fixed_admitted_flows:
        if f.flow_class.value != "tt" or f.flow_id not in fixed_offsets_ns:
            continue
        fp = resolve_flow_path(topology, f)
        for link_id, s, e in flow_intervals(fp, fixed_offsets_ns[f.flow_id], hyperperiod_ns):
            occupancy.setdefault(link_id, LinkOccupancy()).insert(s, e)

    feasible: dict[str, bool] = {}
    by_id = {f.flow_id: f for f in new_flows}
    for flow_id, decision in predicted.items():
        f = by_id.get(flow_id)
        if f is None or not decision["admitted"]:
            feasible[flow_id] = True  # rejecting is always "feasible" (never invalid)
            continue
        if f.flow_class.value != "tt":
            feasible[flow_id] = True  # AVB admission has no offset to validate here
            continue
        if decision["offset_us"] is None:
            feasible[flow_id] = False  # claimed ADMIT for a TT flow but gave no offset
            continue
        fp = resolve_flow_path(topology, f)
        offset_ns = round(decision["offset_us"] * 1000)
        if offset_ns < 0 or offset_ns + fp.total_delay_ns > fp.deadline_ns:
            feasible[flow_id] = False
            continue
        intervals = flow_intervals(fp, offset_ns, hyperperiod_ns)
        collision = any(occupancy.get(link_id, LinkOccupancy()).overlaps(s, e) for link_id, s, e in intervals)
        feasible[flow_id] = not collision
        if not collision:
            for link_id, s, e in intervals:
                occupancy.setdefault(link_id, LinkOccupancy()).insert(s, e)
    return feasible


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the S1 LoRA-fine-tuned model")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="held_out_test",
                     choices=["train", "val", "test", "held_out_test"])
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter-dir", default=None, help="omit to evaluate the base model, no LoRA")
    ap.add_argument("--out", default="results/s1_eval.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                     help="near_saturation instances have 100+ flows; at ~11 tokens/completion "
                          "line the old default of 1024 truncated mid-list past ~90 flows "
                          "(confirmed at full scale, see PROGRESS.md) -- 4096 covers the largest "
                          "observed instances (~120 flows) with headroom")
    ap.add_argument("--limit", type=int, default=None, help="evaluate at most N instances")
    ap.add_argument("--load-factor", default=None, choices=["low", "medium", "near_saturation"],
                     help="restrict to instances whose ID contains this load factor -- for "
                          "targeted re-runs (e.g. confirming a fix on just near_saturation) "
                          "without a full-split run")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    instance_files = sorted((Path(args.data) / args.split).glob("*/instance.json"))
    if args.load_factor:
        instance_files = [f for f in instance_files if f"_{args.load_factor}_" in f.parent.name]
    if args.limit:
        instance_files = instance_files[:args.limit]
    print(f"Evaluating {len(instance_files)} instances from {args.split}"
          f"{f' (load_factor={args.load_factor})' if args.load_factor else ''}")

    # "w": each run of this CLI evaluates a complete split and should produce
    # a complete, fresh log of it -- not accumulate across reruns (bit us
    # once already: results/baselines.jsonl's manifest.jsonl equivalent, see
    # tsn_bench/dataset/build_dataset.py's fix and run_baselines.py's
    # existing unlink-before-open, which this now matches).
    logger = JsonlLogger(args.out, mode="w")
    for instance_file in instance_files:
        with open(instance_file, encoding="utf-8") as fh:
            instance = json.load(fh)

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
        gen_time_s = time.perf_counter() - t0
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        predicted = parse_completion(generated)
        new_flows = [Flow.from_dict(d) for d in instance["new_flows"]]
        gt = instance["ground_truth"]

        topology = Topology.from_dict(instance["topology_after"])
        base_flows_all = [Flow.from_dict(d) for d in instance["base_flows"]]
        admitted_ids = set(instance["base_admitted_flow_ids"])
        fixed_admitted_flows = [f for f in base_flows_all if f.flow_id in admitted_ids]
        fixed_offsets_ns = {fid: off for fid, off in gt["offsets_ns"].items() if fid in admitted_ids}

        feasibility = check_feasibility(topology, fixed_admitted_flows, fixed_offsets_ns, new_flows, predicted)

        n_flows = len(new_flows)
        n_parsed = len(predicted)
        n_admission_correct = sum(
            1 for f in new_flows
            if f.flow_id in predicted and predicted[f.flow_id]["admitted"] == (f.flow_id in gt["admitted"])
        )
        # Feasibility is only a meaningful question for flows the model
        # claimed to ADMIT (check_feasibility marks every rejected flow
        # trivially "feasible" since rejecting is never invalid) -- counting
        # those into the numerator while dividing by n_parsed would credit
        # rejections as if they were valid schedules and can even push the
        # ratio above 1.0 relative to the admitted count. Restrict both
        # sides to admitted flows.
        n_predicted_admitted = sum(1 for d in predicted.values() if d["admitted"])
        n_feasible = sum(1 for fid, v in feasibility.items() if v and predicted.get(fid, {}).get("admitted"))

        logger.log({
            "instance_id": instance["instance_id"],
            "split": args.split,
            "switch_count": instance["switch_count"],
            "n_flows": n_flows,
            "parse_success_rate": n_parsed / n_flows if n_flows else None,
            "admission_accuracy": n_admission_correct / n_flows if n_flows else None,
            "feasibility_rate": n_feasible / n_predicted_admitted if n_predicted_admitted else None,
            "generation_time_s": gen_time_s,
            "raw_generated_text": generated,
        })
        feasible_str = f"{n_feasible}/{n_predicted_admitted}" if n_predicted_admitted else "n/a (nothing admitted)"
        print(f"{instance['instance_id']}: parsed {n_parsed}/{n_flows}, "
              f"admission_acc={n_admission_correct/n_flows:.2f} feasible={feasible_str}")

    logger.close()
    print(f"Done. Wrote results to {args.out}")


if __name__ == "__main__":
    main()
