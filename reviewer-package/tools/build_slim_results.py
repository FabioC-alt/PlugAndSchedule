"""Maintainer tool: turn the raw cluster logs in <repo>/results/ into the slim,
per-instance files shipped in reviewer-package/results/.

Raw logs carry the model's full generated text (megabytes). The slim files keep
only what the paper's numbers need. Re-runs that replace truncated generations
are merged here, explicitly, so the merge rule is visible in one place.

Usage (from the repo root):  python reviewer-package/tools/build_slim_results.py
"""
import json
import sys
from pathlib import Path

RAW = Path("results")
OUT = Path("reviewer-package/results")
LOADS = ("low", "medium", "near_saturation")


def read(name):
    p = RAW / f"{name}.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def load_of(instance_id):
    return next(l for l in LOADS if f"_{l}_" in instance_id)


def merge(base, *replacements):
    """Rows of `base` whose instance_id is not in any replacement, plus all replacement rows."""
    replaced = {r["instance_id"] for rows in replacements for r in rows}
    return [r for r in base if r["instance_id"] not in replaced] + [r for rows in replacements for r in rows]


def slim(rows, latency_key):
    keep = []
    for r in rows:
        keep.append({
            "instance_id": r["instance_id"], "split": r["split"], "switch_count": r["switch_count"],
            "load": load_of(r["instance_id"]), "n_flows": r["n_flows"],
            "accuracy": r["admission_accuracy"], "feasibility": r["feasibility_rate"],
            "latency_s": r.get(latency_key), "used_tool": r.get("used_tool"),
        })
    return sorted(keep, key=lambda r: r["instance_id"])


def write(name, rows, expect):
    assert len(rows) == expect, f"{name}: {len(rows)} rows, expected {expect}"
    assert len({r["instance_id"] for r in rows}) == expect, f"{name}: duplicate instance ids"
    with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {name}: {expect} rows")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # S1: held-out run used max-new-tokens 4096. The test-split run of 2026-09-16 used 1024 and
    # truncated the medium and near-saturation answers, so those rows are replaced by re-runs.
    write("S1_held_out_test", slim(read("s1_eval_held_out_test_full_scale_3ep"), "generation_time_s"), 720)
    near, med = read("s1_eval_test_near_saturation_retest"), read("s1_eval_test_medium_retest")
    if near and med:
        write("S1_test", slim(merge(read("s1_eval_test"), near, med), "generation_time_s"), 226)
    else:
        print("S1_test NOT written: the re-run of the medium-load test instances is missing", file=sys.stderr)
    # S2: first runs used a 512-token budget, which truncated near-saturation tool calls.
    write("S2_held_out_test", slim(merge(read("s2_eval_held_out_test"), read("s2_eval_held_out_test_near_saturation_retest")), "generation_time_s"), 720)
    write("S2_test", slim(merge(read("s2_eval_test"), read("s2_eval_test_near_saturation_retest")), "generation_time_s"), 226)
    write("S2_no_tool_held_out_test", slim(read("s2_eval_held_out_test_no_tool"), "generation_time_s"), 720)
    write("S2b_held_out_test", slim(read("s2b_eval_held_out_test_full"), "generation_time_s"), 720)
    write("S2b_test", slim(read("s2b_eval_test_full"), "generation_time_s"), 226)
    write("S3_held_out_test", slim(read("s3_eval_held_out_test"), "generation_time_s"), 720)
    write("S3_test", slim(read("s3_eval_test"), "generation_time_s"), 226)
    write("S4_held_out_test", slim(read("s4_eval_held_out_test"), "generation_time_s"), 720)
    write("S4_test", slim(read("s4_eval_test"), "generation_time_s"), 226)
    write("B4_held_out_test", slim(read("b4_eval_held_out_test"), "decision_latency_s"), 720)
    write("B4_test", slim(read("b4_eval_test"), "decision_latency_s"), 226)
    base = [r for r in read("baselines") if r["split"] in ("test", "held_out_test")]
    with (OUT / "baselines.jsonl").open("w", encoding="utf-8") as fh:
        for r in sorted(base, key=lambda r: (r["method"], r["instance_id"])):
            fh.write(json.dumps({k: r[k] for k in ("instance_id", "split", "switch_count", "method", "admission_rate", "decision_latency_s")}) + "\n")
    print(f"wrote baselines: {len(base)} rows")


if __name__ == "__main__":
    main()
