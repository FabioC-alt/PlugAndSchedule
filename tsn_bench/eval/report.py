"""Report CLI (research plan Phase 5.3): aggregates results/baselines.jsonl
into per-(method, switch_count, load_factor) summary tables and runs paired
Wilcoxon signed-rank tests between methods on admission rate and decision
latency, matched per-instance (same topology+flow instance across methods).

Usage:
    python -m tsn_bench.eval.report --results results/baselines.jsonl --out results/report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from tsn_bench.logging_utils import read_jsonl

METRIC_COLUMNS = [
    "admission_rate", "deadline_compliance_rate", "mean_latency_us", "jitter_us",
    "gcl_total_entries", "gcl_max_per_link", "decision_latency_s",
]


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distribution"] = df["split"].apply(lambda s: "held_out" if s == "held_out_test" else "in_distribution")
    grouped = df.groupby(["method", "distribution", "switch_count", "load_factor_name"])[METRIC_COLUMNS]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{c}_{stat}" for c, stat in summary.columns]
    return summary.reset_index()


def paired_wilcoxon(df: pd.DataFrame, metric: str, method_a: str, method_b: str) -> dict:
    """Paired per-instance comparison of `metric` between two methods."""
    pivot = df.pivot_table(index="instance_id", columns="method", values=metric)
    if method_a not in pivot.columns or method_b not in pivot.columns:
        return {"metric": metric, "method_a": method_a, "method_b": method_b, "error": "missing method"}
    paired = pivot[[method_a, method_b]].dropna()
    if len(paired) < 2 or (paired[method_a] == paired[method_b]).all():
        return {
            "metric": metric, "method_a": method_a, "method_b": method_b,
            "n_pairs": len(paired), "statistic": None, "p_value": None,
            "note": "insufficient variation for Wilcoxon test",
        }
    stat, p = wilcoxon(paired[method_a], paired[method_b])
    return {
        "metric": metric, "method_a": method_a, "method_b": method_b,
        "n_pairs": len(paired), "statistic": float(stat), "p_value": float(p),
        "mean_a": float(paired[method_a].mean()), "mean_b": float(paired[method_b].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate baseline results and run statistical tests")
    ap.add_argument("--results", default="results/baselines.jsonl")
    ap.add_argument("--out", default="results/report")
    args = ap.parse_args()

    records = read_jsonl(args.results)
    if not records:
        print(f"No records found in {args.results} — run eval.run_baselines first.")
        return
    df = pd.DataFrame(records)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary_table(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    summary.to_markdown(out_dir / "summary.md", index=False)
    print(f"Wrote summary table ({len(summary)} rows) to {out_dir/'summary.csv'}")

    methods = sorted(df["method"].unique())
    tests = []
    for metric in ("admission_rate", "decision_latency_s"):
        for i, method_a in enumerate(methods):
            for method_b in methods[i + 1:]:
                tests.append(paired_wilcoxon(df, metric, method_a, method_b))
    with open(out_dir / "wilcoxon_tests.json", "w", encoding="utf-8") as fh:
        json.dump(tests, fh, indent=2)
    print(f"Wrote {len(tests)} Wilcoxon test results to {out_dir/'wilcoxon_tests.json'}")


if __name__ == "__main__":
    main()
