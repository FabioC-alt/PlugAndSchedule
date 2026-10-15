"""Recompute every number in the paper's Results section from the shipped per-instance files.

Tables need only the Python standard library. The Wilcoxon p-values need scipy and are skipped
(with a message) if scipy is not installed:

    python reproduce.py            # print the tables, check them against expected.json
    python reproduce.py --latex    # also print the LaTeX rows of the main results table
"""
import argparse
import json
from pathlib import Path
from statistics import fmean

HERE = Path(__file__).parent
RES = HERE / "reported_results"
SPLITS = ("held_out_test", "test")
METHODS = ["S2b", "S1", "S3", "S2", "S4", "B4"]
LOADS = ["low", "medium", "near_saturation"]


def read(name):
    p = RES / f"{name}.jsonl"
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else None


def mean_pct(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]  # feasibility is None when nothing was admitted
    return 100 * fmean(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    got = {}  # key -> recomputed value, compared with expected.json at the end

    # ---- classical baselines: the admission rate covers the whole topology, so it is not comparable with accuracy
    base = read("baselines")
    print("== Classical baselines (admission %, latency s)")
    for m, name in [("B1_full_ilp", "B1"), ("B2_incremental_ilp", "B2"), ("B3_heuristic", "B3")]:
        for s in ("test", "held_out_test"):
            sub = [r for r in base if r["method"] == m and r["split"] == s]
            adm, lat = 100 * fmean(r["admission_rate"] for r in sub), fmean(r["decision_latency_s"] for r in sub)
            got[f"{name}.{s}.adm"], got[f"{name}.{s}.lat"] = adm, lat
            print(f"{name:3s} {s:14s} n={len(sub):3d}  adm {adm:6.2f}  lat {lat:.3f}")

    data = {(m, s): read(f"{m}_{s}") for m in METHODS for s in SPLITS}

    # ---- main results table
    print("\n== Main results: accuracy %, feasibility %, latency s")
    for m in METHODS:
        line = f"{m:4s}"
        for s in SPLITS:
            rows = data[(m, s)]
            if rows is None:
                line += f"  |  {s}: not shipped yet (see README, Known gaps)"
                continue
            acc, fe, lat = mean_pct(rows, "accuracy"), mean_pct(rows, "feasibility"), fmean(r["latency_s"] for r in rows)
            got.update({f"{m}.{s}.acc": acc, f"{m}.{s}.feas": fe, f"{m}.{s}.lat": lat})
            line += f"  |  {s}: n={len(rows)}  {acc:5.1f}  {fe:5.1f}  {lat:5.1f}"
        print(line)
    if args.latex:
        print("\n% LaTeX rows: held-out acc, feas, lat, then test acc, feas, lat")
        for m in METHODS:
            cells = []
            for s in SPLITS:
                k = f"{m}.{s}"
                cells += [f"{got[f'{k}.{x}']:.1f}" if f"{k}.acc" in got else "TBD" for x in ("acc", "feas", "lat")]
            print(f"{m:3s} & " + " & ".join(cells) + r" \\")

    # ---- by load level, held-out test
    print("\n== Held-out test by load level: accuracy % / latency s")
    for m in METHODS:
        parts = []
        for l in LOADS:
            sub = [r for r in data[(m, "held_out_test")] if r["load"] == l]
            a, t = mean_pct(sub, "accuracy"), fmean(r["latency_s"] for r in sub)
            got[f"{m}.load.{l}.acc"], got[f"{m}.load.{l}.lat"] = a, t
            parts.append(f"{l}: {a:5.1f} / {t:5.1f}")
        print(f"{m:4s} " + "   ".join(parts))

    # ---- S4: runs that ended without a solver call, held-out near-saturation
    s4 = [r for r in data[("S4", "held_out_test")] if r["load"] == "near_saturation"]
    ok = [r for r in s4 if r["used_tool"]]
    got["S4.ns.no_solver_call_pct"] = 100 * (1 - len(ok) / len(s4))
    got["S4.ns.acc_when_solver_called"] = mean_pct(ok, "accuracy")
    print(f"\n== S4, held-out near-saturation: no solver call in {got['S4.ns.no_solver_call_pct']:.1f}% of {len(s4)} instances; "
          f"accuracy when the solver was called {got['S4.ns.acc_when_solver_called']:.1f}% (n={len(ok)})")

    # ---- S2 without the solver tool
    nt = read("S2_no_tool_held_out_test")
    usable = sum(1 for r in nt if (r["accuracy"] or 0) > 0 or r["used_tool"])
    got["S2.no_tool.usable"] = usable
    print(f"== S2 without the tool: {usable} of {len(nt)} held-out instances gave a usable answer")

    # ---- paired Wilcoxon signed-rank tests on per-instance accuracy, held-out test (needs scipy)
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("\n== Wilcoxon tests skipped: scipy is not installed (pip install scipy)")
    else:
        print("\n== Wilcoxon signed-rank on per-instance accuracy, held-out test")
        acc = {m: {r["instance_id"]: r["accuracy"] for r in data[(m, "held_out_test")]} for m in METHODS}
        for a, b in [("S2b", "S2"), ("S2b", "S3"), ("S2b", "S4"), ("S2b", "S1"), ("S2", "S3"), ("S2", "S4"), ("S3", "S4")]:
            ids = sorted(acc[a])
            p = wilcoxon([acc[a][i] for i in ids], [acc[b][i] for i in ids]).pvalue
            got[f"p.{a}.{b}"] = p
            print(f"{a:3s} vs {b:3s}  p = {p:.3g}")

    # ---- compare with the numbers printed in the paper
    # expected.json: key -> [value printed in the paper, tolerance]; a string such as "<1e-9" is an upper bound
    exp = json.loads((HERE / "expected.json").read_text())
    bad = 0
    print("\n== Check against the numbers printed in the paper")
    for k, (e, tol) in exp.items():
        if k not in got:
            print(f"  SKIP {k} (not computed in this run)")
            continue
        ok_ = got[k] < float(e[1:]) if isinstance(e, str) else abs(got[k] - e) <= tol
        if not ok_:
            bad += 1
            print(f"  FAIL {k}: paper {e}, recomputed {got[k]:.4g}")
    print("  all listed values match" if not bad else f"  {bad} mismatch(es)")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
