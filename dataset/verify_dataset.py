"""Check a dataset directory against the shipped fingerprints.

The fingerprint of an instance covers what the seeded generators produce (topology, base flows, new
flows, new nodes), not the solver outputs (CP-SAT schedules can differ between machines and threads).

    python dataset/verify_dataset.py <data-dir>             # compare with fingerprints.jsonl
    python dataset/verify_dataset.py <data-dir> --verbose   # list every differing or missing instance
    python dataset/verify_dataset.py <data-dir> --write     # (maintainers) write fingerprints.jsonl
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
PARTS = ("instance_id", "seed", "scenario", "load_factor_name", "switch_count",
         "topology_after", "new_node_ids", "base_flows", "new_flows")


def fingerprints(data_dir):
    out = {}
    for f in sorted(Path(data_dir).glob("*/*/instance.json")):
        inst = json.loads(f.read_text(encoding="utf-8"))
        blob = json.dumps({k: inst[k] for k in PARTS}, sort_keys=True, separators=(",", ":"))
        out[inst["instance_id"]] = (f.parent.parent.name, hashlib.sha256(blob.encode()).hexdigest())
    return out


def main():
    data_dir, write = sys.argv[1], "--write" in sys.argv
    fp = fingerprints(data_dir)
    path = HERE / "fingerprints.jsonl"
    if write:
        path.write_text("".join(json.dumps({"instance_id": i, "split": s, "sha256": h}) + "\n" for i, (s, h) in sorted(fp.items())))
        print(f"wrote {len(fp)} fingerprints")
        return
    ref = {r["instance_id"]: r for r in map(json.loads, path.read_text().splitlines())}
    missing, changed = sorted(set(ref) - set(fp)), [i for i in ref if i in fp and fp[i][1] != ref[i]["sha256"]]
    extra = sorted(set(fp) - set(ref))
    print(f"{len(ref) - len(missing) - len(changed)} of {len(ref)} instances match; "
          f"{len(missing)} missing, {len(changed)} different, {len(extra)} not in the reference")
    for label, ids in (("different", changed), ("missing", missing), ("not in the reference", extra)):
        shown = ids if "--verbose" in sys.argv else ids[:10]
        for i in shown:
            print(f"  {label}: {i}")
        if len(ids) > len(shown):
            print(f"  ... {len(ids) - len(shown)} more {label} (use --verbose to list all)")
    sys.exit(1 if missing or changed else 0)


if __name__ == "__main__":
    main()
