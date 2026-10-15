"""Collect (prompt_text, completion_text) pairs out of the dataset generator's
instance.json files into a flat JSONL ready for supervised fine-tuning.

Every data/<split>/<instance_id>/instance.json already contains these under
`prompt`/`completion` (see tsn_bench/serialization/text_format.py) — this
script does no new generation, it just gathers train/val into training
shape while leaving held_out_test completely untouched (S1 must never see
those topology sizes, per the zero-shot generalization test the whole
project is built around).

Usage:
    python -m slm_future.s1_lora.build_sft_dataset --data data --out slm_future/s1_lora/sft_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect_split(data_dir: Path, split: str) -> list[dict]:
    examples = []
    split_dir = data_dir / split
    if not split_dir.exists():
        return examples
    for instance_file in sorted(split_dir.glob("*/instance.json")):
        with open(instance_file, encoding="utf-8") as fh:
            instance = json.load(fh)
        prompt_text = instance["prompt"]["prompt_text"]
        completion_text = instance["completion"]["completion_text"]
        if not completion_text.strip():
            continue  # no new flows requested in this instance, nothing to learn from
        examples.append({
            "instance_id": instance["instance_id"],
            "switch_count": instance["switch_count"],
            "prompt_text": prompt_text,
            "completion_text": completion_text,
        })
    return examples


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the S1 LoRA SFT dataset from generated instances")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="slm_future/s1_lora/sft_data")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        examples = collect_split(data_dir, split)
        out_path = out_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex) + "\n")
        print(f"{split}: {len(examples)} examples -> {out_path}")

    held_out_count = len(list((data_dir / "held_out_test").glob("*/instance.json"))) \
        if (data_dir / "held_out_test").exists() else 0
    print(f"held_out_test: {held_out_count} instances found, intentionally NOT written "
          f"(reserved for zero-shot eval, never trained on)")


if __name__ == "__main__":
    main()
