"""Split assignment: held-out switch counts always land in `held_out_test`
(never touched during training), in-distribution sizes are split by the
configured train/val/test ratios."""
from __future__ import annotations

import random


def assign_split(switch_count: int, instance_seed: int, cfg: dict) -> str:
    held_out = set(cfg["topology"]["held_out_switch_counts"])
    if switch_count in held_out:
        return "held_out_test"

    ratios = cfg["splits"]
    r = random.Random(instance_seed).random()
    if r < ratios["train"]:
        return "train"
    if r < ratios["train"] + ratios["val"]:
        return "val"
    return "test"
