# S5: node-count-invariant fine-tuning curriculum (not implemented yet, stretch goal)

Train S1-S4 with a curriculum mixing multiple topology sizes, then test
zero-shot on the held-out sizes, directly targeting GCN-TD3's stated
retraining limitation (B4).

## What exists already

`tsn_bench/dataset/splits.py::assign_split` already guarantees the held-out
switch counts ({12, 16} by default, see `configs/default.yaml`) never appear
in `train`/`val`/`test` — they only ever land in `held_out_test`. This is the
dataset-level guarantee S5's whole experiment depends on; no changes needed
there, S5 is purely about how S1-S4's training loops sample across the
in-distribution sizes (curriculum ordering) and how they're evaluated on
`held_out_test`.

## What's missing

1. S1 (at minimum) must exist and train successfully first.
2. A curriculum sampler over `data/train` stratified by `switch_count`
   (easy-to-hard or mixed-batch — decide empirically).
3. Zero-shot eval on `data/held_out_test`, reporting the generalization gap
   (§5.1) against B4/B5's retrain-required numbers.

## Compute

Same as whichever of S1-S4 this curriculum is applied to.
