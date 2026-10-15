# S1: SLM baseline, LoRA fine-tuned, no tools/retrieval

Isolates: can a small open model learn TSN admission/offset assignment from
text alone, with no graph structure or external solver?

## Status: pipeline verified end-to-end on disi; full-scale run complete (see below)

Code is in place, unit-tested (pure-Python parsing/feasibility logic
verified locally without a GPU), and now also **run for real on disi**:
1 epoch of LoRA fine-tuning on the full in-distribution train split (989
examples), then evaluated on 5 zero-shot held-out (12-switch, never seen in
training) instances. Results (`smoketest_training_log.json`,
`results/s1_eval_smoketest.jsonl` at the repo root):

- Train loss 0.269 → 0.158 over 1 epoch (eval_loss 0.163) — learns cleanly.
- **Parse success: 100%**, **admission accuracy: 100%** on all 5 held-out
  instances (127/127 flows) — the model reliably reproduces the trained
  output format and its ADMIT/REJECT calls exactly match the B1 ground
  truth, even on topology sizes it never trained on.
- **Feasibility rate: 20-46% per instance, 32.7% average** — when it
  commits to a specific numeric offset for an admitted TT flow, that offset
  collides with another flow's schedule or misses deadline most of the
  time. (This number survived a round of fixing real bugs in the
  evaluation harness itself, prompted by S2's numbers initially looking
  implausible — see `../s2_solver_tool/README.md`'s "Status" section for
  the full account. S1's number barely moved, since admission was already
  ~100% on both sides here, leaving little room for the harness bugs to
  distort it — but don't assume that holds for every instance mix.)

This is only a 1-epoch/989-example smoke test on 5 eval instances, not a
rigorous result — but the *pattern* (near-perfect admission decisions,
poor numeric offset feasibility) lines up exactly with the research plan's
hypothesized SLM-only failure mode, on the very first run. That's the
concrete case for S2: hand the numeric/geometric reasoning to a real
solver instead of asking the model to hallucinate a collision-free offset.
**Confirmed** — S2's feasibility rate on the same 5 instances is 100%.

## Full-scale confirmation (both splits, 946 instances, 2026-09-17)

| | `test` (in-distribution, 226) | `held_out_test` (zero-shot, 720) |
|---|---|---|
| admission accuracy | 74.8% | 74.3% |
| feasibility rate | 33.4% | 35.0% |
| decision time | 26.2s | 25.6s |

**Zero-shot generalization to unseen topology sizes is essentially a
non-issue for S1** — `held_out_test` (12/16-switch, never trained on)
scores within a point of the in-distribution `test` split on every
metric, and within `held_out_test`, 12-switch and 16-switch score the
same as each other (74.3%/34.9% vs. 74.4%/35.0%). Whatever's wrong with
S1 isn't about topology size memorization; it's structural.

By load factor (both splits pooled — the pattern is identical in each):

| load factor | admission accuracy | feasibility rate |
|---|---|---|
| low | 99.9% | ~40% |
| medium | ~78% | ~33% |
| near_saturation | 45.4% | ~31% |

**Feasibility is flat at ~31-40% regardless of load or topology size** —
not something that degrades further under harder instances, just
consistently poor. That confirms this really is a fundamental limitation
of asking an SLM to do exact numeric offset placement, not an artifact of
the easy 5-instance smoke test.

**Admission accuracy originally collapsed with load** (99.9% → 45.4%), and
the cause wasn't a modeling weakness — it was `--max-new-tokens 1024`
being too small for `near_saturation` instances (100+ candidate flows vs.
the smoke test's 23-27; each completion line costs ~11 tokens, so
generation truncated mid-list past ~90 flows and everything past that
point never got parsed). See `../s2_solver_tool/README.md`'s full-scale
section, which hit the same class of bug and got a much worse outcome
from it (S2's truncated tool-call JSON failed entirely rather than
degrading gracefully like S1's did here).

**Fix confirmed, 2026-09-17**: raised `--max-new-tokens` and re-ran the
complete 240-instance `near_saturation` subset of `held_out_test`.
Admission accuracy **45.4% → 99.0%**, matching the low/medium-load
numbers — the truncation really was the whole story at this load factor.
Feasibility rate held flat at **29.1%**, exactly as expected: raising the
token budget fixes parsing, not numeric placement, so it shouldn't (and
didn't) move. See `PROGRESS.md`'s 2026-09-17 entry.

- `build_sft_dataset.py` — collects `prompt_text`/`completion_text` pairs out
  of `data/<split>/*/instance.json` (already produced by
  `tsn_bench/serialization/text_format.py`, nothing new to generate) into
  flat JSONL for training. Deliberately only reads `train`/`val`/`test` —
  `held_out_test` is never touched here, reserved for zero-shot eval.
- `train_lora.py` — LoRA SFT of **Qwen2.5-3B-Instruct** (confirmed target).
  Uses the model's chat template, masks loss to the assistant turn only
  (the model isn't scored on reproducing the prompt). Target modules:
  q/k/v/o/gate/up/down projections (standard for Qwen2-style architecture).
- `evaluate.py` — generates decisions on a split and scores two ways:
  **admission accuracy** (does ADMIT/REJECT match the B1 ground truth?) and
  **feasibility rate** (is the offset the model gave actually collision-free
  and deadline-compliant, reusing `tsn_bench.scheduling.simulator.LinkOccupancy`
  / `flow_intervals` — the same interval-overlap logic B3's heuristic uses).
  Feasibility is the metric that matters more: admission accuracy alone
  would reward copying the one particular optimal solution the model was
  trained on, when many equally-valid offset assignments exist for the same
  instance. This is the "reduce hallucinated/invalid schedules" check the
  research plan calls out as the SLM failure mode to watch for.

## Confirmed cluster environment (disi)

- GPU: `l40s` partition, **NVIDIA L40S, 46GB VRAM**, driver 580.95, CUDA
  13.0 — plenty of headroom for bf16 LoRA fine-tuning of a 3B model, no
  quantization needed.
- Compute nodes have internet access (`huggingface.co` reachable) — the
  model downloads directly from the Hub at job time, no pre-staging needed.
- Needs its own venv (`.venv-gpu`, separate from the CPU-only foundation
  `.venv`) — see `scripts/cluster/submit_s1_train.sbatch` for the exact
  install command (PyTorch cu121 wheel, confirmed compatible with this
  driver) plus `requirements-gpu.txt`.
- **Two environment gotchas hit and fixed while getting the smoke test to
  run** (both now baked into `submit_s1_train.sbatch`/`submit_s1_eval.sbatch`,
  don't rediscover these):
  1. `huggingface_hub` defaults to caching downloads under `~/.cache`, but
     disi home dirs have a 10GB quota — this account's was already
     exhausted by an unrelated pre-existing cache. Fixed by setting
     `HF_HOME` to a scratch-based cache dir.
  2. torch 2.14's native-AOT Triton kernels (`bmm_outer_product`, used by
     Qwen2's RoPE) JIT-compile against `Python.h` on first use, and disi's
     nodes don't have `python3-dev` installed — no fallback, it just
     crashes mid-training. Fixed with `TORCH_DISABLE_NATIVE_JIT=1`
     (confirmed via the matching upstream report,
     [pytorch/pytorch#196977](https://github.com/pytorch/pytorch/issues/196977)).
  3. Also: the **login node's CPU is a restricted "Common KVM processor"**
     (no SSE4.2/AVX at all) and can't even import numpy 2.x — don't debug
     GPU-stack issues by testing imports on the login node, the compute
     nodes (confirmed: Xeon Platinum 8480+, full AVX-512) are entirely
     different hardware.

## Next steps

1. **Full-scale training run**: more epochs (the smoke test only ran 1)
   and/or a larger LoRA rank, on the full 989-example train split — compare
   validation loss and, more importantly, held-out feasibility rate across
   configurations, not just loss.
2. Eval the **base model with no adapter** (`TSN_ADAPTER_DIR=` empty) as a
   zero-shot baseline, to isolate what LoRA actually bought vs. what
   Qwen2.5-3B already does out of the box (its 100% admission accuracy
   *could* partly be prompt-following the instruction's ADMIT-by-default
   framing rather than something LoRA taught — worth checking).
3. ~~Run eval on the full `held_out_test` (720 instances) and `test`
   (226 instances) splits.~~ Done 2026-09-17, both splits, see above.
4. Feed the fixed (post-truncation-fix) `results/s1_eval_*.jsonl` into the
   web UI's Baseline Comparison tab alongside B1-B3, not just the Progress
   tab's summary cards (`webui/backend/main.py::results_summary` currently
   only aggregates `results/baselines.jsonl`).
5. Then S2 (solver tool) — see `../s2_solver_tool/README.md`. The
   feasibility-rate gap found here is the concrete motivation for it.
