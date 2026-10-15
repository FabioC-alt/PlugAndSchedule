# S2b: S2's solver-tool agent, with OpenNod typed-logits flow selection

Isolates: does S2's slowest step -- generating a tool-call JSON that
lists which candidate flow IDs to submit to the solver -- actually need
to be free-text generation? Or can a typed-decision model make the same
per-flow "submit or not" call, faster, with the same solver-backed
feasibility guarantee?

Named S2b (a variant of S2), not S3 -- `slm_future/s3_graphrag/` already
holds that slot in the original research plan's S1-S5 lineup. This is an
additional method that wasn't part of that plan, motivated by root-causing
S2's near_saturation zero-request residual (see
`../s2_solver_tool/README.md`): 80% of those failures were the model
hallucinating near-total-wrong flow IDs while still calling the tool
correctly -- an ID-*copying* failure, not a scheme mismatch. That pointed
directly at typed-decision scoring as the fix, not more prompt tuning.

## What's built

- `agent.py` -- `select_flow_ids()`: scores every candidate flow's
  admit/skip decision in one `score_shared()` call (OpenNod,
  github.com/TheoLeeCJ/OpenNod) sharing the topology description as a
  common prefix, one row per flow. Returns whichever flow IDs score
  yes > no.
- `evaluate.py` -- same two-metric scoring as S1/S2 (admission accuracy,
  feasibility rate), same JSONL logging discipline (`mode="w"`, one run
  = one complete file), S1/S2-style CLI (`--data`, `--split`, `--limit`,
  `--load-factor`).

**The solver tool itself is completely unchanged** -- reused directly
from S2 (`slm_future.s2_solver_tool.tool_schema.solve_subschedule_tool`,
which wraps the same `tsn_bench.solvers.b2_incremental_ilp.solve_b2`). No
new scheduling logic anywhere in this method; only the "which flows to
ask the solver about" decision changed.

## Prototype result (10 instances, 2026-09-19)

A throwaway prototype script (same logic, outside the repo) was run
first on 10 instances across two load regimes to decide whether this was
worth building for real:

| | S2 baseline | S2b prototype |
|---|---|---|
| Low-load (5 instances, S2's original baseline set) | 9.3s/instance, 100%/100%/100% | **0.93s/instance** (0.75s steady-state), 100%/100%/100% |
| near_saturation (5 instances, 104-111 candidates each) | ~36s/instance, 100% feasibility | **3.35s/instance**, 100%/100%/100% |

~10-12x speedup, zero quality loss, on both easy and hard instances --
worth productionizing.

## Full-scale confirmation (946 instances, 2026-09-19)

Ran this package (jobs 52122/52123) on the complete `held_out_test`
(720) and `test` (226) splits, every load factor -- both clean exits:

| | `held_out_test` (zero-shot, 720) | `test` (in-distribution, 226) |
|---|---|---|
| admission accuracy | **99.79%** | **100.00%** |
| feasibility rate | **100.00%** | **100.00%** |
| coverage | 100.00% | 100.00% |
| decision time | 2.01s | 1.79s |

**S2b strictly dominates S2 at full scale**, not just the prototype
sample: admission accuracy 99.8-100% vs. S2's 69.0% blended (S2's own
best post-fix number, 88.7% at near_saturation specifically, is still
well below this); feasibility matches at a perfect 100%; decision time
~1.8-2.0s vs. S2's 9.9s blended -- about 5x faster even against S2's
already-fixed numbers. Zero-shot generalization is a non-issue, same
pattern S1/S2 already showed -- the two splits score almost identically.

**Caveat still standing**: OpenNod's own base model (Qwen3.5-4B) differs
from S2's (Qwen2.5-3B-Instruct). This validates the typed-logits
*mechanism* for this task at real scale, not a same-model ablation. A
cleaner isolation would need a model that supports both OpenNod's
single-token-slot readout and S2's tool-calling chat template -- worth
noting in the eventual paper's limitations, not something that
undermines this result.

## Environment setup (separate from S1/S2's `.venv-gpu`)

OpenNod pins `torch==2.10.0`/`transformers==5.17.0`, a different stack
from S1/S2's environment (and from disi's `TORCH_DISABLE_NATIVE_JIT=1`
workaround, which doesn't apply here since OpenNod's torch version
doesn't hit that bug). This needs its own venv:

```bash
# on disi, outside this repo (sibling directory), pinned to the commit
# this was developed and verified against:
git clone https://github.com/TheoLeeCJ/OpenNod.git
cd OpenNod && git checkout ca3ba65f142967030ecb453346e94d6f476a69df

python3 -m venv .venv                 # on the LOGIN node -- venv creation
source .venv/bin/activate             # silently fails from inside an
pip install --upgrade pip             # sbatch job on the compute node,
pip install -e '.[test]'              # confirmed while setting this up
pip install ortools networkx peft     # tsn_bench's + S1's runtime deps
pip install 'protobuf==7.36.1'        # ortools installs an older pin;
                                       # 7.36.1 works fine for both in
                                       # practice despite pip's warning
```

Then run this package's `evaluate.py` with that venv active and
TSNAgent's repo root on `PYTHONPATH` (or `sys.path`), e.g.:

```bash
cd /path/to/OpenNod
source .venv/bin/activate
PYTHONPATH=/path/to/TSNAgent python -m slm_future.s2b_opennod.evaluate \
    --data /path/to/TSNAgent/data --split held_out_test \
    --out /path/to/TSNAgent/results/s2b_eval_held_out.jsonl
```

## Next steps

1. ~~Run the full `held_out_test`/`test` splits.~~ Done 2026-09-19 --
   see above, strictly dominates S2.
2. ~~Add a confirmed-at-scale entry to `MILESTONES.md` Phase 4.~~ Done.
3. Worth trying with a model that supports both OpenNod's
   single-token-slot readout and S2's tool-calling chat template, to
   remove the base-model caveat above and get a cleaner apples-to-apples
   comparison against S2.
4. Feed S2b's results into the web UI's Baseline Comparison tab
   alongside B1-B3/S1/S2 (same existing `TEST_PLAN.md` item covers this
   for all the SLM methods, not S2b-specific).
