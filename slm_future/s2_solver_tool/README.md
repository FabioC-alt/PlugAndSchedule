# S2: SLM + solver-tool ("hybrid")

Isolates: does removing exact numeric reasoning from the SLM (giving it a
callable schedulability solver instead) fix S1's failure mode? S1's smoke
test found 100% admission accuracy but only 20-46% feasibility on the
offsets it committed to — that gap is exactly what S2 targets.

## Status: verified on disi — feasibility hypothesis confirmed, coverage gap found and fixed

**Result (5 zero-shot held-out instances, same instances S1 was evaluated
on, `results/sample/s2_eval_smoketest.jsonl`):**

| | S1 (direct) | S2, first prompt | S2, fixed prompt |
|---|---|---|---|
| admission accuracy | **100%** | 84% | **100%** |
| feasibility rate | 32.7% | 100% | **100%** |
| coverage | n/a | 84% | **100%** |
| avg. decision time | 15.6s | 12.7s | **9.3s** |

**Feasibility jumps from 33% to 100%, exactly as hypothesized** — every
offset S2 commits to is real, solver-computed, collision-free and
deadline-checked, not a plausible-looking guess. That's the headline
result: routing the numeric reasoning to the tool eliminates hallucinated
schedules entirely, not just reduces them.

**A second, distinct failure mode showed up on the first prompt: admission
accuracy actually *dropped* below S1's.** `coverage`
(`requested_flow_count / candidate_flow_count`) was also 84.0% — an exact
match to admission accuracy, because every candidate flow the model never
submits to the tool is scored as an implicit reject. So S2 wasn't wrong
about *how* to schedule what it requested (100% feasible) — it was
under-using the tool, silently dropping ~16% of flows it was never told
not to submit. The original prompt invited this ("decide which flows to
submit... you may call it more than once, e.g. to split the request"),
giving the model an implicit judgment call it has no principled basis to
make. **Fixed** by rewriting the prompt to be unambiguous: submit every
listed flow ID in one call, with an explicit count reminder ("N flows
listed — flow_ids must include all N") right before the tool-call turn.
Re-running the same 5 instances: coverage 100%, admission accuracy 100%,
feasibility held at 100%, and decision time *also* dropped (12.7s → 9.3s,
plausibly because the model no longer spends tokens deliberating over
which flows to include). One prompt change, improvement on every metric,
no tradeoff found. S2 now strictly dominates S1 on every measured axis —
at this still-5-instance scale; a full-scale run is needed before trusting
that a "worked once we phrased it right" result generalizes.

Getting this result took four rounds of fixing the *evaluation harness*
itself, not the agent — worth reading if extending this further:
1. `check_feasibility` was called with the dataset's stored ground-truth
   offsets as the "existing schedule" reference, but the tool actually
   solved against an independently-recomputed previous schedule (CP-SAT
   can pick different specific offsets for the same admitted set across
   separate solves — multiple optima). Fixed: check against what the tool
   actually solved against.
2. The feasibility_rate *formula* divided "feasible count summed over all
   flows including trivially-feasible rejects" by "admitted count" —
   produced impossible ratios like 22 feasible / 5 admitted. Fixed:
   restrict both sides to admitted flows only. (This also affected S1's
   evaluate.py, though S1's original 20-46% smoke-test numbers turned out
   close to correct once fixed, since admission was already ~100% both
   sides, leaving few rejects to inflate the old numerator.)
3. `check_feasibility`'s verification window (hyperperiod) was computed
   from *all* candidate flows, but a solver call scoped to a requested
   subset only guarantees collision-freedom within *its own* hyperperiod —
   checking more periodic repetitions than were ever solved produces
   phantom collisions. Fixed: compute the window only from flows that
   actually occupy the schedule.
4. `solve_b2` can legitimately reposition (or un-admit) previously-admitted
   base flows within the new node's k-hop neighborhood — that's the point
   of incremental re-optimization. The tool wrapper discarded that updated
   schedule and returned only the requested flows' decisions, so later
   checks (and later tool calls in a multi-call session) used a stale
   pre-call schedule. Fixed: thread the full `ScheduleResult` through
   `run_agent` and `evaluate.py`, and grow the "existing traffic" context
   list across rounds so a 2nd/3rd tool call in the same session knows
   about the 1st call's own admissions (this was the one still showing
   57% feasibility after fixes 1-3, at 4/7 admitted — went to 6/6 after
   this fix).

Also caught and fixed: `evaluate.py`'s JSONL logger defaulted to append
mode, so every `sbatch` rerun during this debugging silently accumulated
onto the same output file instead of replacing it (six runs → 30 lines in
a file meant to hold 5) — averaging over it produced meaningless blended
numbers. Fixed to open in write mode, one run = one complete file, matching
the fix already applied to `tsn_bench/dataset/build_dataset.py`'s manifest
for the same reason.

## Full-scale confirmation (1892 instances, 2026-09-17)

The 5-instance result held up on the metric that matters most, and
exposed a new bug on the one that doesn't threaten the core hypothesis.
Ran S1 and S2 on both `held_out_test` (720 zero-shot instances) and
`test` (226 in-distribution), every load factor — 946 instances each:

| | S1 held_out (zero-shot) | S1 test (in-dist) | S2 held_out (zero-shot) | S2 test (in-dist) |
|---|---|---|---|---|
| feasibility rate (overall) | 35.0% | 33.4% | **100.0%** | **100.0%** |
| feasibility, every load factor | flat, no exceptions | flat, no exceptions | **100.0%, no exceptions** | **100.0%, no exceptions** |
| admission accuracy (overall) | 74.3% | 74.8% | 68.2% | 71.7% |
| admission acc. @ low load | 99.9% | 99.9% | 99.1% | 99.9% |
| admission acc. @ medium load | 77.7% | 77.5% | 97.1% | 99.1% |
| admission acc. @ near_saturation | 45.4% | 45.4% | **8.4%** | **9.9%** |
| avg. decision time | 25.6s | 26.2s | 9.9s | 9.9s |

S1's numbers are essentially identical across both splits — its zero-shot
generalization to unseen (12/16-switch) topology sizes isn't the problem;
whatever's wrong with it is structural, not about topology-size
memorization. S2 combined across all 946 of its instances: **feasibility
100.0%, admission accuracy 69.0%, decision time 9.9s** (2.6x faster than
S1's 25.7s combined).

**The headline result is unconditional at scale**: across all 946 S2
instances evaluated, feasibility rate is exactly 100.0% with zero
exceptions, at every load factor including `near_saturation`. S1's
feasibility sits flat at 30-38% regardless of load — not something that
gets worse with harder instances, just consistently bad, confirming this
is a real weakness of SLM-only numeric placement rather than a scale
artifact either way.

**But a bug specific to `near_saturation` instances (100+ candidate
flows, vs. the smoke test's 23-27) originally tanked admission accuracy
for both methods, S2 worse than S1**: both hit their `max_new_tokens`
generation budget. S1 (1024 tokens, ~11/flow) truncated mid-list past ~90
flows but degraded *gracefully* — it still parsed whatever it managed to
generate. S2 (512 tokens, ~7/flow ID) truncated the tool-call JSON
*before its closing tag*, so the regex never matched at all — *total*
failure, not partial, which is why S2's near_saturation admission accuracy
(8-10%) was actually worse than S1's (45%) at the same load, despite S2's
schedules being unconditionally valid whenever it did succeed.

**Fix confirmed, 2026-09-17**: raised `max_new_tokens` and re-ran the
complete 240-instance `near_saturation` subset. Admission accuracy
**8.4-9.9% → 88.7%** — a big jump, though not full parity with S1's 99.0%
on the same fix. Feasibility held at a perfect **100.0%**, unchanged, as
expected. A new, smaller failure mode showed up once truncation stopped
masking it: **20/240 instances (8.3%) requested zero flows** from the
tool — an all-or-nothing non-attempt, not partial truncation. Excluding
those 20, coverage on the remaining 220 is **96.8%**, close to the ~100%
seen at lower loads. See `PROGRESS.md`'s 2026-09-17 entry for the full
numbers.

**Root-caused, 2026-09-19 — two distinct mechanisms, not random noise.**
80% (16/20) called the tool successfully but with near-total
*hallucinated* flow IDs (hallucinated count ≈ candidate count, not
partial) — the real ID format (`new96`, `new55`...) is correct in style,
so the model isn't confused about the scheme, it fails to *copy the
specific IDs* out of a 100+-item candidate list. This is the concrete
case for the OpenNod prototype (see `FUTURE_LOOK.md`/`TEST_PLAN.md`): a
typed-choice interface sidesteps ID-copying entirely, rather than asking
the model to reproduce exact ID strings by generation. The other 20%
(4/20) had malformed/truncated tool-call JSON — one dropped-quote
sampling glitch, three clear degenerate repetition loops (the same short
ID subsequence repeated dozens of times, burning the token budget before
the JSON could close). Failure rate isn't uniform: ~4x higher on
single-scenario instances than batch, ~2.3x higher on 12-switch
topologies than 16-switch, and clusters hard on specific topologies (one
held-out topology fails on 4/6 of its near_saturation instances). Not
pursued further for now.

## What's built

- `tool_schema.py` — the one tool the model gets, `solve_subschedule(flow_ids)`.
  It's a thin wrapper around the **already-existing** B2 incremental ILP
  solver (`tsn_bench.solvers.b2_incremental_ilp.solve_b2`) — no new
  scheduling logic. The model's job narrows to deciding *which* candidate
  flows to hand the solver; whatever it requests gets a real,
  collision-checked, deadline-checked answer, not a hallucinated one.
- `agent.py` — the tool-calling loop: prompts the model (with the tool
  attached via the chat template's `tools=` argument) and parses Qwen's
  `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` convention.
  The prompt directs the model to submit every candidate flow ID in one
  call (see the coverage-gap fix below for why), but `max_tool_calls`
  still allows up to N rounds if it splits the request anyway. If it never
  calls the tool, its direct text answer is parsed with S1's fallback
  parser so the instance is still scored, not discarded. Any candidate
  flow never requested is scored as an implicit reject.
- `evaluate.py` — same two-metric scoring as S1 (admission accuracy vs. B1
  ground truth, feasibility rate via the same collision/deadline check,
  reusing `slm_future.s1_lora.evaluate.check_feasibility`), plus S2-specific
  metrics: whether the tool was used at all, how many calls, how many
  candidate flows were actually submitted vs. silently dropped, and
  hallucinated flow IDs (requested an ID that isn't a real candidate).

**Design choice: uses the base Qwen2.5-3B-Instruct model, not S1's LoRA
adapter.** S1's adapter was fine-tuned to imitate direct-decision
completions, not to call tools — reusing it here would confound "does tool
access help" with "does this specific fine-tune generalize to a different
task." The base instruct model's native tool-calling ability is the
cleaner variable to isolate for a first result. A natural follow-up is
fine-tuning specifically for tool-calling (S2+LoRA) once this baseline
exists.

**Unit-tested locally without a GPU** (stubbed torch/transformers/peft,
same pattern as S1): tool-call regex/JSON parsing (well-formed, no-call,
and malformed cases), and the real solver-backed tool execution against a
generated instance — requesting a subset of flows returns decisions only
for that subset, a hallucinated flow ID is silently dropped rather than
crashing, an empty request returns `{}`, and `full_schedule` correctly
reflects each new solve.

## No-tool control: the tool isn't a convenience, it's load-bearing

`evaluate.py --no-tool` runs the base model with the tool never offered
at all, using S1's exact direct-decision prompt/parser instead —
distinct from the existing `used_tool=False` rows, where the model had
the tool but chose not to call it. **Confirmed at full scale 2026-09-19
(job 52051, 720/720 held_out_test instances, clean exit): admission
accuracy is exactly 0.0000 on every single instance (100%)** —
feasibility rate was never even computable, since nothing was ever
admitted. A 5-instance smoke test (job 52044) already showed this
starkly; the full run confirms it's not sampling noise. The base model
writes free-form planning prose instead of the required decision format
(sometimes degenerating into a fabricated sequential flow-ID list well
past the real count), never converging on anything the parser can read.
**This settles the "does tool access actually matter" question
unconditionally**: it isn't shaving latency off an otherwise-workable
approach, it's the only thing making the base model's output usable at
all in this setup.

## Decision latency: better, but still slower than B1-B3, by a lot

S2's 9.3s/instance (after the prompt fix) beats S1's 15.6s, but both remain
~50-500x slower than B1/B2/B3's 0.02-0.3s (see the Baseline Comparison tab
in the web UI). LLM generation time still completely dominates — the
actual solver call stays negligible (well under 0.1s). This remains the
open problem for any SLM-based method to be practically competitive on
decision latency, not just a footnote.

**Tried and ruled out, 2026-09-19: a terser instruction.**
`evaluate.py --terse-prompt` swaps in `agent.py`'s
`AGENT_INSTRUCTION_TERSE` (~1/3 the words, no justification content).
Same 5 instances as the baseline above: quality fully preserved (100%
coverage/admission accuracy/feasibility) but **no speedup** (10.8s mean,
9.5s excluding a likely cold-start outlier, vs. 9.3s — essentially tied).
The model was never generating verbose reasoning preamble in the first
place, so instruction length wasn't the bottleneck — generation time
scales with candidate flow count (one token run per ID in the tool-call
JSON), not prompt phrasing. Rules out prompt length as the lever; a real
fix needs to shrink/restructure the *output* itself, which is exactly
what the OpenNod prototype (typed logits instead of generated JSON) is
aimed at.

**Solved (mechanism confirmed), 2026-09-19: OpenNod typed-logits
prototype.** Swapped the "which flow IDs to submit" decision for
OpenNod's (github.com/TheoLeeCJ/OpenNod) single-forward-pass typed
scoring, keeping `solve_subschedule` itself completely unchanged. On the
same 5 low-load instances as the baseline above: **100% coverage,
admission accuracy, and feasibility — identical to S2's tool-calling
result — at 0.93s/instance (0.75s steady-state) vs. 9.3s, a ~10x
speedup.** Confirmed on the harder near_saturation case too (5
instances, 104-111 candidates each): same perfect quality, 3.35s vs.
~36s, a 10.7x speedup. Caveat: OpenNod's base model (Qwen3.5-4B) differs
from S2's (Qwen2.5-3B-Instruct), so this validates the mechanism, not a
same-model ablation, and the sample is 10 instances, not a full-scale
run yet. Being productionized into `slm_future/s2b_opennod/` (tracked in
`TEST_PLAN.md`).

## Next steps

1. ~~Fix the `near_saturation` truncation bug.~~ Done 2026-09-17 — see
   above. Admission accuracy 8.4-9.9% -> 88.7%.
2. ~~Investigate the S2-specific zero-request residual.~~ Done 2026-09-19
   — see above. Two mechanisms found (ID-copying hallucination,
   repetition-loop truncation); not pursued further for now.
3. **Investigate decision latency** — try a terser prompt/system message
   aimed specifically at shortening the response before the tool call;
   9.9s (blended across all load factors) is much better than the pre-fix
   12.7s but still far from B1-B3. Note near_saturation specifically now
   runs ~36s/instance, since it's both the biggest instances and the ones
   needing the largest `max_new_tokens` budget.
4. Then S3 (GraphRAG) — see `../s3_graphrag/README.md`.
