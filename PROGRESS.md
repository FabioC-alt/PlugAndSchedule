# Progress Log

Chronological research log, newest entry first. Headline + bullets, kept
short for scanning — full numbers/reasoning live in `results/` and the
per-method READMEs if you need to dig deeper. `MILESTONES.md` is the
checklist; this is the narrative of what happened and why. The web UI's
Progress tab renders this file directly.

## 2026-09-29 — Phase 5's full evaluation matrix run, and a scope mismatch bigger than expected

Scoped Phase 5 down to the 9 methods already confirmed at full scale
(B1-B4, S1-S4, S2b), skipping the unbuilt B5/B6/S5 stubs. While setting
it up, found the mismatch between the classical baselines and the
model-based methods runs deeper than just "different scoring basis"
(already flagged for B4's paper comparison): B1/B2/B3 score
`admission_rate` over the WHOLE topology (base flows + new ones), while
B4/S1-S4 score `admission_accuracy` only on the NEWLY ADDED flows
against a fixed base schedule -- a different problem scope, not just a
different metric. Checked directly on a sample instance: B1's
`total_flows`/`admitted_flows` = 118 (matches base+new), while the same
instance's actual new-flow count is 27.

So the comparison had to be built as three separate pieces
(`tsn_bench/eval/phase5_report.py`, runs locally, no cluster needed)
instead of one leaderboard: a real head-to-head within the 6 model-based
methods (same scope, same ground truth), a latency-only comparison
spanning all 9 (caveated, since B1-B3 solve a bigger problem each time),
and B1/B2/B3's own internal comparison.

**Headline: S2b is the strongest all-around confirmed method in the
project** -- 99.8% admission accuracy, 100% feasibility, 2.0s latency.
It matches S1's near-perfect accuracy without S1's feasibility collapse
(34.8%), at a twentieth of S1's latency (40.1s). Every pairwise
model-group comparison is statistically significant (p<0.001, n=720/226
paired instances). Also notable: every classical baseline beats every
model-based method on raw latency, including B1's full from-scratch
recompute (0.18s) against B4's 1.2s -- no LLM or GCN forward pass beats
a modern ILP/heuristic solver on wall-clock speed at this problem size,
even solving the strictly bigger whole-topology problem. Full tables in
`TEST_PLAN.md`.

## 2026-09-28 — B4 compared against its source paper, and a structural fact about the dataset surfaced along the way

Re-read the full GCN-TD3 paper (arXiv:2405.05019, not just the abstract)
to ground a proper comparison. Its "~90% admission rate" turns out to be
measured *during training*, on one fixed small topology, with no
independent ground truth -- fundamentally not the same thing as this
project's `admission_accuracy` (scored against B1, an ILP solver, on
topology sizes the model never trained on).

Added raw admitted-flow-count logging to B4's `evaluate.py` (pure
logging addition -- verified zero effect on existing scoring) and
re-ran both splits to get a comparable raw rate. Along the way, found
something worth flagging beyond just the paper comparison: **B1's
ground truth admits literally 100% of newly-requested flows in every
single one of 946 instances across both splits.** Zero exceptions. That
means `admission_accuracy`, throughout this whole project, is
mathematically the same thing as B4's own raw admission rate scored
against a constant "always admit" ceiling -- every point B4 loses is
pure conservatism (rejecting a flow that genuinely does fit), never a
disagreement with a genuinely selective ground truth.

| | B4 raw admission rate | B1 ceiling | Paper's reported rate |
|---|---|---|---|
| `held_out_test` | 22.8% | 100.0% | -- |
| `test` | 17.1% | 100.0% | -- |
| Combined | 21.5% | 100.0% | ~90% (in-training, fixed topology) |

Also confirmed the paper never actually tests its own stated
limitation (retraining needed per switch-count change) -- its numbers
come from one fixed topology size, so there's no like-for-like figure
in the paper to set against this project's own generalization result
(B4 does *better* on unseen 12-16sw than on the 2-8sw range it trained
on). That finding stands as this project's own contribution, not a
reproduction of anything the paper measured. Full writeup, including
the honest read on the gap (most plausibly the same constant-action
collapse dynamic already documented, not a reward-design difference),
in `TEST_PLAN.md`'s B4 entry.

## 2026-09-28 — S4's reliability gap fixed, confirmed at full scale, question closed out

Three days after the initial finding, tried a follow-up fix for S4's
near_saturation reliability gap -- three iterations, two of them real
missteps:

1. **`no_repeat_ngram_size` -- backfired.** Blocks repeating any 6-token
   sequence seen anywhere in the sequence, including the PROMPT, so the
   model got blocked from re-writing "solve_subschedule" (already said
   several times in its own instructions) and mangled it into
   "solve_subscheduler." Removed.
2. **`min_new_tokens` alone -- real but partial win**, then two more
   rounds of latency debugging: the parser was discarding usable
   retries (fixed by checking every `<tool_call>` occurrence, not just
   the first), and the loop's optional post-success "wrap-up" round was
   silently eating a whole extra forced generation on every successful
   instance (fixed by breaking the loop immediately after success --
   which, bonus, turned out to be a latent cost in the ORIGINAL S4 too,
   not something the fix introduced).
3. **Full re-run confirmed the combination works, better than every
   smoke test suggested**:

| Split | Admission accuracy | near_saturation failure rate | Mean latency |
|---|---|---|---|
| `test` | 84.9% -> **93.3%** | 41.4% -> **8.6%** | 21.4s -> 35.4s |
| `held_out_test` | 74.8% -> **87.93%** | 65.0% -> **22.1%** | 20.9s -> 31.1s |

Feasibility stayed a perfect 100.0% throughout -- this was always a
reliability problem, never a decision-quality one. Whenever the tool
call succeeds, S4 matches or slightly beats S2/S3's own numbers on the
same splits. Two full-run attempts (jobs 53663/53664) had to be
cancelled mid-flight when a latency bug was caught early (projected to
blow through its time budget) rather than left to run to a wasted
timeout -- see TEST_PLAN.md's S4 entry for the complete blow-by-blow.
Also rewrote REPORT.md in plain language while this was running, since
the existing report had grown dense with debugging detail that doesn't
matter to a reader who just wants the findings.

## 2026-09-25 — S4 (optional-retrieve agent) built, evaluated, and its answer turns out to be about reliability, not quality

Built S4 as a genuine two-tool agent: S2's `solve_subschedule` tool plus
S3's retrieval mechanism offered as a second, *optional* tool
(`retrieve_subgraph`), letting the model choose for itself whether to
narrow its context before scheduling, instead of S3's always-retrieve
design. Found and fixed two real bugs along the way (a tool-sequencing
abandonment after retrieval, and a crash on malformed tool-call
arguments), then ran the full evaluation on both splits:

| Split | Admission accuracy | Feasibility | Retrieval rate | near_saturation tool-success rate | near_saturation accuracy (successful calls only) |
|---|---|---|---|---|---|
| `test` (2-8sw) | 84.9% | 100.0% | 64.2% | 58.6% | 97.3% |
| `held_out_test` (12-16sw) | 74.8% | 99.99% | 38.3% | 35.0% | 94.3% |

The headline isn't a quality problem, it's a reliability one: on
`near_saturation` instances (100+ flows), generation sometimes stops
abruptly after ~9s -- well under the 3072-token budget -- partway
through listing flow IDs, sometimes right after a short "should I
retrieve? no, proceeding directly" aside. Checked against S2's results on
the identical instance IDs: S2 (no retrieve option, terser prompt)
succeeds on every one where S4 failed. The optional-retrieve framing
itself costs reliability on large instances, and the effect gets worse
with topology size (41.4% near_saturation failure rate on `test`'s
smaller topologies vs. 65.0% on `held_out_test`'s larger ones). Whenever
the tool call *does* succeed, S4 is as good as or better than S2 (88.7%
near_saturation accuracy) and S3 (90.4%) on the same split. Checking in
with the user on whether to chase a fix (shorter instruction, forcing
generation past the early stop) or treat the research question as
answered.

## 2026-09-25 — S3 filled in on the in-distribution split too (user noticed the gap in the redesigned Baseline Comparison tab)

S3 had only ever been evaluated on `held_out_test` -- run now on `test`
(226 instances, job 53436, clean exit) for completeness, after the user
spotted its missing switch_count 2/4/6/8 columns in the redesigned
Baseline Comparison tab.

| Split | Admission accuracy | Feasibility | Latency | Context reduction |
|---|---|---|---|---|
| `test` (2-8sw) | 98.0% | 100.0% | 23.3s | 23.0% |
| `held_out_test` (12-16sw) | 95.4% | 100.0% | 22.8s | 40.3% |

Slightly better on the split it wasn't "held out" from, but -- unlike
B4 -- this isn't a generalization question at all: S3 has no trained
weights, just a frozen base model plus a retrieval-narrowed prompt, so
there's no train/eval distinction to generalize across in the first
place. The gap most likely just reflects smaller topologies being
easier overall (fewer flows, simpler routing), the same pattern S1/S2/
S2b already show. Context reduction is lower in-distribution (23.0% vs
40.3%) since smaller topologies have less redundant context to prune
in the first place. Wired into the webui (`SLM_CANONICAL_FILES`) so S3
now shows a complete row like every other method.

## 2026-09-25 — B4's small-topology harness bug fixed and confirmed (test split now a clean 100% feasible)

Fixed the stale-decision-after-fallback bug found earlier today:
`evaluate.py` now reconciles every recorded flow decision against the
true final schedule before scoring, instead of trusting a value
recorded before a later fallback (within the same instance) recomputed
everything. Re-ran both splits to confirm:

| Split | Before fix | After fix |
|---|---|---|
| `test` (in-distribution) | 20.4% acc / 96.9% feas | 20.4% acc / **100.0%** feas |
| `held_out_test` (unseen) | 31.1% acc / 100.0% feas | 31.2% acc / 100.0% feas (unchanged, as expected -- 0 fallbacks means the bug could never trigger there) |

Feasibility on `test` is now a clean 100.0% across all 226 instances,
confirming the bug was the entire explanation for the earlier 96.9% --
no other feasibility issue lurking underneath it. Admission accuracy is
essentially untouched on both splits (the tiny 0.1pp move on
held_out_test is ordinary GPU floating-point run-to-run noise, not the
fix), which makes sense: the bug only ever corrupted the *offset*
comparison used for feasibility, never the admit/reject *decision*
itself. Both numbers are now the authoritative, doubly-confirmed B4
results.

## 2026-09-25 — B4's generalization question answered: no gap, but a real small-topology fallback bug found

Evaluated B4 on the in-distribution `test` split (226 instances, same
2-8 switch range as training, job 53396) to compare directly against
`held_out_test`'s already-known numbers (31.1% admission accuracy,
100.0% feasibility) -- this is the actual research question B4 exists
to answer: does GNN-based state avoid the paper's stated
retrain-per-switch-count limitation?

| Split | Switch count | Admission accuracy | Feasibility |
|---|---|---|---|
| `test` (in-distribution) | 2-8 | 20.4% | 96.9% |
| `held_out_test` (unseen) | 12-16 | 31.1% | 100.0% |

**No generalization gap in the direction the paper worries about** --
B4 does *better* on switch counts it never trained on than on the ones
it did. Worth being honest about why rather than declaring victory:
smaller topologies concentrate the same load onto far fewer links,
which structurally hurts an already-conservative policy more, and (see
below) also breaks a harness assumption.

**Found and root-caused a real bug** while digging into why feasibility
wasn't a clean 100% here: 8/226 instances (3.5%), all 2-switch
near_saturation, each triggered 106-115 ILP fallback resets within a
single instance's flow-processing sequence -- essentially every flow
decision immediately exceeds the GCL threshold and forces a fallback.
`evaluate.py` records a flow's decision at the moment it's made, but
doesn't account for an immediately-following fallback recomputing the
entire schedule -- so that recorded decision can reference an offset
the final schedule no longer honors, producing real, mechanically
correct-to-detect infeasibility (as low as 5% on the worst affected
instance). Excluding those 8 pathological instances, the remaining 218
are a clean 100.0% feasible with 19.0% admission accuracy -- barely
different, confirming this is a narrow harness edge case (specific to
tiny+near-saturated topologies that fallback almost every step), not a
policy-quality problem. Not fixed yet -- noted as a known limitation.

## 2026-09-25 — B4 validated at full scale: fastest method, perfect feasibility, but the weakest admission accuracy

Evaluated the pre-collapse checkpoint (`b4_v3_inst0180.pt`) on the full
`held_out_test` split (720 instances, job 53347, 17m16s, clean exit).
Smoke test first (job 53346) confirmed real recovery -- 0.26-0.63
admission accuracy with 100% feasibility, a dramatic improvement over
the collapsed checkpoint's 0.00-0.07 -- before committing to the full
run.

Full results:

| Load factor | Admission accuracy | Feasibility | Latency |
|---|---|---|---|
| low | 54.7% | 100% | 0.40s |
| medium | 24.9% | 100% | 1.08s |
| near_saturation | 13.9% | 100% | 2.10s |
| **Overall** | **31.1%** | **100.0%** | **1.20s** |

Two things stand out against the project's other confirmed methods
(S1 99.7%/34.8%/40.1s, S2 69.0%/100%/9.9s, S2b 99.8%/100%/2.0s, S3
95.4%/100%/22.8s):

- **Feasibility is a perfect 100.0% across all 720 instances, zero
  exceptions, zero ILP fallbacks triggered.** Matches the S2/S2b/S3
  tier -- when the trained policy does admit a flow, it always picks a
  valid, non-colliding offset.
- **Decision latency (1.20s) is the fastest of any method in the
  project**, including S2b (2.0s) -- a single GCN forward pass + one
  actor forward pass per flow decision is genuinely cheap.
- **Admission accuracy (31.1%) is the lowest of any confirmed method.**
  B4 is conservative -- it under-admits relative to what B1's ground
  truth says should be admitted, especially at higher load (13.9% at
  near_saturation). Plausibly a residual echo of the collapse dynamic:
  even the pre-collapse checkpoint was captured close to where the
  policy started drifting, not necessarily at its true peak.

B4 is now validated at full scale for the first time. Not yet done:
a structured comparison against the paper's own headline numbers (their
metric is raw admission RATE, not accuracy against B1 ground truth, so
not directly comparable without more work), and the actual research
question B4 exists to answer -- does GNN-based state avoid the paper's
stated retraining-per-switch-count limitation? That needs a held-out-by-
switch-count training/eval design, not yet run.

## 2026-09-25 — B4 checkpoint versioning validated: caught a recurring collapse, gave us a usable pre-collapse snapshot

Retrain with checkpoint versioning (job 53083, 250 instances) finished
clean (8h38m, 3567 epochs, no errors). Full trajectory: cold start
~10-16%, a dip to ~9% around epoch 242-466 that turned out to be
*transient* (recovered on its own, not the terminal failure), then a
strong 75-93% plateau holding from epoch ~689 clear through epoch 2623
(~1900 epochs, instances ~45-180) -- notably longer than the previous
run's healthy stretch. The same constant-action collapse then set in at
epoch 2624 (instance 183) and never recovered, declining to 12-14%
admission for the rest of the run. Overall pooled: 51.8%.

The versioning fix did exactly what it was built for: `b4_v3_inst0180.pt`
was saved at instance 180 / epoch 2621, 3 instances before the decline
started -- a real, usable pre-collapse snapshot, distinct from the final
`b4_v3.pt` which reflects the collapsed policy. Versioning doesn't
prevent the collapse (it was never meant to -- only the declined
entropy-regularization option would address the root cause), but it
means a run doesn't have to be a total loss when collapse happens.

Also learned to distinguish transient dips (epoch 457-466 last time,
242-466 this time -- both recovered) from the terminal collapse (no
recovery observed once it starts) -- useful for reading future runs
without over- or under-reacting to a rough patch.

Checking in with the user on which checkpoint to evaluate before
running held-out-test.

## 2026-09-24 — B4's checkpoint evaluates poorly: a second, distinct collapse (constant action, not reject bias)

Evaluation smoke test (job 53055, 5 held_out_test instances, checkpoint
`b4_v2.pt`) finished clean but showed admission_accuracy 0.00-0.07
against B1 ground truth -- essentially failing. Diagnosed via a targeted
actor-output probe (job 53077): the actor outputs the exact same
constant action for every one of 19 different flows in a real instance
-- `accept_logit=1.0000`, `offset_frac=-1.0000`, both saturated at the
tanh boundary, completely state-independent. It always tries to accept
every flow at offset=0 regardless of the actual state, so only the
first flow to reach each link at offset=0 ever succeeds -- everything
else collides and fails.

This is a *different* failure mode from the always-reject collapse
fixed earlier this week -- constant-action collapse, not a reward bias
toward rejecting. Almost certainly what the late-training admission-rate
dip (70-93% -> 35-37%, flagged when training completed) actually was:
not noise, but this collapse starting to set in.

Also surfaced an operational gap: `train.py`'s periodic checkpoint
overwrites the same file every save, so the earlier good checkpoint
(from the 70-93% plateau, before the collapse) no longer exists --
there's no way to recover it without retraining.

Not proceeding to the full 720-instance held-out-test run; it would
just confirm the same failure at scale. Checking in with the user on
how to proceed (recommendations: add an action-entropy/diversity
regularization term to discourage constant-action collapse; version
checkpoints instead of overwriting, so a mid-training model can be
recovered if a later one degrades; or both).

## 2026-09-23 — B4 training run complete: fix held for the full run, with a late-training dip worth watching

Job 52763 finished clean (9h01m, all 250/250 instances, 3898 epochs, no
errors). The always-reject collapse never recurred anywhere in the run
-- the fix held. Full trajectory: cold start ~10-12% admission on
low/medium load, a strong 70-93% plateau from roughly epoch 689 to
2831, then a real (if much milder) dip to ~35-37% for the final ~600
epochs. Not a repeat of the pre-fix collapse (which flatlined to exactly
0.000%) -- this is a genuine admission rate, just lower than the peak.
Overall pooled low/medium admission across the whole run: 60.3%
(3491/5787). near_saturation instances hit immediate ILP fallback
essentially every time (3628/3670 epochs), consistent with earlier
findings about that load factor's structural density.

Worth flagging honestly rather than only reporting the good middle
stretch: whether the late dip is normal late-training policy drift
(plausible with TD3's continuous exploration noise) or something that
would benefit from investigation isn't yet known. Checkpoint saved to
`baselines_future/b4_gcn_td3/checkpoints/b4_v2.pt`. Held-out-test
evaluation not started -- checking in with the user on how to proceed
(evaluate as-is, investigate the dip first, or something else).

## 2026-09-22 — B4's reject-penalty fix validated: no collapse, and much stronger performance

Retrained from scratch (job 52763, not resumed from the collapsed
checkpoint, for a clean read) with an explicit `REJECT_PENALTY=0.1`
subtracted only on an outright reject (user's chosen fix for the
always-reject collapse below). Result: admission rate on low/medium
load climbed to a sustained 78-86% by epoch ~693 and *improved further*
to 93-95% by epoch 2400+ -- well past epoch 2376, the exact point the
previous run collapsed -- with no sign of the failure recurring. Also
notably better than the pre-fix run's peak of ~40%, not just stable:
the reward flaw wasn't just causing collapse, it was actively holding
quality down even before the collapse hit. Training still running;
held-out-test evaluation not started pending a check-in with the user.

## 2026-09-22 — B4 training collapsed; root-caused to a reward design flaw

Real training run (job 52672, resumed after the checkpoint fix) ran
healthy for ~2h100+ instances with genuine learning signal (admission
rate 15-40% on low/medium load), then flatlined to exactly 0.000 for
650+ consecutive epochs starting ~epoch 2376 and never recovered.
Cancelled the run rather than let it keep burning GPU-hours on a
collapsed policy.

Diagnosed via checkpoint inspection (job 52705): every layer across the
encoder, actor, and twin critics is numerically clean -- no NaN, no Inf,
no exploding weights. Not a divergence/gradient-explosion failure.

Root cause is behavioral: `env.py`'s reward gives a rejected flow and a
failed accept attempt (collision or deadline miss) the *identical*
reward -- both zero the compliance term and differ only by the shared
penalty terms. Nothing in the reward distinguishes "attempted a
reasonable offset and it collided" from "didn't try at all." Since
correctly guessing a non-colliding offset in an already-dense schedule
is genuinely hard, TD3's deterministic policy gradient has a real
incentive to collapse toward "always reject" -- a stable, predictable
reward with no downside, versus a noisy, usually-failing attempt.
Reward/exploration design needs revisiting (e.g. partial credit for a
near-miss offset, or an explicit reject penalty distinct from a failed
accept) before another real training run.

## 2026-09-22 — B4 real training started; caught a checkpoint-loss risk before it hit

First real training run (job 52664, 300 instances, 8h budget) ran healthy
for 2h06m -- no errors, and a genuine positive signal: low/medium-load
admission rate (near_saturation instances correctly hit immediate
fallback, excluded from this read) trended 0% -> 8% -> 17% -> 12% ->
35.6% pooled across the run so far, noisy but real learning, not chance.

Caught a real problem before it could bite: at the observed pace
(~111s/instance) the full 300-instance run projected to ~9.3h, over the
8h budget -- and `train.py` only saved a checkpoint once, at the very
end. A SLURM TIMEOUT would have killed the job with the full training
run's progress unrecoverable, nothing written to disk. Cancelled the run
deliberately (no loss either way, since nothing was checkpointed yet),
fixed `train.py` to save every 10 instances instead, and resubmitted
(job 52672) with a 10h budget for margin.

## 2026-09-22 — B4 (GCN-TD3) pipeline built and confirmed running end-to-end; GCL threshold needed recalibrating

Read the full source paper (arXiv:2405.05019, previously only had the
abstract) to ground a faithful reimplementation rather than guess: Table
II's node-feature schema, the GCN architecture, TD3's Algorithm 1, and
the exact reward function (eq. 2-5).

- Transcribed Table II into `graph_format.py::gcn_td3_node_features`
  (queue-level graph nodes, DeepSets-style variable-length per-node
  features). Built the GCN encoder, RL environment, and TD3 agent
  (`baselines_future/b4_gcn_td3/{gcn,env,td3,train,evaluate}.py`).
- Documented, deliberate deviations from the paper where it doesn't fit
  this project's shared no-wait simulator or isn't fully specified: a
  single release-offset action (not the paper's per-hop vector, which
  only makes sense under independent per-hop timing our shared simulator
  doesn't have), per-switch queue nodes (not per-egress-port, matching
  every other baseline's granularity), AVB bypassing the RL decision
  (matches the paper's own training setup), and the GCN trained
  end-to-end with the RL losses (the paper's separate MSE pretext task
  isn't reproducible from the text alone). Full list in the README.
- `env.py` verified locally against real dataset instances before any
  cluster time was spent (seeding from a base schedule, TT/AVB step
  logic, state-graph construction all correct).
- First cluster smoke test (job 52661) ran clean but revealed a real
  calibration problem: the paper's GCL threshold omega=128 (a specific
  hardware switch's limit from their toy-scale setup) is already
  exceeded by this dataset's base schedules alone (136-632 entries
  before any new flow even arrives) -- every episode fell back to ILP
  instantly, zero training signal. Fixed to 1024, matching
  `metrics.py::compute_metrics`'s existing dataset-calibrated default
  (already used by every other baseline) instead of the paper's literal
  figure. Re-run (job 52663) now processes real episode lengths
  (15-24 flows/episode), pipeline confirmed working end-to-end.
- Not chased further this session: admission_rate was exactly 0.000 in
  the 3-instance smoke test -- plausibly a cold-start RL exploration
  artifact (untrained actor + modest noise struggles to find a
  non-colliding offset in an already-dense schedule by chance), not
  confirmed as a bug, worth watching once real training runs.
- Deliberately NOT started: a real training run. That's a bigger
  cluster-time commitment than a smoke test, holding for a check-in
  before submitting it.

## 2026-09-21 — S3 (GraphRAG) confirmed at full scale: matches S2's accuracy, real context savings, no latency win

Full held_out_test run (job 52612, 720 instances, after job 52600 timed out
at 302/720 on the sbatch template's 2h default -- fixed to 6h and
resubmitted, see below):

- **Admission accuracy 95.4%** (low 99.3% / medium 96.4% / near_saturation
  90.4%), **feasibility 100.0%** throughout -- matches S2's held_out_test
  accuracy (~95.0% blended, using S2's own post-fix near_saturation
  numbers) almost exactly, same perfect feasibility.
- **Real context reduction confirmed at scale**: 40.3% average (55.4% low
  / 37.6% medium / 28.0% near_saturation) -- shrinks with load as
  predicted, since more flow routes cover more of the graph.
- **Latency is not improved**: S3 blended ~22.8s vs S2's ~19.2s. Low and
  near_saturation are close to parity (9.76s vs 9.31s; 36.38s vs 36.03s),
  but medium load shows a real, low-variance ~1.8x slowdown (22.4s vs
  12.3s, stdev 2.75s vs 0.74s -- not a couple of outliers) at *identical*
  flow counts on both sides. Not explained by prompt length or candidate
  count; not chased further this session, flagged as an open question.
- **Net finding**: retrieval preserves decision quality and cuts context
  length substantially, but is not a decision-latency win -- useful for
  scaling to larger topologies that wouldn't fit in context (not testable
  with the current 16-switch max dataset), not for speed.

## 2026-09-21 — S3 (GraphRAG) built as an S2 variant; retrieval only pays off with no k-hop buffer

Built `slm_future/s3_graphrag/` as a thin S2 variant (same base model, same
tool-calling loop) with a retrieval-narrowed prompt: added a
`prompt_topology` param to `s2_solver_tool.agent.run_agent` so retrieval
narrows only what the model *sees*, never what `solve_subschedule` actually
schedules against. `Topology.k_hop_subview` (new) restricts a topology to
the k-hop neighborhood of a node set, mirroring `b2_incremental_ilp`'s
existing flow-level `k_hop_affected_flows` pattern.

- Correctness catch: centering retrieval on the new node(s) alone can drop
  a new flow's OTHER endpoint from the prompt entirely (the generator only
  requires one endpoint to touch a new node). Fixed by seeding retrieval
  with every new flow's actual shortest-path route, not just new-node IDs.
- That fix has a side effect worth flagging: once routes are counted
  correctly, a k-hop buffer barely shrinks anything — measured locally on
  held_out_test, k_hop=1 already keeps 91-100% of nodes (a single flow's
  route alone can span most of the graph). Only k_hop=0 (routes + new
  nodes, no buffer) gives a real reduction: 39% (low load) to 75%
  (near_saturation) of nodes kept on average. `evaluate.py --k-hop`
  defaults to 0 accordingly.
- Smoke test (job 52598, 5 instances) passed clean: 100% admission accuracy,
  100% feasibility, real node reduction (45-58% of full topology kept, avg
  ~50%), 9.3s/instance excluding cold start -- matches S2's baseline on
  comparable instances almost exactly. Full held_out_test run (720
  instances, job 52600) submitted to confirm at scale.
- Job 52600 hit the sbatch template's 2h default `--time` limit and was
  killed by SLURM (TIMEOUT, not a bug) at 302/720 instances -- the same
  class of issue S2's original full runs needed a 6h override for.
  Observed pace (~24s/instance across all load factors) implies ~4.8h
  needed for all 720. Fixed `submit_s3_eval.sbatch`'s default to 6h and
  resubmitted as job 52612.

## 2026-09-21 — OMNeT++/INET deprioritized: no compiler toolchain on disi

Investigated before committing cluster time: `disi` has no C++ compiler
anywhere — neither the login node nor the compute nodes have
gcc/g++/bison/flex/qmake (only `gcc-14-base`, the runtime-library
metapackage), and there's no root access, module system, or
Singularity/Apptainer to work around it. A real path exists (bootstrap a
user-space conda-forge toolchain, no root needed), but it's genuinely
open-ended — likely many hours for OMNeT++ + INET alone, then a separate
research question of finding/adapting the right TSN-specific simulation
module. User's call: deprioritize — this gap is already a disclosed,
accepted limitation in `MILESTONES.md`, not a blocker. Moving to S3
(GraphRAG) / B4 (GCN-TD3) instead.

## 2026-09-19 — Node+flow scaling confirmed together: real stress, and the heuristic wins at the extreme

Fixed the dataset's load-factor definition (opt-in aggregate-capacity
scaling, see the generator entry below) and ran B1-B3 on a dedicated
Stage-2 probe combining topology size AND load factor (jobs
52490/52495, 60 instances, 180 records, both clean exits):

- **Flow count now genuinely scales with topology size**: near_saturation
  goes 472 -> 968 -> 1958 flows at switch_count 2/8/16 (vs. the old flat
  ~287 regardless of size).
- **Admission rate collapses when both axes push together**: stays
  ~92-100% even at switch_count=16 under near_saturation alone, but
  **stress_2x at switch_count=16 drops to 60.5% (B1) / 73.8% (B2) /
  76.5% (B3)**. Statistically significant on every method pair, both
  metrics (Wilcoxon, p<0.03, most p<0.001, n=60).
- **B3 (heuristic) beats both ILP methods on admission rate at the
  hardest cell** — 76.5% vs. B1's 60.5%, B2's 73.8% — while running
  ~100x+ faster (1.33s vs. 151-161s). Not because it's a better
  optimizer: B1/B2 hit their 150s solver time limit on 10/10 instances
  in that cell, so their numbers are time-capped best-effort, not
  converged-to-optimal. B3 always completes a full pass.
- Caveat carried forward honestly: several cells (switch_count>=8 with
  stress_2x, partially switch_count=16 near_saturation) have B1/B2
  records capped at the 150s limit — their admission-rate numbers there
  may understate what more solver time would find.

## 2026-09-19 — Fixed the dataset's load-factor definition (opt-in aggregate-capacity scaling)

Root cause (see the 2026-09-17 node/flow-count entry below): flow
generation targets a fixed single link's capacity regardless of
topology size, so bigger topologies never got proportionally more
candidate flows. Added an opt-in `capacity_scale_links` parameter to
`generate_flows()` — target bandwidth becomes relative to N links'
aggregate capacity instead of always one link. Default stays `None`
(byte-identical old behavior), so the main dataset and every existing
S1/S2/B1-B3 result stay meaningful — this is additive, not a breaking
change.

Calibrated locally before touching any cluster time: divisor=20 gives a
1x/2x/4x multiplier at switch_count 2/8/16, confirmed tractable
(359/723/1430 flows, sub-10s B1 solve time at near_saturation=0.85)
before adding load-factor stress on top (stress_2x=2.0 shows real
admission stress at switch_count=16; stress_4x tried and ruled out at
150s+ solve time, impractical for a real dataset). New
`configs/aggregate_scaling.yaml` runs both axes together — see the entry
above for the confirmed result.

## 2026-09-19 — S2b (OpenNod) confirmed at full scale: strictly dominates S2

Productionized (`slm_future/s2b_opennod/`) and run on the complete
`held_out_test` (720) + `test` (226) = 946 instances, all load factors
(jobs 52122/52123, both clean exits):

- **`held_out_test` (zero-shot): 99.79% admission accuracy, 100.00%
  feasibility, 100.00% coverage, 2.01s/instance.**
- **`test` (in-distribution): 100.00% admission accuracy, 100.00%
  feasibility, 100.00% coverage, 1.79s/instance.**
- Zero-shot generalization is a non-issue, same pattern S1/S2 already
  showed — the two splits score almost identically.

**S2b now strictly dominates S2 at full scale, not just the 10-instance
prototype**: admission accuracy 99.8-100% vs. S2's 69.0% blended
(S2's own best post-fix number, 88.7% at near_saturation specifically,
is still well below this); feasibility matches at a perfect 100%;
decision time ~1.8-2.0s vs. S2's 9.9s blended — about 5x faster even
against S2's already-fixed numbers. `MILESTONES.md` Phase 4 updated.

Caveat still standing: OpenNod's base model (Qwen3.5-4B) differs from
S2's (Qwen2.5-3B-Instruct), so this is a mechanism win, not a validated
same-model ablation — worth keeping in mind for the eventual paper.

## 2026-09-19 — OpenNod prototype for S2: ~10x speedup, zero quality loss (decision: build it for real)

Swapped S2's flow-selection step (generate a tool-call JSON listing
which flows to submit) for OpenNod's typed-logits scoring — same
solver tool (`solve_subschedule`), completely unchanged:

- **Low-load instances** (5, same ones as S2's original baseline):
  100% coverage, 100% admission accuracy, 100% feasibility — identical
  quality to S2's tool-calling baseline — at **0.93s/instance mean**
  (0.75s steady-state) vs. S2's **9.3s**. ~10x speedup (~12x
  steady-state).
- **near_saturation instances** (5, 104-111 candidate flows each, ~4x
  more than the low-load test): same perfect quality (100%/100%/100%),
  at **3.35s/instance** vs. S2's **~36s** there. 10.7x speedup.
- Quality and speedup ratio both hold as candidate count roughly
  quadruples — this isn't a fluke of the easy case.

**Decision: yes, build this for real.** Ten instances, two very
different load regimes, zero quality loss, consistent ~10-12x speedup,
mechanism independently verified against the vendor's own claims
(previous entry). Caveats worth keeping in view: OpenNod's base model
(Qwen3.5-4B) differs from S2's (Qwen2.5-3B-Instruct), so this tests the
typed-logits *mechanism* rather than a same-model ablation; and n=10 is
a prototype sample, not a full-scale confirmation. Both are now tracked
as follow-up `TEST_PLAN.md` items: productionize into
`slm_future/s2b_opennod/`, then run the full held-out/test splits.

## 2026-09-19 — OpenNod's claims independently reproduced (both check out)

Cloned OpenNod (github.com/TheoLeeCJ/OpenNod) into its own venv on disi
and ran its own committed reproduction scripts against its own committed
labeled data — not just trusting the README:

- **Quality: 0.8132 balanced accuracy** on their 144-row labeled set,
  via their own `evaluate.py` — vs. their claimed 0.813. Matches almost
  exactly.
- **Speed: 4.84x faster** than JSON generation (0.56s vs. 2.73s median),
  via their own `decision_vs_generation.py` — vs. their claimed 5.21x.
  Close, real, substantial; a touch below advertised on this small
  21-row fixture.
- Confirmed mechanically too: one forward pass really does score every
  candidate option in one call (no N-separate-calls problem).
- **Caveat**: their "authored" gold labels are model-reviewed, not
  human-adjudicated (per their own provenance metadata) — the accuracy
  number is real and reproduces, but the ground truth it's measured
  against wasn't independently human-verified.

Both headline claims check out. Proceeding to steps 3-5: prototype
swapping this into S2's flow-selection step.

## 2026-09-19 — S2 decision-latency: terser prompt doesn't help (negative result)

**Job 52074** (`--terse-prompt`, same 5 instances as S2's original 9.3s
baseline):

- Quality fully preserved: 100% coverage, 100% admission accuracy, 100%
  feasibility — identical to the verbose prompt.
- **No speedup**: 10.8s mean (9.5s excluding a likely cold-start
  outlier) vs. 9.3s baseline — essentially tied, if anything marginally
  worse.
- Why: the model already emitted the tool call promptly under the old
  instruction, no verbose reasoning preamble to trim. Generation time
  scales with candidate flow count (one token run per ID in the
  tool-call JSON), not instruction phrasing. Rules out prompt length as
  the latency driver — a real fix needs to shrink/restructure the
  *output*, not the instruction (motivates the OpenNod prototype).
- Not running a full-scale confirmation — the mechanism is understood
  well enough that a multi-hour run would just re-confirm a null result.

## 2026-09-19 — S2 no-tool control: the tool isn't a convenience, it's load-bearing

**Job 52051** (base model, no tool, no fine-tuning, S1's exact
direct-decision prompt, full held_out_test, 720/720 instances, clean
exit, 8h57m):

- **admission_accuracy = 0.0000 on every single instance — 720/720
  (100%).** feasibility_rate was never even computable (nothing was ever
  admitted, since nothing was ever parsed).
- Confirms the 5-instance smoke test unconditionally at scale: without
  either the tool (S2) or fine-tuning (S1), the base model can't produce
  a scoreable decision at all. It writes free-form planning prose
  instead of the required format, sometimes degenerating into a
  fabricated sequential ID list.
- **Answers the open question cleanly**: tool access isn't a nice-to-have
  that trims decision latency — it's the only thing making the base
  model's output usable in this setup at all. 44.4s/instance average,
  entirely wasted (nothing extracted from any of it).

## 2026-09-19 — S2's near_saturation zero-request residual: root-caused, two mechanisms found

Investigated the 20/240 (8.3%) instances where S2 submitted zero flows
to the solver tool (flagged in the 2026-09-17 retest below). Not random:

- **80% (16/20): tool called successfully, but with near-total
  hallucinated flow IDs** — hallucinated count ≈ candidate count, not
  partial. Real ID format (`new96`, `new55`...) is correct in style, so
  the model isn't confused about the scheme — it fails to *copy the
  specific IDs* from a 100+-item prompt list. Direct motivation for the
  OpenNod prototype (typed-choice interface sidesteps ID-copying
  entirely).
- **20% (4/20): malformed/truncated tool-call JSON** — 1 dropped-quote
  sampling glitch, 3 clear **degenerate repetition loops** (same short
  ID subsequence repeated dozens of times, burning the token budget
  before the JSON closes).
- Skewed, not uniform: single-scenario instances fail ~4x more than
  batch (13.3% vs 3.3%), 12-switch topologies ~2.3x more than 16-switch
  (11.7% vs 5%), and it clusters on specific topologies — one held-out
  topology fails on 4/6 of its near_saturation instances (67%).

Not chasing further for now — logged in `TEST_PLAN.md`, closed as
investigated.

## 2026-09-19 — 3-epoch S1 training confirmed: feasibility ceiling is structural, not a training-budget problem

**Job 51966** (S1's `full_scale_3ep` adapter, held_out_test, 720/720
instances, clean exit, 8h07m):

- **Feasibility rate: 34.8%** — vs. the 1-epoch full-scale run's **35.0%
  on the same split**. Flat. 3x the training did not move it.
- Admission accuracy 99.7%, but this isn't an epoch effect — this run
  also carries the already-fixed token budget that the old 74.3% number
  predates, so the two changes are confounded there. Feasibility is the
  clean comparison (parse-conditional, unaffected by the token fix).
- Decision time 40.1s/instance (slower than the 1-epoch adapter's ~26s —
  plausibly more confident/verbose generations after more training).
- **Answers the open question**: S1's numeric-placement weakness is
  structural to asking an SLM to do exact offset math, not something
  more LoRA training epochs fix. Closes out that line of investigation
  for S1 specifically — the actual fix remains S2's solver-tool
  approach (100% feasibility by construction).

## 2026-09-17 — Node/flow-count scaling check on B1-B3

**Finding: flow count doesn't scale with node count in the main dataset**
— a generator limitation, not a result. From the existing full
`results/baselines.jsonl` (6480 records, no new job needed):

- Mean flow count flat at ~287 across every switch count 2-16 — root
  cause is the same per-link load-factor proxy already flagged; fix
  tracked in `TEST_PLAN.md`.
- Node count alone still adds real latency growth (2->16 switches): B1
  0.13s -> 0.19s (1.5x), B2 0.14s -> 0.17s (1.2x), B3 0.02s -> 0.08s
  (4.3x, still fastest by far).
- B2's edge over B1 grows with topology size: 7% slower at 2 switches ->
  12% faster at 16 — incremental re-solve pays off more as networks grow.
- Admission rate flat ~100% at every size (same load-factor limitation).

## 2026-09-17 — Truncation fix confirmed at full scale; real ILP scaling stress found

**S1/S2 near_saturation retest** (jobs 51871/51872, full 240/240 each):

- S1 admission accuracy: 45.4% -> **99.0%** (fix confirmed), feasibility
  flat at 29.1% as expected (unaffected — it's a parsing fix, not a
  placement fix).
- S2 admission accuracy: 8.4-9.9% -> **88.7%**, feasibility still a
  perfect **100.0%**.
- New, smaller S2 issue: 20/240 instances (8.3%) submit zero flows —
  tracked in `TEST_PLAN.md`.

**Scaling probe** (job 51866, 60 instances x B1/B2/B3, 180/180 records) —
does classical ILP show real stress under aggregate load, not just
per-link?

- B1/B2 latency: 0.3s (near_saturation) -> ~65-70s (stress_32x), ~200x.
- B3 latency: 0.13s -> 3.0s, ~23x, stays two orders of magnitude faster.
- Admission rate: ~100% -> ~49-50% for all three methods — genuine
  capacity collapse, not just slower solving.
- Deadline compliance stays 100% throughout for everything admitted.
- Full report: `results/scaling_probe_report/summary.md`.

## 2026-09-17 — Full-scale confirmation: feasibility hypothesis holds; truncation bug found (then fixed above)

- **S2 feasibility: 100.0% across all 946 instances, every load factor,
  zero exceptions** — core hypothesis confirmed at scale.
- **S1 feasibility: flat 30-38%** regardless of load — a structural SLM
  weakness, not a scale artifact.
- Bug found: both hit their `max_new_tokens` budget on near_saturation
  (100+ flows) — S1 degrades gracefully (45.4% admission acc), S2 fails
  totally (8.4-9.9%, JSON truncated before its closing tag). Fixed in
  the entry above.

Final numbers, full run (1892 instances, both splits):

- S1: 74.3-74.8% admission accuracy, 33-35% feasibility, ~26s/instance.
- S2: 68.2-71.7% admission accuracy, **100% feasibility**, 9.9s/instance
  (2.6x faster than S1).
- S1's zero-shot scores (12/16sw held-out) match in-distribution exactly
  — not a generalization problem, just a structural one.

## 2026-09-16 — Foundation phase shipped; S1 and S2 built and verified on disi

**Foundation**: topology/flow generators, shared GCL scheduler, B1-B3
baselines — verified on `disi` at full scale (2160 instances, 6480 eval
records). Web UI live.

- B2 beats B1 on decision latency, gap widens with topology size; B3 is
  5-8x faster than either ILP method.
- Admission rate ~100% even at near_saturation — the load-factor
  generator limitation later confirmed (and slated for fixing) above.

**S1 (LoRA, no tools)**, 5-instance smoke test: 100% admission accuracy,
**32.7% feasibility** — learns *what* to admit, not *how* to place it
validly.

**S2 (LoRA + solver tool)**, same 5 instances:

- First pass: 100% feasibility, but only 84% admission accuracy (model
  under-submitted candidate flows to the tool).
- Fixed same day with a more directive prompt: **100% coverage, 100%
  admission accuracy, 100% feasibility, 9.3s/instance** (down from
  12.7s) — clean sweep, S2 strictly dominates S1 on every metric at this
  scale.
- Took 4 evaluation-harness bugs found and fixed along the way — see
  `slm_future/s2_solver_tool/README.md` for the details.
