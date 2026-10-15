# B4: GCN-TD3

Primary SOTA baseline to reproduce — Islam & Bin Muslim, IFIP/IEEE
Networking 2024 ([arXiv:2405.05019](https://arxiv.org/abs/2405.05019)).
This is the paper whose stated limitation (must retrain when switch count
changes) the whole research plan is aimed at.

## What it is

A GCN state encoder + TD3 (Fujimoto et al. 2018) actor-critic agent that
processes dynamically-arriving TT flow requests one at a time, with an
ILP fallback (`solve_b1`) that recomputes the whole schedule whenever any
link's GCL length exceeds a threshold ω=128, matching the paper's
Algorithm 1 (`until β = ω`).

- `tsn_bench/serialization/graph_format.py::gcn_td3_node_features` — the
  paper's Table II node-feature schema, transcribed from the paper's
  marked columns exactly (not from Fig. 3's derived shorthand formula,
  which doesn't fully reconcile against the marked table for the BE-queue
  row and isn't reproducible without the paper's own code).
- `gcn.py` — the GCN encoder (2 layers, 64→32, ReLU, Kipf & Welling
  propagation), with a DeepSets-style per-node-type input layer.
- `env.py` — the RL environment: one MDP step per arriving flow, reward
  function eq. (2)-(5) from the paper, ILP-fallback reset on GCL overflow.
- `td3.py` — TD3 agent: actor, twin critics, replay buffer, Algorithm 1's
  training step (delayed policy updates, target soft-updates, twin-critic
  min-target).
- `train.py` / `evaluate.py` — CLI entry points, mirroring S1-S3's
  conventions (JSONL logging, `--data`/`--split`/`--limit`).

## Documented deviations from the paper (and why)

These aren't shortcuts taken silently — each one is forced by either (a)
information the paper doesn't give (no released code, no full
hyperparameter table beyond a citation), or (b) this project's own
standing principle that every method shares one environment
(`MILESTONES.md`), which the paper's setup wasn't designed around.

1. **Action space: one offset, not a per-hop vector.** The paper's action
   includes a *separate* continuous gate-opening-time per bridge on the
   path (literally indexed "first bridge," "second bridge" — it doesn't
   generalize past 2 hops in the paper text). That only makes sense under
   a per-hop-independent timing model. This project's shared scheduling
   simulator (`tsn_bench/scheduling/simulator.py`) is explicitly a
   "no-wait" model: a TT flow's entire path timing is a deterministic
   function of ONE decision variable, its release offset — every other
   baseline (B1-B3) and every SLM method (S1-S3) already treats
   "schedule this flow" as "pick one offset." B4's action is
   `(accept: bool, offset_ns: float in [0, period_ns))` — same decision
   variable as everything else in the project, chosen by a learned policy
   instead of a solver. See `env.py`'s module docstring for the full
   reasoning.
2. **Queue nodes are per-switch, not per-egress-port.** The paper's graph
   puts scheduled/BE queue nodes at egress-port granularity (Fig. 1b
   references `BR1.Q2`, `BR1.Q3` as distinct per-link-direction nodes).
   `tsn_bench`'s `Topology`/`Node` model tracks queues per-switch
   (`Node.num_queues`), the same granularity every other baseline already
   uses. We emit one `scheduled_queue` node and one `be_queue` node per
   switch, aggregating across its egress ports.
3. **No RL action for AVB flows.** Matches the paper's own training setup
   ("Two AVB flows... scheduled statically... using ILP") — AVB flows go
   through the existing deterministic reserved-bandwidth check
   (`cp_sat_common.admit_avb_flows`), same as every other baseline.
4. **An explicit reject penalty was added (not in the paper).** A real
   training run collapsed to always-rejecting every TT flow for 650+
   consecutive epochs, root-caused to eq. (2) scoring an outright reject
   and a failed accept attempt (collision/deadline miss) identically --
   both zero the compliance term, so TD3's policy gradient had a real
   incentive to always reject rather than keep attempting offsets that
   usually collide in an already-dense schedule. Checkpoint inspection
   confirmed this wasn't numerical divergence (no NaN/Inf, all weights
   small and unremarkable) -- a legitimate local optimum of the reward as
   transcribed. Fixed by subtracting a small constant (`REJECT_PENALTY`,
   default 0.1) on an outright reject only, not on a failed accept
   attempt, so the two are no longer reward-equivalent. See `env.py`.
5. **r(B) (BE-queue preemption-guard term) is fixed at 1.** This project
   generates no BE-class flows and has no credit-based-shaping
   simulation (same documented simplification already in place for AVB
   traffic) — there's nothing for r(B) to check.
6. **The GCN encoder is trained end-to-end with the RL losses, not
   pretrained separately.** The paper describes a separate MSE loss
   against an unspecified "actual graph representation" target (Sec.
   IV-B) that isn't reproducible without more detail. We use one shared
   `GCNStateEncoder`, soft-updated alongside the target actor/critics —
   standard shared-feature-extractor actor-critic design.
7. **Graph embedding width is a hyperparameter (default 32), not
   hardcoded to 22.** Fig. 3's "22" output is specific to the paper's own
   toy example (falls out of *their* example's flow/gate-event counts),
   not a stated architectural constant.
8. **Hyperparameters** (γ, τ, policy noise, learning rates) follow
   Fujimoto et al. 2018's defaults, since the paper states it aligned its
   own hyperparameters with that source (`[24, Tab. 3]`) rather than
   listing them itself.

## Status

Pipeline built and syntax-checked locally (2026-09-22); `env.py` verified
end-to-end against a real dataset instance (seeding from a base schedule,
processing TT/AVB flow decisions, GCL threshold tracking, state-graph
construction all confirmed working). `gcn.py`/`td3.py` need actual torch
execution to validate (no local GPU venv) — next step is a small smoke
test on `disi`.

## Not yet done

1. Smoke-test `train.py`/`evaluate.py` on `disi` (a handful of instances)
   to confirm the GCN/TD3 code actually runs and losses are finite.
2. Real training run, then evaluate on `held_out_test`/`test`.
3. Validate against source-paper numbers (admission rate ~90%, jitter as
   low as ~2us, convergence within ~4000 epochs) — on comparable
   topology sizes, acknowledging our topologies/flow distributions aren't
   identical to the paper's toy 2-switch demo, so this is a shape/
   direction comparison, not an exact-number reproduction.
4. Explicitly test the retraining-cost question this whole project
   targets: train a checkpoint at one switch-count range, evaluate
   zero-shot at another (mirroring S1-S3's held-out-by-size split), and
   measure how much accuracy degrades vs. retraining from scratch —
   this is the actual "does GNN-based state let you avoid retraining"
   question, and where B4 differs most directly from S1-S5's approach.

## Compute

Needs a GPU (RL training). `scripts/cluster/submit_b4_train.sbatch` /
`submit_b4_eval.sbatch` target `disi`'s `l40s` partition, reusing the same
`.venv-gpu` as S1/S2/S2b (already has torch + ortools + networkx).
