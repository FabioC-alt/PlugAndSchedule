# Milestones

Living checklist for the "Language-Model-Based Dynamic Scheduling for TSN Node Addition" research
program. Mirrors §8 of the original research plan. The web UI home page parses this file — keep the
checkbox syntax (`- [ ]` / `- [x]`) exact.

## Foundation phase (this repo, current session)

- [x] Repo scaffold (`tsn_bench` package, configs, `.gitignore`, cluster scripts, web UI skeleton)
- [x] Topology generator (Craciunas "config 3" style, sizes {2,4,6,8,12,16}, held-out split)
- [x] Flow generator (TT + AVB, load-factor control, randomized PCP/src/dst/arrival order)
- [x] Shared time-slotted GCL scheduling simulator + metrics (admission, jitter, latency, GCL length)
- [x] B3 heuristic incremental scheduler (flexibility-along-path admissibility check)
- [x] B2 incremental ILP (CP-SAT, re-solves only the affected sub-schedule)
- [x] B1 full ILP (CP-SAT, recompute-everything baseline)
- [x] Dataset generator CLI (train/val/test + held-out-node-count split, ground truth via B1)
- [x] Eval CLI (B1–B3 across the topology×load×scenario×seed matrix, JSONL logging)
- [x] Report CLI (aggregation, Wilcoxon signed-rank tests)
- [x] Docker web UI (milestones, dataset stats, topology/instance viewer, baseline comparison)
- [x] Cluster scripts (`disi` SLURM: env setup, sbatch job templates, results sync)
- [x] Cluster smoke test executed end-to-end on `disi` (dataset gen + B1–B3 via `sbatch`)

## Later phases (scaffolded only — not implemented this session)

- [x] Phase 2: B4 GCN-TD3 reimplementation, validated against source-paper numbers.
      IN PROGRESS (2026-09-22): full pipeline built (GCN encoder, TD3
      agent, RL environment reusing the shared no-wait simulator, ILP
      fallback), grounded directly in the paper (arXiv:2405.05019) with
      documented deviations (single-offset action space, per-switch queue
      nodes, end-to-end-trained GCN -- see `baselines_future/b4_gcn_td3/README.md`).
      Confirmed running end-to-end on disi after fixing a GCL-threshold
      calibration bug (paper's omega=128 was already exceeded by this
      dataset's base schedules; recalibrated to 1024, matching
      `metrics.py`'s existing default). Two distinct RL failure modes
      were found and fixed in training (always-reject collapse -- a
      reward design flaw, not numerical divergence -- fixed with an
      explicit reject penalty; then a constant-action collapse -- the
      actor saturating to the same action regardless of state -- worked
      around with checkpoint versioning so a pre-collapse snapshot
      survives even when a later one doesn't). Full story in PROGRESS.md
      2026-09-22 through 2026-09-25.

      **Validated at full scale (2026-09-25, job 53347, 720 held_out_test
      instances, checkpoint `b4_v3_inst0180.pt`)**: admission accuracy
      31.1% (low 54.7% / medium 24.9% / near_saturation 13.9%) --
      lowest of any confirmed method in the project. Feasibility a
      PERFECT 100.0% across every instance, zero ILP fallbacks --
      matches S2/S2b/S3's tier. Decision latency 1.20s mean -- the
      FASTEST of any method, faster than S2b's 2.0s. B4 is fast and
      generally never infeasible, but conservative/under-admits relative
      to B1 ground truth.

      **Generalization question answered (2026-09-25, job 53396,
      in-distribution `test` split, 226 instances)**: 20.4% admission
      accuracy, LOWER than held_out_test's 31.1% -- B4 does not show the
      paper's stated retrain-per-switch-count limitation; it performs at
      least as well on unseen switch counts as on the ones it trained on.
      Also found a real (if narrow) harness bug: 8/226 instances (all
      2-switch near_saturation, each triggering 106+ ILP fallback resets)
      had degraded feasibility (down to 5%) because `evaluate.py` didn't
      account for a flow's recorded decision going stale when a fallback
      immediately follows it. **Fixed and confirmed (2026-09-25)**:
      `evaluate.py` now reconciles decisions against the true final
      schedule. Re-run on both splits -- `test` is now a clean 100.0%
      feasible (was 96.9%), `held_out_test` unchanged at 100.0% (as
      expected, since it had 0 fallbacks). Admission accuracy essentially
      untouched on both (20.4% / 31.2%), confirming the bug only ever
      affected feasibility, never the underlying decisions.

      **Compared against the paper's own numbers (2026-09-28)**: the
      paper's "~90% admission rate" is measured during training itself,
      on one fixed small topology, with no independent ground truth --
      not comparable on its face to `admission_accuracy` (scored against
      B1). Added raw admission-count logging and found B1's ground truth
      admits literally 100% of new flows in every one of 946 instances
      across both splits (zero exceptions) -- meaning
      `admission_accuracy` in this dataset already *is* B4's own raw
      admission rate, scored against a constant "always admit" ceiling.
      B4's flow-pooled raw admission rate: 22.8% (`held_out_test`), 17.1%
      (`test`), 21.5% combined -- well under both B1's 100% ceiling and
      the paper's own reported ~90%. Also: the paper never tests its own
      stated retrain-per-switch-count limitation (no held-out-by-size
      split in the paper at all), so this project's generalization
      result above has no corresponding number in the paper to compare
      against -- it's an original contribution, not a reproduction. Full
      writeup and honest interpretation (most plausibly explained by the
      same constant-action collapse dynamic the pre-collapse checkpoint
      sits close to, not by the paper's reward design) in `TEST_PLAN.md`.
- [ ] Phase 2 (optional): B5 GCN-DDQN / B6 GCN-DDPG ablations
- [x] Phase 4: S1 — SLM + LoRA (no tools/retrieval) trained and evaluated. **Full-scale
      confirmation across both splits (946 instances)**: feasibility rate flat at ~34-35%
      regardless of load factor AND regardless of topology size (zero-shot 12sw/16sw and
      in-distribution all score the same) — confirms this is a structural weakness, not a
      scale or generalization artifact. Admission accuracy originally collapsed with load
      (low 99.9% -> medium 77.5% -> near_saturation 45.4%) due to a generation-length
      truncation bug at 100+ flows; **fix confirmed 2026-09-17 on the full 240-instance
      near_saturation retest — admission accuracy 45.4% -> 99.0%, feasibility unchanged
      at 29.1% as expected**. **Training-budget question closed 2026-09-19**: a 3-epoch
      re-train (vs. the original 1 epoch) shows the same flat feasibility on the same
      held_out_test split (35.0% -> 34.8%) — confirms the ceiling is structural, not
      fixable by more training. See `slm_future/s1_lora/README.md`.
- [x] Phase 4: S2 — SLM + solver-tool ("hybrid") implemented and evaluated. **Full-scale
      confirmation across 946 instances (held_out_test 720 + test 226, every load
      factor): feasibility rate is exactly 100.0% with zero exceptions** — the
      solver-guaranteed-valid property holds unconditionally at scale, not just in the
      5-instance smoke test. Admission accuracy originally collapsed even harder than
      S1's at near_saturation load (8-10%, worse than S1's 45%) due to the same class of
      generation-length truncation bug, but total (S2's truncated tool-call JSON failed
      the closing-tag regex entirely, vs. S1's graceful partial degradation); **fix
      confirmed 2026-09-17 on the full 240-instance near_saturation retest — admission
      accuracy 8.4-9.9% -> 88.7%, feasibility still 100.0%**. A smaller residual issue
      (20/240 instances request zero flows) was root-caused 2026-09-19 (ID-copying
      hallucination + degenerate repetition loops, not chased further). **No-tool control
      confirmed 2026-09-19 (720/720 instances)**: without the tool, the base model scores
      admission_accuracy = 0.0000 on every instance — the tool isn't a convenience, it's
      the only thing making the base model's output usable at all. See
      `slm_future/s2_solver_tool/README.md`.
- [x] Phase 4 (additional, not in the original S1-S5 lineup): S2b — S2's solver-tool
      agent with OpenNod (github.com/TheoLeeCJ/OpenNod) typed-logits flow selection
      instead of generated tool-call JSON; solver tool itself unchanged. Motivated by
      root-causing S2's near_saturation zero-request residual (80% was an ID-*copying*
      failure, not a scheme mismatch) — pointed straight at typed-decision scoring as
      the fix. **Full-scale confirmation across the same 946 instances (held_out_test
      720 + test 226, every load factor), 2026-09-19: admission accuracy 99.79%
      (held_out_test) / 100.00% (test), feasibility 100.00% on both, decision time
      ~1.8-2.0s/instance.** Strictly dominates S2: ~30pp higher admission accuracy,
      matching perfect feasibility, ~5x faster. Caveat: OpenNod's base model
      (Qwen3.5-4B) differs from S2's (Qwen2.5-3B-Instruct) — a mechanism win, not a
      validated same-model ablation. See `slm_future/s2b_opennod/README.md`.
- [x] Phase 4: S3 — SLM + GraphRAG implemented and evaluated. S2 variant:
      same base model + tool loop, prompt shows only a retrieval-narrowed
      subtopology (new node(s) + every new flow's shortest-path route, no
      k-hop buffer -- a buffer was found to wash out most of the
      reduction once routes are counted correctly). **Full-scale
      confirmation, held_out_test (720 instances), 2026-09-21: admission
      accuracy 95.4%, feasibility 100.0% -- matches S2's held_out_test
      accuracy (~95.0%) almost exactly, with a genuine 40.3% average
      context reduction on top.** Decision latency is NOT improved
      (~22.8s vs S2's ~19.2s blended), including an unexplained ~1.8x
      slowdown at medium load specifically that isn't a prompt-length or
      flow-count effect (open question, not chased further). Net: a
      quality-preserving context-reduction technique, not a latency win.
      **In-distribution `test` split added 2026-09-25 for completeness**:
      98.0% accuracy, 100.0% feasibility, 23.3s latency, 23.0% context
      reduction -- slightly better than held_out_test, but not a
      generalization story (S3 has no trained weights); just reflects
      smaller topologies being easier, same as S1/S2/S2b. See
      `slm_future/s3_graphrag/README.md`.
- [x] Phase 4: S4 — full agent (GNN-tool + solver-tool) implemented,
      evaluated, and its reliability gap fixed (stretch). S3's retrieval
      mechanism offered as a second, OPTIONAL tool alongside S2's
      `solve_subschedule` -- the model decides for itself whether to
      retrieve, instead of S3's always-retrieve. **First evaluation
      (2026-09-25)** found a real reliability gap, not a quality one: on
      `near_saturation` instances, generation sometimes stopped abruptly
      well under the token budget partway through listing flow IDs
      (confirmed via comparison against S2 on identical instance IDs --
      S2 succeeded where S4 failed), scaling with topology size (41.4% on
      `test`'s 2-8sw vs. 65.0% on `held_out_test`'s 12-16sw).
      **Fixed and reconfirmed at full scale (2026-09-28)**, after two
      false starts (a token-repetition penalty that corrupted the tool's
      own name; an early latency fix that missed an unbounded wrap-up
      round already latent in the original design) -- full story in
      `TEST_PLAN.md`. Final numbers, both splits:

      | Split | Admission accuracy | Feasibility | near_saturation failure rate |
      |---|---|---|---|
      | `test` (226) | 93.3% | 100.0% | 8.6% |
      | `held_out_test` (720) | 87.93% | 100.0% | 22.1% |

      Whenever the tool call succeeds, S4 matches or beats S2 (88.7%
      near_saturation accuracy) and S3 (90.4%) on the same split --
      97.8%/92.74% among successful calls. Latency cost is real but
      bounded (~1.5x baseline). See `slm_future/s4_full_agent/README.md`
      and `TEST_PLAN.md`.
- [x] Phase 5: full evaluation matrix run across B1–B6/S1–S5, all metrics +
      statistical tests. Scoped to the 9 methods confirmed at full scale
      (B5/B6/S5 remain unbuilt stubs). Built `tsn_bench/eval/phase5_report.py`
      -- runs locally, pure pandas/scipy, no cluster needed. Key finding
      beyond the numbers: B1/B2/B3 and B4/S1-S4 aren't just scored
      differently, they're evaluated on different PROBLEM SCOPES (whole
      topology vs. new-flows-only against a fixed base schedule), so the
      report produces three separate comparisons rather than one
      leaderboard -- see `TEST_PLAN.md` for the full tables. Headline:
      **S2b is the strongest all-around confirmed method** (99.8%
      accuracy, 100% feasibility, 2.0s latency -- matches S1's accuracy
      without S1's feasibility collapse, at 1/20th the latency). Every
      pairwise model-group comparison is significant at p<0.001. Every
      classical baseline (B1/B2/B3) beats every model-based method on
      raw latency, despite solving a strictly bigger problem each time.
- [ ] Phase 7: paper draft assembled from results

## Known simplifications (track before writing the paper's limitations section)

- Scheduling environment is a custom pure-Python/NetworkX time-slotted GCL model, not OMNeT++/INET
  (GCN-TD3's original environment). Same environment used for every method, so comparisons stay fair,
  but absolute numbers won't be directly comparable to OMNeT++-based papers.
- ILP baselines use OR-Tools CP-SAT, not CPLEX (no license available). Documented per plan §6.
- AVB flows are admitted via a reserved-bandwidth budget check, not a full credit-based-shaping (CBS)
  simulation.
- B1/B2's constraint model is a defensible simplification of Craciunas et al.'s formulation (interval
  variables + no-overlap per link/queue + deadline windows), not a line-for-line reproduction.

## Cluster notes (disi)

- Deploy under `/scratch.hpc/<username>/TSNAgent`, not the home directory — home has a 10GB quota
  that was already exhausted by unrelated files on this account before this project touched it.
- No `rsync` in this project's local Git Bash shell; `scripts/cluster/sync_results.sh` falls back to
  tar-over-ssh automatically. `git`/`docker` work fine locally.
- End-to-end verified 2026-09-16: `sbatch scripts/cluster/submit_dataset_gen.sbatch` (job 51771) then
  `submit_baselines.sbatch` (job 51772) on the `amdepyc7713` CPU partition, both completed cleanly and
  reproduced byte-identical instance IDs to the local `--quick` run (confirms deterministic seeding).
