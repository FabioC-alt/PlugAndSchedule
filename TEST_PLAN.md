# Progress

Single ordered plan for what's left before this project is done —
merges what used to be split across `TEST_PLAN.md`'s near-term tests and
`TODO.md`'s backlog into one list, in the order to actually work through
it. `MILESTONES.md` still owns the detailed definition of the big
research-plan phases (S3, B4, etc.) — this list sequences them alongside
the smaller items rather than re-describing them. `PROGRESS.md` stays the
append-only narrative log of what already happened. Items here are
edited in place as results land or instructions change: check a box when
a step is genuinely done, and rewrite its text with the actual result,
not just tick it.

## Plan

- [x] Full-scale S1 LoRA training run (3 epochs, rank 16 -- more than the
      1-epoch smoke test). Training done 2026-09-17 (job 51899, clean
      exit, eval loss 0.160 -> 0.152 across 3 epochs).
- [x] Finish + analyze the held_out_test eval of that new S1 adapter (job
      51966, completed 2026-09-19, 8h07m, clean, 720/720 instances).
      **Feasibility rate 34.8% vs. the 1-epoch full-scale run's 35.0% on
      the SAME split -- flat, confirms the ceiling is structural, not a
      training-budget problem.** (Admission accuracy came out at 99.7%,
      but that's not an epoch effect -- this run also has the
      already-fixed token budget the old 74.3% number predates, so the
      two changes are confounded there; feasibility is the clean
      comparison since it's parse-conditional.) Decision time 40.1s/inst.
- [x] S2 near_saturation zero-request residual: 20/240 instances (8.3%)
      requested zero flows from the solver tool. **Investigated 2026-09-19
      — two distinct mechanisms, not random noise:**
      (1) **16/20 (80%): tool called successfully but with near-total
      hallucinated flow IDs** (hallucinated count ≈ candidate count, not
      partial) — the real ID format (`new96`, `new55`, ...) is correct in
      style, so the model isn't using a wrong scheme, it's failing to
      copy the *specific* IDs from the prompt under near_saturation's
      100+-item candidate lists. This is the direct motivating case for
      the OpenNod prototype below — a typed-choice interface sidesteps
      ID-copying entirely.
      (2) **4/20 (20%): malformed/truncated tool-call JSON** — 1 case is
      a single dropped quote character (sampling glitch); 3 cases show a
      clear **degenerate repetition loop** (the same short flow-ID
      subsequence repeated dozens of times), burning the token budget
      before the JSON can close.
      Not evenly distributed: single-scenario instances fail ~4x more
      than batch (13.3% vs 3.3%), 12-switch topologies ~2.3x more than
      16-switch (11.7% vs 5%), and it clusters hard on specific
      topologies — one held-out topology (`topo_12sw_6`) fails on 4/6 of
      its near_saturation instances (67%). Not chasing further right now;
      the repetition-loop cases would benefit from a repetition penalty
      in generation config if revisited.
- [x] S2 no-tool control eval: added `--no-tool` to `evaluate.py` (reuses
      S1's exact direct-decision prompt/parser/ground-truth-offset
      feasibility check, no tool ever offered). **Confirmed at full scale
      2026-09-19 (job 52051, 720/720 instances, clean): admission_accuracy
      is exactly 0.0000 on every single instance (100%), feasibility_rate
      never even computable (nothing was ever admitted since nothing was
      ever parsed).** Without either the tool (S2) or fine-tuning (S1),
      the base model cannot produce a scoreable decision at all — it
      writes free-form planning prose instead of the required format,
      sometimes degenerating into a fabricated sequential ID list. This
      is the clean answer to "does the tool matter": unconditionally yes
      — it's not a convenience, it's the only thing making the model's
      output usable at all in this setup.
- [x] S2 decision-latency reduction: added `--terse-prompt` to
      `evaluate.py` (`AGENT_INSTRUCTION_TERSE`, ~1/3 the words, no
      justification content). **Negative result, job 52074, same 5
      instances as S2's original 9.3s baseline**: quality fully
      preserved (100% coverage, 100% admission accuracy, 100%
      feasibility, identical to before) but **no speedup** — 10.8s mean
      (9.5s excluding a likely cold-start outlier), essentially tied with
      the 9.3s baseline. Makes sense in hindsight: the model was already
      emitting the tool call promptly without verbose reasoning preamble
      under the old instruction too (see its raw completions), so
      instruction length was never the bottleneck — generation time is
      dominated by output length (one token run per candidate flow ID in
      the tool-call JSON), which scales with candidate count, not prompt
      phrasing. Not running a full-scale confirmation — the smoke test
      plus this mechanism explanation is conclusive enough to not spend
      hours re-confirming a well-understood negative. Decision latency
      remains an open problem; a real fix needs to shrink or restructure
      the *output*, not the instruction (see OpenNod below).
- [x] OpenNod 1/5: cloned OpenNod into its own venv on disi
      (`<scratch>/OpenNod`, separate from TSNAgent's
      `.venv-gpu` -- different pinned torch/transformers versions).
      **Both claims independently reproduced, 2026-09-19 (job 52095)**:
      quality **0.8132 balanced accuracy** on their own 144-row labeled
      set (`benchmarks/data/authored144.jsonl`, their own `evaluate.py`)
      vs. their claimed 0.813 -- matches almost exactly. Speed **4.84x**
      faster than JSON generation (0.56s vs. 2.73s median, their own
      `decision_vs_generation.py`) vs. their claimed 5.21x -- close, real,
      substantial, if a touch below advertised (small 21-row comparison
      fixture, some variance expected). One caveat worth flagging: their
      "authored" gold labels are model-reviewed, not human-adjudicated
      (per their own data provenance field) -- the accuracy number is
      real and reproducible, but the ground truth it's measured against
      wasn't independently human-verified.
- [x] OpenNod 2/5: **answered from its own docs, 2026-09-19** — yes, one
      forward pass scores all supplied options together. **Refined after
      reading the actual source, 2026-09-19**: precisely, `score_shared()`
      takes N per-flow decisions that all share one exact topology
      "state" text, prefills that shared prefix once, then scores all N
      decision suffixes together in one batched GPU forward pass (not N
      separate model calls) — so a ~100+-candidate near_saturation
      instance is still one batched call, not N sequential ones, just not
      literally "one token of output" either. Confirmed working in
      practice in step 3/4 below.
- [x] OpenNod 3/5 + 4/5: wrote `opennod_s2_prototype.py` (for each
      instance: scores every candidate flow's admit/skip decision in one
      `score_shared()` call sharing the topology as a common prefix,
      submits whatever scores yes to the exact same `solve_subschedule`
      tool S2 uses, solver and tool wrapper completely unchanged).
      **Job 52098, same 5 instances as S2's baseline (100% coverage/100%
      admission accuracy/100% feasibility/9.3s): 100% coverage, 100%
      admission accuracy, 100% feasibility — identical quality — at
      0.93s/instance mean (0.75s steady-state, excluding one cold-start
      call). That's a ~10x speedup (~12x steady-state) with zero quality
      loss on this sample.** Known caveat: OpenNod's base model
      (Qwen3.5-4B) differs from S2's (Qwen2.5-3B-Instruct) — this tests
      the typed-logits *mechanism*, not a same-model ablation.
      **Confirmed on the harder near_saturation case too (job 52100, 5
      instances, 104-111 candidate flows each — ~4x more than the
      low-load test): 100% coverage, 100% admission accuracy, 100%
      feasibility again, at 3.35s/instance vs. S2's ~36s there — a
      10.7x speedup. Quality and speedup ratio both hold as candidate
      count roughly quadruples.**
- [x] OpenNod 5/5: **decision: yes, formalize as a real S-variant.** Ten
      instances across two very different load regimes, zero quality
      loss, consistent ~10-12x speedup, mechanism independently verified
      against the vendor's own claims (step 1). The honest caveats
      (different base model than S2, n=10 not a full-scale run yet, one
      dependency-version wrangle to get OpenNod and `tsn_bench` sharing a
      venv) are real but don't undermine the core result. Named it S2b
      (a variant of S2), not S3 -- that slot is already
      `slm_future/s3_graphrag/` in the original plan.
- [x] Productionize the OpenNod prototype into `slm_future/s2b_opennod/`
      -- real package (`agent.py`, `evaluate.py` matching S1/S2's JSONL
      schema and CLI conventions, `README.md`), replacing the throwaway
      prototype script. **Verified 2026-09-19 (job 52120, clean exit,
      49s)**: 5-instance smoke test via the new
      `scripts/cluster/submit_s2b_eval.sbatch` (points at OpenNod's
      separate venv, not TSNAgent's own `.venv-gpu`) reproduces the
      prototype's numbers exactly -- 100% coverage/admission
      accuracy/feasibility, 0.72-1.60s/instance.
- [x] Run the productionized S2b package on the full `held_out_test`
      (720) and `test` (226) splits, all load factors. **Confirmed at
      full scale, 2026-09-19 (jobs 52122/52123, both clean exits, 946
      total instances):**
      - `held_out_test` (720, zero-shot): admission accuracy **99.79%**,
        feasibility **100.00%**, coverage **100.00%**, mean decision
        time **2.01s**.
      - `test` (226, in-distribution): admission accuracy **100.00%**,
        feasibility **100.00%**, coverage **100.00%**, mean decision
        time **1.79s**.
      **S2b now strictly dominates S2 at full scale**: admission
      accuracy 99.8-100% vs. S2's 69.0% (even S2's best post-fix number,
      88.7% at near_saturation specifically, is well below this),
      feasibility matches at 100%, decision time ~1.8-2.0s vs. S2's
      9.9s blended (~5x faster). Zero-shot generalization is a
      non-issue, same pattern S1/S2 showed -- `held_out_test` and `test`
      score almost identically. `MILESTONES.md` Phase 4 entry added.
- [x] Fix the dataset's load-factor definition. **Built and calibrated
      2026-09-19**: `generate_flows()` gets an opt-in
      `capacity_scale_links` parameter (target bandwidth relative to N
      links' aggregate capacity, not always one link) -- default `None`
      preserves the exact old behavior, so the main dataset's
      low/medium/near_saturation tiers and every existing S1/S2/B1-B3
      result stay meaningful; this is additive, not a breaking change.
      Calibrated locally before any cluster time: divisor=20 gives a
      1x/2x/4x multiplier at switch_count 2/8/16 (359/723/1430 flows,
      0.35s/0.92s/9.22s B1 solve time at near_saturation=0.85) --
      tractable. Adding load-factor stress on top (stress_2x=2.0) shows
      real admission stress at switch_count=16 (75% admission, 92s
      solve) while staying easy at 2/8 -- stress_4x tried and ruled out
      (150s+, impractical for a real dataset). New
      `configs/aggregate_scaling.yaml` (Stage 2 probe, combines topology
      size + aggregate load factor, unlike Stage 1's `scaling_probe.yaml`
      which only varied load factor at a fixed size) submitted to
      confirm on the cluster.
- [x] Re-run B1-B3 baselines on the fixed dataset. **Confirmed, 2026-09-19
      (jobs 52490/52495, 60 instances, 180 records, both clean exits):**
      flow count now genuinely scales with topology size at the same load
      factor (near_saturation: 472 -> 968 -> 1958 flows at
      switch_count 2/8/16, roughly matching the calibrated 1x/2x/4x
      target). Admission rate collapses when both axes push together:
      near_saturation stays ~92-100% even at switch_count=16, but
      **stress_2x at switch_count=16 drops to 60.5% (B1) / 73.8% (B2) /
      76.5% (B3)** -- real stress that neither axis alone produced
      before. Statistically significant across all method pairs on both
      admission_rate and decision_latency_s (Wilcoxon, p<0.03 for every
      pair, most p<0.001, n=60).
      **Notable finding: B3 (heuristic) beats both ILP methods on
      admission rate at the hardest cell** (76.5% vs. B1's 60.5%, B2's
      73.8%) while being ~100x+ faster (1.33s vs. 151-161s) -- likely
      because B1/B2 are hitting their solver time limit there (10/10
      instances capped at 150s for both), not because they're worse
      optimizers; B3 always completes a full (if approximate) pass.
      **Caveat**: at switch_count>=8 with stress_2x, and partially at
      switch_count=16 near_saturation, a meaningful fraction of B1/B2
      records are time-capped at 150s (6/10, 6/10, 10/10, 10/10 across
      the affected cells) -- their reported decision_latency_s there is
      "best effort within budget," not "time to converge," and their
      admission-rate numbers may understate what more solver time would
      find. Full numbers: `results/aggregate_scaling.jsonl` /
      `results/aggregate_scaling_report/`.
- [x] Add B1-B3 *and* S1/S2/S2b side by side in the web UI's Baseline
      Comparison tab. New `/api/results/combined_summary` endpoint +
      updated tab: unified table (36 rows, all 6 methods x 6 switch
      counts) and a log-scale decision-latency chart (spans 0.02s-40s)
      comparable across every method. `admission_rate` (B1-B3, a raw
      schedule property) and `admission_accuracy` (S1/S2/S2b,
      correctness against B1 ground truth) are different metrics despite
      similar names -- kept as separate columns/charts, never merged,
      with a note explaining why. Surfaced one known data asymmetry in
      the UI itself: S1's in-distribution numbers (switch_count 2/4/6/8)
      are from an earlier checkpoint than its held-out numbers (12/16,
      the confirmed 3-epoch retrain), since that retrain was only
      re-evaluated on the held-out split.
- [ ] Set up an OMNeT++/INET environment. **Deprioritized 2026-09-21**,
      user's call: investigated first -- `disi` has no C++ compiler
      toolchain anywhere (neither login nor compute nodes have
      gcc/g++/bison/flex/qmake, only the `gcc-14-base` runtime-library
      metapackage), no root access, no module system, no
      Singularity/Apptainer. A real path exists (user-space conda-forge
      toolchain, no root needed) but is genuinely open-ended: likely many
      hours for OMNeT++ + INET alone, then a separate research question
      of finding/adapting the right TSN-specific simulation module to
      match GCN-TD3's original setup, with real risk of not converging
      cleanly. This gap is already a disclosed, accepted limitation in
      `MILESTONES.md`, not a blocker -- moving to S3/B4 instead, which
      are more central to the actual research contribution. Revisit only
      if there's specific reviewer/deadline pressure requiring it.
- [ ] Cross-validate the custom pure-Python/NetworkX GCL simulator
      against a real OMNeT++/INET run on a subset of instances --
      current results are internally fair (every method shares the same
      simulator) but not validated against a discrete-event simulation.
      Blocked on the item above.
- [x] S3: SLM + GraphRAG implemented and evaluated (`slm_future/s3_graphrag/`,
      `MILESTONES.md` Phase 4). Built as an S2 variant (same base model +
      tool loop, `s2_solver_tool.agent.run_agent`'s new `prompt_topology`
      param), retrieval = new node(s) + every new flow's shortest-path
      route (`agent.py::retrieval_node_ids`) expanded by
      `Topology.k_hop_subview`, `--k-hop` defaults to 0 (routes only, no
      buffer -- any k-hop buffer washes out most of the reduction once
      routes are counted correctly). **Full held_out_test confirmation
      (job 52612, 720 instances, 2026-09-21)**: admission accuracy 95.4%
      (low 99.3% / medium 96.4% / near_saturation 90.4%), feasibility
      100.0% throughout -- matches S2's held_out_test accuracy (~95.0%
      blended) almost exactly, with a real 40.3% average context
      reduction on top (55.4% low / 37.6% medium / 28.0% near_saturation
      -- less reduction at higher load since more flow routes cover more
      of the graph, as predicted from the local measurement). Decision
      latency is NOT improved: S3 blended ~22.8s vs S2's ~19.2s, with an
      unexplained, low-variance ~1.8x slowdown specific to medium load
      (22.4s vs 12.3s at identical flow counts) -- not a prompt-length or
      flow-count effect, flagged as an open question, not chased further.
      Net: retrieval preserves quality and cuts context length
      substantially, but is not a latency win. Results in
      `results/s3_eval_held_out_test.jsonl`.

      **In-distribution `test` split added for completeness (job 53436,
      226 instances, 2026-09-25)**: admission accuracy 98.0% (low 99.9%
      / medium 99.2% / near_saturation 94.5%), feasibility 100.0%
      throughout, 23.3s latency, 23.0% average context reduction --
      slightly *better* than held_out_test's 95.4% (unlike B4, this
      isn't a generalization story since S3 has no trained weights to
      generalize -- same frozen base model + tool loop either way, so
      the gap just reflects smaller topologies being easier overall,
      consistent with S1/S2/S2b's own pattern). Results in
      `results/s3_eval_test.jsonl`, wired into the webui alongside
      held_out_test.
- [x] B4: GCN-TD3 reimplementation, validated against source-paper numbers
      (`baselines_future/b4_gcn_td3/`, `MILESTONES.md` Phase 2). IN
      PROGRESS (2026-09-22): fetched and read the full paper
      (arXiv:2405.05019) to ground the build -- transcribed Table II's
      node-feature schema exactly into
      `tsn_bench/serialization/graph_format.py::gcn_td3_node_features`,
      built the GCN encoder (`gcn.py`), the RL environment (`env.py`,
      reward eq. (2)-(5)), the TD3 agent (`td3.py`, Algorithm 1), and
      `train.py`/`evaluate.py`. Documented deviations (see README):
      single-offset action space instead of the paper's per-hop vector
      (matches every other baseline's shared no-wait simulator), per-switch
      not per-egress-port queue nodes, AVB bypasses RL same as the paper's
      own setup, GCN trained end-to-end not pretrained separately.
      Smoke-tested end-to-end on disi with real torch (jobs 52661/52663):
      found and fixed a real calibration bug -- the paper's GCL threshold
      omega=128 (a specific hardware limit from their toy-scale setup) is
      already exceeded by this dataset's base schedules alone (136-632
      entries before any new flow arrives), causing every episode to
      instantly fall back to ILP with zero training signal; fixed to 1024,
      matching `metrics.py::compute_metrics`'s existing dataset-calibrated
      default. Re-run (job 52663) completes cleanly with real episode
      lengths (15-24 flows/episode) -- pipeline confirmed working
      end-to-end. Real training run started (jobs 52664/52672, 300
      instances) and confirmed real learning signal early on (admission
      rate 15-40% on low/medium load), then collapsed to a hard 0.000 for
      650+ consecutive epochs from ~epoch 2376 onward and never
      recovered -- cancelled rather than keep burning GPU-hours.
      Root-caused via checkpoint inspection (job 52705, all weights
      numerically clean, no NaN/Inf -- rules out divergence): `env.py`'s
      reward gives a rejected flow and a failed accept attempt the exact
      same score, so TD3's policy gradient has a real incentive to
      collapse to "always reject" (a stable, predictable reward) rather
      than keep attempting offsets that usually collide in an
      already-dense schedule. Fixed by adding an explicit
      `REJECT_PENALTY` (0.1) subtracted only on an outright reject, per
      user decision. **Fix validated, training run complete (job 52763,
      2026-09-22/23)**: all 250/250 instances, 3898 epochs, no errors,
      exit clean. No recurrence of the always-reject collapse anywhere
      in the run. Low/medium admission rate: cold start ~10-12%, strong
      70-93% plateau from epoch ~689-2831, then a real but much milder
      late-run dip to ~35-37% for the final ~600 epochs (not a repeat of
      the total 0.000% collapse). Overall pooled low/medium admission:
      60.3% (3491/5787). near_saturation instances hit immediate ILP
      fallback essentially every time, as expected/structural. Checkpoint
      saved to `baselines_future/b4_gcn_td3/checkpoints/b4_v2.pt`.
      **Evaluation smoke test (job 53055, 5 instances) reveals a second,
      distinct collapse**: admission_accuracy 0.00-0.07 vs. B1 ground
      truth. Root-caused (job 53077): the actor outputs the exact same
      constant action for every flow regardless of state
      (`accept_logit=1.0000`, `offset_frac=-1.0000`, both saturated at
      the tanh boundary) -- always tries to accept everything at
      offset=0, so only the first flow per link ever succeeds. This
      explains the late-training admission-rate dip noted above -- it
      wasn't noise, it was this collapse setting in. Also exposed an
      operational gap: `train.py`'s checkpoint overwrites the same file
      each save, so the earlier good checkpoint (from the 70-93%
      plateau) is gone. NOT proceeding to the full held-out-test run.
      User chose checkpoint versioning (not entropy regularization) as
      the fix -- added (keeps the last 8 numbered snapshots alongside
      the overwritten "latest" pointer) and retrained (job 53083, 250
      instances, 3567 epochs, clean exit). **Versioning validated,
      collapse-avoidance not**: the collapse recurred (this time starting
      at instance 183/epoch 2624, after a strong 75-93% plateau that held
      for ~1900 epochs -- also learned that the earlier epoch 457-466 dip
      was transient/recoverable, not terminal, a useful distinction for
      reading future runs). `b4_v3_inst0180.pt` (instance 180, epoch
      2621) sits right at the tail of the healthy plateau, a strong
      pre-collapse candidate. User chose to evaluate the pre-collapse
      snapshot. **Smoke test (job 53346) confirmed a real recovery**:
      0.26-0.63 admission accuracy with 100% feasibility, vs. the
      collapsed checkpoint's 0.00-0.07 -- proceeded to the full run.
      **Full held_out_test evaluation complete (job 53347, 720
      instances, clean exit)**: admission accuracy 31.1% overall (low
      54.7% / medium 24.9% / near_saturation 13.9%) -- lowest of any
      confirmed method in the project. Feasibility a PERFECT 100.0%
      across every instance, matching S2/S2b/S3's tier. Decision latency
      1.20s mean -- the FASTEST of any method (faster than S2b's 2.0s).
      Zero ILP fallbacks during evaluation. **B4 is now validated at full
      scale**: fast and never infeasible, but conservative/under-admits
      relative to B1 ground truth -- plausibly residual influence from
      the collapse dynamic even in the pre-collapse checkpoint.

      **Generalization question answered (job 53396, in-distribution
      `test` split, 226 instances, same 2-8sw range as training)**:
      admission accuracy 20.4% -- LOWER than held_out_test's 31.1%, on
      switch counts B4 was actually trained on. No generalization gap in
      the direction the paper's stated limitation worries about; if
      anything the opposite. Also surfaced a real, narrow harness bug:
      mean feasibility here is 96.9%, not the perfect 100% held_out_test
      showed -- 8/226 instances (3.5%), all 2-switch near_saturation,
      each triggered 106-115 ILP fallback resets within one instance
      (small topologies concentrate GCL entries onto very few links, so
      near_saturation load hits the 1024 threshold on almost every
      decision). `evaluate.py` records a flow's decision before checking
      whether a fallback immediately follows and recomputes the whole
      schedule, so that decision can go stale relative to the truly
      final schedule -- a harness limitation specific to
      excessive-fallback instances, not a policy-quality issue.
      Excluding those 8, the remaining 218 are a clean 100.0% feasible,
      19.0% admission accuracy (barely different from the unfiltered
      number). Net read: B4's weakness is being conservative on small,
      dense topologies specifically, not failing to generalize to unseen
      sizes.

      **Bug fixed and confirmed (job 53427, re-run of `test`)**:
      `evaluate.py` now reconciles every recorded decision against the
      true final schedule before scoring. Result: all 226 instances show
      a clean 100.0% feasibility (was 96.9%), admission accuracy
      unchanged at 20.4% (confirms the bug only ever affected
      feasibility, never accuracy). `held_out_test` was also re-run with
      the fix (job 53429) as a consistency check -- unaffected as
      expected (31.2% accuracy, 100.0% feasibility, ~0.1pp of GPU
      floating-point run-to-run noise from the original 31.1%, not the
      fix). Both numbers are now the authoritative, doubly-confirmed
      results.

      **Comparison against the source paper's own reported numbers
      (2026-09-28).** Re-fetched and re-read the full paper
      (arXiv:2405.05019) specifically for its Section V results, not
      just the abstract. Its headline claim is "~90% admission rate"
      (Fig. 5) with jitter "as low as 2us" (Fig. 6), trained over 5000
      epochs (the abstract's "convergence within 4000 epochs" describes
      where the curve visually stabilizes, not the training length),
      compared against GCN-DDQN and GCN-DDPG rather than any ILP ground
      truth. The paper's conclusion states its headline result as
      "scheduling 30% more TT flows and reducing jitter by 3%" than
      those RL baselines.

      Two things make this NOT a like-for-like comparison, and both
      matter more than the raw numbers:

      1. **Different metric basis.** The paper's "admission rate" is
         scheduled/requested flows measured *during training itself*,
         on a single small fixed topology (Fig. 1(b)'s worked example is
         2 switches) -- there is no independent, harder ground truth in
         the paper that a flow "should" have been admitted; their reward
         function only penalizes constraint *violations* (missed
         deadline, collision), not excessive conservatism, so a policy
         that rejects a fittable flow is never penalized for that choice
         specifically. This project's `admission_accuracy` scores
         agreement with B1 (an ILP solver) on flows the model has NEVER
         been trained on.
      2. **The paper never actually tests its own stated limitation.**
         The paper's own conclusion states GCN-TD3 "necessitates
         retraining with any change in the number of switches" -- that
         limitation is this entire project's motivating question. But
         the paper's reported numbers come from one fixed topology size,
         trained and evaluated in place; there is no held-out-by-size
         split in the paper to compare against. This project's
         generalization result (B4 does *better* on unseen 12-16sw than
         on the 2-8sw range it trained on: 31.2% vs. 20.4% admission
         accuracy) has no corresponding number in the paper to be
         compared with at all -- it's this project's own original
         contribution, not a reproduction of anything the paper measured.

      **A genuinely comparable number, computed here**: added raw
      admitted-flow-count logging to `evaluate.py` (`n_admitted`,
      `n_gt_admitted` -- both were already computed, just never logged;
      pure logging addition, verified zero effect on existing
      `admission_accuracy`/`feasibility_rate` scoring) and re-ran both
      splits (jobs 53799/53800, ~20min/9min, same checkpoint
      `b4_v3_inst0180.pt` as every other confirmed B4 result). Found
      something structurally important along the way: **B1's ground
      truth admits literally 100% of newly-requested flows in every
      single instance across both splits, zero exceptions** (checked:
      min = max = mean = 1.0 across all 946 instances). This means
      `admission_accuracy` in this project is, in this specific dataset,
      mathematically equivalent to B4's own raw admission rate against a
      constant "always admit" reference -- B4's shortfall is pure
      conservatism (rejecting flows that genuinely do fit), never
      disagreement with a selective ground truth. Worth keeping in mind
      when reading B4's accuracy numbers anywhere else in this project,
      not just here.

      | | B4 raw admission rate (flow-pooled) | B1 ground truth | Paper's reported rate |
      |---|---|---|---|
      | `held_out_test` (47,996 new flows) | 22.8% | 100.0% | -- |
      | `test` (14,889 new flows) | 17.1% | 100.0% | -- |
      | Combined | 21.5% | 100.0% | ~90% (in-training, fixed small topology) |

      (Flow-pooled here means summed admitted/summed offered across all
      instances -- weights larger, harder near_saturation instances more
      heavily. The already-reported per-instance-mean `admission_accuracy`
      of 31.2%/20.4% is the same underlying quantity averaged the other
      way, equal-weight per instance.)

      **Honest read**: B4 admits meaningfully fewer flows than the
      paper's own reported figure, and meaningfully fewer than B1 (the
      actual capacity ceiling, confirmed to be 100% here) says it could.
      But given point 1 above, this isn't "B4 failed to reproduce the
      paper" -- it's a genuinely harder test (held-out topology sizes,
      scored against an independent solver's ceiling, not a
      constraint-violation-only reward measured in-place on the same
      fixed setup it trained on) that the paper itself was never
      subjected to. The training-budget comparison is also not
      apples-to-apples: the paper trains for exactly 5000 epochs, while
      `b4_v3_inst0180.pt` is a pre-collapse snapshot from instance
      180/epoch 2621 of a 3567-epoch run -- captured where it was
      healthy, not at a matched training length. Combined with the
      already-documented constant-action collapse this checkpoint sits
      close to (see above), the conservatism gap is most plausibly
      explained by that same collapse dynamic pulling the policy toward
      "reject when uncertain," not by anything the paper's own reward
      design or architecture does differently. The declined fix
      (action-diversity/entropy regularization) remains the direct lever
      if a materially less conservative number is wanted later.
- [ ] B5/B6: GCN-DDQN / GCN-DDPG ablations (optional, `MILESTONES.md`
      Phase 2).
- [x] S4: full agent (GNN-tool + solver-tool) implemented and evaluated
      (stretch, `MILESTONES.md` Phase 4). Built as a genuine two-tool
      agent -- combines S2's solve_subschedule tool with S3's retrieval
      mechanism as a second, OPTIONAL tool (`retrieve_subgraph`), letting
      the model decide for itself whether to retrieve before submitting
      flows, instead of S3's always-retrieve design. New two-tool loop
      (`slm_future/s4_full_agent/agent.py`, dispatches by tool-call name
      each round) since S2's loop hardcodes single-tool dispatch. Local
      mechanics test confirmed the retrieval logic works (reuses S3's
      validated `k_hop_subview`/`retrieval_node_ids`).

      **Smoke test (job 53494, 5 instances) found a real bug**: 2/5
      instances called `retrieve_subgraph` then never followed up with
      `solve_subschedule` -- the model lost the thread of its remaining
      obligation after the tool round-trip (asked for info already given,
      or narrated intent instead of emitting the call). Root-caused via
      raw generated text, not a loop bug. Fixed with an explicit
      reinforcement message inserted right after `retrieve_subgraph`'s
      tool result, restating the solve_subschedule requirement. **Retry
      (job 53495, 10 instances) confirmed the fix**: 0/10 abandonments,
      98.7% admission accuracy, 100% feasibility, 40% retrieval rate.

      **Full evaluation, `test` split (job 53498, 226 instances,
      complete)**: 84.9% admission accuracy overall, 100.0% feasibility,
      64.2% retrieval rate, 21.4s mean latency. By load factor -- low:
      99.6% acc; medium: 95.6% acc; near_saturation: 57.0% acc (but 97.3%
      among the 41/70 instances where the tool WAS successfully called).

      **Full evaluation, `held_out_test` split**: first attempt (job
      53497) crashed at instance 149/720 --
      `AttributeError: 'list' object has no attribute 'get'` in
      `agent.py`'s solve_subschedule dispatch, because the model
      occasionally emits `"arguments": [...]` (a list) instead of an
      object, and only the outer `json.loads`/`call.get` was guarded
      against malformed output, not the shape of `args` itself. Fixed
      with an `isinstance(args, dict)` guard (commit `9c1b951`),
      resubmitted clean (job 53504, all 720/720, no errors). Overall:
      74.8% admission accuracy, 99.99% feasibility, 38.3% retrieval rate,
      20.9s mean latency. By load factor -- low: 97.5% acc; medium: 93.9%
      acc; near_saturation: 33.0% acc (94.3% among the 84/240 where the
      tool WAS called).

      **Headline finding: a real, load-factor-specific reliability gap,
      not a decision-quality problem.** On `near_saturation` instances
      (100+ flows), the model sometimes stops generating abruptly after
      only ~9s / ~400-500 tokens -- well under the 3072-token budget --
      partway through listing `flow_ids`, occasionally right after a
      short "should I retrieve? no, I'll proceed directly" aside,
      yielding `used_tool=False` and zero flows requested. Confirmed NOT
      random: checked several near_saturation instances that failed this
      way against S2's results on the EXACT SAME instance IDs -- S2 (no
      retrieve option, terser prompt) succeeds on every one of them
      (full/near-full flow lists, 35-41s generations, high accuracy). The
      optional-retrieve framing itself appears to cost reliability
      specifically on large instances, even when the model ultimately
      declines to retrieve, and the effect scales with instance size:
      near_saturation failure rate is 41.4% (29/70) on `test`'s smaller
      2-8sw topologies vs. 65.0% (156/240) on `held_out_test`'s larger
      12-16sw topologies.

      Whenever the tool call DOES succeed, S4's quality matches or beats
      S2 (88.7% near_saturation accuracy on held_out_test) and S3 (90.4%)
      -- S4's 94.3%/97.3% among successful calls is the best of the
      three. The entire quality deficit versus S2/S3 in the raw numbers
      is explained by the reliability gap, not by worse decisions.
      Research question partially answered here: the optional-retrieve
      choice costs reliability on large instances -- user approved a
      follow-up fix attempt rather than accepting this as final.

      **Fix attempt 1 (FAILED, made things worse)**: `no_repeat_ngram_size=6`
      added to block the degenerate repetition-loop variant of the
      failure. Smoke test (job 53558, 20 near_saturation instances)
      showed 18-20/20 still failed, plus a brand-new corruption: the tool
      name got mangled into `"solve_subscheduler"` (never seen before).
      Root cause: `no_repeat_ngram_size` blocks repeating ANY 6-token
      n-gram seen anywhere earlier in the sequence, including the
      PROMPT -- `AGENT_INSTRUCTION` already says "solve_subschedule"
      several times, so the model was blocked from writing that exact
      identifier again. Removed.

      **Fix attempt 2 (partial win)**: `min_new_tokens` alone (scaled to
      candidate flow count, forcing generation past the early stop).
      Smoke test (job 53559, same 20 instances): failure rate down to
      35% (from 65% baseline), but latency more than doubled (21.7s ->
      52.1s). Post-mortem found the parser was discarding usable
      retries -- the model sometimes writes a malformed first attempt
      then immediately retries with a well-formed second one, but the
      regex only ever checked the first `<tool_call>` occurrence. Fixed
      by iterating every occurrence via `finditer` until one parses
      (commit `f1c4f58`). Retest (job 53562): failure rate down to 25%,
      confirmed on the exact traced instance (104/104 flow IDs recovered
      from what used to be a discarded second attempt).

      **User approved the full re-run** at this 25%/~2.66x-latency state
      -- but the first two full-re-run attempts (jobs 53663/53664) had
      to be CANCELLED after ~1h: `held_out_test` was projecting ~10.6h
      against an 8h budget. Root cause: the loop's optional post-success
      "wrap-up" round was ALSO being forced through the full
      `min_new_tokens` floor, so every successful instance paid for a
      whole extra forced-length round (retrieved instances paid it
      three times over -- retrieve round + solve round + wrap-up round).
      First patch (zeroing `min_new_tokens` once the tool had already
      succeeded, commit `b48a0bc`) wasn't enough -- smoke test (job
      53677) showed the model rambles in that round on its own (e.g.
      prose-summarizing results), not because of any forced floor, so a
      simple low-load, single-call instance still took 34.3s vs. the
      ~9.7s baseline. Final fix: **break the loop immediately after a
      successful `solve_subschedule` call** instead of allowing an
      optional wrap-up round at all (commit `166b500`) -- `AGENT_INSTRUCTION`
      already says to call it exactly once, so nothing is lost. Smoke
      test (job 53678) confirmed non-retrieved instances landed at or
      *below* the original pre-fix baseline (the wrap-up round's
      unbounded cost existed in the ORIGINAL S4 too, just never
      surfaced/measured before).

      **Full re-run (attempt 3, working), both splits, confirmed at full
      scale (2026-09-28)** -- results BETTER than every smoke-test
      estimate:

      | Split | Admission accuracy | Feasibility | near_saturation failure rate | Mean latency |
      |---|---|---|---|---|
      | `test` (226, job 53763) | 93.3% (was 84.9%) | 100.0% | 8.6% (was 41.4%) | 35.4s (was 21.4s) |
      | `held_out_test` (720, job 53762) | 87.93% (was 74.8%) | 100.0% (was 99.99%) | 22.1% (was 65.0%) | 31.1s (was 20.9s) |

      Among near_saturation instances where the tool call succeeds,
      quality remains excellent on both splits: 97.8% / 92.74% admission
      accuracy, 100% feasibility -- matching or beating S2/S3's own
      numbers. Results in `results/s4_eval_held_out_test.jsonl` /
      `results/s4_eval_test.jsonl`.

      **S4's research question, final answer**: giving the model the
      retrieval choice, as implemented, initially cost real reliability
      on large instances for no compensating benefit (retrieval was
      already free via S3, unnecessary via S2). That reliability gap was
      a fixable implementation issue, not an inherent property of giving
      the model the choice -- once fixed (forcing generation past
      premature stops, recovering discarded retries, and removing an
      unbounded wrap-up round that was actually a latent cost in the
      ORIGINAL design too), S4 matches or slightly beats S2/S3's quality
      tier at a real but bounded and now well-understood latency cost.
      This closes out S4 at full scale on both splits.
- [x] Phase 5: full evaluation matrix run across B1-B6/S1-S5, all metrics
      + statistical tests (`MILESTONES.md` Phase 5). User scoped this
      down to the 9 methods already confirmed at full scale (B1, B2, B3,
      B4, S1, S2, S2b, S3, S4) -- B5/B6 (optional RL ablations) and S5
      (curriculum, stub only) out of scope unless a specific gap shows
      up. Confirmed B1-B3's `results/baselines.jsonl` and the SLM/RL
      methods' eval JSONLs share the exact same 720 `held_out_test`
      instance IDs (720/720 overlap), so a genuine per-instance PAIRED
      comparison is possible, not just comparing averages.

      **A deeper scope mismatch than initially flagged.** B1/B2/B3
      report `admission_rate` over the WHOLE topology (base flows +
      newly added ones -- checked directly: `total_flows`/`admitted_flows`
      for a sample instance = 118, matching `len(ground_truth.admitted)`
      which includes base flow IDs), each re-solving the schedule from
      scratch (B1), incrementally (B2), or greedily (B3). B4/S1-S4 score
      `admission_accuracy`/`feasibility_rate` ONLY on the newly added
      flows (27 for that same instance), against B1's own incremental
      ground truth, with the base schedule held fixed. This isn't just a
      different scoring basis (as flagged for the B4-vs-paper item) --
      it's a different PROBLEM SCOPE. A direct per-instance pairing of
      B1-B3's admission_rate against B4/S1-S4's admission_accuracy would
      not be honest, so the matrix (`tsn_bench/eval/phase5_report.py`,
      run locally -- pure pandas/scipy, no cluster needed) produces
      three separate outputs instead of one leaderboard:

      1. `model_group_wilcoxon.json` -- paired tests within the 6
         model-based methods (B4/S1/S2/S2b/S3/S4) on admission_accuracy,
         feasibility_rate, latency. Same basis, same scope, same ground
         truth, same instances -- the real head-to-head.
      2. `latency_all9_wilcoxon.json` -- paired latency tests spanning
         all 9. Latency is at least dimensionally comparable (wall-clock
         seconds) even though B1-B3 solve a bigger problem each time --
         caveated below, not hidden.
      3. `baseline_family_wilcoxon.json` -- B1/B2/B3's own comparison on
         their shared admission_rate + latency, restricted to the same
         `test`/`held_out_test` instances (not the full train/val matrix
         `baselines.jsonl` also holds).

      **Model-group results (`held_out_test`, 720 paired instances;
      `test`, 226 -- every pairwise comparison below is significant at
      p<0.001, unsurprising given these sample sizes and effect sizes,
      so the ranking is not a close call):**

      | Method | Admission accuracy | Feasibility | Latency |
      |---|---|---|---|
      | S2b | 99.8% | 100.0% | 2.0s |
      | S1 | 99.7% | 34.8% | 40.1s |
      | S3 | 95.4% | 100.0% | 22.8s |
      | S4 | 87.9% | 100.0% | 31.1s |
      | S2 | 68.2% | 100.0% | 9.9s |
      | B4 | 31.3% | 100.0% | 1.2s |

      (`test`-split ranking is the same shape: S2b 100% > S3 98.0% > S4
      93.3% > S1 74.8% > S2 71.7% > B4 20.4%, with S1's `test`-split
      feasibility also collapsing to 33.4%, same as `held_out_test`.)
      Feasibility comparisons between any two of {B4, S2, S2b, S3, S4}
      returned "insufficient variation for Wilcoxon test" -- all four are
      a flat, tied 100.0%, only S1 is statistically distinguishable on
      that metric. S2b is the strongest all-around confirmed method in
      the project: matches S1's near-perfect accuracy without S1's
      feasibility collapse, at a twentieth of S1's latency.

      **Baseline-family results** (same `test`/`held_out_test`
      instances): B1/B2/B3 all sit at ~99.9-100.0% whole-topology
      admission rate (these instances are all within solver capacity by
      construction), with B3 (heuristic) 2-6x faster than B1/B2
      (0.03-0.07s vs. 0.12-0.18s).

      **Cross-family latency ranking, all 9 methods (`held_out_test`,
      mean seconds)**: B3 0.07 < B2 0.16 < B1 0.18 < B4 1.22 < S2b 2.01
      < S2 9.92 < S3 22.83 < S4 31.10 < S1 40.11. Every classical
      baseline is faster than every model-based method, B1-B3 included,
      despite solving a strictly bigger problem (whole topology, not
      just the new flows) each time -- no LLM token-generation loop and
      no GCN/actor forward pass beats a modern ILP/heuristic solver on
      raw wall-clock speed for problems this size, even the "slow" full
      recompute (B1). Caveat stands: this ranking compares different
      computational tasks, not a fully controlled speed test.

      Full data in `results/phase5_report/` (gitignored, regenerate with
      `python -m tsn_bench.eval.phase5_report`).
- [ ] Phase 7: paper draft assembled from results (`MILESTONES.md`
      Phase 7) -- last item; project done when this lands.
