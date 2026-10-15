# Plug and Schedule

Code and results to check the paper *Plug and Schedule: Can Language Models Admit New Machines into
Time-Sensitive Networks?*. There are three levels, from a few seconds to a cluster re-run. Level 1 needs no
GPU, no dataset and nothing beyond a Python 3 interpreter.

The paper compares nine methods for admitting new nodes into a running Time-Sensitive Network (TSN): three
classical baselines (B1 full ILP, B2 incremental ILP, B3 greedy heuristic), a reinforcement learning policy
(B4, a reimplementation of GCN-TD3), and five language-model configurations (S1 fine-tuned model, S2 model
with a solver tool, S2b yes/no scoring per flow, S3 narrowed prompt, S4 optional retrieval). All methods are
scored against the schedule of a full ILP solver on 2160 generated instances.

```
reproduce.py                  Level 1: recompute every number of the Results section, compare with the paper
expected.json                 the values as printed in the paper, with tolerances
reported_results/             per-instance results of all methods (slim: no generated text)
training_logs/               training logs of B4 (3567 epochs) and S1 (loss per epoch, training time)
dataset/                      Level 2: manifest and fingerprints of the 2160 instances + verifier
tools/build_slim_results.py   how reported_results/ was derived from the raw cluster logs (merge rules in one place)
tsn_bench/                    dataset generator, solvers B1 to B3, evaluation tools
baselines_future/b4_gcn_td3/  B4
slm_future/                   S1 to S4 (directory names are historical: s2b_opennod holds S2b)
configs/                      dataset configuration (seeds, topology sizes, flow profiles, load levels)
scripts/cluster/              SLURM job scripts used for every run in the paper
```

## Level 1: recompute the tables and figures (seconds)

```bash
python reproduce.py            # tables, by-load numbers, S4 and S2 statistics, p-values, check against the paper
```

The last block of the output compares every value against `expected.json` (the numbers printed in the
paper) and exits with status 1 on a mismatch. Only the Python standard library is needed. The Wilcoxon
p-values need `scipy`; without it those checks are skipped, and the last line says how many (`pip install scipy`). Latencies are measured
wall-clock times: they are recomputed from the shipped per-instance values, not re-measured.

| Paper item | Where it comes from |
|---|---|
| Table of classical baselines | `reported_results/baselines.jsonl`, whole-topology admission rate (not comparable with accuracy) |
| Main results table (accuracy, feasibility, latency) | `reported_results/<method>_<split>.jsonl` |
| Figure: accuracy against latency | held-out rows of the main results table |
| Figure: results by load level | `reproduce.py`, section "by load level" (same files, grouped by `load`) |
| S4 failure rate, 92.7% accuracy when it calls the solver | `reported_results/S4_held_out_test.jsonl`, field `used_tool` |
| S2 without the tool: 0 of 720 usable | `reported_results/S2_no_tool_held_out_test.jsonl` |
| Wilcoxon p-values | paired, per-instance accuracy, held-out split |
| Training facts (B4 epochs, hours, instance 180 and 183; S1 validation losses and training time) | `training_logs/` |

Per-instance fields: `accuracy` (decision match with the ILP ground truth over the new flows), `feasibility`
(valid flows over admitted flows, empty when nothing was admitted), `latency_s` (wall-clock seconds per
instance), `load`, `switch_count`, `n_flows`, `used_tool` (solver tool called at least once).

## Level 2: dataset (verify, or regenerate)

`dataset/manifest.jsonl` lists the 2160 instances (seed, scenario, load, number of base and
new flows). `dataset/fingerprints.jsonl` holds a SHA-256 of what the seeded generators
produce for each instance (topology, base flows, new flows, new nodes). Solver outputs are excluded because
CP-SAT schedules can differ between machines.

```bash
pip install -r requirements.txt
# regenerate (CPU only; about 2160 CP-SAT solves of up to 30 s each, run it on a cluster)
python -m tsn_bench.dataset.build_dataset --config configs/default.yaml --out data --time-limit-s 30
# then compare with the shipped fingerprints
python dataset/verify_dataset.py data
```

On a SLURM cluster (once: `bash scripts/cluster/setup_env.sh` to create `.venv`, and `mkdir -p slurm_logs`):
`sbatch scripts/cluster/submit_dataset_gen.sbatch`. The splits are a seeded random draw
(seed 42 in `configs/default.yaml`): 12 and 16 switches form `held_out_test` (720 instances), and 2 to 8
switches are split by a seeded random draw with ratios 70/15/15 into `train` (989 instances), `val` (225) and
`test` (226).

## Level 3: re-run the methods (GPU cluster)

All commands run from the repository root, after `bash scripts/cluster/setup_env.sh` (creates `.venv`, used by the
CPU jobs) and `mkdir -p slurm_logs` (the job scripts write their logs there). The learned
methods need a GPU environment (`.venv-gpu`) with `torch`, `transformers` and `peft`
(`pip install -r slm_future/s1_lora/requirements-gpu.txt`). Runs were measured on one NVIDIA L40S GPU. The
defaults of each script are the settings used in the paper.

| Method | Command | Notes |
|---|---|---|
| B1, B2, B3 | `sbatch scripts/cluster/submit_baselines.sbatch` | CPU, `tsn_bench.eval.run_baselines` |
| B4 train | `sbatch --export=ALL,TSN_LIMIT=250 scripts/cluster/submit_b4_train.sbatch` | 250 train instances (without `TSN_LIMIT` it trains on all 989). About 8.6 h in total, so it needs several jobs of the 6 h limit: continue with `TSN_RESUME_FROM=baselines_future/b4_gcn_td3/checkpoints/b4.pt`. Snapshots are saved every 10 instances as `baselines_future/b4_gcn_td3/checkpoints/b4_instNNNN.pt` (last 8 kept). The policy collapses after instance 183, and the paper evaluates the snapshot at instance 180 (`b4_inst0180.pt`) |
| B4 eval | `TSN_CHECKPOINT=<snapshot> TSN_SPLIT=held_out_test sbatch --export=ALL scripts/cluster/submit_b4_eval.sbatch` | |
| S1 train | `sbatch --export=ALL,TSN_RUN_NAME=full_scale_3ep scripts/cluster/submit_s1_train.sbatch` | Qwen2.5-3B-Instruct, LoRA r=16, 3 epochs; the log shows 21 min of training, about 27 min for the whole job (loading, validation, saving) |
| S1 eval | `TSN_RUN_NAME=full_scale_3ep TSN_SPLIT=held_out_test sbatch --export=ALL scripts/cluster/submit_s1_eval.sbatch` | max 4096 new tokens |
| S2 | `TSN_SPLIT=held_out_test sbatch --export=ALL scripts/cluster/submit_s2_eval.sbatch` | max 3072 new tokens, 3 tool rounds. No-tool control: add `TSN_NO_TOOL=1,TSN_OUT=results/s2_eval_held_out_test_no_tool.jsonl` to `--export` (without `TSN_OUT` it overwrites the normal S2 log) |
| S2b | `sbatch --export=ALL,OPENNOD_VENV=<path>/.venv,TSN_SPLIT=held_out_test scripts/cluster/submit_s2b_eval.sbatch` | Qwen3.5-4B. `OPENNOD_VENV` is required: a separate virtual environment that contains the third-party scorer (see below) and this repository's dependencies; the script puts the repository on `PYTHONPATH` |
| S3 | `TSN_SPLIT=held_out_test sbatch --export=ALL scripts/cluster/submit_s3_eval.sbatch` | |
| S4 | `TSN_SPLIT=held_out_test sbatch --export=ALL scripts/cluster/submit_s4_eval.sbatch` | max 4 tool rounds |

The default output of each script is `results/<method>_eval_<split>.jsonl`. To rebuild `reported_results/`,
`tools/build_slim_results.py` expects these raw files in `results/` (give the runs these names with `TSN_OUT`,
or rename them):

```
baselines.jsonl
s1_eval_held_out_test_full_scale_3ep.jsonl   s1_eval_test.jsonl   s1_eval_test_near_saturation_retest.jsonl   s1_eval_test_medium_retest.jsonl
s2_eval_held_out_test.jsonl   s2_eval_held_out_test_near_saturation_retest.jsonl   s2_eval_held_out_test_no_tool.jsonl
s2_eval_test.jsonl   s2_eval_test_near_saturation_retest.jsonl
s2b_eval_held_out_test_full.jsonl   s2b_eval_test_full.jsonl
s3_eval_held_out_test.jsonl   s3_eval_test.jsonl   s4_eval_held_out_test.jsonl   s4_eval_test.jsonl
b4_eval_held_out_test.jsonl   b4_eval_test.jsonl
```

Use `TSN_SPLIT=test` for the 226 in-distribution instances and `TSN_LOAD_FACTOR=<low|medium|near_saturation>`
to run one load level. The job scripts write their raw logs to `results/` (not shipped). After a re-run,
`python tools/build_slim_results.py` rebuilds `reported_results/` from them (it asserts the expected number of rows per file).

Third-party code for S2b: the typed-logits scorer `semif_phase1` from SemIf-OpenJev
(`https://github.com/TheoLeeCJ/SemIf-OpenJev`, previously named OpenNod; the repository was renamed and some
files in this repository still use the old name). The commit used for the reported runs is
`ca3ba65f142967030ecb453346e94d6f476a69df`. Check it out next to this repository, create a virtual environment
there with its requirements, add this repository's `requirements.txt`, and point `OPENNOD_VENV` at it. It uses
different versions of `torch` and `transformers` than the other GPU jobs, which is why S2b does not run in
`.venv-gpu`.

## Merge rules applied to the raw logs

Two early runs generated too few tokens and truncated their answers on large instances. Their affected
rows were replaced by re-runs with the corrected budget, and the replacement is done in
`tools/build_slim_results.py`:

* S2, both splits: the near-saturation instances come from the re-run with 3072 new tokens (the first runs
  used 512).
* S1, `test` split: the first run (1024 new tokens) truncated medium and near-saturation answers, so those
  rows are replaced by re-runs with 4096 new tokens (`s1_eval_test_near_saturation_retest`, `s1_eval_test_medium_retest`).
* S1, `held_out_test`: the full run `s1_eval_held_out_test_full_scale_3ep` with 4096 new tokens. A separate re-run of
  its near-saturation instances (`s1_eval_held_out_test_near_saturation_retest`, 98.99% accuracy against 99.26% in
  the full run on those instances) exists in the raw logs and is not used.

## Known gaps

* **The "before the fix" figures for S4** (accuracy 74.8%, and 65.0% of near-saturation runs without a solver
  call, on the held-out split) come from an earlier full run that was overwritten. They cannot be recomputed
  from the shipped files. The "after" figures (87.9% and 22.1%) can.
* **Model weights are not included** (the LoRA adapter and the B4 snapshot). They can be regenerated with
  the commands above, but the B4 result depends on the snapshot choice described in the paper.
* **Run-to-run differences.** Latencies depend on the GPU and on cluster load. CP-SAT is multi-threaded, so
  the exact schedules it returns can differ, while the admission decisions of the ground truth do not change
  in our runs (it admits every new flow in all 2160 instances). The fingerprint check was written against
  the dataset used for the paper, and has not yet been confirmed by a full regeneration.
* All results are from one run per instance (greedy decoding, no repeated seeds for the models).
* Some code comments and script headers refer to files that are not part of this package: an internal
  development log (`PROGRESS.md`, `MILESTONES.md`, `TEST_PLAN.md`) and per-directory READMEs under
  `slm_future/`. The commands above replace the setup instructions those READMEs contained.
* The code and job scripts still use the old name "OpenNod" for the S2b library (`slm_future/s2b_opennod`,
  `OPENNOD_VENV`).
