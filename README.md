# Plug and Schedule: reviewer instructions

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
python reproduce.py --latex    # also prints the LaTeX rows of the main results table
```

The last block of the output compares every value against `expected.json` (the numbers printed in the
paper) and exits with status 1 on a mismatch. Only the Python standard library is needed. The Wilcoxon
p-values need `scipy` and are skipped if it is not installed (`pip install scipy`). Latencies are measured
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

On a SLURM cluster: `sbatch scripts/cluster/submit_dataset_gen.sbatch`. The splits are a seeded random draw
(seed 42 in `configs/default.yaml`): 12 and 16 switches form `held_out_test` (720 instances), and 2 to 8
switches are split 70/15/15 into `train`/`val`/`test` (the `test` split has 226 instances).

## Level 3: re-run the methods (GPU cluster)

All commands run from the repository root. For B1 to B3, run `bash scripts/cluster/setup_env.sh`. The learned
methods need a GPU environment (`.venv-gpu`) with `torch`, `transformers` and `peft`
(`pip install -r slm_future/s1_lora/requirements-gpu.txt`). Runs were measured on one NVIDIA L40S GPU. The
defaults of each script are the settings used in the paper.

| Method | Command | Notes |
|---|---|---|
| B1, B2, B3 | `sbatch scripts/cluster/submit_baselines.sbatch` | CPU, `tsn_bench.eval.run_baselines` |
| B4 train | `sbatch scripts/cluster/submit_b4_train.sbatch` | 250 train instances, about 8.6 h; the policy collapses after instance 183, the paper evaluates the snapshot at instance 180 |
| B4 eval | `TSN_CHECKPOINT=<snapshot> TSN_SPLIT=held_out_test sbatch --export=ALL scripts/cluster/submit_b4_eval.sbatch` | |
| S1 train | `sbatch --export=ALL,TSN_RUN_NAME=full_scale_3ep scripts/cluster/submit_s1_train.sbatch` | Qwen2.5-3B-Instruct, LoRA r=16, 3 epochs, about 27 min |
| S1 eval | `TSN_RUN_NAME=full_scale_3ep TSN_SPLIT=held_out_test sbatch --export=ALL scripts/cluster/submit_s1_eval.sbatch` | max 4096 new tokens |
| S2 | `TSN_SPLIT=held_out_test sbatch --export=ALL scripts/cluster/submit_s2_eval.sbatch` | max 3072 new tokens, 3 tool rounds; no-tool control: `TSN_NO_TOOL=1` |
| S2b | `TSN_SPLIT=held_out_test sbatch --export=ALL scripts/cluster/submit_s2b_eval.sbatch` | Qwen3.5-4B, needs the third-party scorer below |
| S3 | `TSN_SPLIT=held_out_test sbatch --export=ALL scripts/cluster/submit_s3_eval.sbatch` | |
| S4 | `TSN_SPLIT=held_out_test sbatch --export=ALL scripts/cluster/submit_s4_eval.sbatch` | max 4 tool rounds |

Use `TSN_SPLIT=test` for the 226 in-distribution instances and `TSN_LOAD_FACTOR=<low|medium|near_saturation>`
to run one load level. The job scripts write their raw logs to `results/` (not shipped). After a re-run,
`python tools/build_slim_results.py` rebuilds `reported_results/` from them (it asserts the expected number of rows per file).

Third-party code for S2b: the typed-logits scorer `semif_phase1` from SemIf-OpenJev
(`https://github.com/TheoLeeCJ/SemIf-OpenJev`, previously named OpenNod; the repository was renamed and some
files in this repository still use the old name). The commit used for the reported runs is
`ca3ba65f142967030ecb453346e94d6f476a69df`. Put the checkout on `PYTHONPATH`.

## Merge rules applied to the raw logs

Two early runs generated too few tokens and truncated their answers on large instances. Their affected
rows were replaced by re-runs with the corrected budget, and the replacement is done in
`tools/build_slim_results.py`:

* S2, both splits: the near-saturation instances come from the re-run with 3072 new tokens (the first runs
  used 512).
* S1, `test` split: the first run (1024 new tokens) truncated medium and near-saturation answers, so those
  rows are replaced by re-runs with 4096 new tokens. S1 on `held_out_test` was run once with 4096 tokens.

## Known gaps

* **S1 on the `test` split is not shipped yet.** The medium-load re-run has not finished, and
  `build_slim_results.py` refuses to write `S1_test.jsonl` until it has all re-runs. `reproduce.py`
  reports this instead of printing stale values.
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
* Some code comments refer to an internal development log (`PROGRESS.md`, `MILESTONES.md`, `TEST_PLAN.md`)
  that is not part of this package.
