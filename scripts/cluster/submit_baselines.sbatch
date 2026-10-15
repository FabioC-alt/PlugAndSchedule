#!/usr/bin/env bash
#SBATCH --job-name=tsn-baselines
#SBATCH --partition=amdepyc7713
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/baselines_%j.out
#SBATCH --error=slurm_logs/baselines_%j.err

# Runs B1/B2/B3 across the dataset produced by submit_dataset_gen.sbatch, then
# aggregates + runs Wilcoxon tests (Phase 5). CPU-only. Submit from the repo
# root on disi (after the dataset job has completed):
#   sbatch scripts/cluster/submit_baselines.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

DATA="${TSN_DATA:-data}"
RESULTS="${TSN_RESULTS:-results/baselines.jsonl}"
REPORT_OUT="${TSN_REPORT_OUT:-results/report}"
TIME_LIMIT="${TSN_TIME_LIMIT:-30}"

python -m tsn_bench.eval.run_baselines \
  --data "${DATA}" \
  --out "${RESULTS}" \
  --time-limit-s "${TIME_LIMIT}"

python -m tsn_bench.eval.report \
  --results "${RESULTS}" \
  --out "${REPORT_OUT}"
