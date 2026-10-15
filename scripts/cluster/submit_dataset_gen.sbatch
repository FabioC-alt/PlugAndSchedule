#!/usr/bin/env bash
#SBATCH --job-name=tsn-dataset-gen
#SBATCH --partition=amdepyc7713
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/dataset_gen_%j.out
#SBATCH --error=slurm_logs/dataset_gen_%j.err

# Generates the foundation-phase dataset (Phase 3): CPU-only (CP-SAT ground
# truth via OR-Tools), no GPU needed. Submit from the repo root on disi:
#   mkdir -p slurm_logs && sbatch scripts/cluster/submit_dataset_gen.sbatch
# Override the config/output/time-limit with sbatch --export, e.g.:
#   sbatch --export=ALL,TSN_CONFIG=configs/default.yaml,TSN_OUT=data,TSN_TIME_LIMIT=30 \
#       scripts/cluster/submit_dataset_gen.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

CONFIG="${TSN_CONFIG:-configs/default.yaml}"
OUT="${TSN_OUT:-data}"
TIME_LIMIT="${TSN_TIME_LIMIT:-30}"
QUICK_FLAG=""
if [ "${TSN_QUICK:-0}" = "1" ]; then QUICK_FLAG="--quick"; fi

python -m tsn_bench.dataset.build_dataset \
  --config "${CONFIG}" \
  --out "${OUT}" \
  --time-limit-s "${TIME_LIMIT}" \
  ${QUICK_FLAG}
