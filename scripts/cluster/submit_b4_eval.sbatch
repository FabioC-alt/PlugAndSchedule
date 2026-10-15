#!/usr/bin/env bash
#SBATCH --job-name=tsn-b4-eval
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/b4_eval_%j.out
#SBATCH --error=slurm_logs/b4_eval_%j.err

# Evaluates a trained B4 GCN-TD3 checkpoint (default held_out_test).
#
# Submit from the repo root on disi:
#   sbatch --export=ALL,TSN_CHECKPOINT=baselines_future/b4_gcn_td3/checkpoints/b4.pt scripts/cluster/submit_b4_eval.sbatch
# Smoke test:
#   sbatch --export=ALL,TSN_LIMIT=5,TSN_CHECKPOINT=... scripts/cluster/submit_b4_eval.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv-gpu/bin/activate
export TORCH_DISABLE_NATIVE_JIT=1

DATA="${TSN_DATA:-data}"
SPLIT="${TSN_SPLIT:-held_out_test}"
CHECKPOINT="${TSN_CHECKPOINT:?set TSN_CHECKPOINT to a trained checkpoint path}"
OUT="${TSN_OUT:-results/b4_eval_${SPLIT}.jsonl}"
LIMIT_ARG=""
if [ -n "${TSN_LIMIT:-}" ]; then LIMIT_ARG="--limit ${TSN_LIMIT}"; fi
LOAD_FACTOR_ARG=""
if [ -n "${TSN_LOAD_FACTOR:-}" ]; then LOAD_FACTOR_ARG="--load-factor ${TSN_LOAD_FACTOR}"; fi

python -u -m baselines_future.b4_gcn_td3.evaluate \
  --data "${DATA}" \
  --split "${SPLIT}" \
  --checkpoint "${CHECKPOINT}" \
  --out "${OUT}" \
  ${LIMIT_ARG} \
  ${LOAD_FACTOR_ARG}
