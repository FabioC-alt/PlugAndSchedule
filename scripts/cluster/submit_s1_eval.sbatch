#!/usr/bin/env bash
#SBATCH --job-name=tsn-s1-eval
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/s1_eval_%j.out
#SBATCH --error=slurm_logs/s1_eval_%j.err

# Generates + scores S1's decisions on one split (default held_out_test --
# the zero-shot generalization split). Needs the same .venv-gpu as
# submit_s1_train.sbatch.
#
# Submit from the repo root on disi:
#   sbatch scripts/cluster/submit_s1_eval.sbatch
# Override via --export, e.g. to eval the base model with no adapter:
#   sbatch --export=ALL,TSN_ADAPTER_DIR= scripts/cluster/submit_s1_eval.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv-gpu/bin/activate

# see submit_s1_train.sbatch: redirect the HF cache off the quota-limited
# home dir, and disable torch 2.14's native-JIT Triton kernels (crash on
# this cluster — no python3-dev, see pytorch/pytorch#196977)
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export TORCH_DISABLE_NATIVE_JIT=1

DATA="${TSN_DATA:-data}"
SPLIT="${TSN_SPLIT:-held_out_test}"
MODEL_NAME="${TSN_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
RUN_NAME="${TSN_RUN_NAME:-run1}"
ADAPTER_DIR="${TSN_ADAPTER_DIR:-slm_future/s1_lora/checkpoints/${RUN_NAME}/adapter}"
OUT="${TSN_OUT:-results/s1_eval_${SPLIT}.jsonl}"
MAX_NEW_TOKENS="${TSN_MAX_NEW_TOKENS:-4096}"
LIMIT_ARG=""
if [ -n "${TSN_LIMIT:-}" ]; then LIMIT_ARG="--limit ${TSN_LIMIT}"; fi
ADAPTER_ARG=""
if [ -n "${ADAPTER_DIR}" ]; then ADAPTER_ARG="--adapter-dir ${ADAPTER_DIR}"; fi
LOAD_FACTOR_ARG=""
if [ -n "${TSN_LOAD_FACTOR:-}" ]; then LOAD_FACTOR_ARG="--load-factor ${TSN_LOAD_FACTOR}"; fi

# -u: unbuffered stdout/stderr. print() block-buffers when redirected to a
# SLURM log file (not a TTY), so without this a long run shows zero
# progress for the first several minutes/instances despite actively
# working (noticed during the first full-scale run).
python -u -m slm_future.s1_lora.evaluate \
  --data "${DATA}" \
  --split "${SPLIT}" \
  --model-name "${MODEL_NAME}" \
  --out "${OUT}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  ${ADAPTER_ARG} \
  ${LIMIT_ARG} \
  ${LOAD_FACTOR_ARG}
