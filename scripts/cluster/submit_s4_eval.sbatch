#!/usr/bin/env bash
#SBATCH --job-name=tsn-s4-eval
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/s4_eval_%j.out
#SBATCH --error=slurm_logs/s4_eval_%j.err

# Runs the S4 two-tool agent (same base Qwen2.5-Instruct as S2/S3, but with
# BOTH retrieve_subgraph -- optional -- and solve_subschedule -- required --
# offered as tools, letting the model choose whether/when to retrieve
# instead of S3's always-retrieve or S2's never-retrieve). Same .venv-gpu
# as S1/S2/S3.
#
# Submit from the repo root on disi:
#   sbatch scripts/cluster/submit_s4_eval.sbatch
# Smoke test on a handful of instances first:
#   sbatch --export=ALL,TSN_LIMIT=5,TSN_OUT=results/s4_eval_smoketest.jsonl scripts/cluster/submit_s4_eval.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv-gpu/bin/activate

export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export TORCH_DISABLE_NATIVE_JIT=1

DATA="${TSN_DATA:-data}"
SPLIT="${TSN_SPLIT:-held_out_test}"
MODEL_NAME="${TSN_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
OUT="${TSN_OUT:-results/s4_eval_${SPLIT}.jsonl}"
MAX_TOOL_CALLS="${TSN_MAX_TOOL_CALLS:-4}"
MAX_NEW_TOKENS="${TSN_MAX_NEW_TOKENS:-3072}"
LIMIT_ARG=""
if [ -n "${TSN_LIMIT:-}" ]; then LIMIT_ARG="--limit ${TSN_LIMIT}"; fi
LOAD_FACTOR_ARG=""
if [ -n "${TSN_LOAD_FACTOR:-}" ]; then LOAD_FACTOR_ARG="--load-factor ${TSN_LOAD_FACTOR}"; fi

python -u -m slm_future.s4_full_agent.evaluate \
  --data "${DATA}" \
  --split "${SPLIT}" \
  --model-name "${MODEL_NAME}" \
  --out "${OUT}" \
  --max-tool-calls "${MAX_TOOL_CALLS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  ${LIMIT_ARG} \
  ${LOAD_FACTOR_ARG}
