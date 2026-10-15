#!/usr/bin/env bash
#SBATCH --job-name=tsn-s3-eval
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/s3_eval_%j.out
#SBATCH --error=slurm_logs/s3_eval_%j.err

# Runs the S3/GraphRAG agent (same base Qwen2.5-Instruct + solve_subschedule
# tool as S2, but with a k-hop-retrieval-narrowed prompt topology, see
# slm_future/s3_graphrag/) on one split (default held_out_test). Same
# .venv-gpu as S1/S2.
#
# Submit from the repo root on disi:
#   sbatch scripts/cluster/submit_s3_eval.sbatch
# Override via --export, e.g. a tiny smoke test:
#   sbatch --export=ALL,TSN_LIMIT=5 scripts/cluster/submit_s3_eval.sbatch
# Compare against a wider buffer (see evaluate.py's --k-hop default note
# for why 0 is the default -- k_hop>=1 washes out most of the reduction):
#   sbatch --export=ALL,TSN_K_HOP=1,TSN_OUT=results/s3_eval_khop1.jsonl scripts/cluster/submit_s3_eval.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv-gpu/bin/activate

export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export TORCH_DISABLE_NATIVE_JIT=1

DATA="${TSN_DATA:-data}"
SPLIT="${TSN_SPLIT:-held_out_test}"
MODEL_NAME="${TSN_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
OUT="${TSN_OUT:-results/s3_eval_${SPLIT}.jsonl}"
K_HOP="${TSN_K_HOP:-0}"
MAX_TOOL_CALLS="${TSN_MAX_TOOL_CALLS:-3}"
MAX_NEW_TOKENS="${TSN_MAX_NEW_TOKENS:-3072}"
LIMIT_ARG=""
if [ -n "${TSN_LIMIT:-}" ]; then LIMIT_ARG="--limit ${TSN_LIMIT}"; fi
LOAD_FACTOR_ARG=""
if [ -n "${TSN_LOAD_FACTOR:-}" ]; then LOAD_FACTOR_ARG="--load-factor ${TSN_LOAD_FACTOR}"; fi
TERSE_PROMPT_ARG=""
if [ -n "${TSN_TERSE_PROMPT:-}" ]; then TERSE_PROMPT_ARG="--terse-prompt"; fi

python -u -m slm_future.s3_graphrag.evaluate \
  --data "${DATA}" \
  --split "${SPLIT}" \
  --model-name "${MODEL_NAME}" \
  --out "${OUT}" \
  --k-hop "${K_HOP}" \
  --max-tool-calls "${MAX_TOOL_CALLS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  ${LIMIT_ARG} \
  ${LOAD_FACTOR_ARG} \
  ${TERSE_PROMPT_ARG}
