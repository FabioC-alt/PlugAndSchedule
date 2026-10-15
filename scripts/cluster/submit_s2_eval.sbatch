#!/usr/bin/env bash
#SBATCH --job-name=tsn-s2-eval
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/s2_eval_%j.out
#SBATCH --error=slurm_logs/s2_eval_%j.err

# Runs the S2 tool-calling agent (base Qwen2.5-Instruct + solve_subschedule
# tool, see slm_future/s2_solver_tool/) on one split (default held_out_test).
# Needs more CPU than S1's eval since the tool itself runs OR-Tools CP-SAT —
# same .venv-gpu as S1 (it already includes ortools via requirements.txt).
#
# Submit from the repo root on disi:
#   sbatch scripts/cluster/submit_s2_eval.sbatch
# Override via --export, e.g. a tiny smoke test:
#   sbatch --export=ALL,TSN_LIMIT=5 scripts/cluster/submit_s2_eval.sbatch
# No-tool control (see evaluate.py's --no-tool docstring):
#   sbatch --export=ALL,TSN_NO_TOOL=1,TSN_OUT=results/s2_eval_no_tool.jsonl scripts/cluster/submit_s2_eval.sbatch
# Terser-prompt decision-latency experiment:
#   sbatch --export=ALL,TSN_TERSE_PROMPT=1,TSN_OUT=results/s2_eval_terse.jsonl scripts/cluster/submit_s2_eval.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv-gpu/bin/activate

# see submit_s1_train.sbatch for why both of these are needed on disi
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export TORCH_DISABLE_NATIVE_JIT=1

DATA="${TSN_DATA:-data}"
SPLIT="${TSN_SPLIT:-held_out_test}"
MODEL_NAME="${TSN_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
OUT="${TSN_OUT:-results/s2_eval_${SPLIT}.jsonl}"
MAX_TOOL_CALLS="${TSN_MAX_TOOL_CALLS:-3}"
MAX_NEW_TOKENS="${TSN_MAX_NEW_TOKENS:-3072}"
LIMIT_ARG=""
if [ -n "${TSN_LIMIT:-}" ]; then LIMIT_ARG="--limit ${TSN_LIMIT}"; fi
LOAD_FACTOR_ARG=""
if [ -n "${TSN_LOAD_FACTOR:-}" ]; then LOAD_FACTOR_ARG="--load-factor ${TSN_LOAD_FACTOR}"; fi
NO_TOOL_ARG=""
if [ -n "${TSN_NO_TOOL:-}" ]; then NO_TOOL_ARG="--no-tool"; fi
TERSE_PROMPT_ARG=""
if [ -n "${TSN_TERSE_PROMPT:-}" ]; then TERSE_PROMPT_ARG="--terse-prompt"; fi

# -u: see submit_s1_eval.sbatch for why (block-buffered stdout otherwise)
python -u -m slm_future.s2_solver_tool.evaluate \
  --data "${DATA}" \
  --split "${SPLIT}" \
  --model-name "${MODEL_NAME}" \
  --out "${OUT}" \
  --max-tool-calls "${MAX_TOOL_CALLS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  ${LIMIT_ARG} \
  ${LOAD_FACTOR_ARG} \
  ${NO_TOOL_ARG} \
  ${TERSE_PROMPT_ARG}
