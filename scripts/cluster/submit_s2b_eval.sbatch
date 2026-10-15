#!/usr/bin/env bash
#SBATCH --job-name=tsn-s2b-eval
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/s2b_eval_%j.out
#SBATCH --error=slurm_logs/s2b_eval_%j.err

# Runs S2b (OpenNod typed-logits flow selection + S2's unchanged solver
# tool, see slm_future/s2b_opennod/) on one split (default held_out_test).
#
# Unlike every other submit_*.sbatch here, this does NOT use TSNAgent's
# own .venv-gpu -- S2b needs `semif_phase1` (OpenNod), which pins
# torch==2.10.0/transformers==5.17.0, a different stack. It needs a
# sibling OpenNod checkout with its OWN venv that also has this repo's
# runtime deps installed into it (ortools, networkx, peft) -- see
# slm_future/s2b_opennod/README.md's "Environment setup" section for the
# exact one-time setup. OPENNOD_VENV below must point at that venv.
#
# Submit from the repo root on disi:
#   sbatch --export=ALL,OPENNOD_VENV=/scratch.hpc/$USER/OpenNod/.venv scripts/cluster/submit_s2b_eval.sbatch
# Override via --export, e.g. a tiny smoke test:
#   sbatch --export=ALL,OPENNOD_VENV=...,TSN_LIMIT=5 scripts/cluster/submit_s2b_eval.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

if [ -z "${OPENNOD_VENV:-}" ]; then
  echo "OPENNOD_VENV must be set -- see this script's header comment." >&2
  exit 1
fi
source "${OPENNOD_VENV}/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${OPENNOD_VENV}/../.hf_cache"

DATA="${TSN_DATA:-data}"
SPLIT="${TSN_SPLIT:-held_out_test}"
MODEL_NAME="${TSN_MODEL_NAME:-Qwen/Qwen3.5-4B}"
REVISION="${TSN_REVISION:-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
OUT="${TSN_OUT:-results/s2b_eval_${SPLIT}.jsonl}"
LIMIT_ARG=""
if [ -n "${TSN_LIMIT:-}" ]; then LIMIT_ARG="--limit ${TSN_LIMIT}"; fi
LOAD_FACTOR_ARG=""
if [ -n "${TSN_LOAD_FACTOR:-}" ]; then LOAD_FACTOR_ARG="--load-factor ${TSN_LOAD_FACTOR}"; fi

# -u: see submit_s1_eval.sbatch for why (block-buffered stdout otherwise)
python -u -m slm_future.s2b_opennod.evaluate \
  --data "${DATA}" \
  --split "${SPLIT}" \
  --model-name "${MODEL_NAME}" \
  --revision "${REVISION}" \
  --out "${OUT}" \
  ${LIMIT_ARG} \
  ${LOAD_FACTOR_ARG}
