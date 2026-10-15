#!/usr/bin/env bash
#SBATCH --job-name=tsn-b4-train
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/b4_train_%j.out
#SBATCH --error=slurm_logs/b4_train_%j.err

# Trains the B4 GCN-TD3 agent (baselines_future/b4_gcn_td3/) -- a GCN state
# encoder + TD3 actor/critic, no LLM involved, so no transformers/peft
# needed. Reuses the same .venv-gpu as S1/S2/S2b (already has torch +
# ortools + networkx via requirements.txt/requirements-gpu.txt).
#
# Submit from the repo root on disi:
#   sbatch scripts/cluster/submit_b4_train.sbatch
# Smoke test on a handful of instances first:
#   sbatch --export=ALL,TSN_LIMIT=5,TSN_CHECKPOINT=baselines_future/b4_gcn_td3/checkpoints/smoketest.pt scripts/cluster/submit_b4_train.sbatch
# Resume a run that hit its time limit before finishing (see train.py's
# --resume-from docstring -- weights only, not replay buffer/optimizer
# state, but skips instances already in the log so nothing is reprocessed):
#   sbatch --export=ALL,TSN_LIMIT=300,TSN_RESUME_FROM=baselines_future/b4_gcn_td3/checkpoints/b4.pt scripts/cluster/submit_b4_train.sbatch
#
# Every periodic save also writes a numbered snapshot (default: keep the
# last 8) alongside the overwritten "latest" checkpoint, so a policy that
# collapses late in a run (see PROGRESS.md 2026-09-24) can be rolled back
# to an earlier, healthy snapshot without retraining from scratch. Set
# TSN_KEEP_VERSIONED=0 to disable and match the old overwrite-only behavior.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv-gpu/bin/activate
export TORCH_DISABLE_NATIVE_JIT=1  # see submit_s1_train.sbatch for why

DATA="${TSN_DATA:-data}"
SPLIT="${TSN_SPLIT:-train}"
CHECKPOINT="${TSN_CHECKPOINT:-baselines_future/b4_gcn_td3/checkpoints/b4.pt}"
LOG="${TSN_LOG:-results/b4_train_log.jsonl}"
BATCH_SIZE="${TSN_BATCH_SIZE:-256}"
LIMIT_ARG=""
if [ -n "${TSN_LIMIT:-}" ]; then LIMIT_ARG="--limit ${TSN_LIMIT}"; fi
RESUME_ARG=""
if [ -n "${TSN_RESUME_FROM:-}" ]; then RESUME_ARG="--resume-from ${TSN_RESUME_FROM}"; fi
KEEP_VERSIONED_ARG=""
if [ -n "${TSN_KEEP_VERSIONED:-}" ]; then KEEP_VERSIONED_ARG="--keep-versioned-checkpoints ${TSN_KEEP_VERSIONED}"; fi

python -u -m baselines_future.b4_gcn_td3.train \
  --data "${DATA}" \
  --split "${SPLIT}" \
  --out-checkpoint "${CHECKPOINT}" \
  --log "${LOG}" \
  --batch-size "${BATCH_SIZE}" \
  ${LIMIT_ARG} \
  ${RESUME_ARG} \
  ${KEEP_VERSIONED_ARG}
