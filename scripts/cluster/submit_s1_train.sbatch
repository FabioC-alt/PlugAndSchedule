#!/usr/bin/env bash
#SBATCH --job-name=tsn-s1-train
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/s1_train_%j.out
#SBATCH --error=slurm_logs/s1_train_%j.err

# LoRA fine-tunes Qwen2.5-3B-Instruct on the S1 SFT dataset. Needs a GPU venv
# separate from the CPU-only foundation venv (see requirements-gpu.txt) —
# create it once with:
#   python3 -m venv .venv-gpu && source .venv-gpu/bin/activate
#   pip install -r requirements.txt -r slm_future/s1_lora/requirements-gpu.txt \
#       --extra-index-url https://download.pytorch.org/whl/cu121
# (confirmed working on disi's l40s nodes: driver 580.95, CUDA 13.0 — the
# cu121 PyTorch wheel runs fine against that driver.)
#
# Submit from the repo root on disi:
#   sbatch scripts/cluster/submit_s1_train.sbatch
# Override via --export, e.g. a tiny smoke-test run:
#   sbatch --export=ALL,TSN_EPOCHS=1,TSN_RUN_NAME=smoketest scripts/cluster/submit_s1_train.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv-gpu/bin/activate

# huggingface_hub defaults to caching model downloads under ~/.cache, but
# disi home directories have a 10GB quota (already exhausted on this
# account by unrelated files, see README's cluster section) — redirect the
# cache to scratch, where the repo itself already lives.
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"

# torch 2.14's new native-AOT Triton kernels (e.g. bmm_outer_product, used
# by Qwen2's RoPE) JIT-compile a small C extension against Python.h the
# first time they run, and disi's nodes don't have python3-dev installed —
# there's no fallback, it just crashes (confirmed upstream:
# https://github.com/pytorch/pytorch/issues/196977). This env var disables
# that native-JIT path and falls back to the standard kernel.
export TORCH_DISABLE_NATIVE_JIT=1

DATA="${TSN_DATA:-data}"
SFT_DATA_DIR="${TSN_SFT_DATA_DIR:-slm_future/s1_lora/sft_data}"
RUN_NAME="${TSN_RUN_NAME:-run1}"
MODEL_NAME="${TSN_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
EPOCHS="${TSN_EPOCHS:-3}"

python -m slm_future.s1_lora.build_sft_dataset --data "${DATA}" --out "${SFT_DATA_DIR}"

python -m slm_future.s1_lora.train_lora \
  --sft-data-dir "${SFT_DATA_DIR}" \
  --output-dir "slm_future/s1_lora/checkpoints/${RUN_NAME}" \
  --model-name "${MODEL_NAME}" \
  --epochs "${EPOCHS}"
