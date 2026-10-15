#!/usr/bin/env bash
# One-time environment setup on the disi SLURM cluster (login node — this
# script only creates a venv and installs CPU-only deps, no heavy compute).
# Usage (from the login node, inside the synced repo directory):
#   bash scripts/cluster/setup_env.sh
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment ready at $(pwd)/.venv"
echo "Foundation-phase jobs (dataset gen, B1-B3 eval) need only CPU — submit via:"
echo "  sbatch scripts/cluster/submit_dataset_gen.sbatch"
echo "  sbatch scripts/cluster/submit_baselines.sbatch"
