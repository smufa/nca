#!/bin/bash
#SBATCH --job-name=nca
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Train one NCA on an H100.
#
#   sbatch slurm/train.sbatch --target "emoji:🦎" --experiment regenerating \
#       --run-name lizard_regen --target-size 40 --padding 16 \
#       --batch-size 16 --iterations 8000 --train-steps 64 96
#
# IMPORTANT: --constraint=h100 keeps us off the V100S nodes (CUDA 13 dropped
# Volta, so the cu13 torch build would fail there).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?run this with sbatch from the repo root}"
source .venv/bin/activate
# The cluster's Lmod environment injects a Python 3.12 site-packages tree via
# PYTHONPATH, which shadows the venv's packages (numpy/torch). Drop it.
unset PYTHONPATH PYTHONHOME

# A `1g.10gb` slice has 9.75 GiB, and a `hidden=256` run sits at ~9.3 GiB, where
# allocator fragmentation alone is enough to OOM. Expandable segments costs nothing
# and removes that failure mode (`docs/findings/training-perf.md`).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job ${SLURM_JOB_ID} on $(hostname) | $(date -Is)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python main.py --device cuda "$@"
