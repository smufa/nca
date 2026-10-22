#!/bin/bash
#SBATCH --job-name=ncaprof
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Batch-size / throughput / memory probe for the training loop.
#
#   sbatch slurm/profile.sbatch --steps 80 --batches 16 32 64 128
#   sbatch -p mig-priority --constraint=mig --gres=gpu:1g.10gb:1 slurm/profile.sbatch ...
#
# Uses `source .venv/bin/activate` + `python`, NOT `uv run`: the lock pins torch
# to the CPU index ([tool.uv.sources]) and `uv run` will happily replace the
# cluster's CUDA build with a CPU wheel.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run this with sbatch from the repo root}"
source .venv/bin/activate
unset PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job ${SLURM_JOB_ID} on $(hostname) | $(date -Is)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python scripts/profile_train.py --breakdown --variants "$@"
