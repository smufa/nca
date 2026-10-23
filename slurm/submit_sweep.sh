#!/bin/bash
# Submit one sweep file to Slurm: `bash slurm/submit_sweep.sh slurm/sweeps/<name>.toml`.
#
# Everything a job needs lives in the TOML ([resources] = placement, [[run]] =
# what to train), so this wrapper only forwards its arguments. Placement can
# still be overridden from the environment, as the old per-sweep scripts allowed:
#
#   PARTITION=mig-priority CONSTRAINT=mig GRES=gpu:1g.10gb:1 \
#     bash slurm/submit_sweep.sh slurm/sweeps/cross.toml
#   bash slurm/submit_sweep.sh slurm/sweeps/cross.toml --only cross8.mismatch
#   DEPEND=20754359 bash slurm/submit_sweep.sh slurm/sweeps/cross.toml
#   bash slurm/submit_sweep.sh slurm/sweeps/emoji-sweep.toml --dry-run
#
# Replaces submit_{emoji,capacity,multitarget,crosstarget}_sweep.sh. Rationale and
# the equivalent old command lines: docs/TRAINING.md.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# Lmod exports a Python 3.12 PYTHONPATH on ARNES login nodes; it shadows the
# venv's own packages and `import numpy` then fails from the wrong interpreter.
# `slurm/train.sbatch` unsets the same two for the job itself.
unset PYTHONPATH PYTHONHOME

# Same interpreter the repo uses everywhere: the venv if present, else `uv run`,
# else whatever `python` is on PATH (the cluster has an activated venv already).
if [ -n "${PYTHON:-}" ]; then
  exec $PYTHON -m nca.sweep "$@"
elif [ -x .venv/bin/python ]; then
  exec .venv/bin/python -m nca.sweep "$@"
elif command -v uv >/dev/null; then
  exec uv run python -m nca.sweep "$@"
else
  exec python3 -m nca.sweep "$@"
fi
