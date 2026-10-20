#!/bin/bash
# Submit a sweep of experiments 1-3 across a set of emoji targets.
#
#   bash slurm/submit_sweep.sh              # full sweep (18 jobs)
#   bash slurm/submit_sweep.sh crab         # just one target (3 jobs)
#
# Each job trains one model on an H100 and keeps intermediate checkpoints.

set -euo pipefail
mkdir -p logs

# ascii-name -> emoji (new set, distinct from the paper's)
declare -A TARGETS=(
  [crab]="🦀"
  [octopus]="🐙"
  [trex]="🦖"
  [mushroom]="🍄"
  [cactus]="🌵"
  [turtle]="🐢"
)
EXPERIMENTS=(growing persistent regenerating)

if [[ $# -gt 0 ]]; then
  names=("$@")
else
  names=("${!TARGETS[@]}")
fi

for name in "${names[@]}"; do
  emoji="${TARGETS[$name]:?unknown target '$name'}"
  for exp in "${EXPERIMENTS[@]}"; do
    sbatch slurm/train.sbatch \
      --target "emoji:${emoji}" \
      --experiment "$exp" \
      --target-size 40 --padding 16 \
      --batch-size 16 --iterations 8000 --train-steps 64 96 \
      --ckpt-every 250 --log-every 25 \
      --run-name "${name}_${exp}"
  done
done

echo "submitted. monitor with: squeue -u \$USER"
