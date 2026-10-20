#!/bin/bash
# Submit a sweep of experiments 1-3 across a set of emoji targets.
#
#   bash slurm/submit_sweep.sh                 # all targets, default partition (gpu/H100)
#   bash slurm/submit_sweep.sh crab            # just one target
#
# Override the placement with environment variables, e.g. to use an idle MIG
# slice while the H100s are busy:
#
#   PARTITION=mig-priority CONSTRAINT=mig GRES=gpu:1g.10gb:1 TIME=02:00:00 \
#     bash slurm/submit_sweep.sh
#
# Targets are given by codepoint so no locale/UTF-8 issues over ssh.

set -euo pipefail
mkdir -p logs

# ascii-name -> emoji codepoint (new set, distinct from the paper's)
declare -A TARGETS=(
  [crab]="1f980"
  [octopus]="1f419"
  [trex]="1f996"
  [mushroom]="1f344"
  [cactus]="1f335"
  [turtle]="1f422"
)
EXPERIMENTS=(growing persistent regenerating)

PARTITION="${PARTITION:-gpu}"
CONSTRAINT="${CONSTRAINT:-h100}"
GRES="${GRES:-gpu:1}"
TIME="${TIME:-12:00:00}"
MEM="${MEM:-64G}"
CPUS="${CPUS:-8}"

if [[ $# -gt 0 ]]; then
  names=("$@")
else
  names=("${!TARGETS[@]}")
fi

for name in "${names[@]}"; do
  code="${TARGETS[$name]:?unknown target '$name'}"
  for exp in "${EXPERIMENTS[@]}"; do
    sbatch -p "$PARTITION" --constraint="$CONSTRAINT" --gres="$GRES" --time="$TIME" \
      --mem="$MEM" --cpus-per-task="$CPUS" \
      slurm/train.sbatch \
      --target "emoji:${code}" \
      --experiment "$exp" \
      --target-size 40 --padding 16 \
      --batch-size 16 --iterations 8000 --train-steps 64 96 \
      --ckpt-every 250 --log-every 25 \
      --run-name "${name}_${exp}"
  done
done

echo "submitted to $PARTITION. monitor with: squeue -u \$USER"
