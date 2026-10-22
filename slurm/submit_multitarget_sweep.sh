#!/bin/bash
# Sweep how many emoji one conditional model can reproduce.
#
# Each run gets the first N of a fixed emoji list (so the runs are nested), and
# the model receives an N-way one-hot selector appended to its perception.
#
#   bash slurm/submit_multitarget_sweep.sh
#   PARTITION=mig-priority CONSTRAINT=mig GRES=gpu:1g.10gb:1 MEM=16G CPUS=4 \
#     bash slurm/submit_multitarget_sweep.sh
#
# Runs are named multi2 / multi4 / multi6 / multi8.

set -euo pipefail
mkdir -p logs

PARTITION="${PARTITION:-mig-priority}"
CONSTRAINT="${CONSTRAINT:-mig}"
GRES="${GRES:-gpu:1g.10gb:1}"
TIME="${TIME:-02:00:00}"
MEM="${MEM:-16G}"
CPUS="${CPUS:-4}"
AMP="${AMP:-off}"
ITERATIONS="${ITERATIONS:-8000}"

# Fixed order: multi<N> always uses the first N of this list.
EMOJIS=(
  "emoji:1f980"  # crab
  "emoji:1f419"  # octopus
  "emoji:1f996"  # t-rex
  "emoji:1f422"  # turtle
  "emoji:1f335"  # cactus
  "emoji:1f344"  # mushroom
  "emoji:1f41d"  # bee
  "emoji:1f9a9"  # flamingo
)

for n in 2 4 6 8; do
  specs=("${EMOJIS[@]:0:$n}")
  sbatch -p "$PARTITION" --constraint="$CONSTRAINT" --gres="$GRES" --time="$TIME" \
    --mem="$MEM" --cpus-per-task="$CPUS" \
    slurm/train.sbatch \
    --targets "${specs[@]}" --experiment regenerating \
    --target-size 40 --padding 16 \
    --batch-size 16 --iterations "$ITERATIONS" --train-steps 64 96 \
    --amp "$AMP" \
    --ckpt-every 500 --log-every 25 \
    --run-name "multi${n}"
done

echo "submitted multi-target sweep (N=2,4,6,8) to $PARTITION"
