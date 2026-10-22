#!/bin/bash
# Cross-target sweep: same conditional models as submit_multitarget_sweep.sh, but
# a slice of every batch is asked to convert an already-grown emoji into a
# *different* one (--mismatch-p).
#
# Rationale: with self-consistent (state, cond) pairs, a small MLP can satisfy the
# task by memorising N attractors and treating the one-hot as an index. Breaking
# the pairing forces the conditioning to steer the dynamics.
#
#   bash slurm/submit_crosstarget_sweep.sh
#   FAILOVER=1 bash slurm/submit_crosstarget_sweep.sh
#   DEPEND=20754359:20754360 bash slurm/submit_crosstarget_sweep.sh   # after those jobs
#
# Runs are named cross2 / cross4 / cross6 / cross8 so they don't collide with the
# matched multi{2,4,6,8} runs.

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
MISMATCH_P="${MISMATCH_P:-0.3}"
DEPEND="${DEPEND:-}"

# Fixed order, matching submit_multitarget_sweep.sh so the runs are comparable.
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
  sbatch ${DEPEND:+--dependency=afterany:$DEPEND} \
    -p "$PARTITION" --constraint="$CONSTRAINT" --gres="$GRES" --time="$TIME" \
    --mem="$MEM" --cpus-per-task="$CPUS" \
    slurm/train.sbatch \
    --targets "${specs[@]}" --experiment regenerating --mismatch-p "$MISMATCH_P" \
    --target-size 40 --padding 16 \
    --batch-size 16 --iterations "$ITERATIONS" --train-steps 64 96 \
    --amp "$AMP" \
    --ckpt-every 500 --log-every 25 \
    --run-name "cross${n}"
done

echo "submitted cross-target sweep (N=2,4,6,8, mismatch_p=$MISMATCH_P) to $PARTITION"
