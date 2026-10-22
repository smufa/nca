#!/bin/bash
# Sweep the network capacity (hidden width and channel count) on one target, to
# find how small the update rule can get before the pattern degrades.
#
#   bash slurm/submit_capacity_sweep.sh
#   PARTITION=mig-priority CONSTRAINT=mig GRES=gpu:1g.10gb:1 MEM=16G CPUS=4 \
#     bash slurm/submit_capacity_sweep.sh
#
# Runs the same regenerating crab as the full-size baseline (hidden 128, 16
# channels, 8336 params) so the numbers are directly comparable.
#
# NOTE: models with channels != 16 or hidden != 128 run on the WASM/WebGPU(ORT)
# engines; the native `shader` engine currently assumes C=16 and hidden=128.

set -euo pipefail
mkdir -p logs

TARGET="${TARGET:-emoji:1f980}"          # crab
PARTITION="${PARTITION:-mig-priority}"
CONSTRAINT="${CONSTRAINT:-mig}"
GRES="${GRES:-gpu:1g.10gb:1}"
TIME="${TIME:-02:00:00}"
MEM="${MEM:-16G}"
CPUS="${CPUS:-4}"

submit() { # name channels hidden
  sbatch -p "$PARTITION" --constraint="$CONSTRAINT" --gres="$GRES" --time="$TIME" \
    --mem="$MEM" --cpus-per-task="$CPUS" \
    slurm/train.sbatch \
    --target "$TARGET" --experiment regenerating \
    --target-size 40 --padding 16 \
    --batch-size 16 --iterations 8000 --train-steps 64 96 \
    --amp bf16 \
    --ckpt-every 500 --log-every 25 \
    --channels "$2" --hidden "$3" \
    --run-name "cap_c${2}_h${3}"
}

# hidden-width sweep at the paper's 16 channels
for h in 8 16 32 64 128; do submit "cap_c16_h${h}" 16 "$h"; done
# channel-count sweep at the paper's 128 hidden units
for c in 6 8 10 12; do submit "cap_c${c}_h128" "$c" 128; done

echo "submitted capacity sweep to $PARTITION. monitor: squeue -u \$USER"
