#!/bin/bash
# usage: scripts/train_weekly.sh [seed ...]   (default: all five seeds)
set -euo pipefail
export JAX_PLATFORMS=cpu

declare -A TEST_SEED=([123456]=265464 [364988]=379841 [457896]=569784 [698754]=898462 [789413]=978543)
SEEDS=${*:-"123456 364988 457896 698754 789413"}

for seed in $SEEDS; do
  args=(--config-name weekly seed=$seed train_file=train_noise01_4w_seed_$seed
        test_file=test_noise01_1w_odd_seed_${TEST_SEED[$seed]})
  python entrypoint.py "${args[@]}"                   # Perpetua*
  python entrypoint.py "${args[@]}" models=perpetua   # Perpetua
done
