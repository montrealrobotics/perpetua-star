#!/bin/bash
# usage: [SEEDS="<seed> ..."] scripts/reproduce_table1.sh [params_dir (default: models/weekly_schedule)] [results_dir (default: results/weekly)]
set -euo pipefail
export JAX_PLATFORMS=cpu

PARAMS_DIR=${1:-models/weekly_schedule}
RESULTS_DIR=${2:-results/weekly}
SEEDS=${SEEDS:-"123456 364988 457896 698754 789413"}
declare -A TEST_SEED=([123456]=265464 [364988]=379841 [457896]=569784 [698754]=898462 [789413]=978543)

for seed in $SEEDS; do
  args=(--config-name test_weekly seed=$seed train_file=train_noise01_4w_seed_$seed
        test_file=test_noise01_1w_odd_seed_${TEST_SEED[$seed]} mask_weekends=true
        params_dir=$PARAMS_DIR results_dir=$RESULTS_DIR)
  python entrypoint.py "${args[@]}" models=perpetua  # Perpetua
  python entrypoint.py "${args[@]}" fremen=true      # FreMEn
  python entrypoint.py "${args[@]}"                  # Perpetua* (FreMEn prior)
  python entrypoint.py "${args[@]}" llm_prior=true   # Perpetua* (LLM prior)
  python entrypoint.py "${args[@]}" gt_prior=true    # Perpetua* (ground-truth prior)
done

# Figure 4 shows seed 698754
FIG_SEED=698754
[[ " $SEEDS " == *" $FIG_SEED "* ]] || FIG_SEED=${SEEDS%% *}
python figures/table1.py --results-dir "$RESULTS_DIR"
python figures/weekly.py --results-dir "$RESULTS_DIR" --output-dir "$RESULTS_DIR/figures" --seed $FIG_SEED --test-file test_noise01_1w_odd_seed_${TEST_SEED[$FIG_SEED]}
