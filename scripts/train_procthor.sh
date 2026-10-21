#!/bin/bash
# usage: scripts/train_procthor.sh <layout> <SemiStaticSim privileged folder of the layout> [hydra overrides...]
set -euo pipefail
export JAX_PLATFORMS=cpu

python entrypoint.py --config-name train_procthor run_id="$1" sequence_path="$2" "${@:3}"
