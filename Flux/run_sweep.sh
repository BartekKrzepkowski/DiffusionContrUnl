#!/usr/bin/env bash
# ============================================================================
# Flux InTAct Sweep Runner
# ============================================================================
# Usage:
#   cd Flux
#   ./run_sweep.sh sweep_concept
#   ./run_sweep.sh sweep_class

set -e

# Redirect all caches to /shared/results to avoid home-directory quota issues
export HF_HOME="/net/tscratch/people/plgphelm/unl/.cache/huggingface"
export TORCH_HOME="/net/tscratch/people/plgphelm/unl/.cache/torch"
export XDG_CACHE_HOME="/net/tscratch/people/plgphelm/unl/.cache"
export WANDB_DIR="/net/tscratch/people/plgphelm/unl/.cache/wandb"
export WANDB_CACHE_DIR="/net/tscratch/people/plgphelm/unl/.cache/wandb"
export CLIP_CACHE_DIR="/net/tscratch/people/plgphelm/unl/.cache/clip"

MAIN_DIR="configs/intact"
PROJECT_NAME="intact-flux"

run_sweep_and_agent() {
  SWEEP_NAME="$1"

  YAML_PATH="$MAIN_DIR/${SWEEP_NAME}.yaml"

  if [ ! -f "$YAML_PATH" ]; then
    echo "Error: YAML file '${SWEEP_NAME}.yaml' not found in $MAIN_DIR"
    exit 1
  fi

  echo "Running wandb sweep for: $SWEEP_NAME"
  wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "$YAML_PATH" > ${SWEEP_NAME}_temp_output.txt 2>&1

  SWEEP_ID=$(awk '/wandb agent/{ match($0, /wandb agent (.+)/, arr); print arr[1]; }' ${SWEEP_NAME}_temp_output.txt)

  rm ${SWEEP_NAME}_temp_output.txt

  echo "Starting WandB agent for sweep ID: $SWEEP_ID"
  wandb agent "$SWEEP_ID"
}

if [ $# -eq 0 ]; then
  echo "Usage: $0 <sweep_name>"
  echo "  e.g.: $0 sweep_concept"
  echo "  e.g.: $0 sweep_class"
  echo "  e.g.: $0 sweep_nsfw"  # NSFW erasure using dataset paths
  exit 1
fi

for SWEEP_NAME in "$@"; do
  run_sweep_and_agent "$SWEEP_NAME"
done
