#!/bin/bash
# ============================================================================
# DDPM Sweep Runner
# ============================================================================
# Usage:
#   ./run_sweep.sh sweep

set -e

# Redirect all caches to /shared/results to avoid home-directory quota issues
export HF_HOME="/shared/results/common/miksa/.cache/huggingface"
export TORCH_HOME="/shared/results/common/miksa/.cache/torch"
export XDG_CACHE_HOME="/shared/results/common/miksa/.cache"
export WANDB_DIR="/shared/results/common/miksa/.cache/wandb"
export WANDB_CACHE_DIR="/shared/results/common/miksa/.cache/wandb"
export CLIP_CACHE_DIR="/shared/results/common/miksa/.cache/clip"

MAIN_DIR="configs"
PROJECT_NAME="intact-ddpm"

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

# Main execution
if [ $# -eq 0 ]; then
  echo "Usage: $0 <sweep_name>"
  echo "Available sweeps:"
  echo "  - sweep"
  exit 1
fi

SWEEP_NAME="$1"
run_sweep_and_agent "$SWEEP_NAME"
