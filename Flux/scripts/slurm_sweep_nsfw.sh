#!/bin/bash
# ============================================================================
# SLURM Job – NSFW Sweep (Flux) — Single GPU
# ============================================================================
# Runs a full wandb sweep for NSFW erasure on Flux with H100 / 256 GB nodes.
# The sweep configuration lives at `configs/intact/sweep_nsfw.yaml`.
#
# For PARALLEL multi-GPU execution, use:
#   sbatch scripts/slurm_sweep_nsfw_parallel.sh
#
# Usage:
#   cd Flux
#   sbatch scripts/slurm_sweep_nsfw.sh
# ============================================================================

#SBATCH --job-name=flux-nsfw-sweep
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --partition=dgxh100

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flux          # adjust to the appropriate env if needed (was "ldm" previously)
cd $HOME/InTAct-Unl/Flux
export PYTHONPATH="$HOME/InTAct-Unl:$PYTHONPATH"

# wandb configuration (must have permission to create sweeps here)
export WANDB_ENTITY="oneandzero24"    # adjust to your account/org
PROJECT_NAME="intact-flux"

# caches to shared storage
export HF_HOME="/net/tscratch/people/plgphelm/unl/.cache/huggingface"
export TORCH_HOME="/net/tscratch/people/plgphelm/unl/.cache/torch"
export XDG_CACHE_HOME="/net/tscratch/people/plgphelm/unl/.cache"
export WANDB_DIR="/net/tscratch/people/plgphelm/unl/.cache/wandb"
export WANDB_CACHE_DIR="/net/tscratch/people/plgphelm/unl/.cache/wandb"
export CLIP_CACHE_DIR="/net/tscratch/people/plgphelm/unl/.cache/clip"

echo "Starting Flux NSFW sweep on $(hostname)"

echo "=== launching wandb sweep ==="
SWEEP_NAME="sweep_nsfw"
YAML_PATH="configs/intact/${SWEEP_NAME}.yaml"

echo "Using config: $YAML_PATH"
set -x
# capture both stdout and stderr because wandb prints ID to stderr
SWEEP_OUT=$(wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "$YAML_PATH" 2>&1)
echo "$SWEEP_OUT"
# try a couple of patterns for the agent command or ID
SWEEP_ID=$(echo "$SWEEP_OUT" | awk '/wandb agent/{ match($0, /wandb agent ([^ ]+)/, arr); print arr[1]; }')
if [ -z "$SWEEP_ID" ]; then
    SWEEP_ID=$(echo "$SWEEP_OUT" | awk '/Creating sweep with ID/{ match($0, /ID: ([^ ]+)/, arr); print arr[1]; }')
fi
if [ -z "$SWEEP_ID" ]; then
    echo "Failed to parse sweep ID from output"
    exit 1
fi

echo "Starting WandB agent for sweep ID: $SWEEP_ID"
wandb agent "$SWEEP_ID"

# wandb agent blocks until the sweep completes or is stopped

echo "Flux NSFW sweep finished."
