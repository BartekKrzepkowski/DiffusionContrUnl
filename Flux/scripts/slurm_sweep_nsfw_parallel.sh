#!/bin/bash
# ============================================================================
# SLURM Job – Parallelised NSFW Sweep (Flux)
# ============================================================================
# Runs a wandb sweep for NSFW erasure with MULTIPLE sweep agents in parallel,
# one per GPU on the same node.  Each agent picks up the next hyperparameter
# combo automatically from the wandb controller.
#
# How it works:
#   1. Creates the sweep once (or reuses an existing SWEEP_ID).
#   2. Launches N_GPUS wandb agents in parallel, each pinned to its own GPU.
#   3. Each agent runs one sweep trial at a time (`--count 1` loop) and exits
#      when the sweep is exhausted or stopped from the wandb UI.
#
# Usage:
#   cd Flux
#   sbatch scripts/slurm_sweep_nsfw_parallel.sh          # auto-create sweep
#   sbatch scripts/slurm_sweep_nsfw_parallel.sh <id>     # reuse existing sweep
#
# To change GPU count edit N_GPUS below **and** the --gres line.
# ============================================================================

#SBATCH --job-name=flux-nsfw-sweep-par
#SBATCH --gres=gpu:4                   # ← number of GPUs on this node
#SBATCH --cpus-per-task=32             # 8 CPUs per GPU × 4
#SBATCH --mem=256GB
#SBATCH --partition=plgrid-gpu-a100

set -euo pipefail
N_GPUS=4                               # must match --gres=gpu:N above

# ---- Environment ----
source /net/tscratch/people/plgphelm/miniconda3/bin/activate
conda activate flux
cd "$HOME/repo/InTAct-Unl/Flux"
export PYTHONPATH="$HOME/repo/InTAct-Unl:${PYTHONPATH:-}"

# wandb settings
export WANDB_ENTITY="oneandzero24"
PROJECT_NAME="intact-flux"

# caches → shared storage
export HF_HOME="/net/tscratch/people/plgphelm/unl/.cache/huggingface"
export TORCH_HOME="/net/tscratch/people/plgphelm/unl/.cache/torch"
export XDG_CACHE_HOME="/net/tscratch/people/plgphelm/unl/.cache"
export WANDB_DIR="/net/tscratch/people/plgphelm/unl/.cache/wandb"
export WANDB_CACHE_DIR="/net/tscratch/people/plgphelm/unl/.cache/wandb"
export CLIP_CACHE_DIR="/net/tscratch/people/plgphelm/unl/.cache/clip"


echo "================================================================"
echo "Flux NSFW parallel sweep on $(hostname)  –  ${N_GPUS} GPUs"
echo "================================================================"

# ---- Create or reuse sweep ----
SWEEP_NAME="sweep_nsfw"
YAML_PATH="configs/intact/${SWEEP_NAME}.yaml"

if [ -n "${1:-}" ]; then
    # Reuse existing sweep ID passed as argument
    SWEEP_ID="$1"
    echo "Reusing existing sweep ID: $SWEEP_ID"
else
    echo "Creating new wandb sweep from $YAML_PATH …"
    SWEEP_OUT=$(wandb sweep --project "$PROJECT_NAME" --name "$SWEEP_NAME" "$YAML_PATH" 2>&1)
    echo "$SWEEP_OUT"

    SWEEP_ID=$(echo "$SWEEP_OUT" | awk '/wandb agent/{ match($0, /wandb agent ([^ ]+)/, arr); print arr[1]; }')
    if [ -z "$SWEEP_ID" ]; then
        SWEEP_ID=$(echo "$SWEEP_OUT" | awk '/Creating sweep with ID/{ match($0, /ID: ([^ ]+)/, arr); print arr[1]; }')
    fi
    if [ -z "$SWEEP_ID" ]; then
        echo "ERROR: Failed to parse sweep ID"
        exit 1
    fi
    echo "Created sweep ID: $SWEEP_ID"
fi

# ---- Launch one agent per GPU ----
PIDS=()
for GPU_IDX in $(seq 0 $((N_GPUS - 1))); do
    echo "Launching wandb agent on GPU ${GPU_IDX} …"
    (
        export CUDA_VISIBLE_DEVICES=$GPU_IDX
        # --count 0 = run until sweep is exhausted / stopped
        wandb agent --count 0 "$SWEEP_ID"
    ) &
    PIDS+=($!)
done

echo "All ${N_GPUS} agents launched.  PIDs: ${PIDS[*]}"
echo "Monitor at: https://wandb.ai/${WANDB_ENTITY}/${PROJECT_NAME}/sweeps"

# ---- Wait for all agents ----
FAIL=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || ((FAIL++))
done

if [ "$FAIL" -gt 0 ]; then
    echo "WARNING: ${FAIL} agent(s) exited with errors."
else
    echo "All agents finished successfully."
fi

echo "Flux NSFW parallel sweep finished."
