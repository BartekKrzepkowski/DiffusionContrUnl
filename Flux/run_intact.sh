#!/usr/bin/env bash
# =============================================================================
# Run Flux InTAct Pipeline (Concept Erasure)
# =============================================================================
# Usage:
#   cd Flux
#   bash run_intact.sh [--config path/to/config.yaml] [--no-wandb]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Redirect all caches to /shared/results to avoid home-directory quota issues
export HF_HOME="/net/tscratch/people/plgphelm/unl/.cache/huggingface"
export TORCH_HOME="/net/tscratch/people/plgphelm/unl/.cache/torch"
export XDG_CACHE_HOME="/net/tscratch/people/plgphelm/unl/.cache"
export WANDB_DIR="/net/tscratch/people/plgphelm/unl/.cache/wandb"
export WANDB_CACHE_DIR="/net/tscratch/people/plgphelm/unl/.cache/wandb"
export CLIP_CACHE_DIR="/net/tscratch/people/plgphelm/unl/.cache/clip"

CONFIG="${1:-configs/intact/pipeline_concept.yaml}"
shift 2>/dev/null || true

echo "============================================"
echo " Flux InTAct Pipeline"
echo " Config: $CONFIG"
echo "============================================"

python intact_pipeline.py --config "$CONFIG" "$@"
