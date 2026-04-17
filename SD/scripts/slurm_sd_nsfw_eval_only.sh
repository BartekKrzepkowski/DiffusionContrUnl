#!/bin/bash
# ============================================================================
# SLURM Array Job – SD NSFW Eval-Only (I2P Benchmark, pre-trained models)
# ============================================================================
# Evaluates already-trained models with the I2P benchmark protocol:
#   Generate 4703 I2P images (1 per prompt) → NudeNet I2P (thr=0.6, detailed)
#   → MS-COCO 10K FID & CLIP → probe images → wandb
#
# Skips the unlearning step entirely.
#
# HOW TO USE:
#   Fill in MODEL_PATHS array with your trained model checkpoint paths
#   sbatch scripts/slurm_sd_nsfw_eval_only.sh
# ============================================================================

#SBATCH --job-name=sd-nsfw-eval
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=dgxa100
#SBATCH --array=0-1

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
cd $HOME/InTAct-Unl/SD
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# ============================================================================
# Pre-trained model paths (diffusers .pt files)
# ============================================================================
COMBO_NUMBERS=(19 15)
MODEL_PATHS=(
    "/shared/results/common/miksa/intact/SD/models/compvis-intact-nsfw-targets_attn2.to_q_attn2.to_k_attn2.to_v-lambda_1-lr_5e-06/diffusers-intact-nsfw-targets_attn2.to_q_attn2.to_k_attn2.to_v-lambda_1-lr_5e-06.pt"
    "/shared/results/common/miksa/intact/SD/models/compvis-intact-nsfw-targets_attn2.to_q_attn2.to_k_attn2.to_v-lambda_0.5-lr_5e-06/diffusers-intact-nsfw-targets_attn2.to_q_attn2.to_k_attn2.to_v-lambda_0.5-lr_5e-06.pt"
)
# ============================================================================

IDX=${SLURM_ARRAY_TASK_ID}
COMBO_NUM=${COMBO_NUMBERS[$IDX]}
MODEL_PATH=${MODEL_PATHS[$IDX]}

# Extract model name from path
MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_NAME=$(basename "$MODEL_DIR")

echo "============================================"
echo "SD NSFW Eval-Only (I2P) – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  Combo #${COMBO_NUM}"
echo "  Model: ${MODEL_NAME}"
echo "  Path: ${MODEL_PATH}"
echo "============================================"

# Verify model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

# ---- Build eval-only config ----
TMPCONFIG="/tmp/sd_nsfw_eval_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

python - <<PYEOF
import yaml, os

with open("configs/pipeline_nsfw_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# Tag the wandb run
cfg["wandb"]["tags"].append("eval-only")
cfg["wandb"]["tags"].append("i2p")
cfg["wandb"]["tags"].append("combo${COMBO_NUM}")
cfg["wandb"]["group"] = "nsfw-eval-only-i2p"

# Unique output dir per job
suffix = f"eval_combo{${COMBO_NUM}}"
cfg["paths"]["output_dir"] = os.path.join(cfg["paths"]["output_dir"], suffix)

# Set explicit model_name so get_model_name() is bypassed in eval-only mode
cfg["pipeline"]["model_name"] = "${MODEL_NAME}"

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

# ---- Run eval-only pipeline ----
# --eval-only: skip unlearning, use pre-trained model weights
# The model_name is derived from the config; the weights at MODEL_PATH
# must match the expected path pattern: <model_save_dir>/<model_name>/diffusers-*.pt
python pipeline.py \
    --config "${TMPCONFIG}" \
    --eval-only

echo "SD NSFW Eval-Only (I2P) – Job ${IDX} complete."
