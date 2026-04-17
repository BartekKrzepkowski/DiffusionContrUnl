#!/bin/bash
# ============================================================================
# SLURM Array Job – SD Class Forgetting Eval-Only (pre-trained models)
# ============================================================================
# Evaluates already-trained class-forgetting models:
#   Generate Imagenette images (10 per prompt) → UA (classification)
#   → FID (remaining classes, full reference) → wandb
#
# Skips the unlearning step entirely.
#
# HOW TO USE:
#   1. Fill in MODEL_PATHS array with your trained model diffusers .pt paths
#   2. Set --array=0-<N-1> where N = number of models
#   3. sbatch scripts/slurm_sd_class_eval_only.sh
# ============================================================================

#SBATCH --job-name=sd-class-eval
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
COMBO_NUMBERS=(1 2)
MODEL_PATHS=(
    "/shared/results/common/miksa/intact/SD/models/compvis-intact-rl-class_0-targets_attn2.to_q_attn2.to_k_attn2.to_v-lambda_1.0-epochs_5-lr_5e-06/diffusers-intact-rl-class_0-targets_attn2.to_q_attn2.to_k_attn2.to_v-lambda_1.0-epochs_5-lr_5e-06.pt"
    "/shared/results/common/miksa/intact/SD/models/compvis-intact-rl-class_0-targets_attn2.to_q_attn2.to_k_attn2.to_v-lambda_0.5-epochs_3-lr_1e-05/diffusers-intact-rl-class_0-targets_attn2.to_q_attn2.to_k_attn2.to_v-lambda_0.5-epochs_3-lr_1e-05.pt"
)
# ============================================================================

IDX=${SLURM_ARRAY_TASK_ID}
COMBO_NUM=${COMBO_NUMBERS[$IDX]}
MODEL_PATH=${MODEL_PATHS[$IDX]}

# Extract model name from path (directory name = compvis-* model name)
MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_NAME=$(basename "$MODEL_DIR")

echo "============================================"
echo "SD Class Eval-Only – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
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
TMPCONFIG="/tmp/sd_class_eval_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

python - <<PYEOF
import yaml, os

with open("configs/pipeline_class_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# Tag the wandb run
cfg["wandb"]["tags"].append("eval-only")
cfg["wandb"]["tags"].append("combo${COMBO_NUM}")
cfg["wandb"]["group"] = "class-eval-only"

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
python pipeline.py \
    --config "${TMPCONFIG}" \
    --eval-only

echo "SD Class Eval-Only – Job ${IDX} complete."
