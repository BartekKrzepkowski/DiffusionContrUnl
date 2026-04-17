#!/bin/bash
# ============================================================================
# SLURM – SD NSFW Eval-Only COCO30k (FID & CLIP)
# ============================================================================
# Loads a specific model, runs COCO30k evaluation (FID & CLIP) without
# regenerating any I2P/unsafe-prompt images.  The script points the
# pipeline at pre-generated I2P and COCO directories and sets skip_i2p.
# Usage:
#   sbatch scripts/slurm_sd_nsfw_eval_coco30k.sh
# ============================================================================

#SBATCH --job-name=sd-nsfw-coco30k
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=dgxa100

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
cd $HOME/InTAct-Unl/SD
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
MODEL_NAME="compvis-intact-nsfw-targets_attn2.to_q_attn2.to_k_attn2.to_v-lambda_1-lr_5e-06"
MODEL_PATH="/shared/results/common/miksa/intact/SD/models/${MODEL_NAME}/diffusers-intact-nsfw-targets_attn2.to_q_attn2.to_k_attn2.to_v-lambda_1-lr_5e-06.pt"

COCO_CSV="prompts/coco_30k.csv"
COCO_REF_DIR="/shared/results/common/miksa/intact/SD/data/coco_val2014_30k_ref"
EVAL_OUTPUT_DIR="/shared/results/common/miksa/intact/SD/coco30k_eval"
# directory containing already-generated COCO30k images for this model
COCO_GEN_DIR="${EVAL_OUTPUT_DIR}/coco_generated/${MODEL_NAME}"

# ---- Paths: I2P logic from reeval_combo19 ----
EVAL_ROOT="/shared/results/common/miksa/intact/SD/fulleval/eval_combo19"
I2P_DIR=$(find "${EVAL_ROOT}/generated" -mindepth 1 -maxdepth 1 -type d | head -1)
if [ -z "$I2P_DIR" ]; then
    I2P_DIR="${EVAL_ROOT}/generated"
fi

# Verify model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

# ---- Build eval-only config ----
TMPCONFIG="/tmp/sd_nsfw_eval_coco30k_${SLURM_JOB_ID}.yaml"

python - <<PYEOF
import yaml, os

with open("configs/pipeline_nsfw_eval_coco30k.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["wandb"]["tags"].append("eval-only")
cfg["wandb"]["tags"].append("coco30k")
cfg["wandb"]["group"] = "nsfw-coco30k-eval"
cfg["paths"]["output_dir"] = "${EVAL_OUTPUT_DIR}"
cfg["paths"]["coco_images_dir"] = "${COCO_REF_DIR}"
cfg["pipeline"]["model_name"] = "${MODEL_NAME}"
cfg["evaluate"]["coco"]["pregenerated_prompts_csv"] = "${COCO_CSV}"
cfg["evaluate"]["coco"]["n_captions"] = 30000
# point pipeline at already-generated COCO images so generation is skipped
cfg["evaluate"]["coco"]["pregenerated_images_path"] = "${COCO_GEN_DIR}"
# also disable any I2P production (safety net)
cfg.setdefault("evaluate", {})["skip_i2p"] = True

# Add I2P path (used for class/nsfw images)
cfg["evaluate"]["pregenerated_images_path"] = "${I2P_DIR}"

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

python pipeline.py \
    --config "${TMPCONFIG}" \
    --eval-only \
    --pregenerated-images "${I2P_DIR}" \
    --pregenerated-coco-images "${COCO_GEN_DIR}" \
    --pregenerated-coco-prompts-csv "${COCO_CSV}" \
    --fid-batch-size 64

echo "I2P images:  ${I2P_DIR}"
echo "SD NSFW Eval-Only COCO30k – done."
