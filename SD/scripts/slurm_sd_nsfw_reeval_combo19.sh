#!/bin/bash
# ============================================================================
# SLURM – Re-evaluate combo19 from pre-generated images
# ============================================================================
# Runs NudeNet I2P counts + MS-COCO FID & CLIP on already-generated images.
# No model loading, no image generation — pure evaluation.
#
#   sbatch scripts/slurm_sd_nsfw_reeval_combo19.sh
# ============================================================================

#SBATCH --job-name=sd-reeval-c19
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxa100

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
cd $HOME/InTAct-Unl/SD
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# ---- Fix onnxruntime thread-affinity errors on SLURM ----
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# ---- Paths ----
EVAL_ROOT="/shared/results/common/miksa/intact/SD/fulleval/eval_combo19"

# Auto-detect the model-name subfolder inside generated/ and coco_generated/
I2P_DIR=$(find "${EVAL_ROOT}/generated" -mindepth 1 -maxdepth 1 -type d | head -1)
COCO_DIR=$(find "${EVAL_ROOT}/coco_generated" -mindepth 1 -maxdepth 1 -type d | head -1)
COCO_CSV="${EVAL_ROOT}/coco_prompts.csv"

# Fallback: if images are directly in generated/ (no subfolder)
if [ -z "$I2P_DIR" ]; then
    I2P_DIR="${EVAL_ROOT}/generated"
fi
if [ -z "$COCO_DIR" ]; then
    COCO_DIR="${EVAL_ROOT}/coco_generated"
fi

echo "============================================"
echo "SD NSFW Re-eval combo19"
echo "  I2P images:  ${I2P_DIR}"
echo "  COCO images: ${COCO_DIR}"
echo "  COCO CSV:    ${COCO_CSV}"
echo "  I2P count:   $(find ${I2P_DIR} -name '*.png' -o -name '*.jpg' | wc -l)"
echo "  COCO count:  $(find ${COCO_DIR} -name '*.png' -o -name '*.jpg' | wc -l)"
echo "============================================"

# ---- Build minimal eval-only config ----
TMPCONFIG="/tmp/sd_nsfw_reeval_combo19_${SLURM_JOB_ID}.yaml"

python - <<PYEOF
import yaml, os

with open("configs/pipeline_nsfw_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# wandb tags
cfg["wandb"]["tags"] = ["sd", "nsfw", "intact", "reeval", "combo19"]
cfg["wandb"]["group"] = "nsfw-reeval"

# Point output_dir to the existing eval directory
cfg["paths"]["output_dir"] = "${EVAL_ROOT}"

# Pipeline settings
cfg["pipeline"]["eval_only"] = True
cfg["pipeline"]["model_name"] = "reeval-combo19"  # arbitrary, not used for loading

# Pre-generated I2P images (NudeNet will scan these)
cfg.setdefault("evaluate", {})["pregenerated_images_path"] = "${I2P_DIR}"

# Pre-generated COCO images (FID + CLIP)
cfg["evaluate"].setdefault("coco", {})
cfg["evaluate"]["coco"]["enabled"] = True
cfg["evaluate"]["coco"]["pregenerated_images_path"] = "${COCO_DIR}"
cfg["evaluate"]["coco"]["pregenerated_prompts_csv"] = "${COCO_CSV}"

# NudeNet enabled
cfg["evaluate"].setdefault("nudenet", {})
cfg["evaluate"]["nudenet"]["enabled"] = True
cfg["evaluate"]["nudenet"]["threshold"] = 0.6
cfg["evaluate"]["nudenet"]["detailed"] = True

# Probe: skip (already generated)
cfg["evaluate"]["probe"] = {"enabled": False}

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

echo "---- Running eval-only pipeline ----"
python pipeline.py \
    --config "${TMPCONFIG}" \
    --eval-only \
    --pregenerated-images "${I2P_DIR}" \
    --pregenerated-coco-images "${COCO_DIR}" \
    --pregenerated-coco-prompts-csv "${COCO_CSV}" \
    --fid-batch-size 64

echo "SD NSFW Re-eval combo19 – done."
