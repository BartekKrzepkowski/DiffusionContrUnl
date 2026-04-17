#!/bin/bash
# ============================================================================
# SLURM Array Job – Full eval (UA + FID) for all Imagenette classes 0-9
# ============================================================================
#   Hyperparameters: lr=5e-6, epochs=2, lambda=10
#   500 images/class (10 batch × 50 outer), FID(feature=64), no subsampling
#
# Usage:
#   cd SD
#   sbatch scripts/slurm_fulleval_class_all.sh
# ============================================================================

#SBATCH --job-name=sd-fulleval
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --partition=dgxh100
#SBATCH --array=0-9

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
cd $HOME/InTAct-Unl/SD
export PYTHONPATH="$HOME/InTAct-Unl:$PYTHONPATH"

CLASS=${SLURM_ARRAY_TASK_ID}

CLASSES=("tench" "english_springer" "cassette_player" "chain_saw" "church"
         "french_horn" "garbage_truck" "gas_pump" "golf_ball" "parachute")
CLASS_NAME=${CLASSES[$CLASS]}

echo "============================================"
echo "Full eval – class ${CLASS} (${CLASS_NAME})"
echo "  lr=1e-5  epochs=5  lambda=1"
echo "  Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "============================================"

# ---- Build per-class config ----
TMPCONFIG="/tmp/sd_fulleval_${SLURM_ARRAY_JOB_ID}_${CLASS}.yaml"

python - "$CLASS" "$TMPCONFIG" <<'PYEOF'
import yaml, sys

cls = int(sys.argv[1])
out = sys.argv[2]

with open("configs/pipeline_class_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# Hyperparameters
cfg["unlearn"]["class_to_forget"] = cls
cfg["unlearn"]["lr"] = 1.0e-5
cfg["unlearn"]["epochs"] = 3
cfg["intact"]["lambda_interval"] = 1.0

# 100 images per class: 10 batch × 10 outer → 900 fake (9 remaining classes)
# Subsample real Imagenette to 900 to match
cfg["evaluate"]["num_samples_per_prompt"] = 10
cfg["evaluate"]["n_outer"] = 10
cfg["evaluate"]["fid"]["max_real"] = 900
cfg["evaluate"]["fid"]["max_fake"] = None

# wandb tags
cfg["wandb"]["group"] = "class-fulleval-lr5e6-ep2-lam10"
cfg["wandb"]["tags"] = ["sd", "class-wise", "intact", "fulleval", "salun-protocol"]

# per-class output dir to avoid collisions
cfg["paths"]["output_dir"] = cfg["paths"]["output_dir"] + f"/class_{cls}"

with open(out, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to {out}")
PYEOF

# ---- Run pipeline ----
python pipeline.py --config "${TMPCONFIG}"

echo "Class ${CLASS} (${CLASS_NAME}) – done."
