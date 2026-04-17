#!/bin/bash
# ============================================================================
# SLURM Array Job – SD NSFW Full Evaluation (I2P Benchmark)
# ============================================================================
# Runs the full SD NSFW unlearning pipeline:
#   Unlearn → Generate 4703 I2P images (1 per prompt) → NudeNet I2P (thr=0.6)
#   per-class counts → MS-COCO 10K FID & CLIP → probe images → wandb
#
# HOW TO USE:
#   1. Fill in the HPARAM arrays below with your best sweep results
#   2. Set --array=0-<N-1> where N = number of hparam combos
#   3. sbatch scripts/slurm_sd_nsfw_fulleval.sh
# ============================================================================

#SBATCH --job-name=sd-nsfw-fulleval
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxa100
#SBATCH --array=0-1

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
cd $HOME/InTAct-Unl/SD
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# ============================================================================
# Best hyperparameter combinations from sweep
# ============================================================================
COMBO_NUMBERS=(     19        15       )  # Original sweep combo IDs
LEARNING_RATES=(    5e-6      5e-6     )
EPOCHS=(            3         3        )
LAMBDA_INTERVALS=(  1.0       0.5      )
# ============================================================================

IDX=${SLURM_ARRAY_TASK_ID}

COMBO_NUM=${COMBO_NUMBERS[$IDX]}
LR=${LEARNING_RATES[$IDX]}
EP=${EPOCHS[$IDX]}
LAMBDA=${LAMBDA_INTERVALS[$IDX]}

echo "============================================"
echo "SD NSFW Full Eval – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  Combo #${COMBO_NUM}:  lr=${LR}  epochs=${EP}  lambda=${LAMBDA}"
echo "============================================"

# ---- Build per-job config by patching the full-eval template ----
TMPCONFIG="/tmp/sd_nsfw_fulleval_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

python - <<PYEOF
import yaml, os, sys

with open("configs/pipeline_nsfw_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# Override hyperparams
cfg["unlearn"]["lr"]     = float("${LR}")
cfg["unlearn"]["epochs"] = int("${EP}")
cfg["intact"]["lambda_interval"] = float("${LAMBDA}")

# Tag the wandb run
cfg["wandb"]["tags"].append("fulleval-best")
cfg["wandb"]["group"] = "nsfw-fulleval"

# Unique output dir per job
suffix = f"combo{${COMBO_NUM}}_lr{${LR}}_ep{${EP}}_lam{${LAMBDA}}"
cfg["paths"]["output_dir"] = os.path.join(cfg["paths"]["output_dir"], suffix)

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

# ---- Run pipeline ----
python pipeline.py --config "${TMPCONFIG}"

echo "SD NSFW Full Eval – Job ${IDX} complete."
