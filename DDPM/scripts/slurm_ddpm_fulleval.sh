#!/bin/bash
# ============================================================================
# SLURM Array Job – DDPM Full Evaluation (best hyperparameter combos)
# ============================================================================
# Runs the full DDPM unlearning pipeline (unlearn → sample → eval → wandb)
# with reference-paper data sizes (5000 FID / 500 classifier per class).
#
# HOW TO USE:
#   1. Fill in the HPARAM arrays below with your best sweep results
#   2. Set --array=0-<N-1> where N = number of hparam combos
#   3. sbatch scripts/slurm_ddpm_fulleval.sh
# ============================================================================

#SBATCH --job-name=ddpm-fulleval
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxa100
#SBATCH --array=0-8

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# ============================================================================
# Best hyperparameter combinations from sweep
# ============================================================================
COMBO_NUMBERS=(     25      27      18      17      16      9       7       6       4      )  # Original sweep combo IDs
LEARNING_RATES=(    1e-4    1e-4    1e-4    1e-4    1e-4    1e-4    1e-4    1e-3    1e-3   )
N_ITERS=(           1000    3000    3000    1500    1000    3000    1000    3000    1000   )
LAMBDA_INTERVALS=(  5.0     5.0     1.0     1.0     1.0     0.1     0.1     0.1     0.1    )
METHODS=(           "rl"    "rl"    "rl"    "rl"    "rl"    "rl"    "rl"    "rl"    "rl"   )
# ============================================================================

IDX=${SLURM_ARRAY_TASK_ID}

COMBO_NUM=${COMBO_NUMBERS[$IDX]}
LR=${LEARNING_RATES[$IDX]}
NITERS=${N_ITERS[$IDX]}
LAMBDA=${LAMBDA_INTERVALS[$IDX]}
METHOD=${METHODS[$IDX]}

echo "============================================"
echo "DDPM Full Eval – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  Combo #${COMBO_NUM}:  lr=${LR}  n_iters=${NITERS}  lambda=${LAMBDA}  method=${METHOD}"
echo "============================================"

# ---- Build per-job config by patching the full-eval template ----
TMPCONFIG="/tmp/ddpm_fulleval_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

python - <<PYEOF
import yaml, copy, sys

with open("configs/pipeline_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# Override hyperparams
cfg["unlearn"]["lr"]     = float("${LR}")
cfg["unlearn"]["n_iters"] = int("${NITERS}")
cfg["unlearn"]["method"] = "${METHOD}"
cfg["intact"]["lambda_interval"] = float("${LAMBDA}")

# Tag the wandb run
cfg["wandb"]["tags"].append("fulleval-best")
cfg["wandb"]["group"] = "cifar10-fulleval"

# Unique output dirs per job
import os
suffix = f"combo{${COMBO_NUM}}_lr{${LR}}_ni{${NITERS}}_lam{${LAMBDA}}"
cfg["paths"]["output_dir"]     = os.path.join(cfg["paths"]["output_dir"], suffix)
cfg["paths"]["checkpoint_dir"] = os.path.join(cfg["paths"]["checkpoint_dir"], suffix)

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

# ---- Run pipeline ----
python pipeline.py --config "${TMPCONFIG}"

echo "DDPM Full Eval – Job ${IDX} complete."
