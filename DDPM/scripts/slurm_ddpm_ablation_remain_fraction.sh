#!/bin/bash
# ============================================================================
# SLURM Array Job – DDPM CIFAR-10 InTAct Ablation: remain-set fraction
# ============================================================================
# Goal:
#   Measure how the remain-set size used for ACTUAL BOUNDS affects unlearning.
#   We keep class-forgetting hyperparameters fixed and only vary remain fraction.
#
# Fixed hyperparameters:
#   lr=1e-4, n_iters=3000, lambda_interval=5.0, method=rl, use_actual_bounds=true
#
# Remain fractions tested:
#   1.0 (full), 0.5, 0.1
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_ddpm_ablation_remain_fraction.sh
# ============================================================================

#SBATCH --job-name=ddpm-remain-ablate
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxa100
#SBATCH --array=0-2

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# ---- Fixed setup ----
FORGET_CLASS=0
LR=1e-4
NITERS=3000
LAMBDA=5.0
METHOD="rl"
USE_ACTUAL_BOUNDS=true

# ---- Ablation axis ----
REMAIN_FRACTIONS=(1.0 0.75 0.5 0.25 0.1)
SUBSET_SEED=1234

IDX=${SLURM_ARRAY_TASK_ID}
REMAIN_FRAC=${REMAIN_FRACTIONS[$IDX]}

echo "============================================"
echo "DDPM Remain-Fraction Ablation – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  forget_class=${FORGET_CLASS}  lr=${LR}  n_iters=${NITERS}  lambda=${LAMBDA}"
echo "  method=${METHOD}  use_actual_bounds=${USE_ACTUAL_BOUNDS}"
echo "  remain_fraction=${REMAIN_FRAC}  subset_seed=${SUBSET_SEED}"
echo "============================================"

TMPCONFIG="/tmp/ddpm_ablate_remain_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

python - <<PYEOF
import os
import yaml

with open("configs/pipeline_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

forget_class = int("${FORGET_CLASS}")
lr = float("${LR}")
niters = int("${NITERS}")
lam = float("${LAMBDA}")
method = "${METHOD}"
use_actual_bounds = "${USE_ACTUAL_BOUNDS}".lower() == "true"
remain_frac = float("${REMAIN_FRAC}")
subset_seed = int("${SUBSET_SEED}")

cfg["unlearn"]["label_to_forget"] = forget_class
cfg["unlearn"]["lr"] = lr
cfg["unlearn"]["n_iters"] = niters
cfg["unlearn"]["method"] = method

cfg.setdefault("intact", {})
cfg["intact"]["lambda_interval"] = lam
cfg["intact"]["use_actual_bounds"] = use_actual_bounds

# New training knobs consumed by datasets.get_forget_dataset()
cfg["intact"]["remain_fraction"] = remain_frac
cfg["intact"]["remain_subset_seed"] = subset_seed

# Keep full-eval budget (5k FID / 500 classifier)
cfg.setdefault("evaluate", {}).setdefault("fid", {})["n_samples_per_class"] = 5000
cfg["evaluate"]["n_samples_per_class"] = 500
cfg["evaluate"].setdefault("classifier", {})["n_samples_per_class"] = 500

cfg.setdefault("wandb", {})
cfg["wandb"]["group"] = "cifar10-ablate-remain-fraction"
cfg["wandb"]["tags"] = list(cfg["wandb"].get("tags", [])) + [
    "ablation", "remain-fraction", f"remain_{remain_frac}",
    "lr_1e-4", "iters_3000", "lambda_5.0"
]

suffix = f"ablate_remain_fc{forget_class}_rf{remain_frac}_lr{lr}_ni{niters}_lam{lam}"
cfg["paths"]["output_dir"] = os.path.join(cfg["paths"]["output_dir"], suffix)
cfg["paths"]["checkpoint_dir"] = os.path.join(cfg["paths"]["checkpoint_dir"], suffix)

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

python pipeline.py --config "${TMPCONFIG}"

echo "Remain-fraction ablation job ${IDX} complete."
