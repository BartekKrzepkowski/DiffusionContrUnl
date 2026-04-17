#!/bin/bash
# ============================================================================
# SLURM Array Job – DDPM CIFAR-10 InTAct Ablation: no-actual-bounds grid
# ============================================================================
# Goal:
#   Disable remain-set-based ACTUAL bounds and search interval hyperparameters.
#
# Fixed hyperparameters:
#   lr=1e-4, n_iters=3000, lambda_interval=5.0, method=rl, use_actual_bounds=false
#
# Grid axis:
#   lower_percentile, upper_percentile, infinity_scale
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_ddpm_ablation_no_actual_bounds_grid.sh
# ============================================================================

#SBATCH --job-name=ddpm-nobounds-grid
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxa100
#SBATCH --array=0-6

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
USE_ACTUAL_BOUNDS=false
REDUCED_DIM=32
NORMALIZE_PROTECTION=true

# ---- Grid (7 combos) ----
LOWER_PCTS=(   0.10 0.05 0.01 0.10 0.01 0.01 0.05 )
UPPER_PCTS=(   0.90 0.95 0.99 0.90 0.99 0.99 0.95 )
INFTY_SCALES=( 10   20   20   40   40   80   80   )

IDX=${SLURM_ARRAY_TASK_ID}
LOWER=${LOWER_PCTS[$IDX]}
UPPER=${UPPER_PCTS[$IDX]}
INF_SCALE=${INFTY_SCALES[$IDX]}

echo "============================================"
echo "DDPM No-Actual-Bounds Grid – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  forget_class=${FORGET_CLASS}  lr=${LR}  n_iters=${NITERS}  lambda=${LAMBDA}"
echo "  method=${METHOD}  use_actual_bounds=${USE_ACTUAL_BOUNDS}"
echo "  lower=${LOWER}  upper=${UPPER}  infinity_scale=${INF_SCALE}"
echo "============================================"

TMPCONFIG="/tmp/ddpm_ablate_nobounds_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

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
lower = float("${LOWER}")
upper = float("${UPPER}")
inf_scale = float("${INF_SCALE}")
reduced_dim = int("${REDUCED_DIM}")
normalize_protection = "${NORMALIZE_PROTECTION}".lower() == "true"

cfg["unlearn"]["label_to_forget"] = forget_class
cfg["unlearn"]["lr"] = lr
cfg["unlearn"]["n_iters"] = niters
cfg["unlearn"]["method"] = method

cfg.setdefault("intact", {})
cfg["intact"]["lambda_interval"] = lam
cfg["intact"]["use_actual_bounds"] = use_actual_bounds
cfg["intact"]["lower_percentile"] = lower
cfg["intact"]["upper_percentile"] = upper
cfg["intact"]["infinity_scale"] = inf_scale
cfg["intact"]["reduced_dim"] = reduced_dim
cfg["intact"]["normalize_protection"] = normalize_protection

# Keep full-eval budget (5k FID / 500 classifier)
cfg.setdefault("evaluate", {}).setdefault("fid", {})["n_samples_per_class"] = 5000
cfg["evaluate"]["n_samples_per_class"] = 500
cfg["evaluate"].setdefault("classifier", {})["n_samples_per_class"] = 500

cfg.setdefault("wandb", {})
cfg["wandb"]["group"] = "cifar10-ablate-nobounds-grid"
cfg["wandb"]["tags"] = list(cfg["wandb"].get("tags", [])) + [
    "ablation", "no-actual-bounds", "grid-search",
    f"lower_{lower}", f"upper_{upper}", f"infscale_{inf_scale}",
    "lr_1e-4", "iters_3000", "lambda_5.0"
]

suffix = (
    f"ablate_nobounds_fc{forget_class}_l{lower}_u{upper}_inf{inf_scale}"
    f"_lr{lr}_ni{niters}_lam{lam}"
)
cfg["paths"]["output_dir"] = os.path.join(cfg["paths"]["output_dir"], suffix)
cfg["paths"]["checkpoint_dir"] = os.path.join(cfg["paths"]["checkpoint_dir"], suffix)

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

python pipeline.py --config "${TMPCONFIG}"

echo "No-actual-bounds grid job ${IDX} complete."
