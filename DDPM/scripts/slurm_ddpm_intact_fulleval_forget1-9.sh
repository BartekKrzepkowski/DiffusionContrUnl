#!/bin/bash
# ============================================================================
# SLURM Array Job – DDPM InTAct Full Eval (5k FID) – Forget Classes 1-9
# ============================================================================
# Runs InTAct unlearning + full evaluation (5000 samples/class FID, 500 clf)
# with fixed hyperparams: lr=1e-4, n_iters=3000, lambda=5.0, method=rl
# for each forget class 1 through 9 (skipping class 0 / airplane).
#
# HOW TO USE:
#   cd DDPM
#   sbatch scripts/slurm_ddpm_intact_fulleval_forget1-9.sh
# ============================================================================

#SBATCH --job-name=ddpm-intact-f1-9
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
# Fixed hyperparameters
# ============================================================================
LR=1e-4
NITERS=3000
LAMBDA=5.0
METHOD="rl"

# Forget classes 1-9 (array index 0 → class 1, index 8 → class 9)
FORGET_CLASSES=(1 2 3 4 5 6 7 8 9)
# ============================================================================

IDX=${SLURM_ARRAY_TASK_ID}
FORGET_CLASS=${FORGET_CLASSES[$IDX]}

echo "============================================"
echo "DDPM InTAct Full Eval – Job ${SLURM_ARRAY_JOB_ID}_${IDX}"
echo "  lr=${LR}  n_iters=${NITERS}  lambda=${LAMBDA}  method=${METHOD}"
echo "  Forget Class: ${FORGET_CLASS}"
echo "============================================"

# ---- Build per-job config by patching the full-eval template ----
TMPCONFIG="/tmp/ddpm_intact_fulleval_fc${FORGET_CLASS}_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

python - <<PYEOF
import yaml, os

with open("configs/pipeline_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# Override hyperparams
cfg["unlearn"]["lr"]              = float("${LR}")
cfg["unlearn"]["n_iters"]         = int("${NITERS}")
cfg["unlearn"]["method"]          = "${METHOD}"
cfg["unlearn"]["label_to_forget"] = int("${FORGET_CLASS}")
cfg["intact"]["lambda_interval"]  = float("${LAMBDA}")

# Adjust ref dataset path for the correct forget class
# Pattern: cifar10_without_label_<N>
base_ref = os.path.dirname(cfg["paths"]["ref_dataset_dir"])
cfg["paths"]["ref_dataset_dir"] = os.path.join(base_ref, f"cifar10_without_label_{${FORGET_CLASS}}")

# Tag the wandb run
cfg["wandb"]["tags"].append("fulleval-forget1-9")
cfg["wandb"]["group"] = "cifar10-fulleval-forget1-9"

# Unique output dirs per job
suffix = f"fulleval_fc{${FORGET_CLASS}}_lr{${LR}}_ni{${NITERS}}_lam{${LAMBDA}}"
cfg["paths"]["output_dir"]     = os.path.join(cfg["paths"]["output_dir"], suffix)
cfg["paths"]["checkpoint_dir"] = os.path.join(cfg["paths"]["checkpoint_dir"], suffix)

with open("${TMPCONFIG}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to ${TMPCONFIG}")
PYEOF

# ---- Run pipeline (train + sample + full eval + log) ----
python pipeline.py --config "${TMPCONFIG}"

echo "DDPM InTAct Full Eval – Forget Class ${FORGET_CLASS} – Job ${IDX} complete."
