#!/bin/bash
# ============================================================================
# SLURM Array Job – Full eval (UA + FID) for Imagenette class 0 (tench) with multiple param combos
# ============================================================================
#   Each job runs a different (lambda, epochs, lr) combo for class 0 (tench)
#
# Usage:
#   cd SD
#   sbatch scripts/slurm_fulleval_class_tench_grid.sh
# ============================================================================

#SBATCH --job-name=sd-fulleval-tench-grid
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --partition=dgxh100
#SBATCH --array=0-7

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
cd $HOME/InTAct-Unl/SD
export PYTHONPATH="$HOME/InTAct-Unl:$PYTHONPATH"

CLASS=0  # tench
CLASS_NAME="tench"

LAMBDAS=(5 10 1 0.5 10 5 5 1)
EPOCHS=(3 3 2 2 5 2 5 5)
LRS=(0.000005 0.000005 0.000005 0.000005 0.000005 0.00001 0.00001 0.00001)

IDX=${SLURM_ARRAY_TASK_ID}
LAMBDA=${LAMBDAS[$IDX]}
EPOCH=${EPOCHS[$IDX]}
LR=${LRS[$IDX]}

PARAM_TAG="lam${LAMBDA}-ep${EPOCH}-lr${LR}"

TMPCONFIG="/tmp/sd_fulleval_tench_${SLURM_ARRAY_JOB_ID}_${IDX}.yaml"

python - "$CLASS" "$LAMBDA" "$EPOCH" "$LR" "$TMPCONFIG" <<'PYEOF'
import yaml, sys
cls = int(sys.argv[1])
lambda_val = float(sys.argv[2])
epochs = int(sys.argv[3])
lr = float(sys.argv[4])
out = sys.argv[5]

with open("configs/pipeline_class_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["unlearn"]["class_to_forget"] = cls
cfg["unlearn"]["lr"] = lr
cfg["unlearn"]["epochs"] = epochs
cfg["intact"]["lambda_interval"] = lambda_val

cfg["evaluate"]["num_samples_per_prompt"] = 10
cfg["evaluate"]["n_outer"] = 10
cfg["evaluate"]["fid"]["max_real"] = 900
cfg["evaluate"]["fid"]["max_fake"] = None

cfg["wandb"]["group"] = f"tench-fulleval-{lambda_val}-{epochs}-{lr}"
cfg["wandb"]["tags"] = ["sd", "class-wise", "intact", "fulleval", "salun-protocol", "tench", f"lambda_{lambda_val}", f"epochs_{epochs}", f"lr_{lr}"]

cfg["paths"]["output_dir"] = cfg["paths"]["output_dir"] + f"/tench_{lambda_val}_{epochs}_{lr}"

with open(out, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to {out}")
PYEOF

python pipeline.py --config "${TMPCONFIG}"

echo "Tench (${PARAM_TAG}) – done."
