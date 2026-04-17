#!/bin/bash
# ============================================================================
# SLURM Array Job – Full eval grid search across ALL Imagenette classes
# ============================================================================
#   Fully parallelised: each job = one (hyperparam combo, class) pair.
#
#   This script now runs two sweeps in one array:
#   1) FULL sweep across all 10 Imagenette classes with NON-ZERO alpha.
#   2) FOCUSED sweep on {tench, chain_saw, golf_ball} with alpha=0 and
#      alternative lambda/epoch/lr combos.
#
#   Indexing:
#     - TASK_ID < FOCUS_JOBS         -> focused alpha=0 sweep
#     - otherwise                    -> full non-zero-alpha sweep
#
# Usage:
#   cd SD
#   sbatch scripts/slurm_fulleval_grid_all_classes.sh
# ============================================================================

#SBATCH --job-name=sd-grid-all
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --partition=dgxh100
#SBATCH --array=0-97

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm
cd $HOME/InTAct-Unl/SD
export PYTHONPATH="$HOME/InTAct-Unl:$PYTHONPATH"

# ---- Full-grid hyperparameters: non-zero alpha across all classes ----
FULL_ALPHAS=( 0.05 0.05 0.10 0.10 0.20 0.20 0.30 0.30 )
FULL_LAMBDAS=(5    10   5    10   3    7    3    7   )
FULL_EPOCHS=( 2    2    3    3    4    4    5    5   )
FULL_LRS=(    5e-6 1e-5 5e-6 1e-5 5e-6 1e-5 5e-6 1e-5)

NUM_FULL_COMBOS=${#FULL_ALPHAS[@]}  # 8
NUM_CLASSES=10

# ---- Focused alpha=0 grid for selected classes only ----
FOCUS_CLASS_IDS=(0 3 8)  # tench, chain_saw, golf_ball
FOCUS_ALPHAS=( 0.0 0.0 0.0 0.0 0.0 0.0 )
FOCUS_LAMBDAS=(0.5 1   2   5   10  15  )
FOCUS_EPOCHS=( 2   3   3   5   5   7   )
FOCUS_LRS=(    2e-6 5e-6 1e-5 1e-5 2e-5 2e-5)

NUM_FOCUS_COMBOS=${#FOCUS_ALPHAS[@]}      # 6
NUM_FOCUS_CLASSES=${#FOCUS_CLASS_IDS[@]}  # 3

FULL_JOBS=$(( NUM_FULL_COMBOS * NUM_CLASSES ))
FOCUS_JOBS=$(( NUM_FOCUS_COMBOS * NUM_FOCUS_CLASSES ))
TOTAL_JOBS=$(( FULL_JOBS + FOCUS_JOBS ))

TASK_ID=${SLURM_ARRAY_TASK_ID}

if (( TASK_ID >= TOTAL_JOBS )); then
    echo "Task ${TASK_ID} is outside active range [0, $((TOTAL_JOBS - 1))]. Exiting."
    exit 0
fi

CLASSES=("tench" "english_springer" "cassette_player" "chain_saw" "church"
         "french_horn" "garbage_truck" "gas_pump" "golf_ball" "parachute")

if (( TASK_ID < FOCUS_JOBS )); then
    SWEEP_KIND="focus-alpha0"
    LOCAL_ID=${TASK_ID}
    COMBO_IDX=$(( LOCAL_ID / NUM_FOCUS_CLASSES ))
    FOCUS_CLASS_POS=$(( LOCAL_ID % NUM_FOCUS_CLASSES ))
    CLASS=${FOCUS_CLASS_IDS[$FOCUS_CLASS_POS]}

    ALPHA=${FOCUS_ALPHAS[$COMBO_IDX]}
    LAMBDA=${FOCUS_LAMBDAS[$COMBO_IDX]}
    EPOCH=${FOCUS_EPOCHS[$COMBO_IDX]}
    LR=${FOCUS_LRS[$COMBO_IDX]}
else
    SWEEP_KIND="full-nonzero-alpha"
    LOCAL_ID=$(( TASK_ID - FOCUS_JOBS ))
    COMBO_IDX=$(( LOCAL_ID / NUM_CLASSES ))
    CLASS=$(( LOCAL_ID % NUM_CLASSES ))

    ALPHA=${FULL_ALPHAS[$COMBO_IDX]}
    LAMBDA=${FULL_LAMBDAS[$COMBO_IDX]}
    EPOCH=${FULL_EPOCHS[$COMBO_IDX]}
    LR=${FULL_LRS[$COMBO_IDX]}
fi

CLASS_NAME=${CLASSES[$CLASS]}
PARAM_TAG="a${ALPHA}-lam${LAMBDA}-ep${EPOCH}-lr${LR}"

echo "============================================"
echo "Grid search (${SWEEP_KIND}) – combo ${COMBO_IDX} (${PARAM_TAG}), class ${CLASS} (${CLASS_NAME})"
echo "  Total active tasks: ${TOTAL_JOBS} (full=${FULL_JOBS}, focus=${FOCUS_JOBS})"
echo "  Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "============================================"

# ---- Build per-job config ----
TMPCONFIG="/tmp/sd_grid_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.yaml"

python - "$CLASS" "$ALPHA" "$LAMBDA" "$EPOCH" "$LR" "$PARAM_TAG" "$CLASS_NAME" "$SWEEP_KIND" "$TMPCONFIG" <<'PYEOF'
import yaml, sys

cls        = int(sys.argv[1])
alpha_val  = float(sys.argv[2])
lambda_val = float(sys.argv[3])
epochs     = int(sys.argv[4])
lr         = float(sys.argv[5])
param_tag  = sys.argv[6]
cls_name   = sys.argv[7]
sweep_kind = sys.argv[8]
out        = sys.argv[9]

with open("configs/pipeline_class_fulleval.yaml") as f:
    cfg = yaml.safe_load(f)

# Hyperparameters
cfg["unlearn"]["class_to_forget"] = cls
cfg["unlearn"]["alpha"]           = alpha_val
cfg["unlearn"]["lr"]              = lr
cfg["unlearn"]["epochs"]          = epochs
cfg["intact"]["lambda_interval"]  = lambda_val

# Evaluation budget (same as single-class fulleval)
cfg["evaluate"]["num_samples_per_prompt"] = 10
cfg["evaluate"]["n_outer"]                = 10
cfg["evaluate"]["fid"]["max_real"]        = 900
cfg["evaluate"]["fid"]["max_fake"]        = None

# wandb – group by hyperparam combo so cross-class averages are trivial
cfg["wandb"]["group"] = f"grid-{sweep_kind}-{param_tag}"
cfg["wandb"]["tags"]  = [
    "sd", "class-wise", "intact", "fulleval", "grid-search", sweep_kind,
    f"alpha_{alpha_val}", f"lambda_{lambda_val}", f"epochs_{epochs}", f"lr_{lr}",
    cls_name,
]

# Unique output dir per (combo, class)
cfg["paths"]["output_dir"] = (
    cfg["paths"]["output_dir"] + f"/grid/{sweep_kind}/{param_tag}/class_{cls}"
)

with open(out, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Config written to {out}")
PYEOF

# ---- Run pipeline ----
python pipeline.py --config "${TMPCONFIG}"

echo "${SWEEP_KIND}: combo ${COMBO_IDX} (${PARAM_TAG}), class ${CLASS} (${CLASS_NAME}) – done."
