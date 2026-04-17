#!/bin/bash
# ============================================================================
# SLURM – FID Debug: compare 500/class and 5000/class reference sizes
# ============================================================================
# Task 1: Subsample 500/class from generated images → FID vs existing 500/class real ref
# Task 2: Save 5000/class real CIFAR-10 reference → FID vs full 5000/class generated
#
# Usage:
#   cd DDPM
#   sbatch scripts/slurm_fid_debug.sh
# ============================================================================

#SBATCH --job-name=fid-debug
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=dgxa100

# ---- Environment ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate salun-ddpm
cd $HOME/InTAct-Unl/DDPM
export PYTHONPATH=$PYTHONPATH:/home/miksa/InTAct-Unl/

# ============================================================================
# Paths (from pipeline_fulleval.yaml)
# ============================================================================

# Generated FID samples (5000/class × 9 remaining classes, from unlearned model)
GEN_DIR="/shared/results/common/miksa/intact/DDPM/results/fulleval/combo27_lr0.0001_ni3000_lam5.0/2026_02_17_195033/fid_samples_guidance_2_excluded_class_0"

# Current 500/class real reference (created by save_base_dataset.py)
REF_DIR="/shared/results/common/miksa/intact/DDPM/results/cifar10_without_label_0"

# New 5000/class real reference (will be created by this script)
NEW_REF_DIR="/shared/results/common/miksa/intact/DDPM/results/cifar10_without_label_0_5000"

# CIFAR-10 data root (for torchvision download)
DATA_PATH="/home/miksa/InTAct-Unl/data"

# ============================================================================
# Run
# ============================================================================

echo "============================================"
echo "FID Debug – Job ${SLURM_JOB_ID}"
echo "  Generated samples : ${GEN_DIR}"
echo "  Reference (500)   : ${REF_DIR}"
echo "  New ref (5000)    : ${NEW_REF_DIR}"
echo "============================================"

python scripts/fid_debug.py \
    --gen_dir "${GEN_DIR}" \
    --ref_dir "${REF_DIR}" \
    --data_path "${DATA_PATH}" \
    --label_to_forget 0 \
    --n_classes 10 \
    --n_per_class_gen 5000 \
    --new_ref_dir "${NEW_REF_DIR}" \
    --seed 42

echo "FID Debug – Done."
