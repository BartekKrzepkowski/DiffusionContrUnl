#!/bin/bash
# Generate the synthetic NSFW / not-NSFW image folders used by SD NSFW unlearning.
#
# Historical setup in NOTES.md uses 800 SD v1.4 samples per prompt:
#   - "a photo of a nude person" -> nsfw_data
#   - "a photo of a person wearing clothes" -> not_nsfw_data
#
# Usage:
#   bash SD/scripts/slurm_sd_generate_nsfw_data.sh
#   bash SD/scripts/slurm_sd_generate_nsfw_data.sh --samples 800
#   sbatch SD/scripts/slurm_sd_generate_nsfw_data.sh --samples 800

#SBATCH --job-name=sd-nsfw-data
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=/net/storage/pr3/plgrid/plggdnnp/plgkrzepk/SD/slurm_raw/%x-%j.out
#SBATCH --error=/net/storage/pr3/plgrid/plggdnnp/plgkrzepk/SD/slurm_raw/%x-%j.err

set -euo pipefail

SCRIPT_DIR_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH_SELF="${SCRIPT_DIR_SELF}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR_SELF}/../.." && pwd)"

SAMPLES="${SAMPLES:-800}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
DDIM_STEPS="${DDIM_STEPS:-100}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
DEVICE="${DEVICE:-cuda:0}"
STORAGE_ROOT="${STORAGE_ROOT:-/net/storage/pr3/plgrid/plggdnnp}"
SD_STORAGE_ROOT="${SD_STORAGE_ROOT:-${STORAGE_ROOT}/${USER}/SD}"
NSFW_DATA="${NSFW_DATA:-${SD_STORAGE_ROOT}/data/nsfw}"
NOT_NSFW_DATA="${NOT_NSFW_DATA:-${SD_STORAGE_ROOT}/data/not-nsfw}"
STAGING_DIR="${STAGING_DIR:-${SD_STORAGE_ROOT}/data/nsfw_generation_staging}"
SD_CKPT="${SD_CKPT:-${STORAGE_ROOT}/SD/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt}"
SD_CONFIG="${SD_CONFIG:-configs/stable-diffusion/v1-intact.yaml}"
DRY_RUN="${DRY_RUN:-0}"
FORWARD_ARGS=()
WRAPPER_SBATCH_ARGS=()

print_help() {
    cat <<'HELP_EOF'
Uzycie:
  bash SD/scripts/slurm_sd_generate_nsfw_data.sh [opcje]
  sbatch SD/scripts/slurm_sd_generate_nsfw_data.sh [opcje]

Opcje:
  --samples N          Laczna liczba obrazow na prompt. Domyslnie 800.
  --batch-size N       Ile obrazow generowac naraz na prompt. Domyslnie 4.
  --nsfw-data PATH     Folder wyjsciowy dla obrazow NSFW.
  --not-nsfw-data PATH Folder wyjsciowy dla obrazow non-NSFW.
  --staging-dir PATH   Folder roboczy na wygenerowane obrazy 0_*.png i 1_*.png.
  --sd-ckpt PATH       Checkpoint SD v1.4 .ckpt.
  --sd-config PATH     Config CompVis SD. Domyslnie configs/stable-diffusion/v1-intact.yaml.
  --ddim-steps N       Domyslnie 100.
  --guidance-scale V   Domyslnie 7.5.
  --image-size N       Domyslnie 512.
  --dry-run            Wypisz ustawienia bez generowania.
  -h, --help           Pomoc.
HELP_EOF
}

require_value() {
    if [[ "$2" -lt 2 ]]; then
        echo "Brak wartosci dla $1" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) print_help; exit 0 ;;
        --samples) require_value "$1" "$#"; SAMPLES="$2"; FORWARD_ARGS+=("--samples" "$2"); shift 2 ;;
        --samples=*) SAMPLES="${1#*=}"; FORWARD_ARGS+=("--samples" "${1#*=}"); shift ;;
        --batch-size) require_value "$1" "$#"; GEN_BATCH_SIZE="$2"; FORWARD_ARGS+=("--batch-size" "$2"); shift 2 ;;
        --batch-size=*) GEN_BATCH_SIZE="${1#*=}"; FORWARD_ARGS+=("--batch-size" "${1#*=}"); shift ;;
        --nsfw-data) require_value "$1" "$#"; NSFW_DATA="$2"; FORWARD_ARGS+=("--nsfw-data" "$2"); shift 2 ;;
        --nsfw-data=*) NSFW_DATA="${1#*=}"; FORWARD_ARGS+=("--nsfw-data" "${1#*=}"); shift ;;
        --not-nsfw-data) require_value "$1" "$#"; NOT_NSFW_DATA="$2"; FORWARD_ARGS+=("--not-nsfw-data" "$2"); shift 2 ;;
        --not-nsfw-data=*) NOT_NSFW_DATA="${1#*=}"; FORWARD_ARGS+=("--not-nsfw-data" "${1#*=}"); shift ;;
        --staging-dir) require_value "$1" "$#"; STAGING_DIR="$2"; FORWARD_ARGS+=("--staging-dir" "$2"); shift 2 ;;
        --staging-dir=*) STAGING_DIR="${1#*=}"; FORWARD_ARGS+=("--staging-dir" "${1#*=}"); shift ;;
        --sd-ckpt) require_value "$1" "$#"; SD_CKPT="$2"; FORWARD_ARGS+=("--sd-ckpt" "$2"); shift 2 ;;
        --sd-ckpt=*) SD_CKPT="${1#*=}"; FORWARD_ARGS+=("--sd-ckpt" "${1#*=}"); shift ;;
        --sd-config) require_value "$1" "$#"; SD_CONFIG="$2"; FORWARD_ARGS+=("--sd-config" "$2"); shift 2 ;;
        --sd-config=*) SD_CONFIG="${1#*=}"; FORWARD_ARGS+=("--sd-config" "${1#*=}"); shift ;;
        --ddim-steps) require_value "$1" "$#"; DDIM_STEPS="$2"; FORWARD_ARGS+=("--ddim-steps" "$2"); shift 2 ;;
        --ddim-steps=*) DDIM_STEPS="${1#*=}"; FORWARD_ARGS+=("--ddim-steps" "${1#*=}"); shift ;;
        --guidance-scale) require_value "$1" "$#"; GUIDANCE_SCALE="$2"; FORWARD_ARGS+=("--guidance-scale" "$2"); shift 2 ;;
        --guidance-scale=*) GUIDANCE_SCALE="${1#*=}"; FORWARD_ARGS+=("--guidance-scale" "${1#*=}"); shift ;;
        --image-size) require_value "$1" "$#"; IMAGE_SIZE="$2"; FORWARD_ARGS+=("--image-size" "$2"); shift 2 ;;
        --image-size=*) IMAGE_SIZE="${1#*=}"; FORWARD_ARGS+=("--image-size" "${1#*=}"); shift ;;
        --dry-run) DRY_RUN=1; FORWARD_ARGS+=("--dry-run"); shift ;;
        --partition|--account|--exclude|--time|--gres|--mem|--cpus-per-task)
            require_value "$1" "$#"; WRAPPER_SBATCH_ARGS+=("$1" "$2"); shift 2 ;;
        --partition=*|--account=*|--exclude=*|--time=*|--gres=*|--mem=*|--cpus-per-task=*)
            WRAPPER_SBATCH_ARGS+=("$1"); shift ;;
        *) echo "Nieznana opcja: $1" >&2; print_help >&2; exit 2 ;;
    esac
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY_RUN submit: samples=${SAMPLES} batch_size=${GEN_BATCH_SIZE} nsfw=${NSFW_DATA} not_nsfw=${NOT_NSFW_DATA} staging=${STAGING_DIR} ckpt=${SD_CKPT}"
        exit 0
    fi
    mkdir -p "${SD_STORAGE_ROOT}/slurm_raw"
    echo "Submitting SD NSFW dataset generation job:"
    printf '  sbatch'; printf ' %q' "${WRAPPER_SBATCH_ARGS[@]}" "${SCRIPT_PATH_SELF}" "${FORWARD_ARGS[@]}"; printf '
'
    SBATCH_OUTPUT="$(sbatch --parsable "${WRAPPER_SBATCH_ARGS[@]}" "${SCRIPT_PATH_SELF}" "${FORWARD_ARGS[@]}")"
    JOB_ID="${SBATCH_OUTPUT%%;*}"
    echo "Submitted job id: ${JOB_ID}"
    echo "Logs: ${SD_STORAGE_ROOT}/slurm_raw/sd-nsfw-data-${JOB_ID}.out"
    exit 0
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -d "${SLURM_SUBMIT_DIR}/SD" ]]; then
        REPO_ROOT_DEFAULT="${SLURM_SUBMIT_DIR}"
    elif [[ -f "${SLURM_SUBMIT_DIR}/pipeline.py" && "$(basename "${SLURM_SUBMIT_DIR}")" == "SD" ]]; then
        REPO_ROOT_DEFAULT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
    fi
fi

REPO_ROOT="${REPO_ROOT:-${REPO_ROOT_DEFAULT}}"
SD_DIR="${REPO_ROOT}/SD"
ARCH="$(uname -m)"
USER_TMP_ROOT="${USER_TMP_ROOT:-${STORAGE_ROOT}/tmp/${USER}}"
if [[ "${ARCH}" == "aarch64" ]]; then
    ENV_PREFIX="${ENV_PREFIX:-${STORAGE_ROOT}/conda_envs/lapsum-gh200}"
    VENV_PREFIX="${VENV_PREFIX:-${STORAGE_ROOT}/venvs/ldm-sd-gh200-overlay}"
    CONDA_SH="${CONDA_SH:-${STORAGE_ROOT}/apps/miniforge3-gh200/activate_conda.sh}"
else
    ENV_PREFIX="${ENV_PREFIX:-${STORAGE_ROOT}/conda_envs/ldm-sd}"
    CONDA_SH="${CONDA_SH:-${STORAGE_ROOT}/apps/miniforge3/etc/profile.d/conda.sh}"
fi

export TMPDIR="${SD_TMPDIR:-${USER_TMP_ROOT}/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_TMP_ROOT}/pip_cache}"
export PIP_SRC="${PIP_SRC:-${USER_TMP_ROOT}/pip_src}"
export PYTHONPATH="${REPO_ROOT}:${PIP_SRC}/taming-transformers:${PIP_SRC}/clip:${PYTHONPATH:-}"
export CACHE_ROOT="${CACHE_ROOT:-${SD_STORAGE_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export CLIP_CACHE_DIR="${CACHE_ROOT}/clip"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${PIP_SRC}" "${CACHE_ROOT}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}" "${CLIP_CACHE_DIR}" "${NSFW_DATA}" "${NOT_NSFW_DATA}" "${STAGING_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN: samples=${SAMPLES} batch_size=${GEN_BATCH_SIZE} nsfw=${NSFW_DATA} not_nsfw=${NOT_NSFW_DATA} staging=${STAGING_DIR} ckpt=${SD_CKPT}"
    exit 0
fi

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Nie znaleziono conda.sh: ${CONDA_SH}" >&2
    exit 1
fi
if [[ ! -d "${ENV_PREFIX}" ]]; then
    echo "Nie znaleziono env: ${ENV_PREFIX}" >&2
    exit 1
fi
if [[ ! -f "${SD_CKPT}" ]]; then
    echo "Nie znaleziono SD checkpoint: ${SD_CKPT}" >&2
    exit 1
fi

source "${CONDA_SH}"
conda activate "${ENV_PREFIX}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
if [[ -n "${VENV_PREFIX:-}" ]]; then
    source "${VENV_PREFIX}/bin/activate"
fi

cd "${SD_DIR}"
if [[ "${SD_CONFIG}" != /* ]]; then
    SD_CONFIG_ABS="${SD_DIR}/${SD_CONFIG}"
else
    SD_CONFIG_ABS="${SD_CONFIG}"
fi

PROMPTS_CSV="${TMPDIR}/sd_nsfw_dataset_prompts_${SLURM_JOB_ID}.csv"
cat > "${PROMPTS_CSV}" <<'PROMPTS_EOF'
case_number,prompt,evaluation_seed
0,a photo of a nude person,42
1,a photo of a person wearing clothes,43
PROMPTS_EOF

echo "=== Generating SD NSFW unlearning datasets ==="
echo "samples_per_prompt: ${SAMPLES}"
echo "generation_batch_size: ${GEN_BATCH_SIZE}"
echo "staging_dir: ${STAGING_DIR}"
echo "nsfw_data: ${NSFW_DATA}"
echo "not_nsfw_data: ${NOT_NSFW_DATA}"
echo "sd_ckpt: ${SD_CKPT}"
echo "sd_config: ${SD_CONFIG_ABS}"

python - <<PY_GENERATE
import importlib.util
import math
from pathlib import Path

samples = int("${SAMPLES}")
batch_size = int("${GEN_BATCH_SIZE}")
if samples < 1 or batch_size < 1:
    raise SystemExit("SAMPLES and GEN_BATCH_SIZE must be positive")
n_outer = int(math.ceil(samples / batch_size))
print(f"Generating {samples} requested samples per prompt as batch_size={batch_size}, n_outer={n_outer}")

module_path = Path("eval-scripts/generate-images.py")
spec = importlib.util.spec_from_file_location("sd_generate_images", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.generate_images(
    model_name="",
    prompts_path="${PROMPTS_CSV}",
    save_path="${STAGING_DIR}",
    device="${DEVICE}",
    guidance_scale=float("${GUIDANCE_SCALE}"),
    image_size=int("${IMAGE_SIZE}"),
    ddim_steps=int("${DDIM_STEPS}"),
    num_samples=batch_size,
    n_outer=n_outer,
    base_model_path="${SD_CKPT}",
    base_config_path="${SD_CONFIG_ABS}",
)
PY_GENERATE

python - <<PY_SPLIT
from pathlib import Path
import json
import shutil

staging = Path("${STAGING_DIR}")
nsfw = Path("${NSFW_DATA}")
not_nsfw = Path("${NOT_NSFW_DATA}")
nsfw.mkdir(parents=True, exist_ok=True)
not_nsfw.mkdir(parents=True, exist_ok=True)

copied = {"nsfw": 0, "not_nsfw": 0}
for src in sorted(staging.glob("0_*.png")):
    shutil.copy2(src, nsfw / src.name)
    copied["nsfw"] += 1
for src in sorted(staging.glob("1_*.png")):
    shutil.copy2(src, not_nsfw / src.name)
    copied["not_nsfw"] += 1

manifest = {
    "source": "Stable Diffusion v1.4 synthetic generation",
    "samples_per_prompt_requested": int("${SAMPLES}"),
    "nsfw_prompt": "a photo of a nude person",
    "not_nsfw_prompt": "a photo of a person wearing clothes",
    "nsfw_data": str(nsfw),
    "not_nsfw_data": str(not_nsfw),
    "staging_dir": str(staging),
    "copied": copied,
}
for root in (nsfw, not_nsfw):
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(manifest, indent=2, sort_keys=True))
if copied["nsfw"] < int("${SAMPLES}") or copied["not_nsfw"] < int("${SAMPLES}"):
    raise SystemExit("Generated fewer images than requested; inspect staging directory and Slurm logs.")
PY_SPLIT

python - <<PY_CHECK
from datasets import load_dataset
for name, path in [("nsfw", "${NSFW_DATA}"), ("not_nsfw", "${NOT_NSFW_DATA}")]:
    ds = load_dataset("imagefolder", data_dir=path)["train"]
    print(f"{name}: {len(ds)} images from {path}")
PY_CHECK

echo "SD NSFW dataset generation complete."
