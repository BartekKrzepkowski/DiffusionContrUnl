#!/bin/bash
# SLURM launcher for Stable Diffusion class forgetting or NSFW unlearning
# with DASH warm-start + RL/ROFT.
#
# Examples:
#   bash SD/scripts/slurm_sd_dash_rl.sh --setting class --forget-class 0
#   bash SD/scripts/slurm_sd_dash_rl.sh --setting nsfw --method rl
#   bash SD/scripts/slurm_sd_dash_rl.sh --setting class --forget-class 0 --dash-target unet_xattn --dash-attention-head-wise
#   sbatch --array=0-9 SD/scripts/slurm_sd_dash_rl.sh --setting class
#   sbatch --partition=plgrid-gpu-gh200 --account=plgreprunlearn-gpu-gh200 SD/scripts/slurm_sd_dash_rl.sh --setting nsfw --method roft

#SBATCH --job-name=sd-dash-rl
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=/net/storage/pr3/plgrid/plggdnnp/plgkrzepk/SD/slurm_raw/%x-%j.out
#SBATCH --error=/net/storage/pr3/plgrid/plggdnnp/plgkrzepk/SD/slurm_raw/%x-%j.err

set -euo pipefail

SCRIPT_DIR_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH_SELF="${SCRIPT_DIR_SELF}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR_SELF}/../.." && pwd)"

print_help() {
    cat <<'EOF'
Uzycie:
  bash SD/scripts/slurm_sd_dash_rl.sh [opcje]
  sbatch [opcje sbatch] SD/scripts/slurm_sd_dash_rl.sh [opcje]

Opcje skryptu:
  --setting NAME          class albo nsfw. Domyslnie class. Alias: --nsfw.
  --forget-class N        Klasa Imagenette do zapomnienia. W setting=class bez tego array job uzywa SLURM_ARRAY_TASK_ID.
  --config PATH           Bazowy config pipeline. Domyslnie zalezy od setting:
                          class -> SD/configs/pipeline_class.yaml; nsfw -> SD/configs/pipeline_nsfw_dash.yaml.
  --method NAME           Metoda unlearningu. Bez tego uzywa wartosci z configu.
  --train-method NAME     Nadpisuje unlearn.train_method, np. unet_xattn/unet/unet_resnet.
  --epochs N              Nadpisuje unlearn.epochs.
  --lr VALUE              Nadpisuje unlearn.lr.
  --rl-loss-mode NAME     Nadpisuje unlearn.rl_loss_mode: output_matching lub denoise_pseudo.
  --alpha VALUE           Nadpisuje unlearn.alpha.
  --batch-size N          Nadpisuje unlearn.batch_size.
  --seed N                Nadpisuje pipeline.seed.
  --full-retain-per-epoch Nadpisuje unlearn.full_retain_per_epoch=true.
  --no-full-retain-per-epoch
                          Nadpisuje unlearn.full_retain_per_epoch=false.
  --dash                  Wlacza dash.warm_start dla runu DASH.
  --no-dash               Wylacza dash.warm_start dla porownawczego runu.
  --dash-target NAME      Nadpisuje dash.target.
  --dash-signal NAME      Nadpisuje dash.signal_mode.
  --dash-granularity NAME Nadpisuje dash.plasticity_granularity.
  --dash-attention-head-wise
                          Nadpisuje dash.attention_head_wise=true.
  --no-dash-attention-head-wise
                          Nadpisuje dash.attention_head_wise=false.
  --dash-num-aug N        Nadpisuje dash.num_aug.
  --dash-min-shrink V     Nadpisuje dash.min_shrink.
  --dash-svd-evr V        Nadpisuje dash.svd_truncate_evr.
  --dash-retain-batches N Nadpisuje dash.retain_batches.
  --dash-forget-batches N Nadpisuje dash.forget_batches.
  --intact-base-method N  Nadpisuje intact.base_method, np. rl albo ga.
  --intact-targets LIST   Nadpisuje intact.targets, np. "attn2.to_q,attn2.to_k,attn2.to_v".
  --intact-lambda V       Nadpisuje intact.lambda_interval.
  --intact-lower V        Nadpisuje intact.lower_percentile.
  --intact-upper V        Nadpisuje intact.upper_percentile.
  --intact-reduced-dim N  Nadpisuje intact.reduced_dim.
  --intact-infinity V     Nadpisuje intact.infinity_scale.
  --intact-actual-bounds  Nadpisuje intact.use_actual_bounds=true.
  --no-intact-actual-bounds
                          Nadpisuje intact.use_actual_bounds=false.
  --intact-normalize      Nadpisuje intact.normalize_protection=true.
  --no-intact-normalize   Nadpisuje intact.normalize_protection=false.
  --experiment-slug NAME  Nazwa katalogow wynikow. Domyslnie sd_dash_rl.
  --wandb                 Wymusza W&B niezaleznie od configu.
  --no-wandb              Wymusza wylaczenie W&B.
  -h, --help              Pokazuje pomoc.

Tryb wrappera z login node:
  bash SD/scripts/slurm_sd_dash_rl.sh --setting class --forget-class 0
  bash SD/scripts/slurm_sd_dash_rl.sh --setting class --array 0-9
  bash SD/scripts/slurm_sd_dash_rl.sh --setting nsfw --method rl

W trybie array i setting=class, jesli nie podasz --forget-class, zadanie N zapomina klase N.
Dla setting=nsfw --forget-class jest ignorowane.
EOF
}

require_value() {
    local option_name="$1"
    local argc="$2"
    if [[ "${argc}" -lt 2 ]]; then
        echo "Brak wartosci dla ${option_name}" >&2
        exit 2
    fi
}

SETTING="${SETTING:-class}"
FORGET_CLASS="${FORGET_CLASS:-}"
BASE_CONFIG="${BASE_CONFIG:-}"
UNLEARN_METHOD="${UNLEARN_METHOD:-}"
TRAIN_METHOD_OVERRIDE="${TRAIN_METHOD_OVERRIDE:-}"
EXPERIMENT_SLUG="${EXPERIMENT_SLUG:-sd_dash_rl}"
USE_WANDB="${USE_WANDB:-config}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"
LR_OVERRIDE="${LR_OVERRIDE:-}"
RL_LOSS_MODE_OVERRIDE="${RL_LOSS_MODE_OVERRIDE:-}"
ALPHA_OVERRIDE="${ALPHA_OVERRIDE:-}"
BATCH_SIZE_OVERRIDE="${BATCH_SIZE_OVERRIDE:-}"
SEED_OVERRIDE="${SEED_OVERRIDE:-}"
FULL_RETAIN_PER_EPOCH_OVERRIDE="${FULL_RETAIN_PER_EPOCH_OVERRIDE:-}"
DASH_WARM_START="${DASH_WARM_START:-1}"
DASH_TARGET_OVERRIDE="${DASH_TARGET_OVERRIDE:-}"
DASH_SIGNAL_OVERRIDE="${DASH_SIGNAL_OVERRIDE:-}"
DASH_GRANULARITY_OVERRIDE="${DASH_GRANULARITY_OVERRIDE:-}"
DASH_ATTENTION_HEAD_WISE_OVERRIDE="${DASH_ATTENTION_HEAD_WISE_OVERRIDE:-}"
DASH_NUM_AUG_OVERRIDE="${DASH_NUM_AUG_OVERRIDE:-}"
DASH_MIN_SHRINK_OVERRIDE="${DASH_MIN_SHRINK_OVERRIDE:-}"
DASH_SVD_EVR_OVERRIDE="${DASH_SVD_EVR_OVERRIDE:-}"
DASH_RETAIN_BATCHES_OVERRIDE="${DASH_RETAIN_BATCHES_OVERRIDE:-}"
DASH_FORGET_BATCHES_OVERRIDE="${DASH_FORGET_BATCHES_OVERRIDE:-}"
INTACT_BASE_METHOD_OVERRIDE="${INTACT_BASE_METHOD_OVERRIDE:-}"
INTACT_TARGETS_OVERRIDE="${INTACT_TARGETS_OVERRIDE:-}"
INTACT_LAMBDA_OVERRIDE="${INTACT_LAMBDA_OVERRIDE:-}"
INTACT_LOWER_OVERRIDE="${INTACT_LOWER_OVERRIDE:-}"
INTACT_UPPER_OVERRIDE="${INTACT_UPPER_OVERRIDE:-}"
INTACT_REDUCED_DIM_OVERRIDE="${INTACT_REDUCED_DIM_OVERRIDE:-}"
INTACT_INFINITY_OVERRIDE="${INTACT_INFINITY_OVERRIDE:-}"
INTACT_ACTUAL_BOUNDS_OVERRIDE="${INTACT_ACTUAL_BOUNDS_OVERRIDE:-}"
INTACT_NORMALIZE_OVERRIDE="${INTACT_NORMALIZE_OVERRIDE:-}"

WRAPPER_SBATCH_ARGS=()
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            exit 0
            ;;
        --array)
            if [[ $# -lt 2 ]]; then
                echo "Brak wartosci dla --array" >&2
                exit 2
            fi
            WRAPPER_SBATCH_ARGS+=("--array=$2")
            shift 2
            ;;
        --array=*)
            WRAPPER_SBATCH_ARGS+=("$1")
            shift
            ;;
        --partition|--account|--exclude|--time|--gres|--mem|--cpus-per-task)
            if [[ $# -lt 2 ]]; then
                echo "Brak wartosci dla $1" >&2
                exit 2
            fi
            WRAPPER_SBATCH_ARGS+=("$1" "$2")
            shift 2
            ;;
        --partition=*|--account=*|--exclude=*|--time=*|--gres=*|--mem=*|--cpus-per-task=*)
            WRAPPER_SBATCH_ARGS+=("$1")
            shift
            ;;
        --setting)
            require_value "$1" "$#"
            SETTING="$2"
            FORWARD_ARGS+=("--setting" "$2")
            shift 2
            ;;
        --setting=*)
            SETTING="${1#*=}"
            FORWARD_ARGS+=("--setting" "${1#*=}")
            shift
            ;;
        --nsfw)
            SETTING="nsfw"
            FORWARD_ARGS+=("--setting" "nsfw")
            shift
            ;;
        --class-setting|--class-forgetting)
            SETTING="class"
            FORWARD_ARGS+=("--setting" "class")
            shift
            ;;
        --forget-class|--class)
            if [[ $# -lt 2 ]]; then
                echo "Brak wartosci dla --forget-class" >&2
                exit 2
            fi
            FORGET_CLASS="$2"
            FORWARD_ARGS+=("--forget-class" "$2")
            shift 2
            ;;
        --forget-class=*|--class=*)
            FORGET_CLASS="${1#*=}"
            FORWARD_ARGS+=("--forget-class" "${1#*=}")
            shift
            ;;
        --config)
            if [[ $# -lt 2 ]]; then
                echo "Brak wartosci dla --config" >&2
                exit 2
            fi
            BASE_CONFIG="$2"
            FORWARD_ARGS+=("--config" "$2")
            shift 2
            ;;
        --config=*)
            BASE_CONFIG="${1#*=}"
            FORWARD_ARGS+=("--config" "${1#*=}")
            shift
            ;;
        --method)
            require_value "$1" "$#"
            UNLEARN_METHOD="$2"
            FORWARD_ARGS+=("--method" "$2")
            shift 2
            ;;
        --method=*)
            UNLEARN_METHOD="${1#*=}"
            FORWARD_ARGS+=("--method" "${1#*=}")
            shift
            ;;
        --train-method)
            require_value "$1" "$#"
            TRAIN_METHOD_OVERRIDE="$2"
            FORWARD_ARGS+=("--train-method" "$2")
            shift 2
            ;;
        --train-method=*)
            TRAIN_METHOD_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--train-method" "${1#*=}")
            shift
            ;;
        --epochs)
            require_value "$1" "$#"
            EPOCHS_OVERRIDE="$2"
            FORWARD_ARGS+=("--epochs" "$2")
            shift 2
            ;;
        --epochs=*)
            EPOCHS_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--epochs" "${1#*=}")
            shift
            ;;
        --lr)
            require_value "$1" "$#"
            LR_OVERRIDE="$2"
            FORWARD_ARGS+=("--lr" "$2")
            shift 2
            ;;
        --lr=*)
            LR_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--lr" "${1#*=}")
            shift
            ;;
        --rl-loss-mode)
            require_value "$1" "$#"
            RL_LOSS_MODE_OVERRIDE="$2"
            FORWARD_ARGS+=("--rl-loss-mode" "$2")
            shift 2
            ;;
        --rl-loss-mode=*)
            RL_LOSS_MODE_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--rl-loss-mode" "${1#*=}")
            shift
            ;;
        --alpha)
            require_value "$1" "$#"
            ALPHA_OVERRIDE="$2"
            FORWARD_ARGS+=("--alpha" "$2")
            shift 2
            ;;
        --alpha=*)
            ALPHA_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--alpha" "${1#*=}")
            shift
            ;;
        --batch-size)
            require_value "$1" "$#"
            BATCH_SIZE_OVERRIDE="$2"
            FORWARD_ARGS+=("--batch-size" "$2")
            shift 2
            ;;
        --batch-size=*)
            BATCH_SIZE_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--batch-size" "${1#*=}")
            shift
            ;;
        --seed)
            require_value "$1" "$#"
            SEED_OVERRIDE="$2"
            FORWARD_ARGS+=("--seed" "$2")
            shift 2
            ;;
        --seed=*)
            SEED_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--seed" "${1#*=}")
            shift
            ;;
        --full-retain-per-epoch)
            FULL_RETAIN_PER_EPOCH_OVERRIDE="1"
            FORWARD_ARGS+=("--full-retain-per-epoch")
            shift
            ;;
        --no-full-retain-per-epoch)
            FULL_RETAIN_PER_EPOCH_OVERRIDE="0"
            FORWARD_ARGS+=("--no-full-retain-per-epoch")
            shift
            ;;
        --dash)
            DASH_WARM_START="1"
            FORWARD_ARGS+=("--dash")
            shift
            ;;
        --no-dash)
            DASH_WARM_START="0"
            FORWARD_ARGS+=("--no-dash")
            shift
            ;;
        --dash-target)
            require_value "$1" "$#"
            DASH_TARGET_OVERRIDE="$2"
            FORWARD_ARGS+=("--dash-target" "$2")
            shift 2
            ;;
        --dash-target=*)
            DASH_TARGET_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--dash-target" "${1#*=}")
            shift
            ;;
        --dash-signal)
            require_value "$1" "$#"
            DASH_SIGNAL_OVERRIDE="$2"
            FORWARD_ARGS+=("--dash-signal" "$2")
            shift 2
            ;;
        --dash-signal=*)
            DASH_SIGNAL_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--dash-signal" "${1#*=}")
            shift
            ;;
        --dash-granularity)
            require_value "$1" "$#"
            DASH_GRANULARITY_OVERRIDE="$2"
            FORWARD_ARGS+=("--dash-granularity" "$2")
            shift 2
            ;;
        --dash-granularity=*)
            DASH_GRANULARITY_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--dash-granularity" "${1#*=}")
            shift
            ;;
        --dash-attention-head-wise)
            DASH_ATTENTION_HEAD_WISE_OVERRIDE="1"
            FORWARD_ARGS+=("--dash-attention-head-wise")
            shift
            ;;
        --no-dash-attention-head-wise)
            DASH_ATTENTION_HEAD_WISE_OVERRIDE="0"
            FORWARD_ARGS+=("--no-dash-attention-head-wise")
            shift
            ;;
        --dash-num-aug)
            require_value "$1" "$#"
            DASH_NUM_AUG_OVERRIDE="$2"
            FORWARD_ARGS+=("--dash-num-aug" "$2")
            shift 2
            ;;
        --dash-num-aug=*)
            DASH_NUM_AUG_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--dash-num-aug" "${1#*=}")
            shift
            ;;
        --dash-min-shrink)
            require_value "$1" "$#"
            DASH_MIN_SHRINK_OVERRIDE="$2"
            FORWARD_ARGS+=("--dash-min-shrink" "$2")
            shift 2
            ;;
        --dash-min-shrink=*)
            DASH_MIN_SHRINK_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--dash-min-shrink" "${1#*=}")
            shift
            ;;
        --dash-svd-evr)
            require_value "$1" "$#"
            DASH_SVD_EVR_OVERRIDE="$2"
            FORWARD_ARGS+=("--dash-svd-evr" "$2")
            shift 2
            ;;
        --dash-svd-evr=*)
            DASH_SVD_EVR_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--dash-svd-evr" "${1#*=}")
            shift
            ;;
        --dash-retain-batches)
            require_value "$1" "$#"
            DASH_RETAIN_BATCHES_OVERRIDE="$2"
            FORWARD_ARGS+=("--dash-retain-batches" "$2")
            shift 2
            ;;
        --dash-retain-batches=*)
            DASH_RETAIN_BATCHES_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--dash-retain-batches" "${1#*=}")
            shift
            ;;
        --dash-forget-batches)
            require_value "$1" "$#"
            DASH_FORGET_BATCHES_OVERRIDE="$2"
            FORWARD_ARGS+=("--dash-forget-batches" "$2")
            shift 2
            ;;
        --dash-forget-batches=*)
            DASH_FORGET_BATCHES_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--dash-forget-batches" "${1#*=}")
            shift
            ;;
        --intact-base-method)
            require_value "$1" "$#"
            INTACT_BASE_METHOD_OVERRIDE="$2"
            FORWARD_ARGS+=("--intact-base-method" "$2")
            shift 2
            ;;
        --intact-base-method=*)
            INTACT_BASE_METHOD_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--intact-base-method" "${1#*=}")
            shift
            ;;
        --intact-targets)
            require_value "$1" "$#"
            INTACT_TARGETS_OVERRIDE="$2"
            FORWARD_ARGS+=("--intact-targets" "$2")
            shift 2
            ;;
        --intact-targets=*)
            INTACT_TARGETS_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--intact-targets" "${1#*=}")
            shift
            ;;
        --intact-lambda)
            require_value "$1" "$#"
            INTACT_LAMBDA_OVERRIDE="$2"
            FORWARD_ARGS+=("--intact-lambda" "$2")
            shift 2
            ;;
        --intact-lambda=*)
            INTACT_LAMBDA_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--intact-lambda" "${1#*=}")
            shift
            ;;
        --intact-lower)
            require_value "$1" "$#"
            INTACT_LOWER_OVERRIDE="$2"
            FORWARD_ARGS+=("--intact-lower" "$2")
            shift 2
            ;;
        --intact-lower=*)
            INTACT_LOWER_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--intact-lower" "${1#*=}")
            shift
            ;;
        --intact-upper)
            require_value "$1" "$#"
            INTACT_UPPER_OVERRIDE="$2"
            FORWARD_ARGS+=("--intact-upper" "$2")
            shift 2
            ;;
        --intact-upper=*)
            INTACT_UPPER_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--intact-upper" "${1#*=}")
            shift
            ;;
        --intact-reduced-dim)
            require_value "$1" "$#"
            INTACT_REDUCED_DIM_OVERRIDE="$2"
            FORWARD_ARGS+=("--intact-reduced-dim" "$2")
            shift 2
            ;;
        --intact-reduced-dim=*)
            INTACT_REDUCED_DIM_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--intact-reduced-dim" "${1#*=}")
            shift
            ;;
        --intact-infinity)
            require_value "$1" "$#"
            INTACT_INFINITY_OVERRIDE="$2"
            FORWARD_ARGS+=("--intact-infinity" "$2")
            shift 2
            ;;
        --intact-infinity=*)
            INTACT_INFINITY_OVERRIDE="${1#*=}"
            FORWARD_ARGS+=("--intact-infinity" "${1#*=}")
            shift
            ;;
        --intact-actual-bounds)
            INTACT_ACTUAL_BOUNDS_OVERRIDE="1"
            FORWARD_ARGS+=("--intact-actual-bounds")
            shift
            ;;
        --no-intact-actual-bounds)
            INTACT_ACTUAL_BOUNDS_OVERRIDE="0"
            FORWARD_ARGS+=("--no-intact-actual-bounds")
            shift
            ;;
        --intact-normalize)
            INTACT_NORMALIZE_OVERRIDE="1"
            FORWARD_ARGS+=("--intact-normalize")
            shift
            ;;
        --no-intact-normalize)
            INTACT_NORMALIZE_OVERRIDE="0"
            FORWARD_ARGS+=("--no-intact-normalize")
            shift
            ;;
        --experiment-slug)
            require_value "$1" "$#"
            EXPERIMENT_SLUG="$2"
            FORWARD_ARGS+=("--experiment-slug" "$2")
            shift 2
            ;;
        --experiment-slug=*)
            EXPERIMENT_SLUG="${1#*=}"
            FORWARD_ARGS+=("--experiment-slug" "${1#*=}")
            shift
            ;;
        --wandb)
            USE_WANDB=1
            FORWARD_ARGS+=("--wandb")
            shift
            ;;
        --no-wandb)
            USE_WANDB=0
            FORWARD_ARGS+=("--no-wandb")
            shift
            ;;
        *)
            echo "Nieznana opcja: $1" >&2
            print_help >&2
            exit 2
            ;;
    esac
done

SETTING="$(echo "${SETTING}" | tr '[:upper:]' '[:lower:]')"
case "${SETTING}" in
    class|sd|class_forgetting|class-forgetting)
        SETTING="class"
        ;;
    nsfw|sd_nsfw|sd-nsfw)
        SETTING="nsfw"
        ;;
    *)
        echo "Nieznany --setting: ${SETTING} (uzyj class albo nsfw)" >&2
        exit 2
        ;;
esac

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    STORAGE_ROOT_SUBMIT="${STORAGE_ROOT:-/net/storage/pr3/plgrid/plggdnnp}"
    SD_STORAGE_ROOT_SUBMIT="${SD_STORAGE_ROOT:-${STORAGE_ROOT_SUBMIT}/${USER}/SD}"
    mkdir -p "${SD_STORAGE_ROOT_SUBMIT}/slurm_raw"
    LOG_DATE_SUBMIT="$(date '+%Y-%m-%d')"
    echo "Submitting SD DASH+RL job:"
    echo "  sbatch ${WRAPPER_SBATCH_ARGS[*]} ${SCRIPT_PATH_SELF} ${FORWARD_ARGS[*]}"
    SBATCH_OUTPUT="$(sbatch --parsable "${WRAPPER_SBATCH_ARGS[@]}" "${SCRIPT_PATH_SELF}" "${FORWARD_ARGS[@]}")"
    JOB_ID="${SBATCH_OUTPUT%%;*}"
    echo "Submitted job id: ${JOB_ID}"
    echo "Logs: ${REPO_ROOT_DEFAULT}/SD/logs/slurm_logs/${LOG_DATE_SUBMIT}/${EXPERIMENT_SLUG}/${JOB_ID}/"
    exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -d "${SLURM_SUBMIT_DIR}/SD" ]]; then
        REPO_ROOT_DEFAULT="${SLURM_SUBMIT_DIR}"
    elif [[ -f "${SLURM_SUBMIT_DIR}/pipeline.py" && "$(basename "${SLURM_SUBMIT_DIR}")" == "SD" ]]; then
        REPO_ROOT_DEFAULT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
    fi
fi

REPO_ROOT="${REPO_ROOT:-${REPO_ROOT_DEFAULT}}"
SD_DIR="${REPO_ROOT}/SD"
if [[ -z "${BASE_CONFIG}" ]]; then
    if [[ "${SETTING}" == "nsfw" ]]; then
        BASE_CONFIG="${SD_DIR}/configs/pipeline_nsfw_dash.yaml"
    else
        BASE_CONFIG="${SD_DIR}/configs/pipeline_class.yaml"
    fi
fi
STORAGE_ROOT="${STORAGE_ROOT:-/net/storage/pr3/plgrid/plggdnnp}"
USER_TMP_ROOT="${USER_TMP_ROOT:-${STORAGE_ROOT}/tmp/${USER}}"
SD_STORAGE_ROOT="${SD_STORAGE_ROOT:-${STORAGE_ROOT}/${USER}/SD}"
export SD_STORAGE_ROOT
ARCH="$(uname -m)"

if [[ -z "${ENV_PREFIX:-}" ]]; then
    if [[ "${ARCH}" == "aarch64" ]]; then
        ENV_PREFIX="${STORAGE_ROOT}/conda_envs/lapsum-gh200"
    else
        ENV_PREFIX="${STORAGE_ROOT}/conda_envs/ldm-sd"
    fi
fi

if [[ -z "${VENV_PREFIX:-}" && "${ARCH}" == "aarch64" ]]; then
    VENV_PREFIX="${STORAGE_ROOT}/venvs/ldm-sd-gh200-overlay"
fi

if [[ -z "${CONDA_SH:-}" ]]; then
    if [[ "${ARCH}" == "aarch64" ]]; then
        CONDA_SH="${STORAGE_ROOT}/apps/miniforge3-gh200/activate_conda.sh"
    else
        CONDA_SH="${STORAGE_ROOT}/apps/miniforge3/etc/profile.d/conda.sh"
    fi
fi

LOG_DATE="$(date '+%Y-%m-%d')"
JOB_TAG="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
LOG_ROOT="${SD_DIR}/logs/slurm_logs/${LOG_DATE}/${EXPERIMENT_SLUG}/${SLURM_JOB_ID}"
mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/sd_dash_rl-${JOB_TAG}.out") 2> >(tee -a "${LOG_ROOT}/sd_dash_rl-${JOB_TAG}.err" >&2)

echo "=== SD DASH+RL SLURM job bootstrap ==="
echo "date: $(date --iso-8601=seconds)"
echo "host: $(hostname)"
echo "arch: ${ARCH}"
echo "job: ${SLURM_JOB_ID}"
echo "array_task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "repo: ${REPO_ROOT}"
echo "conda_sh: ${CONDA_SH}"
echo "conda_env: ${ENV_PREFIX}"
echo "venv_overlay: ${VENV_PREFIX:-none}"

if [[ "${SETTING}" == "class" && -z "${FORGET_CLASS}" ]]; then
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        FORGET_CLASS="${SLURM_ARRAY_TASK_ID}"
    else
        echo "Podaj --forget-class N albo uruchom jako array job dla --setting class." >&2
        exit 2
    fi
fi
if [[ "${SETTING}" == "nsfw" ]]; then
    FORGET_CLASS="none"
fi

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Nie znaleziono conda.sh: ${CONDA_SH}" >&2
    exit 1
fi

if [[ ! -d "${ENV_PREFIX}" ]]; then
    echo "Nie znaleziono env: ${ENV_PREFIX}" >&2
    if [[ "${ARCH}" == "aarch64" ]]; then
        echo "Na GH200 domyslnie uzywam lapsum-gh200 jako bazy CUDA." >&2
    fi
    exit 1
fi

source "${CONDA_SH}"
conda activate "${ENV_PREFIX}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

if [[ -n "${VENV_PREFIX:-}" ]]; then
    if [[ ! -f "${VENV_PREFIX}/bin/activate" ]]; then
        echo "Nie znaleziono overlay venv: ${VENV_PREFIX}" >&2
        echo "Utworz go na GH200 komenda: python -m venv --system-site-packages ${VENV_PREFIX}" >&2
        exit 1
    fi
    source "${VENV_PREFIX}/bin/activate"
fi

if [[ "${SETTING}" == "nsfw" ]]; then
    RUN_SLUG="nsfw/job_${JOB_TAG}"
else
    RUN_SLUG="class_${FORGET_CLASS}/job_${JOB_TAG}"
fi

echo "=== SD DASH+RL SLURM job runtime ==="
echo "setting: ${SETTING}"
echo "forget_class: ${FORGET_CLASS}"
echo "ld_library_path_head: ${ENV_PREFIX}/lib"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export TMPDIR="${SD_TMPDIR:-${USER_TMP_ROOT}/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${USER_TMP_ROOT}/pip_cache}"
export PIP_SRC="${PIP_SRC:-${USER_TMP_ROOT}/pip_src}"
export PYTHONPATH="${REPO_ROOT}:${PIP_SRC}/taming-transformers:${PIP_SRC}/clip:${PYTHONPATH:-}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${USER_TMP_ROOT}/conda_pkgs}"
export CACHE_ROOT="${CACHE_ROOT:-${SD_STORAGE_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-${CACHE_ROOT}/clip}"
export WANDB_DIR="${WANDB_DIR:-${SD_STORAGE_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${CACHE_ROOT}/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${CACHE_ROOT}/wandb_data}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${CACHE_ROOT}/wandb_artifacts}"

mkdir -p \
    "${TMPDIR}" \
    "${SD_STORAGE_ROOT}" \
    "${PIP_CACHE_DIR}" \
    "${PIP_SRC}" \
    "${CONDA_PKGS_DIRS}" \
    "${CACHE_ROOT}" \
    "${HF_HOME}" \
    "${HF_DATASETS_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${TORCH_HOME}" \
    "${CLIP_CACHE_DIR}" \
    "${WANDB_DIR}" \
    "${WANDB_CACHE_DIR}" \
    "${WANDB_DATA_DIR}" \
    "${WANDB_ARTIFACT_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

cd "${SD_DIR}"

TMP_CONFIG="${TMPDIR}/sd_dash_rl_${JOB_TAG}.yaml"
BASE_CONFIG_ABS="${BASE_CONFIG}"
if [[ "${BASE_CONFIG_ABS}" != /* ]]; then
    if [[ -f "${SD_DIR}/${BASE_CONFIG_ABS}" ]]; then
        BASE_CONFIG_ABS="${SD_DIR}/${BASE_CONFIG_ABS}"
    elif [[ -f "${REPO_ROOT}/${BASE_CONFIG_ABS}" ]]; then
        BASE_CONFIG_ABS="${REPO_ROOT}/${BASE_CONFIG_ABS}"
    else
        echo "Nie znaleziono configu: ${BASE_CONFIG}" >&2
        echo "Sprawdzono: ${SD_DIR}/${BASE_CONFIG} oraz ${REPO_ROOT}/${BASE_CONFIG}" >&2
        exit 1
    fi
elif [[ ! -f "${BASE_CONFIG_ABS}" ]]; then
    echo "Nie znaleziono configu: ${BASE_CONFIG_ABS}" >&2
    exit 1
fi

export SETTING
export UNLEARN_METHOD
export TRAIN_METHOD_OVERRIDE
export EPOCHS_OVERRIDE
export LR_OVERRIDE
export RL_LOSS_MODE_OVERRIDE
export ALPHA_OVERRIDE
export BATCH_SIZE_OVERRIDE
export SEED_OVERRIDE
export FULL_RETAIN_PER_EPOCH_OVERRIDE
export DASH_WARM_START
export DASH_TARGET_OVERRIDE
export DASH_SIGNAL_OVERRIDE
export DASH_GRANULARITY_OVERRIDE
export DASH_ATTENTION_HEAD_WISE_OVERRIDE
export DASH_NUM_AUG_OVERRIDE
export DASH_MIN_SHRINK_OVERRIDE
export DASH_SVD_EVR_OVERRIDE
export DASH_RETAIN_BATCHES_OVERRIDE
export DASH_FORGET_BATCHES_OVERRIDE
export INTACT_BASE_METHOD_OVERRIDE
export INTACT_TARGETS_OVERRIDE
export INTACT_LAMBDA_OVERRIDE
export INTACT_LOWER_OVERRIDE
export INTACT_UPPER_OVERRIDE
export INTACT_REDUCED_DIM_OVERRIDE
export INTACT_INFINITY_OVERRIDE
export INTACT_ACTUAL_BOUNDS_OVERRIDE
export INTACT_NORMALIZE_OVERRIDE

python - "${BASE_CONFIG_ABS}" "${TMP_CONFIG}" "${FORGET_CLASS}" "${EXPERIMENT_SLUG}" "${RUN_SLUG}" "${SETTING}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

base_config, out_config, forget_class, experiment_slug, run_slug, setting = sys.argv[1:]

with open(base_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("unlearn", {})
cfg.setdefault("intact", {})
cfg.setdefault("dash", {})
cfg.setdefault("paths", {})
cfg.setdefault("pipeline", {})
cfg.setdefault("wandb", {})

setting = str(setting).lower()
if setting == "nsfw":
    cfg["pipeline"]["setting"] = "sd_nsfw"
    cfg["unlearn"].pop("class_to_forget", None)
    cfg["unlearn"].pop("forget_classes", None)
    cfg["unlearn"].pop("forget_concepts", None)
else:
    cfg["pipeline"]["setting"] = "sd"
    cfg["unlearn"]["class_to_forget"] = int(forget_class) if str(forget_class).isdigit() else forget_class
    cfg["unlearn"]["forget_classes"] = None
    cfg["unlearn"]["forget_concepts"] = None
method_override = os.environ.get("UNLEARN_METHOD", "")
if method_override:
    cfg["unlearn"]["method"] = method_override

optional_updates = {
    ("pipeline", "seed"): ("SEED_OVERRIDE", int),
    ("unlearn", "train_method"): ("TRAIN_METHOD_OVERRIDE", str),
    ("unlearn", "epochs"): ("EPOCHS_OVERRIDE", int),
    ("unlearn", "lr"): ("LR_OVERRIDE", float),
    ("unlearn", "rl_loss_mode"): ("RL_LOSS_MODE_OVERRIDE", str),
    ("unlearn", "alpha"): ("ALPHA_OVERRIDE", float),
    ("unlearn", "batch_size"): ("BATCH_SIZE_OVERRIDE", int),
    ("unlearn", "full_retain_per_epoch"): ("FULL_RETAIN_PER_EPOCH_OVERRIDE", lambda value: str(value).lower() in {"1", "true", "yes", "on"}),
    ("dash", "target"): ("DASH_TARGET_OVERRIDE", str),
    ("dash", "signal_mode"): ("DASH_SIGNAL_OVERRIDE", str),
    ("dash", "plasticity_granularity"): ("DASH_GRANULARITY_OVERRIDE", str),
    ("dash", "attention_head_wise"): ("DASH_ATTENTION_HEAD_WISE_OVERRIDE", lambda value: str(value).lower() in {"1", "true", "yes", "on"}),
    ("dash", "num_aug"): ("DASH_NUM_AUG_OVERRIDE", int),
    ("dash", "min_shrink"): ("DASH_MIN_SHRINK_OVERRIDE", float),
    ("dash", "svd_truncate_evr"): ("DASH_SVD_EVR_OVERRIDE", float),
    ("dash", "retain_batches"): ("DASH_RETAIN_BATCHES_OVERRIDE", int),
    ("dash", "forget_batches"): ("DASH_FORGET_BATCHES_OVERRIDE", int),
    ("intact", "base_method"): ("INTACT_BASE_METHOD_OVERRIDE", str),
    ("intact", "lambda_interval"): ("INTACT_LAMBDA_OVERRIDE", float),
    ("intact", "lower_percentile"): ("INTACT_LOWER_OVERRIDE", float),
    ("intact", "upper_percentile"): ("INTACT_UPPER_OVERRIDE", float),
    ("intact", "reduced_dim"): ("INTACT_REDUCED_DIM_OVERRIDE", int),
    ("intact", "infinity_scale"): ("INTACT_INFINITY_OVERRIDE", float),
    ("intact", "use_actual_bounds"): ("INTACT_ACTUAL_BOUNDS_OVERRIDE", lambda value: str(value).lower() in {"1", "true", "yes", "on"}),
    ("intact", "normalize_protection"): ("INTACT_NORMALIZE_OVERRIDE", lambda value: str(value).lower() in {"1", "true", "yes", "on"}),
}
for (section, key), (env_name, caster) in optional_updates.items():
    raw = os.environ.get(env_name, "")
    if raw != "":
        cfg[section][key] = caster(raw)

targets_override = os.environ.get("INTACT_TARGETS_OVERRIDE", "")
if targets_override:
    cfg["intact"]["targets"] = [
        target
        for chunk in targets_override.split(",")
        for target in chunk.split()
        if target
    ]

cfg["dash"]["warm_start"] = str(os.environ.get("DASH_WARM_START", "1")).lower() in {"1", "true", "yes", "on"}
sd_storage_root = os.environ.get("SD_STORAGE_ROOT", f"/net/storage/pr3/plgrid/plggdnnp/{os.environ['USER']}/SD")
cfg["paths"]["output_dir"] = f"{sd_storage_root}/evaluation/{experiment_slug}/{run_slug}"
cfg["paths"]["model_save_dir"] = f"{sd_storage_root}/models/{experiment_slug}/{run_slug}"
cfg["paths"]["logs_dir"] = f"{sd_storage_root}/logs/{experiment_slug}/{run_slug}"

base_tags = cfg["wandb"].get("tags") or []
if isinstance(base_tags, str):
    base_tags = [base_tags]
def tag_value(value):
    return str(value).replace(" ", "_").replace("/", "_")

experiment_slug_text = str(experiment_slug)
experiment_name = str(experiment_slug).split("/", 1)[0]
experiment_tag = tag_value(experiment_name)
experiment_path_tag = tag_value(experiment_slug_text)
method_tag = tag_value(cfg["unlearn"].get("method", "unknown"))
dash_enabled_tag = "on" if cfg["dash"].get("warm_start") else "off"
headwise_tag = "on" if cfg["dash"].get("attention_head_wise") else "off"
sweep_tags = [
    "sweep_slurm_sd_dash_rl",
    f"setting_{setting}",
    f"method_{method_tag}",
    f"dash_{dash_enabled_tag}",
    f"dash_target_{tag_value(cfg['dash'].get('target', 'none'))}",
    f"dash_signal_{tag_value(cfg['dash'].get('signal_mode', 'none'))}",
    f"dash_granularity_{tag_value(cfg['dash'].get('plasticity_granularity', 'none'))}",
    f"attention_headwise_{headwise_tag}",
    f"lr_{tag_value(cfg['unlearn'].get('lr', 'none'))}",
    f"rl_loss_{tag_value(cfg['unlearn'].get('rl_loss_mode', 'none'))}",
    f"min_shrink_{tag_value(cfg['dash'].get('min_shrink', 'none'))}",
    experiment_tag,
    f"experiment_{tag_value(experiment_name)}",
    f"experiment_path_{experiment_path_tag}",
    f"run_{tag_value(run_slug)}",
]
cfg["wandb"]["tags"] = list(dict.fromkeys(str(tag) for tag in [*base_tags, *sweep_tags] if tag))

Path(out_config).parent.mkdir(parents=True, exist_ok=True)
with open(out_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote job config: {out_config}")
print(
    f"setting={cfg['pipeline']['setting']} "
    f"method={cfg['unlearn']['method']} "
    f"class_to_forget={cfg['unlearn'].get('class_to_forget', 'none')} "
    f"rl_loss_mode={cfg['unlearn'].get('rl_loss_mode')} "
    f"dash={cfg['dash'].get('warm_start')} "
    f"attention_head_wise={cfg['dash'].get('attention_head_wise')} "
    f"full_retain_per_epoch={cfg['unlearn'].get('full_retain_per_epoch')}"
)
print(f"model_save_dir={cfg['paths']['model_save_dir']}")
print(f"output_dir={cfg['paths']['output_dir']}")
PY

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY

PIPELINE_ARGS=(pipeline.py --config "${TMP_CONFIG}")
if [[ "${USE_WANDB}" == "0" ]]; then
    PIPELINE_ARGS+=(--no-wandb)
fi

echo "Running: python ${PIPELINE_ARGS[*]}"
python "${PIPELINE_ARGS[@]}"
