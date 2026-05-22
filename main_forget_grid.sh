#!/bin/bash
# Grid launcher for main_forget using SLURM array + Python grid config.
#
# Purpose:
#   - Pull a task-specific command from a grid config script (e.g. rga_grid_config.py)
#   - Execute that command inside a SLURM array job
#
# Inputs:
#   - SLURM_ARRAY_TASK_ID: selects the sweep configuration
#   - Optional env overrides:
#       GRID_SCRIPT (default: rga_grid_config.py)
#       FORGET_CLASS (optional override, no default)
#       FORGET_CLASSES (optional, retrain grid only)
#       SEEDS (default: 2)
#
# Example:
#   GRID_CONFIG=SEMU/Classification/rga_grid_config.py sbatch --array=0-9 SEMU/Classification/main_forget_grid.sh
#
# Notes:
#   - This script expects the grid config to print a line with the full command.
#   - It does not do any validation; check the printed command before running large arrays.


#SBATCH --job-name=retrain-grid
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgreprunlearn-gpu-gh200
#SBATCH --time=08:30:00
#SBATCH --array=0-5

#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1


## Logi - osobne dla każdego array task, z datą
## UWAGA: SLURM nie rozwiązuje %Y-%m-%d automatycznie, więc używamy zmiennych
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# --- tryb bezpieczny ---
set -e -o pipefail

SCRIPT_DIR_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH_SELF="${SCRIPT_DIR_SELF}/$(basename "${BASH_SOURCE[0]}")"
DEFAULT_GRID_SCRIPT="${GRID_SCRIPT:-}"
DEFAULT_EXPERIMENT_SLUG="${EXPERIMENT_SLUG:-}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

normalize_grid_script_name() {
    local grid_script="$1"
    if [[ "${grid_script}" != *.py ]]; then
        grid_script="${grid_script}.py"
    fi
    echo "${grid_script}"
}

derive_experiment_slug_from_grid_script() {
    local grid_script="$1"
    local script_basename
    script_basename="$(basename "${grid_script}")"
    script_basename="${script_basename%.py}"
    if [[ "${script_basename}" == *_config ]]; then
        echo "${script_basename%_config}_grid"
    else
        echo "${script_basename}_grid"
    fi
}

extract_running_command_from_grid_output() {
    local grid_output="$1"
    printf '%s\n' "${grid_output}" | sed -n '/^Running command:/,/^WANDB_TAGS=/p' | sed '1d;$d'
}

extract_grid_marker_value() {
    local grid_output="$1"
    local marker_name="$2"
    printf '%s\n' "${grid_output}" | grep "^${marker_name}=" | cut -d= -f2- || true
}

require_grid_output_value() {
    local field_name="$1"
    local field_value="$2"
    local grid_script="$3"
    if [[ -n "${field_value}" ]]; then
        return 0
    fi
    echo "Missing ${field_name} in output from ${grid_script}" >&2
    return 1
}

strip_leading_srun() {
    local cmd="$1"
    local tokens=()
    local i=1

    read -r -a tokens <<< "${cmd}"
    if [[ "${tokens[0]:-}" != "srun" ]]; then
        printf '%s\n' "${cmd}"
        return 0
    fi

    while [[ ${i} -lt ${#tokens[@]} ]]; do
        case "${tokens[$i]}" in
            --)
                i=$((i + 1))
                break
                ;;
            --ntasks=*|--nodes=*|--cpus-per-task=*|--gres=*|--mem=*|--mem-per-cpu=*|--time=*|--partition=*|--account=*)
                i=$((i + 1))
                ;;
            --ntasks|--nodes|--cpus-per-task|--gres|--mem|--mem-per-cpu|--time|--partition|--account)
                i=$((i + 2))
                ;;
            -*)
                i=$((i + 1))
                ;;
            *)
                break
                ;;
        esac
    done

    if [[ ${i} -ge ${#tokens[@]} ]]; then
        return 1
    fi

    printf '%s\n' "${tokens[*]:${i}}"
}

# Early help to avoid runtime setup when called from login node.
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Użycie: sbatch main_forget_grid.sh [--forget-class N] [--retain-mode retain|no_retain] [--methods dash|rs_fire|dash,rs_fire] [--exp-tag TAG] [--grid_script NAME] [--exclude-nodes CSV]"
    echo "     lub: bash main_forget_grid.sh [--array 0-0] [--forget-class N] [--retain-mode retain|no_retain] [--methods dash|rs_fire|dash,rs_fire] [--exp-tag TAG] [--grid_script NAME] [--exclude-nodes CSV]"
    echo "Dla retrain: sbatch main_forget_grid.sh --forget-classes CSV --seeds CSV"
    exit 0
fi

# Login-wrapper mode (like submit_tests_gpu.sh): submit and print logs path.
if [[ -z "${SLURM_JOB_ID:-}" && "${1:-}" != "-h" && "${1:-}" != "--help" ]]; then
    LOG_DATE_SUBMIT=$(date '+%Y-%m-%d')
    WRAPPER_EXCLUDE_NODES="${EXCLUDE_NODES}"
    WRAPPER_GRID_SCRIPT="${DEFAULT_GRID_SCRIPT}"
    WRAPPER_EXPERIMENT_SLUG="${DEFAULT_EXPERIMENT_SLUG}"
    WRAPPER_GRID_SCRIPT_CLI_SET=0
    HAS_EXCLUDE_ARG=0
    WRAPPER_SBATCH_ARGS=()
    WRAPPER_FORWARD_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --array)
                if [[ $# -lt 2 ]]; then
                    echo "Brak wartości dla --array" >&2
                    exit 2
                fi
                WRAPPER_SBATCH_ARGS+=("$1" "$2")
                shift 2
                ;;
            --array=*)
                WRAPPER_SBATCH_ARGS+=("$1")
                shift
                ;;
            --exclude)
                if [[ $# -lt 2 ]]; then
                    echo "Brak wartości dla --exclude" >&2
                    exit 2
                fi
                WRAPPER_SBATCH_ARGS+=("$1" "$2")
                HAS_EXCLUDE_ARG=1
                shift 2
                ;;
            --exclude=*)
                WRAPPER_SBATCH_ARGS+=("$1")
                HAS_EXCLUDE_ARG=1
                shift
                ;;
            --exclude-nodes)
                if [[ $# -lt 2 ]]; then
                    echo "Brak wartości dla --exclude-nodes" >&2
                    exit 2
                fi
                WRAPPER_EXCLUDE_NODES="$2"
                shift 2
                ;;
            --exclude-nodes=*)
                WRAPPER_EXCLUDE_NODES="${1#*=}"
                shift
                ;;
            --grid_script|--grid-script)
                if [[ $# -lt 2 ]]; then
                    echo "Brak wartości dla --grid_script" >&2
                    exit 2
                fi
                WRAPPER_GRID_SCRIPT="$(normalize_grid_script_name "$2")"
                WRAPPER_EXPERIMENT_SLUG="$(derive_experiment_slug_from_grid_script "${WRAPPER_GRID_SCRIPT}")"
                WRAPPER_GRID_SCRIPT_CLI_SET=1
                shift 2
                ;;
            --grid_script=*|--grid-script=*)
                WRAPPER_GRID_SCRIPT="$(normalize_grid_script_name "${1#*=}")"
                WRAPPER_EXPERIMENT_SLUG="$(derive_experiment_slug_from_grid_script "${WRAPPER_GRID_SCRIPT}")"
                WRAPPER_GRID_SCRIPT_CLI_SET=1
                shift
                ;;
            *)
                WRAPPER_FORWARD_ARGS+=("$1")
                shift
                ;;
        esac
    done
    if [[ -n "${WRAPPER_GRID_SCRIPT}" ]]; then
        WRAPPER_GRID_SCRIPT="$(normalize_grid_script_name "${WRAPPER_GRID_SCRIPT}")"
    fi
    if [[ -z "${WRAPPER_GRID_SCRIPT}" ]]; then
        echo "Brak GRID_SCRIPT. Podaj --grid_script NAME lub ustaw GRID_SCRIPT w env." >&2
        exit 2
    fi
    if [[ -z "${WRAPPER_EXPERIMENT_SLUG}" ]]; then
        WRAPPER_EXPERIMENT_SLUG="$(derive_experiment_slug_from_grid_script "${WRAPPER_GRID_SCRIPT}")"
    fi
    if [[ "${WRAPPER_GRID_SCRIPT_CLI_SET}" -eq 1 ]]; then
        WRAPPER_FORWARD_ARGS+=("--grid_script" "${WRAPPER_GRID_SCRIPT}")
    fi
    if [[ "${HAS_EXCLUDE_ARG}" -eq 0 && -n "${WRAPPER_EXCLUDE_NODES}" ]]; then
        WRAPPER_SBATCH_ARGS+=("--exclude=${WRAPPER_EXCLUDE_NODES}")
    fi
    echo "Submitting grid job:"
    echo "  sbatch ${WRAPPER_SBATCH_ARGS[*]} ${SCRIPT_PATH_SELF} ${WRAPPER_FORWARD_ARGS[*]}"
    SBATCH_OUTPUT=$(sbatch --parsable "${WRAPPER_SBATCH_ARGS[@]}" "${SCRIPT_PATH_SELF}" "${WRAPPER_FORWARD_ARGS[@]}")
    JOB_ID="${SBATCH_OUTPUT%%;*}"
    echo "Submitted job id: ${JOB_ID}"
    echo "Check logs in: ${SCRIPT_DIR_SELF}/logs/slurm_logs/${LOG_DATE_SUBMIT}/${WRAPPER_EXPERIMENT_SLUG}/${JOB_ID}/"
    exit 0
fi

# Praca z katalogu, z którego wywołano sbatch
cd "${SLURM_SUBMIT_DIR:-$PWD}"

############## Ustawienia eksperymentu (łatwo podmienić dla innych gridów)
# EXPERIMENT_SLUG="${EXPERIMENT_SLUG:-rga}"
# GRID_SCRIPT="${GRID_SCRIPT:-rga_grid_config.py}"
EXPERIMENT_SLUG="${EXPERIMENT_SLUG:-${DEFAULT_EXPERIMENT_SLUG}}"
GRID_SCRIPT="${GRID_SCRIPT:-${DEFAULT_GRID_SCRIPT}}"
GRID_SCRIPT_CLI_SET=0

##############

# Argumenty CLI
# ============================================================================
FORGET_CLASS="${FORGET_CLASS:-}"
FORGET_CLASSES="${FORGET_CLASSES:-}"
RETAIN_MODE="${RETAIN_MODE:-}"
METHODS="${METHODS:-}"
SEEDS="${SEEDS:-2}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-}"

usage() {
    echo "Użycie: sbatch main_forget_grid.sh [--forget-class N] [--retain-mode retain|no_retain] [--methods dash|rs_fire|dash,rs_fire] [--exp-tag TAG] [--grid_script NAME] [--exclude-nodes CSV]"
    echo "     lub: bash main_forget_grid.sh [--array 0-0] [--forget-class N] [--retain-mode retain|no_retain] [--methods dash|rs_fire|dash,rs_fire] [--exp-tag TAG] [--grid_script NAME] [--exclude-nodes CSV]"
    echo "Dla retrain: sbatch main_forget_grid.sh --forget-classes CSV --seeds CSV"
    echo "Zmienne env: FORGET_CLASS=N, RETAIN_MODE=retain|no_retain, METHODS=dash|rs_fire|dash,rs_fire"
    echo "Zmienne env (retrain): FORGET_CLASSES=CSV, SEEDS=CSV"
    echo "Zmienne env (opcjonalne): EXPERIMENT_SLUG=..., GRID_SCRIPT=..., EXPERIMENT_TAG=..., EXCLUDE_NODES=..."
    echo ""
    echo "Aby sprawdzić liczbę kombinacji w gridzie:"
    echo "  python ${GRID_SCRIPT} --total-only"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --forget-class)
            FORGET_CLASS="$2"
            shift 2
            ;;
        --forget-classes)
            FORGET_CLASSES="$2"
            FORGET_CLASSES_SET=1
            shift 2
            ;;
        --retain-mode)
            RETAIN_MODE="$2"
            shift 2
            ;;
        --retain-mode=*)
            RETAIN_MODE="${1#*=}"
            shift
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --methods=*)
            METHODS="${1#*=}"
            shift
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --exp-tag)
            EXPERIMENT_TAG="$2"
            shift 2
            ;;
        --grid_script|--grid-script)
            GRID_SCRIPT="$(normalize_grid_script_name "$2")"
            GRID_SCRIPT_CLI_SET=1
            shift 2
            ;;
        --grid_script=*|--grid-script=*)
            GRID_SCRIPT="$(normalize_grid_script_name "${1#*=}")"
            GRID_SCRIPT_CLI_SET=1
            shift
            ;;
        --exclude-nodes)
            EXCLUDE_NODES="$2"
            shift 2
            ;;
        --exclude-nodes=*)
            EXCLUDE_NODES="${1#*=}"
            shift
            ;;
        --exclude)
            EXCLUDE_NODES="$2"
            shift 2
            ;;
        --exclude=*)
            EXCLUDE_NODES="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Nieznany argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ "${GRID_SCRIPT_CLI_SET}" -eq 1 ]]; then
    EXPERIMENT_SLUG="$(derive_experiment_slug_from_grid_script "${GRID_SCRIPT}")"
fi
if [[ -z "${GRID_SCRIPT}" ]]; then
    echo "Brak GRID_SCRIPT. Podaj --grid_script NAME lub ustaw GRID_SCRIPT w env." >&2
    usage
    exit 2
fi
GRID_SCRIPT="$(normalize_grid_script_name "${GRID_SCRIPT}")"
if [[ -z "${EXPERIMENT_SLUG}" ]]; then
    EXPERIMENT_SLUG="$(derive_experiment_slug_from_grid_script "${GRID_SCRIPT}")"
fi

# Utwórz katalog na logi dopiero po ustaleniu EXPERIMENT_SLUG.
LOG_DATE=$(date '+%Y-%m-%d')
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LOG_DIR="logs/slurm_logs/${LOG_DATE}/${EXPERIMENT_SLUG}/${ARRAY_JOB_ID}"
mkdir -p "${LOG_DIR}"
# Format docelowy: logs/slurm_logs/DATA/<slug>/<slug>-%A_%a.out/.err
TARGET_LOG_OUT="${LOG_DIR}/${EXPERIMENT_SLUG}-${ARRAY_JOB_ID}_${TASK_ID}.out"
TARGET_LOG_ERR="${LOG_DIR}/${EXPERIMENT_SLUG}-${ARRAY_JOB_ID}_${TASK_ID}.err"
exec > >(tee -a "${TARGET_LOG_OUT}") 2> >(tee -a "${TARGET_LOG_ERR}" >&2)
echo "Run logs:"
echo "  out: ${TARGET_LOG_OUT}"
echo "  err: ${TARGET_LOG_ERR}"

# Conda GH200 / aarch64
source "$PLG_GROUPS_STORAGE/plggdnnp/apps/miniforge3-gh200/activate_conda.sh"
conda activate "$PLG_GROUPS_STORAGE/plggdnnp/conda_envs/lapsum-gh200"

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "ERROR: This job must run on GH200/aarch64 node, got: $(uname -m)" >&2
    exit 1
fi

echo "Python executable: $(which python)"
python --version

# Tryb surowy + trace
set -xuo pipefail

# Heartbeat
( while true; do echo "[HB] $(date '+%F %T') $(hostname)"; sleep 30; done ) &
HEARTBEAT_PID=$!

cleanup() {
    if [[ -n "${HEARTBEAT_PID:-}" ]] && kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
        kill "${HEARTBEAT_PID}" 2>/dev/null || true
        wait "${HEARTBEAT_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Env vars
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Cache
export CACHE_ROOT="$PLG_GROUPS_STORAGE/plggdnnp/cache/$USER"

export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
export WANDB_DIR="$PLG_GROUPS_STORAGE/plggdnnp/wandb/$USER"

export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

export TMPDIR="$PLG_GROUPS_STORAGE/plggdnnp/tmp/$USER"

mkdir -p \
  "$WANDB_CACHE_DIR" \
  "$WANDB_DIR" \
  "$TORCH_HOME" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TMPDIR"
mkdir -p "$WANDB_CACHE_DIR" "$TORCH_HOME" "$HF_HOME"

# PYTHONPATH
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# W&B
export WANDB__SERVICE_WAIT=300

# Debug
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-?} ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-?} NODELIST=${SLURM_NODELIST:-?} HOST=$(hostname)"
if command -v scontrol >/dev/null 2>&1; then
    echo "SLURM allocation summary:"
    if [[ "$(uname -m)" == "aarch64" ]]; then
        echo "scontrol summary skipped on aarch64; cluster scontrol binary is x86_64 here."
    else
        scontrol show job "${SLURM_JOB_ID}" \
            | grep -Eo 'NodeList=[^ ]+|Gres=[^ ]+|TresPerNode=[^ ]+|TresPerTask=[^ ]+|TRES=[^ ]+|TresAlloc=[^ ]+' \
            || true
    fi
fi
nvidia-smi || true
python - <<'PY'
import sys, os
try:
    import torch
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
except Exception as e:
    print("torch import failed:", e)
    raise SystemExit(0)
print("exe:", sys.executable)
print("python:", sys.version.split()[0])
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available():", cuda_available)
print("torch.cuda.device_count():", device_count)
if cuda_available and device_count > 0:
    try:
        print("torch.cuda.current_device():", torch.cuda.current_device())
    except Exception as e:
        print("torch.cuda.current_device() failed:", repr(e))
    try:
        print("device_name(0):", torch.cuda.get_device_name(0))
    except Exception as e:
        print("torch.cuda.get_device_name(0) failed:", repr(e))
    try:
        x = torch.tensor([1.0], device="cuda:0")
        print("cuda tensor smoke test:", x)
    except Exception as e:
        print("cuda tensor smoke test failed:", repr(e))
PY

if [[ -n "${SLURM_JOB_ID:-}" && -n "${EXCLUDE_NODES}" ]]; then
    echo "NOTE: exclude nodes requested (${EXCLUDE_NODES}), but allocation already exists."
    echo "      Use wrapper mode (bash main_forget_grid.sh ...) or sbatch --exclude=... to affect placement."
fi

# ============================================================================
# Generuj konfigurację i komendę z ${GRID_SCRIPT}
# ============================================================================
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [[ "${GRID_SCRIPT}" != */* && -f "${SCRIPT_DIR_SELF}/${GRID_SCRIPT}" ]]; then
    GRID_SCRIPT="${SCRIPT_DIR_SELF}/${GRID_SCRIPT}"
fi
if [[ "${GRID_SCRIPT}" != */* && -f "SEMU/Classification/${GRID_SCRIPT}" ]]; then
    GRID_SCRIPT="SEMU/Classification/${GRID_SCRIPT}"
fi
if [[ ! -f "${GRID_SCRIPT}" ]]; then
    echo "Grid script not found: ${GRID_SCRIPT}" >&2
    exit 2
fi

# Wywołaj Python script, który wygeneruje całą konfigurację
if [[ "$(basename "${GRID_SCRIPT}")" == "retrain_grid_config.py" ]]; then
    if ! GRID_OUTPUT=$(python "${GRID_SCRIPT}"         --task-id "$TASK_ID"         --forget-classes "$FORGET_CLASSES"         --seeds "$SEEDS"); then
        echo "Error generating grid configuration!" >&2
        exit 1
    fi
else
    GRID_ARGS=(--task-id "$TASK_ID")
    if [[ -n "$FORGET_CLASS" ]]; then
        GRID_ARGS+=(--forget-class "$FORGET_CLASS")
    fi
    if [[ -n "$RETAIN_MODE" ]]; then
        GRID_ARGS+=(--retain-mode "$RETAIN_MODE")
    fi
    if [[ -n "$METHODS" ]]; then
        GRID_ARGS+=(--methods "$METHODS")
    fi
    if ! GRID_OUTPUT=$(python "${GRID_SCRIPT}" "${GRID_ARGS[@]}"); then
        echo "Error generating grid configuration!" >&2
        exit 1
    fi
fi

# Wyodrębnij informacje z outputu
printf '%s
' "$GRID_OUTPUT"     | grep -v "^WANDB_TAGS="     | grep -v "^RGA_CONFIG_TAG="     | grep -v "^REP_CONFIG_TAG="     | grep -v "^TS_CONFIG_TAG="     | grep -v "^READOUT_CONFIG_TAG="     | grep -v "^COARSE_ANALYSIS_CONFIG_TAG="     | grep -v "^EXIT_CODE_PLACEHOLDER"

# Wyodrębnij komendę (wszystko między "Running command:" a "WANDB_TAGS=")
CMD="$(extract_running_command_from_grid_output "$GRID_OUTPUT")"

# Wyodrębnij tagi (do użycia w podsumowaniu)
WANDB_TAGS="$(extract_grid_marker_value "$GRID_OUTPUT" "WANDB_TAGS")"
GRID_CONFIG_TAG="$(extract_grid_marker_value "$GRID_OUTPUT" "TS_CONFIG_TAG")"
if [ -z "${GRID_CONFIG_TAG}" ]; then
    GRID_CONFIG_TAG="$(extract_grid_marker_value "$GRID_OUTPUT" "RGA_CONFIG_TAG")"
fi
if [ -z "${GRID_CONFIG_TAG}" ]; then
    GRID_CONFIG_TAG="$(extract_grid_marker_value "$GRID_OUTPUT" "REP_CONFIG_TAG")"
fi
if [ -z "${GRID_CONFIG_TAG}" ]; then
    GRID_CONFIG_TAG="$(extract_grid_marker_value "$GRID_OUTPUT" "READOUT_CONFIG_TAG")"
fi
if [ -z "${GRID_CONFIG_TAG}" ]; then
    GRID_CONFIG_TAG="$(extract_grid_marker_value "$GRID_OUTPUT" "COARSE_ANALYSIS_CONFIG_TAG")"
fi
if ! require_grid_output_value "WANDB_TAGS" "$WANDB_TAGS" "$GRID_SCRIPT"; then
    printf '%s
' "$GRID_OUTPUT" >&2
    exit 1
fi
if ! require_grid_output_value "Running command block" "$CMD" "$GRID_SCRIPT"; then
    printf '%s
' "$GRID_OUTPUT" >&2
    exit 1
fi

# Dodaj dodatkowy tag eksperymentu do WandB (jeśli podany)
if [ -n "${EXPERIMENT_TAG}" ]; then
    WANDB_TAGS="${WANDB_TAGS} ${EXPERIMENT_TAG}"
    read -r -a CMD_TOKENS <<< "$CMD"
    NEW_TOKENS=()
    i=0
    while [ $i -lt ${#CMD_TOKENS[@]} ]; do
        token="${CMD_TOKENS[$i]}"
        if [ "$token" = "--wandb_tags" ]; then
            NEW_TOKENS+=("$token")
            i=$((i + 1))
            while [ $i -lt ${#CMD_TOKENS[@]} ] && [[ "${CMD_TOKENS[$i]}" != --* ]]; do
                NEW_TOKENS+=("${CMD_TOKENS[$i]}")
                i=$((i + 1))
            done
            NEW_TOKENS+=("${EXPERIMENT_TAG}")
            continue
        fi
        NEW_TOKENS+=("$token")
        i=$((i + 1))
    done
    CMD="${NEW_TOKENS[*]}"
fi

if [[ "$(uname -m)" == "aarch64" ]]; then
    CMD="$(strip_leading_srun "$CMD")"
fi

# ============================================================================
# Uruchom eksperyment
# ============================================================================
eval "$CMD"

EXIT_CODE=$?

echo "=============================================================="
echo "Experiment completed with exit code: ${EXIT_CODE}"
echo "Grid config tag: ${GRID_CONFIG_TAG}"
echo "Tags: ${WANDB_TAGS}"
echo "=============================================================="

exit ${EXIT_CODE}
