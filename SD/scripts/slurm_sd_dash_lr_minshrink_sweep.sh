#!/bin/bash
# Submit a small grid of SD class or NSFW runs over learning rate,
# dash.min_shrink, rl_loss_mode, DASH on/off, and attention head-wise on/off.
#
# Examples:
#   bash SD/scripts/slurm_sd_dash_lr_minshrink_sweep.sh --setting class --forget-class 0
#   bash SD/scripts/slurm_sd_dash_lr_minshrink_sweep.sh --setting nsfw --method rl
#   LR_VALUES="5e-6 1e-5" MIN_SHRINK_VALUES="0.8" \
#     DASH_ATTENTION_HEAD_WISE_VALUES="off on" DASH_VALUES="on" \
#     bash SD/scripts/slurm_sd_dash_lr_minshrink_sweep.sh --setting class --forget-class 0 --dash-target unet_xattn

set -euo pipefail

SCRIPT_DIR_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR_SELF}/slurm_sd_dash_rl.sh"

# LR_VALUES="${LR_VALUES:-1.5e-6 5e-6 1.5e-5}"
LR_VALUES="${LR_VALUES:-1.5e-6}"
MIN_SHRINK_VALUES="${MIN_SHRINK_VALUES:-0.7}"
# MIN_SHRINK_VALUES="${MIN_SHRINK_VALUES:-0.3 0.6 0.8 0.9}"
# MIN_SHRINK_VALUES="${MIN_SHRINK_VALUES:-0.1 0.2 0.3 0.4}"
RL_LOSS_MODE_VALUES="${RL_LOSS_MODE_VALUES:-output_matching}"
DASH_VALUES="${DASH_VALUES:-on}"
DASH_ATTENTION_HEAD_WISE_VALUES="${DASH_ATTENTION_HEAD_WISE_VALUES:-on}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-sd_dash_lr_minshrink_sweep}"
DEDUP_NO_DASH="${DEDUP_NO_DASH:-1}"
DRY_RUN="${DRY_RUN:-0}"

COMMON_ARGS=()
EXPERIMENT_PREFIX_CLI=""
USER_DASH_VALUE=""
USER_HEADWISE_VALUE=""

print_help() {
    cat <<'EOF'
Uzycie:
  bash SD/scripts/slurm_sd_dash_lr_minshrink_sweep.sh --setting class --forget-class 0 [opcje]
  bash SD/scripts/slurm_sd_dash_lr_minshrink_sweep.sh --setting nsfw --method rl [opcje]

Zmienne srodowiskowe:
  LR_VALUES              Lista learning rate, np. "5e-6 1e-5 5e-5".
  MIN_SHRINK_VALUES      Lista dash.min_shrink, np. "0.1 0.3 0.5".
  RL_LOSS_MODE_VALUES    Lista unlearn.rl_loss_mode, np. "denoise_pseudo output_matching".
  DASH_VALUES            Lista trybow DASH: on/off, true/false, 1/0. Domyslnie "on".
  DASH_ATTENTION_HEAD_WISE_VALUES
                         Lista trybow head-wise attention: on/off, true/false, 1/0. Domyslnie "on".
  DEDUP_NO_DASH          1 = dla dash=off uruchamia tylko jeden job na LR, bo shrink/head-wise sa ignorowane.
  DRY_RUN                1 = wypisuje komendy bez submitowania.
  EXPERIMENT_PREFIX      Prefix katalogow wynikow. Domyslnie sd_dash_lr_minshrink_sweep.

Wszystkie argumenty skryptu sa przekazywane dalej do slurm_sd_dash_rl.sh.
Sweep nadpisuje wartosci z YAML, a jawne flagi terminalowe --dash/--no-dash
oraz --dash-attention-head-wise/--no-dash-attention-head-wise zawezaja sweep.
Ten wrapper sam dodaje --lr, --rl-loss-mode, --dash-min-shrink,
--dash/--no-dash, --dash-attention-head-wise / --no-dash-attention-head-wise oraz --experiment-slug.
EOF
}

slugify() {
    local value="$1"
    value="${value//./p}"
    value="${value//-/m}"
    value="${value//+/}"
    echo "${value}"
}

bool_enabled() {
    local value
    value="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    case "${value}" in
        1|true|yes|on|dash|dash_on|headwise|head_wise)
            return 0
            ;;
        0|false|no|off|none|no_dash|nodash|dash_off|no_headwise|no-headwise)
            return 1
            ;;
        *)
            echo "Nieznana wartosc bool: $1 (uzyj on/off, true/false albo 1/0)" >&2
            exit 2
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            exit 0
            ;;
        --experiment-slug)
            if [[ $# -lt 2 ]]; then
                echo "Brak wartosci dla --experiment-slug" >&2
                exit 2
            fi
            EXPERIMENT_PREFIX_CLI="$2"
            shift 2
            ;;
        --experiment-slug=*)
            EXPERIMENT_PREFIX_CLI="${1#*=}"
            shift
            ;;
        --dash)
            USER_DASH_VALUE="on"
            shift
            ;;
        --no-dash)
            USER_DASH_VALUE="off"
            shift
            ;;
        --dash-attention-head-wise)
            USER_HEADWISE_VALUE="on"
            shift
            ;;
        --no-dash-attention-head-wise)
            USER_HEADWISE_VALUE="off"
            shift
            ;;
        *)
            COMMON_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ! -x "${LAUNCHER}" ]]; then
    echo "Nie znaleziono launchera albo brak uprawnien execute: ${LAUNCHER}" >&2
    exit 1
fi

if [[ -n "${USER_DASH_VALUE}" ]]; then
    DASH_VALUES="${USER_DASH_VALUE}"
fi
if [[ -n "${USER_HEADWISE_VALUE}" ]]; then
    DASH_ATTENTION_HEAD_WISE_VALUES="${USER_HEADWISE_VALUE}"
fi

echo "Submitting SD DASH LR/min_shrink sweep"
echo "  lr values: ${LR_VALUES}"
echo "  min_shrink values: ${MIN_SHRINK_VALUES}"
echo "  rl_loss_mode values: ${RL_LOSS_MODE_VALUES}"
echo "  dash values: ${DASH_VALUES}"
echo "  attention_head_wise values: ${DASH_ATTENTION_HEAD_WISE_VALUES}"
echo "  dedup no-dash: ${DEDUP_NO_DASH}"
if [[ -n "${EXPERIMENT_PREFIX_CLI}" ]]; then
    EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX_CLI}"
fi
echo "  dry run: ${DRY_RUN}"
echo "  experiment prefix: ${EXPERIMENT_PREFIX}"
echo "  common args: ${COMMON_ARGS[*]:-<none>}"

for lr in ${LR_VALUES}; do
    for rl_loss_mode in ${RL_LOSS_MODE_VALUES}; do
        for dash_value in ${DASH_VALUES}; do
            if bool_enabled "${dash_value}"; then
                dash_slug="dash_on"
                for headwise_value in ${DASH_ATTENTION_HEAD_WISE_VALUES}; do
                    headwise_slug="headwise_off"
                    headwise_arg=(--no-dash-attention-head-wise)
                    if bool_enabled "${headwise_value}"; then
                        headwise_slug="headwise_on"
                        headwise_arg=(--dash-attention-head-wise)
                    fi
                    for min_shrink in ${MIN_SHRINK_VALUES}; do
                        lr_slug="$(slugify "${lr}")"
                        rl_slug="$(slugify "${rl_loss_mode}")"
                        min_slug="$(slugify "${min_shrink}")"
                        experiment_slug="${EXPERIMENT_PREFIX}/${dash_slug}/${headwise_slug}/rl_${rl_slug}/lr_${lr_slug}/minshrink_${min_slug}"
                        cmd=(bash "${LAUNCHER}" "${COMMON_ARGS[@]}" --lr "${lr}" --rl-loss-mode "${rl_loss_mode}" --dash-min-shrink "${min_shrink}" --dash "${headwise_arg[@]}" --experiment-slug "${experiment_slug}")
                        echo
                        echo "=== dash=on, attention_head_wise=${headwise_value}, rl_loss_mode=${rl_loss_mode}, lr=${lr}, min_shrink=${min_shrink}, experiment=${experiment_slug} ==="
                        if [[ "${DRY_RUN}" == "1" ]]; then
                            printf 'DRY_RUN:'
                            printf ' %q' "${cmd[@]}"
                            printf '\n'
                        else
                            "${cmd[@]}"
                        fi
                    done
                done
            else
                dash_slug="dash_off"
                min_values="${MIN_SHRINK_VALUES}"
                if [[ "${DEDUP_NO_DASH}" == "1" ]]; then
                    min_values="none"
                fi
                for min_shrink in ${min_values}; do
                    lr_slug="$(slugify "${lr}")"
                    rl_slug="$(slugify "${rl_loss_mode}")"
                    if [[ "${min_shrink}" == "none" ]]; then
                        experiment_slug="${EXPERIMENT_PREFIX}/${dash_slug}/rl_${rl_slug}/lr_${lr_slug}"
                        cmd=(bash "${LAUNCHER}" "${COMMON_ARGS[@]}" --lr "${lr}" --rl-loss-mode "${rl_loss_mode}" --no-dash --experiment-slug "${experiment_slug}")
                        echo
                        echo "=== dash=off, rl_loss_mode=${rl_loss_mode}, lr=${lr}, min_shrink=ignored, experiment=${experiment_slug} ==="
                    else
                        min_slug="$(slugify "${min_shrink}")"
                        experiment_slug="${EXPERIMENT_PREFIX}/${dash_slug}/rl_${rl_slug}/lr_${lr_slug}/minshrink_${min_slug}"
                        cmd=(bash "${LAUNCHER}" "${COMMON_ARGS[@]}" --lr "${lr}" --rl-loss-mode "${rl_loss_mode}" --dash-min-shrink "${min_shrink}" --no-dash --experiment-slug "${experiment_slug}")
                        echo
                        echo "=== dash=off, rl_loss_mode=${rl_loss_mode}, lr=${lr}, min_shrink=${min_shrink} (ignored), experiment=${experiment_slug} ==="
                    fi
                    if [[ "${DRY_RUN}" == "1" ]]; then
                        printf 'DRY_RUN:'
                        printf ' %q' "${cmd[@]}"
                        printf '\n'
                    else
                        "${cmd[@]}"
                    fi
                done
            fi
        done
    done
done
