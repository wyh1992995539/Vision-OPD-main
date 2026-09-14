#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${GRPO_CONFIG_FILE:-${PROJECT_ROOT}/configs/grpo_6241.yaml}"
ARGS=(--preflight-only)
EXTRA=()

while (($#)); do
    case "$1" in
        --config)
            [[ $# -ge 2 ]] || { echo "--config requires a path" >&2; exit 2; }
            CONFIG_FILE="$2"
            shift
            ;;
        --preflight-only) ARGS=(--preflight-only) ;;
        --run) ARGS=(--run) ;;
        -h|--help)
            echo "Usage: scripts/run_grpo_2gpu.sh [--config PATH] [--preflight-only|--run] [Hydra overrides...]"
            exit 0
            ;;
        *) EXTRA+=("$1") ;;
    esac
    shift
done

[[ "$CONFIG_FILE" == /* ]] || CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "vision-opd" ]]; then
    command -v conda >/dev/null 2>&1 || { echo "vision-opd conda environment is required" >&2; exit 1; }
    exec conda run --no-capture-output -n vision-opd bash "$0" --config "$CONFIG_FILE" "${ARGS[@]}" "${EXTRA[@]}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
exec python "${PROJECT_ROOT}/scripts/run_grpo_guarded.py" --config "$CONFIG_FILE" "${ARGS[@]}" "${EXTRA[@]}"
