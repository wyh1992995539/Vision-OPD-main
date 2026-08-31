#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${VOPD_DAY8_RELOAD_CONFIG:-${PROJECT_ROOT}/configs/vopd_day8_reload.yaml}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "vision-opd" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "The vision-opd conda environment is required, but conda was not found." >&2
        exit 1
    fi
    exec conda run --no-capture-output -n vision-opd bash "$0" "$@"
fi

cd "$PROJECT_ROOT"
exec python scripts/vopd_day8_reload.py --config "$CONFIG_FILE" "$@"
