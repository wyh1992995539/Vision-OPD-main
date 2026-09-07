#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/run_vopd_2gpu.sh" \
  --config "$ROOT/configs/cached_prefix_6241.yaml" "$@"
