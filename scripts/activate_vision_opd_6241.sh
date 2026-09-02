#!/usr/bin/env bash

# Source this file so cache and temporary paths persist in the current shell.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Usage: source scripts/activate_vision_opd_6241.sh" >&2
    exit 2
fi

export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export TORCH_HOME=/root/autodl-tmp/torch_cache
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export XDG_CACHE_HOME=/root/autodl-tmp/xdg_cache
export TMPDIR=/root/autodl-tmp/tmp

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" \
    "$TORCH_HOME" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" \
    /root/autodl-tmp/data/vision_opd_6241/raw \
    /root/autodl-tmp/data/vision_opd_6241/images \
    /root/autodl-tmp/data/vision_opd_6241/teacher_images

echo "Vision-OPD 6241 cache environment is active under /root/autodl-tmp."
