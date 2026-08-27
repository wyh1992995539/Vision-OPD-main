#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${VOPD_CONFIG_FILE:-${PROJECT_ROOT}/configs/vopd_1024.yaml}"
MODE="preflight"
EXTRA_ARGS=()

usage() {
    printf '%s\n' \
        "Usage: scripts/run_vopd_2gpu.sh [--preflight-only|--run] [Hydra overrides...]" \
        "" \
        "  --preflight-only  Validate files, Parquet schema, image paths, and config without GPU (default)." \
        "  --run             Run the two-GPU, two-optimizer-step Day 7 smoke after preflight passes."
}

while (($#)); do
    case "$1" in
        --preflight-only)
            MODE="preflight"
            ;;
        --run)
            MODE="run"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            ;;
    esac
    shift
done

if [[ "${CONDA_DEFAULT_ENV:-}" != "vision-opd" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "The vision-opd conda environment is required, but conda was not found." >&2
        exit 1
    fi
    exec conda run --no-capture-output -n vision-opd bash "$0" \
        "$([[ "$MODE" == "run" ]] && echo --run || echo --preflight-only)" \
        "${EXTRA_ARGS[@]}"
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

export PROJECT_ROOT
export CONFIG_FILE

eval "$(python - "$CONFIG_FILE" "$PROJECT_ROOT" <<'PY'
import shlex
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
project_root = Path(sys.argv[2])
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

def resolve(value):
    path = Path(value)
    return str(path if path.is_absolute() else project_root / path)

values = {
    "EXPERIMENT_ID": cfg["experiment"]["id"],
    "SEED": cfg["experiment"]["seed"],
    "MODEL_PATH": resolve(cfg["paths"]["model"]),
    "TRAIN_FILE": resolve(cfg["paths"]["train_file"]),
    "CHAT_TEMPLATE_FILE": resolve(cfg["paths"]["chat_template"]),
    "OUTPUT_DIR": resolve(cfg["paths"]["output_dir"]),
    "EXPECTED_TRAIN_ROWS": cfg["data"]["expected_train_rows"],
    "TRAIN_BATCH_SIZE": cfg["data"]["train_batch_size"],
    "IMAGE_KEY": cfg["data"]["image_key"],
    "TEACHER_IMAGE_KEY": cfg["data"]["teacher_image_key"],
    "MAX_PROMPT_LENGTH": cfg["data"]["max_prompt_length"],
    "MAX_RESPONSE_LENGTH": cfg["data"]["max_response_length"],
    "DATALOADER_NUM_WORKERS": cfg["data"]["dataloader_num_workers"],
    "LR": cfg["actor"]["learning_rate"],
    "PPO_MINI_BATCH_SIZE": cfg["actor"]["ppo_mini_batch_size"],
    "PPO_MAX_TOKEN_LEN_PER_GPU": cfg["actor"]["max_token_length_per_gpu"],
    "ROLLOUT_N": cfg["rollout"]["n"],
    "ROLLOUT_TP_SIZE": cfg["rollout"]["tensor_model_parallel_size"],
    "ROLLOUT_GPU_MEMORY_UTILIZATION": cfg["rollout"]["gpu_memory_utilization"],
    "ROLLOUT_LOGPROB_MICRO_BATCH_SIZE": cfg["rollout"]["log_prob_micro_batch_size_per_gpu"],
    "ROLLOUT_AGENT_NUM_WORKERS": cfg["rollout"]["agent_num_workers"],
    "TOP_K": cfg["self_distillation"]["top_k"],
    "ALPHA": cfg["self_distillation"]["alpha"],
    "TEACHER_MODEL_SOURCE": cfg["self_distillation"]["teacher_model_source"],
    "TEACHER_REGULARIZATION": cfg["self_distillation"]["teacher_regularization"],
    "TEACHER_UPDATE_RATE": cfg["self_distillation"]["teacher_update_rate"],
    "N_NODES": cfg["resources"]["nodes"],
    "N_GPUS": cfg["resources"]["gpus_per_node"],
    "TOTAL_OPTIMIZER_STEPS": cfg["smoke"]["total_optimizer_steps"],
}

for name, value in values.items():
    print(f"{name}={shlex.quote(str(value))}")
PY
)"

PREFLIGHT_DIR="${OUTPUT_DIR}/preflight"
mkdir -p "$PREFLIGHT_DIR"

python - "$CONFIG_FILE" "$PROJECT_ROOT" "$PREFLIGHT_DIR/preflight_summary.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

config_path = Path(sys.argv[1]).resolve()
project_root = Path(sys.argv[2]).resolve()
summary_path = Path(sys.argv[3]).resolve()
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

def resolve(value):
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve()

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

errors = []
model_path = resolve(cfg["paths"]["model"])
train_file = resolve(cfg["paths"]["train_file"])
chat_template = resolve(cfg["paths"]["chat_template"])

required_model_files = [
    "config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
]
missing_model_files = [name for name in required_model_files if not (model_path / name).is_file()]
if missing_model_files:
    errors.append(f"missing model files: {missing_model_files}")
if not list(model_path.glob("model-*.safetensors")) and not list(model_path.glob("model.safetensors-*.safetensors")):
    errors.append("no model safetensors shards found")
if not train_file.is_file():
    errors.append(f"training parquet not found: {train_file}")
if not chat_template.is_file():
    errors.append(f"chat template not found: {chat_template}")

row_count = None
columns = []
missing_image_paths = []
if train_file.is_file():
    table = pq.read_table(train_file)
    row_count = table.num_rows
    columns = table.column_names
    expected_rows = int(cfg["data"]["expected_train_rows"])
    if row_count != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {row_count}")
    required_columns = {"prompt", cfg["data"]["image_key"], cfg["data"]["teacher_image_key"], "extra_info"}
    missing_columns = sorted(required_columns.difference(columns))
    if missing_columns:
        errors.append(f"missing parquet columns: {missing_columns}")
    else:
        for column_name in (cfg["data"]["image_key"], cfg["data"]["teacher_image_key"]):
            for row_index, items in enumerate(table[column_name].to_pylist()):
                for item in items or []:
                    image_path = Path(item["path"])
                    if not image_path.is_file():
                        missing_image_paths.append({"row": row_index, "column": column_name, "path": str(image_path)})
                        if len(missing_image_paths) >= 20:
                            break
                if len(missing_image_paths) >= 20:
                    break
            if len(missing_image_paths) >= 20:
                break
        if missing_image_paths:
            errors.append("one or more image paths are missing; first 20 are recorded")

checks = {
    "prefix_source_online": cfg["experiment"]["prefix_source"] == "online",
    "seed_is_42": int(cfg["experiment"]["seed"]) == 42,
    "two_gpus": int(cfg["resources"]["gpus_per_node"]) == 2,
    "global_batch_is_8": int(cfg["data"]["train_batch_size"]) == 8,
    "rollout_n_is_1": int(cfg["rollout"]["n"]) == 1,
    "two_optimizer_steps": int(cfg["smoke"]["total_optimizer_steps"]) == 2,
    "prompt_limit_is_8192": int(cfg["data"]["max_prompt_length"]) == 8192,
    "response_limit_is_256": int(cfg["data"]["max_response_length"]) == 256,
    "truncation_is_error": cfg["data"]["truncation"] == "error",
    "teacher_uses_bbox_images": cfg["data"]["teacher_image_key"] == "bbox_images",
}
failed_checks = sorted(name for name, passed in checks.items() if not passed)
if failed_checks:
    errors.append(f"frozen config checks failed: {failed_checks}")

summary = {
    "experiment_id": cfg["experiment"]["id"],
    "status": "PASS" if not errors else "FAIL",
    "gpu_used": False,
    "config": str(config_path),
    "config_sha256": sha256(config_path),
    "model_path": str(model_path),
    "train_file": str(train_file),
    "train_file_sha256": sha256(train_file) if train_file.is_file() else None,
    "train_rows": row_count,
    "parquet_columns": columns,
    "chat_template": str(chat_template),
    "checks": checks,
    "missing_image_paths": missing_image_paths,
    "errors": errors,
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"DAY7_PREFLIGHT={summary['status']}")
print(f"SUMMARY={summary_path}")
print(f"TRAIN_ROWS={row_count}")
print(f"MISSING_IMAGE_PATHS={len(missing_image_paths)}")
if errors:
    for error in errors:
        print(f"ERROR={error}", file=sys.stderr)
    raise SystemExit(1)
PY

if [[ "$MODE" == "preflight" ]]; then
    echo "No GPU training started. Re-run with --run only after GPUs are available."
    exit 0
fi

if [[ "${CUDA_VISIBLE_DEVICES:-}" == "" ]]; then
    export CUDA_VISIBLE_DEVICES="0,1"
fi

IFS=',' read -r -a GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
if [[ "${#GPU_IDS[@]}" -ne "$N_GPUS" ]]; then
    echo "Expected $N_GPUS visible GPUs, got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/rollouts" "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/evidence"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONBUFFERED=1
export VLLM_USE_V1=1
unset VLLM_ATTENTION_BACKEND
ulimit -c 0

MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
TRAIN_LOG="${OUTPUT_DIR}/logs/train.log"

echo "Starting $EXPERIMENT_ID on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG_FILE"
echo "Training data: $TRAIN_FILE"
echo "Log: $TRAIN_LOG"

python -m verl.trainer.main_ppo --config-name vopd \
    "data.train_files=[\"${TRAIN_FILE}\"]" \
    'data.val_files=[]' \
    data.filter_overlong_prompts=False \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.truncation=error \
    data.shuffle=True \
    data.trust_remote_code=True \
    data.return_multi_modal_inputs=True \
    data.image_key="$IMAGE_KEY" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.custom_chat_template_file="$CHAT_TEMPLATE_FILE" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.data_loader_seed="$SEED" \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=vopd \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.self_distillation.distillation_topk="$TOP_K" \
    actor_rollout_ref.actor.self_distillation.alpha="$ALPHA" \
    actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
    actor_rollout_ref.actor.self_distillation.teacher_model_source="$TEACHER_MODEL_SOURCE" \
    actor_rollout_ref.actor.self_distillation.teacher_regularization="$TEACHER_REGULARIZATION" \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate="$TEACHER_UPDATE_RATE" \
    actor_rollout_ref.actor.self_distillation.teacher_image_key="$TEACHER_IMAGE_KEY" \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.actor.self_distillation.log_prob_dump_dir="${OUTPUT_DIR}/evidence/log_probs" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_AGENT_NUM_WORKERS" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    reward_model.enable=False \
    reward_model.use_reward_loop=False \
    custom_reward_function.path=null \
    critic.model.path="$MODEL_PATH" \
    trainer.project_name=Vision-OPD \
    trainer.group_name=E-D7-001 \
    trainer.experiment_name="$EXPERIMENT_ID" \
    'trainer.logger=["console","tensorboard"]' \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes="$N_NODES" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_OPTIMIZER_STEPS" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
    trainer.rollout_data_dir="${OUTPUT_DIR}/rollouts" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$TRAIN_LOG"

