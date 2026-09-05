#!/usr/bin/env bash

set -euo pipefail

if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${VOPD_CONFIG_FILE:-${PROJECT_ROOT}/configs/vopd_1024.yaml}"
MODE="preflight"
EXTRA_ARGS=()

usage() {
    printf '%s\n' \
        "Usage: scripts/run_vopd_2gpu.sh [--config PATH] [--preflight-only|--run] [Hydra overrides...]" \
        "" \
        "  --config PATH     Select an auditable experiment config (default: configs/vopd_1024.yaml)." \
        "  --preflight-only  Validate model, data, images, hashes, and training contract (default)." \
        "  --run             Internal training entry; use scripts/run_vopd_guarded.py for formal training."
}

while (($#)); do
    case "$1" in
        --config)
            if (($# < 2)); then
                echo "--config requires a path" >&2
                exit 2
            fi
            CONFIG_FILE="$2"
            shift
            ;;
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

if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
fi
export VOPD_CONFIG_FILE="$CONFIG_FILE"

if [[ "$MODE" == "run" && "${VOPD_GUARD_ACTIVE:-}" != "1" ]]; then
    echo "Direct --run is blocked: formal training must be owned by scripts/run_vopd_guarded.py." >&2
    exit 2
fi

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
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
training = config.get("training", config.get("smoke"))
if not isinstance(training, dict):
    raise ValueError("config must contain a training or legacy smoke mapping")

def resolve(value):
    path = Path(value)
    return str(path if path.is_absolute() else project_root / path)

values = {
    "EXPERIMENT_ID": config["experiment"]["id"],
    "GROUP_NAME": config["experiment"].get("group_name", config["experiment"]["id"]),
    "SEED": config["experiment"]["seed"],
    "MODEL_PATH": resolve(config["paths"]["model"]),
    "TRAIN_FILE": resolve(config["paths"]["train_file"]),
    "CHAT_TEMPLATE_FILE": resolve(config["paths"]["chat_template"]),
    "OUTPUT_DIR": resolve(config["paths"]["output_dir"]),
    "TRAIN_BATCH_SIZE": config["data"]["train_batch_size"],
    "DATA_SHUFFLE": config["data"]["shuffle"],
    "IMAGE_KEY": config["data"]["image_key"],
    "TEACHER_IMAGE_KEY": config["data"]["teacher_image_key"],
    "MAX_PROMPT_LENGTH": config["data"]["max_prompt_length"],
    "MAX_RESPONSE_LENGTH": config["data"]["max_response_length"],
    "DATALOADER_NUM_WORKERS": config["data"]["dataloader_num_workers"],
    "FULL_COVERAGE_ENABLED": config["data"].get("full_coverage_padding", {}).get("enabled", False),
    "FULL_COVERAGE_MULTIPLE": config["data"].get("full_coverage_padding", {}).get("multiple", config["data"]["train_batch_size"]),
    "FULL_COVERAGE_PADDING_SOURCE_INDEX": config["data"].get("full_coverage_padding", {}).get("padding_source_index", 0),
    "FULL_COVERAGE_EXPECTED_UNIQUE": config["data"].get("full_coverage_padding", {}).get("expected_unique_samples", config["data"]["expected_train_rows"]),
    "FULL_COVERAGE_EXPECTED_PADDING": config["data"].get("full_coverage_padding", {}).get("expected_padding_rows", 0),
    "FULL_COVERAGE_RECEIPT": resolve(config["data"].get("full_coverage_padding", {}).get("receipt_path", config["paths"]["output_dir"] + "/evidence/full_coverage_receipt.json")),
    "LR": config["actor"]["learning_rate"],
    "LR_WARMUP_STEPS": config["actor"].get("lr_warmup_steps", -1),
    "PPO_MINI_BATCH_SIZE": config["actor"]["ppo_mini_batch_size"],
    "CLIP_RATIO_LOW": config["actor"].get("clip_ratio_low", 0.2),
    "CLIP_RATIO_HIGH": config["actor"].get("clip_ratio_high", 0.2),
    "USE_DYNAMIC_BSZ": config["actor"]["use_dynamic_batch_size"],
    "GRADIENT_CHECKPOINTING": config["actor"]["gradient_checkpointing"],
    "PPO_MAX_TOKEN_LEN_PER_GPU": config["actor"]["max_token_length_per_gpu"],
    "ACTOR_PARAM_OFFLOAD": config["actor"]["parameter_offload"],
    "ACTOR_OPTIMIZER_OFFLOAD": config["actor"]["optimizer_offload"],
    "REF_PARAM_OFFLOAD": config["actor"]["reference_parameter_offload"],
    "ROLLOUT_N": config["rollout"]["n"],
    "ROLLOUT_TEMPERATURE": config["rollout"].get("temperature", 1.0),
    "ROLLOUT_TOP_P": config["rollout"].get("top_p", 1.0),
    "ROLLOUT_TOP_K": config["rollout"].get("top_k", -1),
    "ROLLOUT_IGNORE_EOS": config["rollout"].get("ignore_eos", False),
    "ROLLOUT_TP_SIZE": config["rollout"]["tensor_model_parallel_size"],
    "ROLLOUT_GPU_MEMORY_UTILIZATION": config["rollout"]["gpu_memory_utilization"],
    "ROLLOUT_LOGPROB_MICRO_BATCH_SIZE": config["rollout"]["log_prob_micro_batch_size_per_gpu"],
    "ROLLOUT_AGENT_NUM_WORKERS": config["rollout"]["agent_num_workers"],
    "VLLM_FUSE_ALLREDUCE_RMS": config["rollout"].get("engine_kwargs", {}).get("vllm", {}).get("compilation_config", {}).get("pass_config", {}).get("fuse_allreduce_rms", False),
    "VLLM_ENABLE_FLASHINFER_AUTOTUNE": config["rollout"].get("engine_kwargs", {}).get("vllm", {}).get("kernel_config", {}).get("enable_flashinfer_autotune", False),
    "TOP_K": config["self_distillation"]["top_k"],
    "ALPHA": config["self_distillation"]["alpha"],
    "TEACHER_ALWAYS_ON": config["self_distillation"]["teacher_always_on"],
    "TEACHER_MODEL_SOURCE": config["self_distillation"]["teacher_model_source"],
    "TEACHER_REGULARIZATION": config["self_distillation"]["teacher_regularization"],
    "TEACHER_UPDATE_RATE": config["self_distillation"]["teacher_update_rate"],
    "MAX_REPROMPT_LENGTH": config["self_distillation"].get("max_reprompt_length", 10240),
    "DONT_REPROMPT": config["self_distillation"]["dont_reprompt_on_self_success"],
    "INCLUDE_ENVIRONMENT_FEEDBACK": config["self_distillation"]["include_environment_feedback"],
    "IMPORTANCE_SAMPLING_CLIP": config["self_distillation"]["importance_sampling_clip"],
    "N_NODES": config["resources"]["nodes"],
    "N_GPUS": config["resources"]["gpus_per_node"],
    "TOTAL_OPTIMIZER_STEPS": training["total_optimizer_steps"],
    "TOTAL_EPOCHS": training.get("total_epochs", 1),
    "SAVE_FREQUENCY": training["save_frequency"],
    "TEST_FREQUENCY": training["test_frequency"],
    "RESUME_MODE": training.get("resume_mode", "disable"),
    "MAX_ACTOR_CKPT_TO_KEEP": training.get("max_actor_ckpt_to_keep", None),
}

for name, value in values.items():
    if isinstance(value, bool):
        value = str(value).lower()
    elif value is None:
        value = "null"
    print(f"{name}={shlex.quote(str(value))}")
PY
)"

PREFLIGHT_DIR="${OUTPUT_DIR}/preflight"
PREFLIGHT_SUMMARY="${PREFLIGHT_DIR}/preflight_summary.json"
mkdir -p "$PREFLIGHT_DIR"

python "${PROJECT_ROOT}/scripts/vopd_training_preflight.py" \
    --config "$CONFIG_FILE" \
    --project-root "$PROJECT_ROOT" \
    --output "$PREFLIGHT_SUMMARY"

if [[ "$MODE" == "preflight" ]]; then
    echo "No GPU training started. Re-run with --run only after the preflight passes."
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

RUN_MANIFEST="${PREFLIGHT_DIR}/run_invocation.json"
python - "$PREFLIGHT_SUMMARY" "$RUN_MANIFEST" "${EXTRA_ARGS[@]}" <<'PY'
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))

def git(*args):
    result = subprocess.run(
        ["git", "-C", os.environ["PROJECT_ROOT"], *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()

payload = {
    "schema_version": 1,
    "experiment_id": summary["experiment_id"],
    "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "config": summary["config"],
    "config_sha256": summary["config_sha256"],
    "train_file": summary["train_file"],
    "train_file_sha256": summary["train_file_sha256"],
    "sample_ids": summary["sample_ids"],
    "training_contract": summary["training_contract"],
    "git_commit": git("rev-parse", "HEAD"),
    "git_status_porcelain": git("status", "--porcelain"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    "hydra_overrides": sys.argv[3:],
}
output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

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
echo "Audit manifest: $RUN_MANIFEST"
echo "Log: $TRAIN_LOG"

python -m verl.trainer.main_ppo --config-name vopd \
    "data.train_files=[\"${TRAIN_FILE}\"]" \
    'data.val_files=[]' \
    data.filter_overlong_prompts=False \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.truncation=error \
    data.shuffle="$DATA_SHUFFLE" \
    data.seed="$SEED" \
    data.full_coverage_padding.enabled="$FULL_COVERAGE_ENABLED" \
    data.full_coverage_padding.multiple="$FULL_COVERAGE_MULTIPLE" \
    data.full_coverage_padding.padding_source_index="$FULL_COVERAGE_PADDING_SOURCE_INDEX" \
    data.full_coverage_padding.expected_unique_samples="$FULL_COVERAGE_EXPECTED_UNIQUE" \
    data.full_coverage_padding.expected_padding_rows="$FULL_COVERAGE_EXPECTED_PADDING" \
    data.full_coverage_padding.receipt_path="$FULL_COVERAGE_RECEIPT" \
    data.trust_remote_code=True \
    data.return_multi_modal_inputs=True \
    data.image_key="$IMAGE_KEY" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.custom_chat_template_file="$CHAT_TEMPLATE_FILE" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing="$GRADIENT_CHECKPOINTING" \
    actor_rollout_ref.actor.data_loader_seed="$SEED" \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.optim.lr_warmup_steps="$LR_WARMUP_STEPS" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.clip_ratio_low="$CLIP_RATIO_LOW" \
    actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH" \
    actor_rollout_ref.actor.use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" \
    actor_rollout_ref.actor.policy_loss.loss_mode=vopd \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.self_distillation.distillation_topk="$TOP_K" \
    actor_rollout_ref.actor.self_distillation.alpha="$ALPHA" \
    actor_rollout_ref.actor.self_distillation.teacher_always_on="$TEACHER_ALWAYS_ON" \
    actor_rollout_ref.actor.self_distillation.teacher_model_source="$TEACHER_MODEL_SOURCE" \
    actor_rollout_ref.actor.self_distillation.teacher_regularization="$TEACHER_REGULARIZATION" \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate="$TEACHER_UPDATE_RATE" \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len="$MAX_REPROMPT_LENGTH" \
    actor_rollout_ref.actor.self_distillation.teacher_image_key="$TEACHER_IMAGE_KEY" \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success="$DONT_REPROMPT" \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback="$INCLUDE_ENVIRONMENT_FEEDBACK" \
    actor_rollout_ref.actor.self_distillation.is_clip="$IMPORTANCE_SAMPLING_CLIP" \
    actor_rollout_ref.actor.self_distillation.log_prob_dump_dir="${OUTPUT_DIR}/evidence/log_probs" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
    actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
    actor_rollout_ref.rollout.top_k="$ROLLOUT_TOP_K" \
    actor_rollout_ref.rollout.ignore_eos="$ROLLOUT_IGNORE_EOS" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms="$VLLM_FUSE_ALLREDUCE_RMS" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune="$VLLM_ENABLE_FLASHINFER_AUTOTUNE" \
    actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_AGENT_NUM_WORKERS" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold="$IMPORTANCE_SAMPLING_CLIP" \
    reward_model.enable=False \
    reward_model.use_reward_loop=False \
    custom_reward_function.path=null \
    critic.model.path="$MODEL_PATH" \
    trainer.project_name=Vision-OPD \
    trainer.group_name="$GROUP_NAME" \
    trainer.experiment_name="$EXPERIMENT_ID" \
    'trainer.logger=["console","tensorboard"]' \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes="$N_NODES" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.total_training_steps="$TOTAL_OPTIMIZER_STEPS" \
    trainer.save_freq="$SAVE_FREQUENCY" \
    trainer.test_freq="$TEST_FREQUENCY" \
    trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPT_TO_KEEP" \
    trainer.val_before_train=False \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
    trainer.rollout_data_dir="${OUTPUT_DIR}/rollouts" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$TRAIN_LOG"
