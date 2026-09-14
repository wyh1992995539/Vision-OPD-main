#!/usr/bin/env python3
"""Fail-closed static preflight for the Vision-OPD MCQ GRPO candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COLUMNS = {"data_source", "prompt", "images", "ability", "reward_model", "extra_info"}
VALID_OPTIONS = ("A", "B", "C", "D")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str | Path, root: Path = ROOT) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def read_cgroup_limit() -> dict[str, Any]:
    for base in (Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")):
        maximum = base / "memory.max"
        current = base / "memory.current"
        if maximum.is_file():
            raw = maximum.read_text().strip()
            return {
                "path": str(maximum),
                "maximum_bytes": None if raw == "max" else int(raw),
                "current_bytes": int(current.read_text().strip()) if current.is_file() else None,
            }
        maximum = base / "memory.limit_in_bytes"
        current = base / "memory.usage_in_bytes"
        if maximum.is_file():
            return {
                "path": str(maximum),
                "maximum_bytes": int(maximum.read_text().strip()),
                "current_bytes": int(current.read_text().strip()) if current.is_file() else None,
            }
    return {"path": None, "maximum_bytes": None, "current_bytes": None}


def query_gpus() -> dict[str, Any]:
    command = [
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc), "gpus": []}
    rows = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 5:
            rows.append({
                "index": int(fields[0]), "name": fields[1], "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]), "utilization_percent": int(fields[4]),
            })
    return {"available": True, "gpus": rows}


def build_hydra_overrides(config: dict[str, Any], root: Path = ROOT) -> list[str]:
    paths, data, actor = config["paths"], config["data"], config["actor"]
    rollout, algorithm = config["rollout"], config["algorithm"]
    resources, training = config["resources"], config["training"]
    model = resolve(paths["model"], root)
    train = resolve(paths["train_file"], root)
    template = resolve(paths["chat_template"], root)
    reward = resolve(paths["reward_function"], root)
    output = resolve(paths["output_dir"], root)
    max_model_len = int(data["max_prompt_length"]) + int(data["max_response_length"])
    return [
        f'data.train_files=["{train}"]', 'data.val_files=[]',
        f'data.reward_fn_key={data["reward_fn_key"]}',
        f'data.max_prompt_length={data["max_prompt_length"]}',
        f'data.max_response_length={data["max_response_length"]}',
        f'data.train_batch_size={data["train_batch_size"]}',
        f'data.shuffle={str(data["shuffle"]).lower()}', f'data.seed={config["experiment"]["seed"]}',
        f'data.dataloader_num_workers={data["dataloader_num_workers"]}',
        f'data.filter_overlong_prompts={str(data["filter_overlong_prompts"]).lower()}',
        f'data.truncation={data["truncation"]}', f'data.image_key={data["image_key"]}',
        'data.return_multi_modal_inputs=true', 'data.trust_remote_code=true',
        f'actor_rollout_ref.model.path={model}', 'actor_rollout_ref.model.trust_remote_code=true',
        f'actor_rollout_ref.model.custom_chat_template_file={template}',
        'actor_rollout_ref.model.use_remove_padding=true',
        f'actor_rollout_ref.model.enable_gradient_checkpointing={str(actor["gradient_checkpointing"]).lower()}',
        f'actor_rollout_ref.actor.optim.lr={actor["learning_rate"]}',
        f'actor_rollout_ref.actor.optim.lr_warmup_steps={actor["lr_warmup_steps"]}',
        f'actor_rollout_ref.actor.ppo_mini_batch_size={actor["ppo_mini_batch_size"]}',
        f'actor_rollout_ref.actor.ppo_epochs={actor["ppo_epochs"]}',
        f'actor_rollout_ref.actor.policy_loss.loss_mode={actor["policy_loss_mode"]}',
        f'actor_rollout_ref.actor.loss_agg_mode={actor["loss_agg_mode"]}',
        f'actor_rollout_ref.actor.clip_ratio_low={actor["clip_ratio_low"]}',
        f'actor_rollout_ref.actor.clip_ratio_high={actor["clip_ratio_high"]}',
        f'actor_rollout_ref.actor.entropy_coeff={actor["entropy_coeff"]}',
        f'actor_rollout_ref.actor.use_kl_loss={str(actor["use_kl_loss"]).lower()}',
        f'actor_rollout_ref.actor.kl_loss_coef={actor["kl_loss_coef"]}',
        f'actor_rollout_ref.actor.kl_loss_type={actor["kl_loss_type"]}',
        f'actor_rollout_ref.actor.use_dynamic_bsz={str(actor["use_dynamic_batch_size"]).lower()}',
        f'actor_rollout_ref.actor.ppo_max_token_len_per_gpu={actor["max_token_length_per_gpu"]}',
        f'actor_rollout_ref.actor.fsdp_config.param_offload={str(actor["parameter_offload"]).lower()}',
        f'actor_rollout_ref.actor.fsdp_config.optimizer_offload={str(actor["optimizer_offload"]).lower()}',
        f'actor_rollout_ref.ref.fsdp_config.param_offload={str(actor["reference_parameter_offload"]).lower()}',
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1',
        f'actor_rollout_ref.rollout.n={rollout["n"]}',
        f'actor_rollout_ref.rollout.temperature={rollout["temperature"]}',
        f'actor_rollout_ref.rollout.top_p={rollout["top_p"]}',
        f'actor_rollout_ref.rollout.top_k={rollout["top_k"]}',
        f'actor_rollout_ref.rollout.ignore_eos={str(rollout["ignore_eos"]).lower()}',
        'actor_rollout_ref.rollout.name=vllm',
        f'actor_rollout_ref.rollout.tensor_model_parallel_size={rollout["tensor_model_parallel_size"]}',
        f'actor_rollout_ref.rollout.gpu_memory_utilization={rollout["gpu_memory_utilization"]}',
        f'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={rollout["log_prob_micro_batch_size_per_gpu"]}',
        f'actor_rollout_ref.rollout.max_num_batched_tokens={max_model_len}',
        f'actor_rollout_ref.rollout.max_model_len={max_model_len}',
        f'actor_rollout_ref.rollout.response_length={data["max_response_length"]}',
        f'actor_rollout_ref.rollout.calculate_log_probs={str(rollout["calculate_log_probs"]).lower()}',
        f'actor_rollout_ref.rollout.agent.num_workers={rollout["agent_num_workers"]}',
        f'+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes={rollout["engine_kwargs"]["vllm"]["compilation_config"]["cudagraph_capture_sizes"]}',
        f'+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms={str(rollout["engine_kwargs"]["vllm"]["compilation_config"]["pass_config"]["fuse_allreduce_rms"]).lower()}',
        f'+actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune={str(rollout["engine_kwargs"]["vllm"]["kernel_config"]["enable_flashinfer_autotune"]).lower()}',
        f'algorithm.adv_estimator={algorithm["adv_estimator"]}',
        f'algorithm.norm_adv_by_std_in_grpo={str(algorithm["norm_adv_by_std_in_grpo"]).lower()}',
        f'algorithm.use_kl_in_reward={str(algorithm["use_kl_in_reward"]).lower()}',
        'algorithm.rollout_correction.rollout_is=null',
        'algorithm.rollout_correction.rollout_rs=null',
        'algorithm.rollout_correction.bypass_mode=false',
        'critic.enable=false', 'reward_model.enable=false', 'reward_model.use_reward_loop=false',
        f'custom_reward_function.path={reward}', 'custom_reward_function.name=compute_score',
        'trainer.project_name=Vision-OPD-GRPO',
        f'trainer.group_name={config["experiment"]["id"]}',
        f'trainer.experiment_name={config["experiment"]["id"]}',
        f'trainer.n_gpus_per_node={resources["gpus_per_node"]}', f'trainer.nnodes={resources["nodes"]}',
        f'trainer.total_epochs={training["total_epochs"]}',
        f'trainer.total_training_steps={training["outer_iterations"]}',
        f'trainer.save_freq={training["save_frequency"]}', f'trainer.test_freq={training["test_frequency"]}',
        f'trainer.max_actor_ckpt_to_keep={training["max_actor_ckpt_to_keep"]}',
        'trainer.val_before_train=false', f'trainer.resume_mode={training["resume_mode"]}',
        f'trainer.default_local_dir={output / "checkpoints"}',
        f'trainer.rollout_data_dir={output / "rollouts"}',
    ]


def compose_hydra_config(overrides: list[str], root: Path = ROOT) -> str:
    """Compose the exact local verl config without importing the trainer runtime."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(root / "verl/trainer/config")):
        resolved = compose(config_name="baseline_grpo", overrides=overrides)
    return OmegaConf.to_yaml(resolved, resolve=True)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def validate_config(config_path: Path, output: Path | None = None) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths, data, actor = config["paths"], config["data"], config["actor"]
    rollout, algorithm = config["rollout"], config["algorithm"]
    components, resources, training = config["components"], config["resources"], config["training"]
    model, train = resolve(paths["model"]), resolve(paths["train_file"])
    template, reward = resolve(paths["chat_template"]), resolve(paths["reward_function"])
    reward_validation = resolve(paths["reward_validation"])
    errors: list[str] = []
    for path, label in ((model, "model"), (train, "train_file"), (template, "chat_template"),
                        (reward, "reward_function"), (reward_validation, "reward_validation")):
        if not path.exists(): errors.append(f"missing {label}: {path}")
    required_model = ["config.json", "model.safetensors.index.json", "tokenizer_config.json", "preprocessor_config.json"]
    missing_model = [name for name in required_model if not (model / name).is_file()]
    if missing_model: errors.append(f"missing model files: {missing_model}")

    rows = unique_ids = missing_images = gold_leaks = route_errors = 0
    columns: list[str] = []
    gold_distribution = {key: 0 for key in VALID_OPTIONS}
    if train.is_file():
        table = pq.read_table(train)
        rows, columns = table.num_rows, table.column_names
        if set(columns) != EXPECTED_COLUMNS: errors.append(f"unexpected Parquet columns: {columns}")
        seen: set[str] = set()
        for index, row in enumerate(table.to_pylist()):
            try:
                extra, rm = row["extra_info"], row["reward_model"]
                sample_id, gold = str(extra["sample_id"]), str(rm["ground_truth"])
                if sample_id in seen: continue
                seen.add(sample_id)
                if row["data_source"] != data["data_source"] or extra["reward_route"] != data["reward_route"]:
                    route_errors += 1
                if list(extra["valid_options"]) != list(VALID_OPTIONS) or gold not in VALID_OPTIONS:
                    route_errors += 1
                else: gold_distribution[gold] += 1
                prompt = row["prompt"]
                content = prompt[0]["content"] if isinstance(prompt, list) and len(prompt) == 1 else ""
                if not content or content.count("<image>") != 1 or "<answer>" in content.lower(): gold_leaks += 1
                if f"Answer: {gold}" in content or "ground_truth" in content: gold_leaks += 1
                images = row["images"]
                if not isinstance(images, list) or len(images) != 1 or not Path(images[0]["path"]).is_file():
                    missing_images += 1
            except (KeyError, TypeError, IndexError):
                route_errors += 1
        unique_ids = len(seen)
    expected_rows = int(data["expected_train_rows"])
    outer = rows // int(data["train_batch_size"]) if rows else 0
    effective = outer * int(data["train_batch_size"])
    trajectories = effective * int(rollout["n"])
    reward_report: dict[str, Any] = {}
    if reward_validation.is_file():
        reward_report = json.loads(reward_validation.read_text(encoding="utf-8"))
    checks = {
        "data_rows": rows == expected_rows == int(data["expected_unique_samples"]),
        "unique_sample_ids": unique_ids == expected_rows,
        "exact_grpo_schema": set(columns) == EXPECTED_COLUMNS and "bbox_images" not in columns,
        "all_images_exist": missing_images == 0,
        "no_gold_in_policy_prompt": gold_leaks == 0,
        "all_reward_routes_valid": route_errors == 0,
        "reward_gate_bound_and_passed": reward_report.get("status") == "PASS" and reward_report.get("data_sha256") == (sha256_file(train) if train.is_file() else None) and reward_report.get("reward_file_sha256") == (sha256_file(reward) if reward.is_file() else None),
        "batch_contract": outer == 780 and effective == 6240 and rows - effective == 1 and trajectories == 24960,
        "grpo_group_contract": int(data["train_batch_size"]) == 8 and int(rollout["n"]) == 4 and int(actor["ppo_mini_batch_size"]) == 8 and int(actor["ppo_epochs"]) == 1,
        "plain_policy_gradient": actor["policy_loss_mode"] == "vanilla" and actor["loss_agg_mode"] == "token-mean",
        "symmetric_clip": float(actor["clip_ratio_low"]) == float(actor["clip_ratio_high"]) == 0.2,
        "single_kl_path": actor["use_kl_loss"] is True and float(actor["kl_loss_coef"]) == 0.001 and actor["kl_loss_type"] == "low_var_kl" and algorithm["use_kl_in_reward"] is False,
        "grpo_advantage": algorithm["adv_estimator"] == "grpo" and algorithm["norm_adv_by_std_in_grpo"] is True,
        "sampling_contract": float(rollout["temperature"]) == 1.0 and float(rollout["top_p"]) == 1.0 and int(rollout["top_k"]) == -1 and rollout["ignore_eos"] is False,
        "no_forbidden_training_components": all(value is False for value in components.values()),
        "response_limit_candidate": int(data["max_response_length"]) == 128,
        "prompt_limit": int(data["max_prompt_length"]) == 8192,
        "formal_training_not_authorized": config["validation"]["formal_training_authorized"] is False,
    }
    errors.extend(f"failed check: {name}" for name, passed in checks.items() if not passed)
    cgroup = read_cgroup_limit()
    disk = shutil.disk_usage(resolve(paths["output_dir"]).parent)
    gpus = query_gpus()
    cgroup_ready = cgroup["maximum_bytes"] is None or cgroup["maximum_bytes"] >= int(resources["prelaunch_cgroup_minimum_bytes"])
    disk_ready = disk.free >= int(resources["prelaunch_disk_free_minimum_bytes"])
    gpu_ready = gpus["available"] and len(gpus["gpus"]) >= int(resources["gpus_per_node"])
    overrides = build_hydra_overrides(config)
    resolved_hydra_config = ""
    try:
        resolved_hydra_config = compose_hydra_config(overrides)
    except Exception as exc:
        errors.append(f"Hydra composition failed: {type(exc).__name__}: {exc}")
    forbidden = ("vopd", "self_distillation", "bbox_images", "cached_prefix", "teacher_image")
    override_text = "\n".join(overrides).lower()
    if any(token in override_text for token in forbidden): errors.append("Hydra overrides contain forbidden VOPD/Cached/Teacher fields")
    report = {
        "schema_version": 1, "experiment_id": config["experiment"]["id"],
        "status": "PASS" if not errors else "FAIL", "checks": checks, "errors": errors,
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "train_file": str(train), "train_file_sha256": sha256_file(train) if train.is_file() else None,
        "reward_function": str(reward), "reward_function_sha256": sha256_file(reward) if reward.is_file() else None,
        "rows": rows, "unique_sample_ids": unique_ids, "gold_distribution": gold_distribution,
        "training_contract": {"prompt_batch": data["train_batch_size"], "rollout_n": rollout["n"], "responses_per_outer_iteration": int(data["train_batch_size"]) * int(rollout["n"]), "outer_iterations": outer, "effective_prompts": effective, "dropped_prompts": rows - effective, "effective_trajectories": trajectories, "worker_global_mini_batch_after_rollout_expansion": int(actor["ppo_mini_batch_size"]) * int(rollout["n"]), "worker_local_mini_batch_two_way_dp": int(actor["ppo_mini_batch_size"]) * int(rollout["n"]) // int(resources["gpus_per_node"])},
        "resource_snapshot": {"cgroup": cgroup, "cgroup_ready": cgroup_ready, "disk_free_bytes": disk.free, "disk_ready": disk_ready, "gpu": gpus, "gpu_ready": gpu_ready},
        "static_preflight_passed": not errors, "runtime_resources_ready": cgroup_ready and disk_ready and gpu_ready,
        "training_started": False, "gpu_used": False, "pilot_training_authorized": False,
        "formal_training_authorized": False,
        "blocking_gates": [name for name, ready in (("sufficient_cgroup_memory", cgroup_ready), ("two_visible_gpus", gpu_ready), ("disk_floor", disk_ready)) if not ready] + list(config["validation"]["pending_gates"]),
        "hydra_config_name": "baseline_grpo", "hydra_overrides": overrides,
    }
    if output is not None:
        atomic_write(output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        atomic_write(output.parent / "hydra_overrides.json", json.dumps(overrides, ensure_ascii=False, indent=2) + "\n")
        atomic_write(output.parent / "resolved_hydra_config.yaml", resolved_hydra_config)
        command = ["python", "-m", "verl.trainer.main_ppo", "--config-name", "baseline_grpo", *overrides]
        atomic_write(output.parent / "command.txt", shlex.join(command) + "\n")
    if errors: raise ValueError("GRPO static preflight failed:\n- " + "\n- ".join(errors))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/grpo_6241.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output or resolve(config["paths"]["output_dir"]) / "preflight/preflight.json"
    report = validate_config(args.config, output)
    print(f"PASS: GRPO static preflight ({report['rows']} rows, {report['training_contract']['effective_trajectories']} trajectories)")
    print(f"runtime_resources_ready: {str(report['runtime_resources_ready']).lower()}")
    print(f"training_started: {str(report['training_started']).lower()}")
    print(f"report: {output.resolve()}")


if __name__ == "__main__":
    main()
