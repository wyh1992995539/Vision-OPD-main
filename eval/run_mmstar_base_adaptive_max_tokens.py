#!/usr/bin/env python3
"""Adaptively regenerate only truncated Base MMStar answers until none remain."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.paper_aligned_common import (
    checkpoint_identity,
    load_config,
    now_utc,
    prediction_complete,
    read_jsonl_map,
    require_frozen_base_identity,
    sha256_file,
    write_json,
    write_jsonl_map,
)
from eval.run_mmstar_base_max_tokens_diagnostic import (
    load_mmstar_tasks,
    run_inference_stage,
    score_stage,
    stop_service,
    wait_ready,
    write_artifact_hashes,
)


CONFIG = ROOT / "configs/mmstar_base_adaptive_max_tokens.yaml"
FROZEN_CONFIG_SHA256 = "a48e54bca7c03d995f39d44faa7b87b8c5fc536af03eeff6e63d7bc51330079c"


def is_length(item: dict[str, Any]) -> bool:
    return str(item.get("finish_reason") or "").casefold() == "length"


def filter_mmstar(records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {key: value for key, value in records.items() if value.get("benchmark") == "mmstar"}


def load_and_validate() -> dict[str, Any]:
    actual_config_sha = sha256_file(CONFIG)
    if actual_config_sha != FROZEN_CONFIG_SHA256:
        raise ValueError(
            f"adaptive config SHA256 changed: expected {FROZEN_CONFIG_SHA256}, got {actual_config_sha}"
        )
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["experiment"]["id"] != "E-MMSTAR-BASE-MAXTOK-EXT-001":
        raise ValueError("wrong experiment id")

    source = config["source"]
    r3_path, r3 = load_config(source["r3_config"])
    if sha256_file(r3_path) != source["r3_config_sha256"]:
        raise ValueError("R3 config SHA256 mismatch")

    r4_root = ROOT / source["r4_base_run"]
    diagnostic_root = ROOT / source["diagnostic_run"]
    source_files = {
        "r4_predictions": r4_root / "predictions.jsonl",
        "r4_scores": r4_root / "scores.jsonl",
        "selected_2048_predictions": diagnostic_root / "selected_2048/predictions.jsonl",
        "selected_2048_scores": diagnostic_root / "selected_2048/scores.jsonl",
        "residual_4096_predictions": diagnostic_root / "residual_4096/predictions.jsonl",
        "residual_4096_scores": diagnostic_root / "residual_4096/scores.jsonl",
        "r3_config": r3_path,
    }
    expected_hashes = {
        "r4_predictions": source["r4_predictions_sha256"],
        "r4_scores": source["r4_scores_sha256"],
        "selected_2048_predictions": source["selected_2048_predictions_sha256"],
        "selected_2048_scores": source["selected_2048_scores_sha256"],
        "residual_4096_predictions": source["residual_4096_predictions_sha256"],
        "residual_4096_scores": source["residual_4096_scores_sha256"],
        "r3_config": source["r3_config_sha256"],
    }
    actual_hashes = {name: sha256_file(path) for name, path in source_files.items()}
    if actual_hashes != expected_hashes:
        raise ValueError(f"frozen source SHA256 mismatch: {actual_hashes}")

    baseline_predictions, _ = read_jsonl_map(source_files["r4_predictions"], complete=prediction_complete)
    baseline_scores, _ = read_jsonl_map(source_files["r4_scores"])
    p2048, _ = read_jsonl_map(source_files["selected_2048_predictions"], complete=prediction_complete)
    s2048, _ = read_jsonl_map(source_files["selected_2048_scores"])
    p4096, _ = read_jsonl_map(source_files["residual_4096_predictions"], complete=prediction_complete)
    s4096, _ = read_jsonl_map(source_files["residual_4096_scores"])
    baseline_predictions = filter_mmstar(baseline_predictions)
    baseline_scores = filter_mmstar(baseline_scores)
    p2048 = filter_mmstar(p2048)
    s2048 = filter_mmstar(s2048)
    p4096 = filter_mmstar(p4096)
    s4096 = filter_mmstar(s4096)

    if len(baseline_predictions) != 1500 or len(baseline_scores) != 1500:
        raise ValueError("baseline MMStar must contain 1500 predictions and scores")
    original_length = {key for key, value in baseline_predictions.items() if is_length(value)}
    if len(original_length) != int(source["expected_original_length_samples"]):
        raise ValueError(f"expected 163 original length samples, got {len(original_length)}")
    if set(p2048) != original_length or set(s2048) != original_length:
        raise ValueError("2048 keys do not equal the original 1024 length selection")
    residual_2048 = {key for key, value in p2048.items() if is_length(value)}
    if set(p4096) != residual_2048 or set(s4096) != residual_2048:
        raise ValueError("4096 keys do not equal the residual 2048 length selection")
    residual_4096 = {key for key, value in p4096.items() if is_length(value)}
    if len(residual_4096) != int(source["expected_residual_4096_samples"]):
        raise ValueError(f"expected 46 residual 4096 samples, got {len(residual_4096)}")

    identity = checkpoint_identity(Path(config["model"]["path"]))
    require_frozen_base_identity(identity, r3)
    tasks = load_mmstar_tasks(r3)
    if not original_length <= set(tasks):
        raise ValueError("selected keys are missing from frozen MMStar tasks")

    nonselected = set(baseline_predictions) - original_length
    nonselected_correct = sum(bool(baseline_scores[key]["final_is_correct"]) for key in nonselected)
    if len(nonselected) != 1337 or nonselected_correct != 1000:
        raise ValueError("unexpected nonselected baseline counts")

    return {
        "config": config,
        "r3": r3,
        "r3_path": r3_path,
        "source_files": source_files,
        "source_hashes": actual_hashes,
        "baseline_predictions": baseline_predictions,
        "baseline_scores": baseline_scores,
        "p2048": p2048,
        "s2048": s2048,
        "p4096": p4096,
        "s4096": s4096,
        "original_length": original_length,
        "residual_4096": residual_4096,
        "nonselected_correct": nonselected_correct,
        "identity": identity,
        "tasks": tasks,
    }


def source_hashes(context: dict[str, Any]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in context["source_files"].items()}


def preflight(context: dict[str, Any]) -> dict[str, Any]:
    import torch

    config = context["config"]
    output = ROOT / config["paths"]["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG, output / "config_snapshot.yaml")
    receipt = {
        "schema_version": 1,
        "experiment_id": config["experiment"]["id"],
        "status": "PASS",
        "checked_at_utc": now_utc(),
        "config_sha256": FROZEN_CONFIG_SHA256,
        "source_sha256": context["source_hashes"],
        "checkpoint_identity": context["identity"],
        "original_length_count": len(context["original_length"]),
        "resolved_at_2048_count": 163 - len(context["p4096"]),
        "residual_4096_count": len(context["residual_4096"]),
        "adaptive_stages": config["inference"]["stages"],
        "success_condition": config["completion"]["success_condition"],
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    write_json(output / "preflight.json", receipt)
    write_artifact_hashes(output)
    return receipt


def annotated(item: dict[str, Any], *, stage: str, max_tokens: int) -> dict[str, Any]:
    result = copy.deepcopy(item)
    result["adaptive_source_stage"] = stage
    result["adaptive_max_tokens"] = max_tokens
    return result


def build_trace(
    context: dict[str, Any],
    stage_predictions: dict[str, dict[str, dict[str, Any]]],
    final_origin: dict[str, tuple[str, int]],
) -> dict[str, dict[str, Any]]:
    trace = {}
    for key in sorted(context["original_length"]):
        attempts = [
            {
                "stage": "baseline_1024",
                "max_tokens": 1024,
                "finish_reason": context["baseline_predictions"][key].get("finish_reason"),
                "completion_tokens": context["baseline_predictions"][key].get("completion_tokens"),
            },
            {
                "stage": "selected_2048",
                "max_tokens": 2048,
                "finish_reason": context["p2048"][key].get("finish_reason"),
                "completion_tokens": context["p2048"][key].get("completion_tokens"),
            },
        ]
        if key in context["p4096"]:
            attempts.append(
                {
                    "stage": "residual_4096",
                    "max_tokens": 4096,
                    "finish_reason": context["p4096"][key].get("finish_reason"),
                    "completion_tokens": context["p4096"][key].get("completion_tokens"),
                }
            )
        for stage, records in stage_predictions.items():
            if key in records:
                max_tokens = int(stage.rsplit("_", 1)[1])
                attempts.append(
                    {
                        "stage": stage,
                        "max_tokens": max_tokens,
                        "finish_reason": records[key].get("finish_reason"),
                        "completion_tokens": records[key].get("completion_tokens"),
                    }
                )
        final_stage, final_max = final_origin[key]
        trace[key] = {
            "schema_version": 1,
            "benchmark": "mmstar",
            "view": "full",
            "sample_uid": context["baseline_predictions"][key]["sample_uid"],
            "attempts": attempts,
            "final_stage": final_stage,
            "final_max_tokens": final_max,
        }
    return trace


def run(context: dict[str, Any]) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA GPU is not available; enable one GPU and rerun")

    config = context["config"]
    r3 = copy.deepcopy(context["r3"])
    r3["execution"]["request_timeout_seconds"] = int(
        config["inference"]["request_timeout_seconds"]
    )
    output = ROOT / config["paths"]["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG, output / "config_snapshot.yaml")
    before = source_hashes(context)

    hybrid_predictions = {
        key: copy.deepcopy(value) for key, value in context["p2048"].items()
    }
    hybrid_scores = {key: copy.deepcopy(value) for key, value in context["s2048"].items()}
    final_origin = {key: ("selected_2048", 2048) for key in hybrid_predictions}
    for key, value in context["p4096"].items():
        hybrid_predictions[key] = copy.deepcopy(value)
        hybrid_scores[key] = copy.deepcopy(context["s4096"][key])
        final_origin[key] = ("residual_4096", 4096)
    residual = set(context["residual_4096"])

    serving = config["serving"]
    model_id = config["model"]["served_model_name"]
    command = [
        "/root/miniconda3/envs/vision-opd/bin/vllm",
        "serve",
        config["model"]["path"],
        "--served-model-name",
        model_id,
        "--tensor-parallel-size",
        str(serving["tensor_parallel_size"]),
        "--gpu-memory-utilization",
        str(serving["gpu_memory_utilization"]),
        "--trust-remote-code",
        "--additional-config",
        json.dumps({"gdn_prefill_backend": serving["gdn_prefill_backend"]}),
        "--host",
        str(serving["host"]),
        "--port",
        str(serving["port"]),
    ]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="8")
    log = output / "vllm_server.log"
    stream = log.open("a", encoding="utf-8")
    service = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    started = now_utc()
    stage_predictions: dict[str, dict[str, dict[str, Any]]] = {}
    stage_summaries = []
    error: BaseException | None = None
    result: dict[str, Any] = {}
    service_exit = None
    try:
        print(f"SERVICE_START experiment={config['experiment']['id']}", flush=True)
        wait_ready(service, model_id, log)
        from openai import OpenAI

        local = threading.local()

        def get_client() -> OpenAI:
            client = getattr(local, "client", None)
            if client is None:
                client = OpenAI(
                    api_key="EMPTY",
                    base_url=f"http://{serving['host']}:{serving['port']}/v1",
                    timeout=float(config["inference"]["request_timeout_seconds"]),
                )
                local.client = client
            return client

        for stage_spec in config["inference"]["stages"]:
            if not residual:
                break
            max_tokens = int(stage_spec["max_tokens"])
            workers = int(stage_spec["workers"])
            name = f"residual_{max_tokens}"
            selected = {key: context["tasks"][key] for key in sorted(residual)}
            starting = len(selected)
            predictions = run_inference_stage(
                name, selected, max_tokens, output, r3, model_id, get_client, workers
            )
            incomplete = [key for key, value in predictions.items() if not prediction_complete(value)]
            if incomplete or set(predictions) != set(selected):
                raise RuntimeError(
                    f"{name}: incomplete inference; missing_or_failed={len(incomplete)}"
                )
            scores, summary = score_stage(
                name, predictions, output, r3, model_id, get_client, workers
            )
            if summary["pending_judge"]:
                raise RuntimeError(f"{name}: pending Judge results remain")
            stage_predictions[name] = predictions
            for key, value in predictions.items():
                hybrid_predictions[key] = copy.deepcopy(value)
                hybrid_scores[key] = copy.deepcopy(scores[key])
                final_origin[key] = (name, max_tokens)
            residual = {key for key, value in predictions.items() if is_length(value)}
            selected_correct = sum(
                bool(value["final_is_correct"]) for value in hybrid_scores.values()
            )
            stage_summary = {
                **summary,
                "max_tokens": max_tokens,
                "workers": workers,
                "starting_length_count": starting,
                "remaining_length_count": len(residual),
                "resolved_this_stage": starting - len(residual),
                "hybrid_selected_correct": selected_correct,
                "counterfactual_full_correct": context["nonselected_correct"] + selected_correct,
                "counterfactual_full_accuracy": (
                    context["nonselected_correct"] + selected_correct
                ) / 1500,
            }
            stage_summaries.append(stage_summary)
            write_json(output / name / "adaptive_summary.json", stage_summary)
            print(
                f"ADAPTIVE_STAGE_DONE max_tokens={max_tokens} start={starting} "
                f"remaining_length={len(residual)} full_correct={stage_summary['counterfactual_full_correct']}",
                flush=True,
            )

        final_dir = output / config["paths"]["final_hybrid_directory"]
        final_dir.mkdir(parents=True, exist_ok=True)
        final_predictions = {}
        final_scores = {}
        for key, value in context["baseline_predictions"].items():
            if key in context["original_length"]:
                stage, max_tokens = final_origin[key]
                final_predictions[key] = annotated(
                    hybrid_predictions[key], stage=stage, max_tokens=max_tokens
                )
                final_scores[key] = annotated(
                    hybrid_scores[key], stage=stage, max_tokens=max_tokens
                )
            else:
                final_predictions[key] = annotated(
                    value, stage="baseline_1024", max_tokens=1024
                )
                final_scores[key] = annotated(
                    context["baseline_scores"][key], stage="baseline_1024", max_tokens=1024
                )
        trace = build_trace(context, stage_predictions, final_origin)
        write_jsonl_map(final_dir / "predictions.jsonl", final_predictions)
        write_jsonl_map(final_dir / "scores.jsonl", final_scores)
        write_jsonl_map(final_dir / "selection_trace.jsonl", trace)

        correct = sum(bool(value["final_is_correct"]) for value in final_scores.values())
        final_length = sum(is_length(value) for value in final_predictions.values())
        source_distribution = {}
        for stage, _ in final_origin.values():
            source_distribution[stage] = source_distribution.get(stage, 0) + 1
        source_distribution["baseline_1024"] = 1500 - len(context["original_length"])
        completed = final_length == 0
        summary = {
            "schema_version": 1,
            "experiment_id": config["experiment"]["id"],
            "status": "PASS" if completed else "INCOMPLETE",
            "decision": "ALL_LENGTH_TRUNCATION_ELIMINATED" if completed else "MAX_STAGES_EXHAUSTED_WITH_LENGTH_REMAINING",
            "total": 1500,
            "correct": correct,
            "incorrect": 1500 - correct,
            "accuracy": correct / 1500,
            "original_length_count": len(context["original_length"]),
            "final_length_count": final_length,
            "source_stage_distribution": dict(sorted(source_distribution.items())),
            "adaptive_stage_summaries": stage_summaries,
            "updated_at_utc": now_utc(),
            "limitation_statement": config["limitations"]["statement"],
        }
        write_json(final_dir / "summary.json", summary)
        after = source_hashes(context)
        result = {
            **summary,
            "config_sha256": FROZEN_CONFIG_SHA256,
            "source_files_unchanged": before == after,
            "source_sha256": after,
            "checkpoint_identity": context["identity"],
            "runner_sha256": sha256_file(Path(__file__)),
        }
        if before != after:
            raise RuntimeError("frozen source files changed during adaptive run")
        write_json(output / "completion_receipt.json", result)
    except BaseException as exc:
        error = exc
        write_json(
            output / "failure.json",
            {
                "schema_version": 1,
                "experiment_id": config["experiment"]["id"],
                "failed_at_utc": now_utc(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
    finally:
        service_exit = stop_service(service)
        stream.close()
        write_json(
            output / "session.json",
            {
                "schema_version": 1,
                "experiment_id": config["experiment"]["id"],
                "status": "FAIL" if error else result.get("status", "UNKNOWN"),
                "started_at_utc": started,
                "finished_at_utc": now_utc(),
                "service_command": command,
                "service_exit_code_after_controlled_shutdown": service_exit,
                "service_log": str(log),
            },
        )
        write_artifact_hashes(output)
    if error is not None:
        raise error
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    context = load_and_validate()
    receipt = preflight(context)
    if args.preflight_only:
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    result = run(context)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
