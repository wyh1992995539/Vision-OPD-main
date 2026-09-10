#!/usr/bin/env python3
"""Run the frozen MMStar Base max-token diagnostic and conditional full gate."""

from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from eval.paper_aligned_common import (
    append_jsonl,
    checkpoint_identity,
    judge_complete,
    load_config,
    now_utc,
    prediction_complete,
    read_jsonl_map,
    record_key,
    require_frozen_base_identity,
    sha256_file,
    write_json,
    write_jsonl_map,
)
from eval.run_paper_aligned_eval import run_one as inference_one
from eval.run_paper_aligned_judge import run_one as judge_one
from eval.score_paper_aligned import load_mathruler, merge_judge, rule_score


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/mmstar_base_max_tokens_diagnostic.yaml"
FROZEN_CONFIG_SHA256 = "166c02f1f1dbc8e537024703058c64993ae1f4d6a7f35907a3a958bfd811bc1d"


def load_frozen() -> tuple[dict[str, Any], Path, dict[str, Any]]:
    if sha256_file(CONFIG) != FROZEN_CONFIG_SHA256:
        raise ValueError("MMStar diagnostic config SHA256 changed")
    diag = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    r3_path, r3 = load_config(diag["source"]["r3_config"])
    if sha256_file(r3_path) != diag["source"]["r3_config_sha256"]:
        raise ValueError("source R3 config SHA256 mismatch")
    return diag, r3_path, r3


def load_mmstar_tasks(r3: dict[str, Any]) -> dict[str, dict[str, Any]]:
    data_path = Path(r3["benchmarks"]["mmstar"]["converted_json"])
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    tasks = {}
    for row in rows:
        task = {
            "benchmark": "mmstar",
            "view": "full",
            "row": row,
            "image_path": Path(row["images"][0]),
        }
        key = record_key(
            {
                "benchmark": "mmstar",
                "view": "full",
                "sample_uid": row["sample_uid"],
            }
        )
        tasks[key] = task
    if len(tasks) != 1500:
        raise ValueError(f"expected 1500 MMStar tasks, got {len(tasks)}")
    return tasks


def wait_ready(process: subprocess.Popen, model_id: str, log: Path) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log.read_text(errors="replace")[-5000:] if log.exists() else ""
            raise RuntimeError(f"vLLM exited with {process.returncode}: {tail}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
                data = json.load(response)
            if model_id in [item["id"] for item in data.get("data", [])]:
                print(f"SERVICE_READY model={model_id}", flush=True)
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("vLLM startup timed out")


def stop_service(process: subprocess.Popen) -> int:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            return process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    return process.wait()


def run_inference_stage(
    name: str,
    selected: dict[str, dict[str, Any]],
    max_tokens: int,
    out: Path,
    r3: dict[str, Any],
    model_id: str,
    get_client: Any,
    workers: int,
) -> dict[str, dict[str, Any]]:
    stage = out / name
    stage.mkdir(parents=True, exist_ok=True)
    path = stage / "predictions.jsonl"
    records, _ = read_jsonl_map(path, complete=prediction_complete)
    records = {key: value for key, value in records.items() if key in selected}
    write_jsonl_map(path, records)
    config = copy.deepcopy(r3)
    config["generation"]["max_tokens"] = max_tokens
    pending = [task for key, task in selected.items() if not prediction_complete(records.get(key, {}))]
    print(
        f"INFERENCE_STAGE name={name} max_tokens={max_tokens} total={len(selected)} "
        f"complete={len(selected)-len(pending)} pending={len(pending)}",
        flush=True,
    )
    completed = 0
    failures = 0
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    inference_one,
                    task,
                    get_client=get_client,
                    config=config,
                    model_id=model_id,
                ): task
                for task in pending
            }
            for future in as_completed(future_map):
                result = future.result()
                key = record_key(result)
                records[key] = result
                append_jsonl(path, result)
                completed += 1
                failures += int(not prediction_complete(result))
                if completed % 10 == 0 or completed == len(pending):
                    print(
                        f"INFERENCE_PROGRESS name={name} {completed}/{len(pending)} "
                        f"failures={failures}",
                        flush=True,
                    )
    write_jsonl_map(path, records)
    metrics = {
        "schema_version": 1,
        "stage": name,
        "max_tokens": max_tokens,
        "request_count": len(records),
        "successful_count": sum(prediction_complete(x) for x in records.values()),
        "error_count": sum(not prediction_complete(x) for x in records.values()),
        "finish_reason_length_count": sum(
            str(x.get("finish_reason") or "").casefold() == "length"
            for x in records.values()
        ),
        "completion_tokens": sum(int(x.get("completion_tokens") or 0) for x in records.values()),
        "updated_at_utc": now_utc(),
    }
    write_json(stage / "inference_metrics.json", metrics)
    return records


def score_stage(
    name: str,
    predictions: dict[str, dict[str, Any]],
    out: Path,
    r3: dict[str, Any],
    model_id: str,
    get_client: Any,
    workers: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    stage = out / name
    config = copy.deepcopy(r3)
    config["judge"]["mcq_parser_revision"] = 4
    grader = load_mathruler()
    scores = {
        key: rule_score(prediction, config, grader)
        for key, prediction in predictions.items()
    }
    judge_path = stage / "judge_results.jsonl"
    judges, _ = read_jsonl_map(judge_path, complete=judge_complete)
    required = {key for key, score in scores.items() if score["judge_required"]}
    judges = {
        key: value
        for key, value in judges.items()
        if key in required and judge_complete(value)
    }
    pending = [key for key in sorted(required) if not judge_complete(judges.get(key, {}))]
    print(
        f"JUDGE_STAGE name={name} required={len(required)} "
        f"complete={len(required)-len(pending)} pending={len(pending)}",
        flush=True,
    )
    completed = 0
    failures = 0
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    judge_one,
                    predictions[key],
                    get_client=get_client,
                    config=r3,
                    model_id=model_id,
                ): key
                for key in pending
            }
            for future in as_completed(future_map):
                result = future.result()
                key = record_key(result)
                result["diagnostic_stage"] = name
                judges[key] = result
                append_jsonl(judge_path, result)
                completed += 1
                failures += int(bool(result.get("error")))
                if completed % 10 == 0 or completed == len(pending):
                    print(
                        f"JUDGE_PROGRESS name={name} {completed}/{len(pending)} "
                        f"failures={failures}",
                        flush=True,
                    )
    write_jsonl_map(judge_path, judges)
    merged = {
        key: merge_judge(score, judges.get(key))
        for key, score in scores.items()
    }
    write_jsonl_map(stage / "scores.jsonl", merged)
    summary = {
        "schema_version": 1,
        "stage": name,
        "request_count": len(merged),
        "correct": sum(bool(x["final_is_correct"]) for x in merged.values()),
        "incorrect": sum(
            x["score_status"] == "scored" and not x["final_is_correct"]
            for x in merged.values()
        ),
        "pending_judge": sum(x["score_status"] != "scored" for x in merged.values()),
        "judge_required": len(required),
        "judge_failures": sum(bool(x.get("error")) for x in judges.values()),
        "route_counts": {
            route: sum(x["rule_source"] == route for x in merged.values())
            for route in sorted({x["rule_source"] for x in merged.values()})
        },
        "updated_at_utc": now_utc(),
    }
    write_json(stage / "score_summary.json", summary)
    return merged, summary


def write_artifact_hashes(out: Path) -> None:
    files = sorted(
        path for path in out.rglob("*")
        if path.is_file()
        and path.name != "artifact_sha256.txt"
        and path.name != "vllm_server.log"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(out)}" for path in files]
    (out / "artifact_sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    diag, r3_path, r3 = load_frozen()
    out = ROOT / diag["paths"]["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_mmstar_tasks(r3)

    r4 = ROOT / diag["source"]["r4_base_run"]
    baseline_predictions, _ = read_jsonl_map(
        r4 / "predictions.jsonl", complete=prediction_complete
    )
    baseline_scores, _ = read_jsonl_map(r4 / "scores.jsonl")
    source_paths = [
        r4 / "predictions.jsonl",
        r4 / "scores.jsonl",
        r4 / "summary.json",
        ROOT / diag["source"]["r3_config"],
    ]
    source_before = {str(path): sha256_file(path) for path in source_paths}

    selected_keys = {
        key
        for key, prediction in baseline_predictions.items()
        if prediction["benchmark"] == "mmstar"
        and str(prediction.get("finish_reason") or "").casefold() == "length"
    }
    if len(selected_keys) != int(diag["source"]["expected_selected_samples"]):
        raise ValueError(f"expected 163 selected samples, got {len(selected_keys)}")
    selected = {key: tasks[key] for key in sorted(selected_keys)}
    nonselected_correct = sum(
        bool(score["final_is_correct"])
        for key, score in baseline_scores.items()
        if score["benchmark"] == "mmstar" and key not in selected_keys
    )
    if nonselected_correct != 1000:
        raise ValueError(f"unexpected nonselected baseline correct: {nonselected_correct}")

    identity = checkpoint_identity(Path(diag["model"]["path"]))
    require_frozen_base_identity(identity, r3)
    model_id = diag["model"]["served_model_name"]
    command = [
        "/root/miniconda3/envs/vision-opd/bin/vllm",
        "serve",
        diag["model"]["path"],
        "--served-model-name",
        model_id,
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.75",
        "--trust-remote-code",
        "--additional-config",
        json.dumps({"gdn_prefill_backend": "triton"}),
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="8")
    log = out / "vllm_server.log"
    stream = log.open("w", encoding="utf-8")
    service = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    service_exit = None
    started = now_utc()
    final = {}
    try:
        print("SERVICE_START diagnostic=mmstar_max_tokens", flush=True)
        wait_ready(service, model_id, log)
        from openai import OpenAI

        local = threading.local()

        def get_client() -> OpenAI:
            client = getattr(local, "client", None)
            if client is None:
                client = OpenAI(
                    api_key="EMPTY",
                    base_url="http://127.0.0.1:8000/v1",
                    timeout=float(r3["execution"]["request_timeout_seconds"]),
                )
                local.client = client
            return client

        workers = int(diag["inference"]["workers"])
        p2048 = run_inference_stage(
            "selected_2048",
            selected,
            int(diag["inference"]["initial_max_tokens"]),
            out,
            r3,
            model_id,
            get_client,
            workers,
        )
        s2048, sum2048 = score_stage(
            "selected_2048", p2048, out, r3, model_id, get_client, workers
        )
        counter2048 = nonselected_correct + sum2048["correct"]
        gate_min = int(diag["gate"]["minimum_counterfactual_correct"])
        pass2048 = (
            sum2048["pending_judge"] == 0
            and counter2048 >= gate_min
            and len(p2048) == len(selected)
        )
        residual_keys = {
            key
            for key, prediction in p2048.items()
            if str(prediction.get("finish_reason") or "").casefold() == "length"
        }
        stages = {
            "selected_2048": {
                **sum2048,
                "counterfactual_full_correct": counter2048,
                "counterfactual_full_accuracy": counter2048 / 1500,
                "gate_pass": pass2048,
                "residual_length_count": len(residual_keys),
            }
        }
        chosen_max_tokens = 2048 if pass2048 else None
        chosen_counter = counter2048

        if not pass2048 and residual_keys:
            residual = {key: tasks[key] for key in sorted(residual_keys)}
            p4096 = run_inference_stage(
                "residual_4096",
                residual,
                int(diag["inference"]["residual_max_tokens"]),
                out,
                r3,
                model_id,
                get_client,
                workers,
            )
            s4096, sum4096 = score_stage(
                "residual_4096", p4096, out, r3, model_id, get_client, workers
            )
            hybrid = dict(s2048)
            hybrid.update(s4096)
            hybrid_correct = sum(bool(x["final_is_correct"]) for x in hybrid.values())
            chosen_counter = nonselected_correct + hybrid_correct
            pass4096 = (
                sum4096["pending_judge"] == 0
                and chosen_counter >= gate_min
                and len(p4096) == len(residual)
            )
            stages["residual_4096"] = {
                **sum4096,
                "hybrid_selected_correct": hybrid_correct,
                "counterfactual_full_correct": chosen_counter,
                "counterfactual_full_accuracy": chosen_counter / 1500,
                "gate_pass": pass4096,
                "residual_length_count": sum(
                    str(x.get("finish_reason") or "").casefold() == "length"
                    for x in p4096.values()
                ),
            }
            if pass4096:
                chosen_max_tokens = 4096

        full_run = None
        if chosen_max_tokens is not None:
            full_predictions = run_inference_stage(
                f"full_{chosen_max_tokens}",
                tasks,
                chosen_max_tokens,
                out,
                r3,
                model_id,
                get_client,
                workers,
            )
            full_scores, full_summary = score_stage(
                f"full_{chosen_max_tokens}",
                full_predictions,
                out,
                r3,
                model_id,
                get_client,
                workers,
            )
            full_run = {
                **full_summary,
                "accuracy": full_summary["correct"] / 1500,
                "finish_reason_length_count": sum(
                    str(x.get("finish_reason") or "").casefold() == "length"
                    for x in full_predictions.values()
                ),
                "max_tokens": chosen_max_tokens,
            }

        source_after = {str(path): sha256_file(path) for path in source_paths}
        final = {
            "schema_version": 1,
            "experiment_id": diag["experiment"]["id"],
            "status": "PASS",
            "decision": (
                "FULL_RUN_COMPLETED"
                if full_run is not None
                else "STOP_AFTER_DIAGNOSTIC_GATE_NOT_MET"
            ),
            "created_at_utc": now_utc(),
            "baseline": {
                "r4_correct": 1020,
                "r4_accuracy": 0.68,
                "selected_length_count": len(selected),
                "selected_length_correct": 20,
                "nonselected_correct": nonselected_correct,
            },
            "gate": diag["gate"],
            "stages": stages,
            "chosen_full_max_tokens": chosen_max_tokens,
            "full_run": full_run,
            "source_files_unchanged": source_before == source_after,
            "source_sha256": source_after,
            "config_sha256": FROZEN_CONFIG_SHA256,
            "r3_config_sha256": sha256_file(r3_path),
            "checkpoint_identity": identity,
            "parser_sha256": sha256_file(ROOT / "eval/mcq_parser_r4.py"),
            "scorer_sha256": sha256_file(ROOT / "eval/score_paper_aligned.py"),
            "runner_sha256": sha256_file(Path(__file__)),
            "limitation_statement": diag["limitations"]["statement"],
        }
        write_json(out / "gate_receipt.json", final)
    finally:
        service_exit = stop_service(service)
        stream.close()

    session = {
        "schema_version": 1,
        "status": final.get("status", "FAIL"),
        "started_at_utc": started,
        "finished_at_utc": now_utc(),
        "service_command": command,
        "service_exit_code_after_controlled_shutdown": service_exit,
        "service_log": str(log),
    }
    write_json(out / "session.json", session)
    write_artifact_hashes(out)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
