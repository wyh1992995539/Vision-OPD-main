#!/usr/bin/env python3
"""Prepare, resume, and finalize the frozen R4 score-only re-evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from eval.paper_aligned_common import (
    append_jsonl,
    checkpoint_identity,
    judge_complete,
    judge_prompt,
    load_config,
    now_utc,
    prediction_complete,
    read_jsonl_map,
    record_key,
    require_frozen_base_identity,
    resolve_path,
    sha256_file,
    write_json,
    write_jsonl_map,
)
from eval.run_paper_aligned_judge import run_one
from eval.score_paper_aligned import merge_judge, rule_score


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs/benchmark_eval_paper_basejudge_r4_rescore.yaml"
FROZEN_R4_CONFIG_SHA256 = "7578a6561f3e20a50c9c386e6aa395b3f66a09d5abbfff399481f1053faf35ca"


def load_r4(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else (ROOT / path).resolve()
    actual = sha256_file(resolved)
    if actual != FROZEN_R4_CONFIG_SHA256:
        raise ValueError(
            f"R4 config SHA256 changed: expected {FROZEN_R4_CONFIG_SHA256}, got {actual}"
        )
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if config["protocol"]["experiment_id"] != "E-PAPER-BASEJUDGE-R4-001":
        raise ValueError("wrong R4 experiment_id")
    if int(config["scoring"]["mcq_parser_revision"]) != 4:
        raise ValueError("R4 requires mcq_parser_revision=4")
    return config


def source_hashes(source: Path) -> dict[str, str]:
    names = ("run_manifest.json", "predictions.jsonl", "judge_results.jsonl", "scores.jsonl")
    result = {}
    for name in names:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = sha256_file(path)
    return result


def valid_judge(record: dict[str, Any] | None, prediction: dict[str, Any], r3: dict[str, Any]) -> bool:
    if not record or not judge_complete(record):
        return False
    return (
        record.get("judge_model_id") == r3["judge"]["model"]["served_model_name"]
        and record.get("judge_prompt") == judge_prompt(r3, prediction)
    )


def stat(items: list[dict[str, Any]]) -> dict[str, Any]:
    pending = sum(item["score_status"] != "scored" for item in items)
    correct = sum(bool(item["final_is_correct"]) for item in items)
    return {
        "total": len(items),
        "correct": correct,
        "incorrect": len(items) - correct - pending,
        "pending_judge": pending,
        "accuracy": correct / len(items) if items and pending == 0 else None,
        "accuracy_lower_bound": correct / len(items) if items else None,
        "accuracy_upper_bound": (correct + pending) / len(items) if items else None,
    }


def write_hashes(out: Path, names: list[str]) -> None:
    lines = []
    for name in names:
        path = out / name
        if path.is_file():
            lines.append(f"{sha256_file(path)}  {name}")
    (out / "artifact_sha256.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )


def prepare_role(
    role: str,
    r4: dict[str, Any],
    r3_path: Path,
    r3: dict[str, Any],
) -> dict[str, Any]:
    source = resolve_path(r4["source_runs"][role]["path"])
    out = resolve_path(r4["paths"]["run_root"]) / role
    out.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes(source)

    source_manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("run_mode") != "formal":
        raise ValueError(f"{role}: source run is not formal")
    expected = int(r4["expected"]["requests_per_role"])

    predictions, prediction_stats = read_jsonl_map(
        source / "predictions.jsonl", complete=prediction_complete
    )
    old_scores, old_score_stats = read_jsonl_map(source / "scores.jsonl")
    old_judges, old_judge_stats = read_jsonl_map(
        source / "judge_results.jsonl", complete=judge_complete
    )
    if len(predictions) != expected or len(old_scores) != expected:
        raise ValueError(f"{role}: source count mismatch")

    prediction_copy = out / r4["paths"]["predictions_name"]
    if prediction_copy.exists():
        if sha256_file(prediction_copy) != hashes["predictions.jsonl"]:
            raise ValueError(f"{role}: R4 prediction copy differs from frozen source")
    else:
        shutil.copyfile(source / "predictions.jsonl", prediction_copy)

    current_judges, current_stats = read_jsonl_map(
        out / r4["paths"]["judge_results_name"], complete=judge_complete
    )
    scoring_config = copy.deepcopy(r3)
    scoring_config["judge"]["mcq_parser_revision"] = 4

    final_scores: dict[str, dict[str, Any]] = {}
    selected_judges: dict[str, dict[str, Any]] = {}
    pending_predictions: dict[str, dict[str, Any]] = {}
    provenance = Counter()

    for key, prediction in predictions.items():
        old = old_scores.get(key)
        if old is None:
            raise ValueError(f"{role}: missing source score for {key}")
        benchmark = prediction["benchmark"]
        if benchmark not in set(r4["scoring"]["mcq_benchmarks"]) or old["rule_source"] == "mathruler":
            score = copy.deepcopy(old)
            score["schema_version"] = 2
            score["r4_derivation"] = "preserved_r3_unaffected_or_mathruler"
            provenance["preserved_r3_score"] += 1
        else:
            score = rule_score(prediction, scoring_config, lambda reference, answer: False)
            score["r4_derivation"] = "rescored_with_mcq_parser_r4"
            provenance[score["rule_source"]] += 1

        if score["judge_required"]:
            chosen = None
            candidate = current_judges.get(key)
            if valid_judge(candidate, prediction, r3):
                chosen = copy.deepcopy(candidate)
                chosen["r4_judge_provenance"] = candidate.get(
                    "r4_judge_provenance", "new_r4"
                )
            else:
                candidate = old_judges.get(key)
                if valid_judge(candidate, prediction, r3):
                    chosen = copy.deepcopy(candidate)
                    chosen["r4_judge_provenance"] = "reused_r3_exact_prompt"
            if chosen is not None:
                selected_judges[key] = chosen
                score = merge_judge(score, chosen)
                provenance[chosen["r4_judge_provenance"]] += 1
            else:
                pending_predictions[key] = prediction
                provenance["new_judge_required"] += 1
        final_scores[key] = score

    write_jsonl_map(out / r4["paths"]["judge_results_name"], selected_judges)
    write_jsonl_map(out / r4["paths"]["scores_name"], final_scores)
    write_jsonl_map(out / r4["paths"]["pending_name"], pending_predictions)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for score in final_scores.values():
        groups[f"{score['benchmark']}/{score['view']}"].append(score)
    actual_groups = {name: len(items) for name, items in groups.items()}
    if actual_groups != r4["expected"]["groups"]:
        raise ValueError(f"{role}: group counts mismatch: {actual_groups}")

    pending_count = len(pending_predictions)
    summary = {
        "schema_version": 2,
        "experiment_id": r4["protocol"]["experiment_id"],
        "protocol_revision": 4,
        "generated_at_utc": now_utc(),
        "decision_status": "complete" if pending_count == 0 else "pending_judge",
        "model_role": role,
        "model_id": source_manifest["model_id"],
        "request_count": len(final_scores),
        "groups": {name: stat(items) for name, items in sorted(groups.items())},
        "scoring_pipeline": dict(sorted(provenance.items())),
        "pending_judge_count": pending_count,
        "source_r3_run": str(source),
        "source_r3_hashes": hashes,
        "frozen_r4_config_sha256": FROZEN_R4_CONFIG_SHA256,
        "limitation_statement": r4["limitations"]["statement"],
    }
    write_json(out / r4["paths"]["summary_name"], summary)

    manifest = {
        "schema_version": 1,
        "experiment_id": r4["protocol"]["experiment_id"],
        "protocol_revision": 4,
        "created_at_utc": now_utc(),
        "model_role": role,
        "model_id": source_manifest["model_id"],
        "status": summary["decision_status"],
        "source_r3_run": str(source),
        "source_r3_hashes": hashes,
        "source_r3_config": str(r3_path),
        "source_r3_config_sha256": sha256_file(r3_path),
        "r4_config": str(DEFAULT_CONFIG),
        "r4_config_sha256": FROZEN_R4_CONFIG_SHA256,
        "parser_path": str((ROOT / "eval/mcq_parser_r4.py").resolve()),
        "parser_sha256": sha256_file(ROOT / "eval/mcq_parser_r4.py"),
        "scorer_path": str((ROOT / "eval/score_paper_aligned.py").resolve()),
        "scorer_sha256": sha256_file(ROOT / "eval/score_paper_aligned.py"),
        "prediction_count": len(predictions),
        "pending_judge_count": pending_count,
        "source_compaction": {
            "predictions": prediction_stats,
            "scores": old_score_stats,
            "judges": old_judge_stats,
            "r4_judges": current_stats,
        },
    }
    write_json(out / r4["paths"]["manifest_name"], manifest)
    write_hashes(
        out,
        [
            r4["paths"]["predictions_name"],
            r4["paths"]["judge_results_name"],
            r4["paths"]["scores_name"],
            r4["paths"]["summary_name"],
            r4["paths"]["pending_name"],
            r4["paths"]["manifest_name"],
        ],
    )
    return {
        "role": role,
        "source": source,
        "out": out,
        "source_hashes": hashes,
        "pending": pending_predictions,
        "summary": summary,
    }


def wait_ready(process: subprocess.Popen, url: str, expected_id: str, log: Path) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log.read_text(errors="replace")[-5000:] if log.exists() else ""
            raise RuntimeError(f"vLLM exited with {process.returncode}: {tail}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.load(response)
            if expected_id in [item["id"] for item in data.get("data", [])]:
                print(f"SERVICE_READY model={expected_id}", flush=True)
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


def run_pending(
    states: list[dict[str, Any]],
    r4: dict[str, Any],
    r3: dict[str, Any],
) -> dict[str, Any]:
    all_jobs = [
        (state, prediction)
        for state in states
        for prediction in state["pending"].values()
    ]
    if not all_jobs:
        return {
            "status": "SKIPPED",
            "reason": "no_pending_judge",
            "submitted_request_count": 0,
        }

    host, port = "127.0.0.1", 8000
    try:
        urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=2)
        raise RuntimeError(f"port {port} is already occupied")
    except urllib.error.URLError:
        pass

    judge_path = resolve_path(r4["judge"]["checkpoint_path"])
    identity = checkpoint_identity(judge_path)
    require_frozen_base_identity(identity, r3)
    model_id = str(r4["judge"]["served_model_name"])
    vllm = "/root/miniconda3/envs/vision-opd/bin/vllm"
    command = [
        vllm,
        "serve",
        str(judge_path),
        "--served-model-name",
        model_id,
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        str(r4["judge"]["gpu_memory_utilization"]),
        "--trust-remote-code",
        "--additional-config",
        json.dumps({"gdn_prefill_backend": r4["judge"]["gdn_prefill_backend"]}),
        "--host",
        host,
        "--port",
        str(port),
    ]
    root = resolve_path(r4["paths"]["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    log = root / "judge_server.log"
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="8")
    started = now_utc()
    service = None
    stream = None
    completed = 0
    failures = 0
    error = None
    service_exit = None
    try:
        stream = log.open("w", encoding="utf-8")
        service = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"SERVICE_START pending={len(all_jobs)}", flush=True)
        wait_ready(service, f"http://{host}:{port}/v1/models", model_id, log)

        from openai import OpenAI

        local = threading.local()

        def get_client() -> OpenAI:
            client = getattr(local, "client", None)
            if client is None:
                client = OpenAI(
                    api_key="EMPTY",
                    base_url=f"http://{host}:{port}/v1",
                    timeout=float(r3["judge"]["request_timeout_seconds"]),
                )
                local.client = client
            return client

        workers = int(r4["judge"]["workers"])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    run_one,
                    prediction,
                    get_client=get_client,
                    config=r3,
                    model_id=model_id,
                ): (state, prediction)
                for state, prediction in all_jobs
            }
            for future in as_completed(future_map):
                state, prediction = future_map[future]
                result = future.result()
                result["r4_judge_provenance"] = "new_r4"
                append_jsonl(
                    state["out"] / r4["paths"]["judge_results_name"], result
                )
                completed += 1
                failures += int(bool(result.get("error")))
                if completed % 10 == 0 or completed == len(all_jobs):
                    print(
                        f"R4_JUDGE_PROGRESS {completed}/{len(all_jobs)} failures={failures}",
                        flush=True,
                    )
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if service is not None:
            service_exit = stop_service(service)
        if stream is not None:
            stream.close()

    return {
        "schema_version": 1,
        "status": "PASS" if error is None else "FAIL",
        "started_at_utc": started,
        "finished_at_utc": now_utc(),
        "submitted_request_count": len(all_jobs),
        "completed_request_count": completed,
        "finalized_failure_count": failures,
        "judge_model_id": model_id,
        "judge_checkpoint_identity": identity,
        "service_command": command,
        "service_log": str(log),
        "service_exit_code_after_controlled_shutdown": service_exit,
        "error": error,
    }


def validate_final(
    states: list[dict[str, Any]],
    r4: dict[str, Any],
    initial_hashes: dict[str, dict[str, str]],
) -> dict[str, Any]:
    checks = {}
    for state in states:
        role = state["role"]
        out = state["out"]
        summary = json.loads((out / r4["paths"]["summary_name"]).read_text())
        scores, stats = read_jsonl_map(out / r4["paths"]["scores_name"])
        explicit_mismatches = [
            item
            for item in scores.values()
            if item.get("rule_source") == "mcq_option_mismatch_r4"
        ]
        checks[role] = {
            "summary_complete": summary["decision_status"] == "complete",
            "request_count": len(scores),
            "score_duplicates": stats["duplicate_keys"],
            "pending_judge_count": summary["pending_judge_count"],
            "explicit_mismatch_count": len(explicit_mismatches),
            "explicit_mismatch_all_incorrect_without_judge": all(
                not item["final_is_correct"] and not item["judge_required"]
                for item in explicit_mismatches
            ),
            "source_r3_unchanged": source_hashes(state["source"]) == initial_hashes[role],
        }
    passed = all(
        item["summary_complete"]
        and item["request_count"] == int(r4["expected"]["requests_per_role"])
        and item["score_duplicates"] == 0
        and item["pending_judge_count"] == 0
        and item["explicit_mismatch_all_incorrect_without_judge"]
        and item["source_r3_unchanged"]
        for item in checks.values()
    )
    return {
        "schema_version": 1,
        "experiment_id": r4["protocol"]["experiment_id"],
        "validated_at_utc": now_utc(),
        "status": "PASS" if passed else "FAIL",
        "frozen_r4_config_sha256": FROZEN_R4_CONFIG_SHA256,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    r4 = load_r4(args.config)
    r3_path, r3 = load_config(r4["protocol"]["source_r3_config"])
    if sha256_file(r3_path) != r4["protocol"]["source_r3_config_sha256"]:
        raise ValueError("frozen R3 config SHA256 mismatch")
    roles = list(r4["source_runs"])
    initial_hashes = {
        role: source_hashes(resolve_path(r4["source_runs"][role]["path"]))
        for role in roles
    }

    states = [prepare_role(role, r4, r3_path, r3) for role in roles]
    print(
        "R4_PREPARED "
        + " ".join(f"{state['role']}={len(state['pending'])}" for state in states),
        flush=True,
    )
    if args.prepare_only:
        return 0

    session = run_pending(states, r4, r3)
    root = resolve_path(r4["paths"]["run_root"])
    write_json(root / r4["paths"]["session_name"], session)

    states = [prepare_role(role, r4, r3_path, r3) for role in roles]
    validation = validate_final(states, r4, initial_hashes)
    write_json(root / r4["paths"]["validation_name"], validation)
    write_json(
        root / "freeze_receipt.json",
        {
            "schema_version": 1,
            "experiment_id": r4["protocol"]["experiment_id"],
            "created_at_utc": now_utc(),
            "status": validation["status"],
            "config_path": str(args.config.resolve()),
            "config_sha256": FROZEN_R4_CONFIG_SHA256,
            "parser_sha256": sha256_file(ROOT / "eval/mcq_parser_r4.py"),
            "scorer_sha256": sha256_file(ROOT / "eval/score_paper_aligned.py"),
            "runner_sha256": sha256_file(Path(__file__)),
            "source_r3_config_sha256": sha256_file(r3_path),
            "session_status": session["status"],
            "validation_status": validation["status"],
        },
    )
    for state in states:
        write_hashes(
            state["out"],
            [
                r4["paths"]["predictions_name"],
                r4["paths"]["judge_results_name"],
                r4["paths"]["scores_name"],
                r4["paths"]["summary_name"],
                r4["paths"]["pending_name"],
                r4["paths"]["manifest_name"],
            ],
        )
    root_hash_names = [
        r4["paths"]["session_name"],
        r4["paths"]["validation_name"],
        "freeze_receipt.json",
    ]
    write_hashes(root, root_hash_names)
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    if validation["status"] != "PASS":
        raise RuntimeError("R4 final validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
