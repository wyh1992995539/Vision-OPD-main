#!/usr/bin/env python3
"""Run the resumable frozen Qwen3.5-4B Base Judge stage."""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from eval.paper_aligned_common import (
    append_jsonl,
    checkpoint_identity,
    judge_complete,
    judge_prompt,
    judge_request_kwargs,
    load_config,
    normalize_judge_decision,
    now_utc,
    prediction_complete,
    read_jsonl_map,
    record_key,
    require_formal_manifest_comparable_with_base,
    require_frozen_base_identity,
    require_frozen_r3_config,
    resolve_path,
    update_cost_from_sessions,
    usage_value,
    write_json,
    write_jsonl_map,
)
from eval.score_paper_aligned import load_mathruler, rule_score


def build_pending_scores(
    predictions: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    grader = load_mathruler()
    return {
        key: rule_score(prediction, config, grader)
        for key, prediction in predictions.items()
    }


def update_judge_resume(
    out: Path,
    config: dict[str, Any],
    required_keys: set[str],
    judge_records: dict[str, dict[str, Any]],
    load_stats: dict[str, int],
) -> None:
    completed = sum(judge_complete(judge_records.get(key, {})) for key in required_keys)
    failures = sum(
        judge_complete(judge_records.get(key, {}))
        and bool(judge_records.get(key, {}).get("error"))
        for key in required_keys
    )
    write_json(
        out / "judge_resume_status.json",
        {
            "schema_version": 1,
            "updated_at_utc": now_utc(),
            "required_judge_count": len(required_keys),
            "completed_judge_count": completed,
            "finalized_failure_count": failures,
            "remaining_judge_count": len(required_keys) - completed,
            "duplicate_judge_keys_compacted": load_stats["duplicate_keys"],
            "malformed_judge_lines_compacted": load_stats["malformed_lines"],
            "resume_complete": completed == len(required_keys),
        },
    )


def run_one(
    prediction: dict[str, Any],
    *,
    get_client: Any,
    config: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    prompt = judge_prompt(config, prediction)
    raw = ""
    normalized = None
    error = None
    prompt_tokens = completion_tokens = 0
    attempt = 0
    max_retries = int(config["judge"]["max_retries"])
    started = time.perf_counter()
    for attempt in range(1, max_retries + 1):
        try:
            response = get_client().chat.completions.create(
                **judge_request_kwargs(model_id=model_id, prompt=prompt, config=config)
            )
            raw = str(response.choices[0].message.content or "").strip()
            prompt_tokens = usage_value(response, "prompt_tokens")
            completion_tokens = usage_value(response, "completion_tokens")
            normalized = normalize_judge_decision(raw)
            if normalized is None:
                raise ValueError(f"Judge output is not exactly Yes or No: {raw!r}")
            error = None
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(float(attempt))
    return {
        "schema_version": 1,
        "benchmark": prediction["benchmark"],
        "view": prediction["view"],
        "sample_uid": prediction["sample_uid"],
        "judge_model_id": model_id,
        "judge_prompt": prompt,
        "raw_judge_output": raw,
        "normalized_decision": normalized,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_seconds": round(time.perf_counter() - started, 6),
        "retry_count": max(0, attempt - 1),
        "error": error,
        "finalized": True,
        "judged_at_utc": now_utc(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--judge-model-id")
    parser.add_argument("--judge-model-path")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--retry-finalized-failures", action="store_true")
    args = parser.parse_args()

    config_path, config = load_config(args.config)
    out = resolve_path(args.input_dir)
    manifest_path = out / config["paths"]["run_manifest_name"]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "E-PAPER-BASEJUDGE-001":
        raise ValueError("input directory is not a paper-aligned run")
    if manifest.get("run_mode") == "formal":
        require_frozen_r3_config(config_path)
    require_formal_manifest_comparable_with_base(manifest, config)

    predictions, prediction_stats = read_jsonl_map(
        out / config["paths"]["predictions_name"], complete=prediction_complete
    )
    expected = int(manifest["expected_request_count"])
    if len(predictions) != expected:
        raise ValueError(f"expected {expected} unique predictions, found {len(predictions)}")
    scores = build_pending_scores(predictions, config)
    required_keys = {key for key, score in scores.items() if score["judge_required"]}

    judge_path = out / config["paths"]["judge_results_name"]
    judge_records, load_stats = read_jsonl_map(judge_path, complete=judge_complete)
    judge_records = {key: value for key, value in judge_records.items() if key in required_keys}
    if args.retry_finalized_failures:
        judge_records = {
            key: value
            for key, value in judge_records.items()
            if not value.get("error")
        }
    write_jsonl_map(judge_path, judge_records)
    update_judge_resume(out, config, required_keys, judge_records, load_stats)

    model_id = args.judge_model_id or str(config["judge"]["model"]["served_model_name"])
    expected_model_id = str(config["judge"]["model"]["served_model_name"])
    if model_id != expected_model_id:
        raise ValueError(f"Judge served model must be {expected_model_id!r}")
    model_path = resolve_path(args.judge_model_path or config["judge"]["model"]["path"])
    identity = checkpoint_identity(model_path)
    require_frozen_base_identity(identity, config)
    judge_identity_path = out / "judge_manifest.json"
    proposed_identity = {
        "schema_version": 1,
        "experiment_id": "E-PAPER-BASEJUDGE-001",
        "judge_model_id": model_id,
        "judge_checkpoint_identity": identity,
        "judge_role": config["judge"]["model"]["role"],
        "system_prompt": None,
        "enable_thinking": False,
        "temperature": 0,
        "max_tokens": 2048,
        "required_judge_count": len(required_keys),
        "paper_judge": config["judge"]["paper_model"],
        "substitution_reason": config["judge"]["substitution_reason"],
    }
    if judge_identity_path.is_file():
        existing_identity = json.loads(judge_identity_path.read_text(encoding="utf-8"))
        if existing_identity != proposed_identity:
            raise ValueError("Judge manifest changed; refusing to mix Judge identities")
    else:
        write_json(judge_identity_path, proposed_identity)

    pending_keys = [
        key for key in sorted(required_keys)
        if not judge_complete(judge_records.get(key, {}))
    ]
    print(
        f"paper-aligned Judge: required={len(required_keys)} "
        f"complete={len(required_keys)-len(pending_keys)} pending={len(pending_keys)}",
        flush=True,
    )
    if args.prepare_only:
        return

    from openai import OpenAI

    local = threading.local()

    def get_client() -> OpenAI:
        client = getattr(local, "client", None)
        if client is None:
            client = OpenAI(
                api_key=args.api_key,
                base_url=args.api_base,
                timeout=float(config["judge"]["request_timeout_seconds"]),
            )
            local.client = client
        return client

    workers = args.workers or int(config["judge"]["parallel_workers"])
    if workers <= 0:
        raise ValueError("--workers must be positive")
    session_started_utc = now_utc()
    session_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_one,
                predictions[key],
                get_client=get_client,
                config=config,
                model_id=model_id,
            ): key
            for key in pending_keys
        }
        for index, future in enumerate(as_completed(futures), 1):
            item = future.result()
            append_jsonl(judge_path, item)
            judge_records[record_key(item)] = item
            print(
                f"[{index}/{len(pending_keys)}] {item['benchmark']}/{item['sample_uid']} "
                f"decision={item['normalized_decision']} error={bool(item['error'])}",
                flush=True,
            )
    append_jsonl(
        out / "run_sessions.jsonl",
        {
            "schema_version": 1,
            "stage": "judge",
            "started_at_utc": session_started_utc,
            "finished_at_utc": now_utc(),
            "wall_seconds": round(time.perf_counter() - session_started, 6),
            "submitted_request_count": len(pending_keys),
            "workers": workers,
            "model_id": model_id,
        },
    )
    update_cost_from_sessions(out, config)
    final_records, final_stats = read_jsonl_map(judge_path, complete=judge_complete)
    final_records = {key: value for key, value in final_records.items() if key in required_keys}
    write_jsonl_map(judge_path, final_records)
    update_judge_resume(out, config, required_keys, final_records, final_stats)
    completed = sum(judge_complete(final_records.get(key, {})) for key in required_keys)
    if completed != len(required_keys):
        raise RuntimeError(
            f"Judge incomplete: {completed}/{len(required_keys)}; rerun the same command to resume"
        )
    print(f"Judge complete: {completed}/{len(required_keys)}", flush=True)


if __name__ == "__main__":
    main()
