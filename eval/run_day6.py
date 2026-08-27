#!/usr/bin/env python3
"""Run resumable E-D6-001 full external evaluation under the frozen amendment."""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from eval.run_smoke import (
    final_answer, image_data_uri, record_key, resolve_path, sha256_file, write_records,
)

EXPECTED = {"zoombench/full": 845, "zoombench/crop": 845, "mmstar/full": 1500, "vstar/full": 191}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            result[record_key(item)] = item
        except (json.JSONDecodeError, KeyError):
            continue
    return result


def load_amendment(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    policy = value.get("effective_reporting_policy", {})
    if not (
        value.get("status") == "effective"
        and value.get("experiment_id") == "E-D6-001"
        and policy.get("vstar_summary_denominator") == 191
        and policy.get("vstar_official_request_count") == 191
        and policy.get("report_deduplicated_diagnostic_separately") is False
    ):
        raise ValueError("invalid V* reporting amendment: require official-only 191-sample policy")
    return value


def load_tasks(config: dict[str, Any], repo: Path) -> list[dict[str, Any]]:
    root = resolve_path(config["paths"]["data_root"], repo)
    tasks: list[dict[str, Any]] = []
    for benchmark in ("zoombench", "mmstar", "vstar"):
        rows = json.loads((root / "converted" / benchmark / f"{benchmark}.json").read_text(encoding="utf-8"))
        expected = int(config["benchmarks"][benchmark]["expected_sample_count"])
        if len(rows) != expected or len({row["sample_uid"] for row in rows}) != expected:
            raise ValueError(f"{benchmark}: converted data does not match frozen sample count")
        for view in (("full", "crop") if benchmark == "zoombench" else ("full",)):
            for row in rows:
                images = row.get("crop_images") if view == "crop" else row.get("images")
                if not images or not Path(str(images[0])).is_file():
                    raise FileNotFoundError(f"{benchmark}/{view}/{row['sample_uid']}: image missing")
                tasks.append({"benchmark": benchmark, "view": view, "row": row, "image": Path(str(images[0]))})
    actual = {name: sum(1 for t in tasks if f"{t['benchmark']}/{t['view']}" == name) for name in EXPECTED}
    if actual != EXPECTED:
        raise ValueError(f"unexpected full-evaluation workload: {actual}")
    return tasks


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Explicit legacy E-D5/E-D6 config path; R3 uses the paper-aligned entrypoints")
    parser.add_argument("--amendment", default="artifacts/runs/E-D6-001/preflight/vstar_reporting_amendment.yaml")
    parser.add_argument("--output-dir", default="artifacts/runs/E-D6-001/base")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-id", default="vision-opd-base")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    config_path = resolve_path(args.config, repo)
    amendment_path = resolve_path(args.amendment, repo)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    amendment = load_amendment(amendment_path)
    tasks = load_tasks(config, repo)
    out = resolve_path(args.output_dir, repo)
    predictions = out / "predictions.jsonl"
    records = read_jsonl(predictions)
    manifest = {
        "schema_version": 1, "experiment_id": "E-D6-001", "created_at_utc": now(),
        "config": str(config_path), "config_sha256_raw_bytes": sha256_file(config_path),
        "vstar_reporting_amendment": str(amendment_path),
        "vstar_reporting_amendment_sha256_raw_bytes": sha256_file(amendment_path),
        "vstar_reporting_policy": amendment["effective_reporting_policy"],
        "output_files": {"predictions": "predictions.jsonl", "scores": "scores.jsonl", "summary": "summary.json"},
        "expected_requests": EXPECTED, "expected_request_count": len(tasks), "resume_key": "benchmark\\0view\\0sample_uid",
    }
    write_json(out / "run_manifest.json", manifest)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", **manifest}, ensure_ascii=False, indent=2))
        return

    generation, system = config["generation"], config["prompt_and_image"]["system_prompt"]
    checkpoint = config["model_under_test"]["base_weight_sha256"]
    local = threading.local()
    def client() -> OpenAI:
        if not hasattr(local, "client"):
            local.client = OpenAI(api_key=args.api_key, base_url=args.api_base, timeout=3600)
        return local.client
    def run_one(task: dict[str, Any]) -> dict[str, Any]:
        row = task["row"]; started = time.perf_counter(); raw = ""; error = None; finish = None; prompt_tokens = completion_tokens = 0
        for attempt in range(1, args.max_retries + 1):
            try:
                reply = client().chat.completions.create(model=args.model_id, seed=int(generation["seed"]), temperature=float(generation["temperature"]), top_p=float(generation["top_p"]), presence_penalty=float(generation["presence_penalty"]), max_tokens=int(generation["max_new_tokens"]), extra_body={"top_k": int(generation["top_k"]), "repetition_penalty": float(generation["repetition_penalty"])}, messages=[{"role": "system", "content": system}, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_data_uri(task["image"])}}, {"type": "text", "text": str(row["query"]).replace("<image>", "").strip()}]}])
                raw = str(reply.choices[0].message.content or "").strip(); finish = reply.choices[0].finish_reason
                prompt_tokens = int(getattr(reply.usage, "prompt_tokens", 0) or 0); completion_tokens = int(getattr(reply.usage, "completion_tokens", 0) or 0)
                if raw: error = None; break
                raise ValueError("empty model response")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_retries: time.sleep(attempt)
        return {"schema_version": 1, "benchmark": task["benchmark"], "view": task["view"], "sample_uid": str(row["sample_uid"]), "source_id": str(row["source_id"]), "official_category": str(row.get("category", "unavailable_official")), "question_format": str(row["question_format"]), "dataset_revision": str(row["source_revision"]), "prompt": str(row["query"]).replace("<image>", "").strip(), "reference_answer": str(row["response"]), "raw_model_answer": raw, "parsed_answer": final_answer(raw), "model_id": args.model_id, "model_checkpoint_sha256": checkpoint, "finish_reason": finish, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "latency_seconds": round(time.perf_counter()-started, 6), "retry_count": attempt-1, "error": error, "generated_at_utc": now()}
    pending = [t for t in tasks if not (record_key({"benchmark":t["benchmark"], "view":t["view"], "sample_uid":t["row"]["sample_uid"]}) in records and not records[record_key({"benchmark":t["benchmark"], "view":t["view"], "sample_uid":t["row"]["sample_uid"]})].get("error"))]
    print(f"Day 6 requests: total={len(tasks)} completed={len(tasks)-len(pending)} pending={len(pending)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, task) for task in pending]
        for index, future in enumerate(as_completed(futures), 1):
            item = future.result(); records[record_key(item)] = item; write_records(predictions, records)
            print(f"[{index}/{len(pending)}] {item['benchmark']}/{item['view']}/{item['sample_uid']} error={bool(item['error'])}", flush=True)
    expected_keys = {record_key({"benchmark":t["benchmark"], "view":t["view"], "sample_uid":t["row"]["sample_uid"]}) for t in tasks}
    final = {key:value for key,value in records.items() if key in expected_keys}; write_records(predictions, final)
    write_json(out / "resume_status.json", {"updated_at_utc": now(), "expected_request_count": len(tasks), "completed_unique_request_count": len(final), "remaining_request_count": len(tasks)-len(final), "retryable_error_count": sum(bool(x.get("error")) for x in final.values()), "duplicate_request_keys": 0})

if __name__ == "__main__": main()
