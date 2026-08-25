#!/usr/bin/env python3
"""Capture the frozen Day 6 Judge runtime evidence while vLLM is live."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import torch


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    preflight = repo / "artifacts/runs/E-D6-001/preflight"
    records = [
        json.loads(line)
        for line in (preflight / "judge_calibration.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    with urlopen("http://127.0.0.1:8000/v1/models", timeout=10) as response:
        models = json.load(response)
    gpus = []
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        gpus.append({
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
        })
    evidence = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_command": (
            "OMP_NUM_THREADS=1 /root/miniconda3/envs/vision-opd/bin/python -m "
            "vllm.entrypoints.openai.api_server --model /root/autodl-tmp/models/Qwen3.5-4B "
            "--served-model-name vision-opd-base --tensor-parallel-size 2 --dtype bfloat16 "
            "--gpu-memory-utilization 0.80 --max-model-len 32768 --max-num-seqs 8 "
            "--seed 42 --trust-remote-code --host 127.0.0.1 --port 8000"
        ),
        "api_base": "http://127.0.0.1:8000/v1",
        "health": "pass",
        "models_response": models,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": gpus,
        "calibration": {
            "record_count": len(records),
            "score_source_counts": dict(Counter(row["pipeline_score"]["score_source"] for row in records)),
            "judge_raw_counts": dict(Counter(
                row["pipeline_score"]["judge_raw"]
                for row in records
                if row["pipeline_score"]["judge_raw"] is not None
            )),
            "judge_failure_count": sum(
                row["pipeline_score"]["score_source"] == "judge_failure" for row in records
            ),
            "pending_count": sum(
                row["pipeline_score"]["score_status"] != "scored" for row in records
            ),
        },
    }
    (preflight / "judge_runtime.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
