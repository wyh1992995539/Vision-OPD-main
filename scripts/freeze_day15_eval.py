#!/usr/bin/env python3
"""Freeze Day15 model identities and evaluation design before new scores exist."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCORE_BEARING = {
    "predictions.jsonl", "judge_results.jsonl", "scores.jsonl",
    "summary.json", "validation.json",
}


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def verify_model(role: str, spec: dict[str, Any]) -> dict[str, Any]:
    model = resolve(spec["merged_model"])
    manifest_path = resolve(spec["merged_manifest"])
    reload_path = resolve(spec["cold_reload_receipt"])
    manifest = json.loads(manifest_path.read_text())
    reload_receipt = json.loads(reload_path.read_text())
    errors = []
    entries = manifest.get("files", [])
    for entry in entries:
        path = model / entry["relative_path"]
        if not path.is_file():
            errors.append(f"missing {path}")
        elif path.stat().st_size != int(entry["size_bytes"]):
            errors.append(f"size mismatch {path}")
        elif sha256(path) != entry["sha256"]:
            errors.append(f"hash mismatch {path}")
    if manifest.get("status") != "PASS" or len(entries) != 7:
        errors.append(f"{role} merged manifest is not PASS with 7 files")
    if reload_receipt.get("status") != "PASS":
        errors.append(f"{role} cold reload is not PASS")
    verification = reload_receipt.get("verification", {})
    if verification.get("prediction_count") != 5 or verification.get("inference_error_count") != 0:
        errors.append(f"{role} cold reload is not 5/5 with zero errors")
    return {
        "status": "PASS" if not errors else "FAIL",
        "model_id": spec["model_id"],
        "model_path": str(model.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "model_weight_sha256": {
            entry["relative_path"]: entry["sha256"]
            for entry in entries if entry["relative_path"].endswith(".safetensors")
        },
        "cold_reload_receipt": str(reload_path.resolve()),
        "cold_reload_receipt_sha256": sha256(reload_path),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/day15_final_eval_freeze.yaml")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text())
    r3 = resolve(config["frozen_r3_config"])
    base = resolve(config["base_run"])
    checks: dict[str, bool] = {}
    checks["freeze_status"] = config["status"] == "pre_score_freeze"
    checks["r3_hash"] = sha256(r3) == config["expected_r3_config_sha256"]
    base_validation = json.loads((base / "validation.json").read_text())
    checks["base_r3_pass"] = base_validation.get("status") == "pass"
    checks["base_complete"] = base_validation.get("run", {}).get("expected_request_count") == 2536
    checks["comparison_script_exists"] = (ROOT / "eval/compare_experiments.py").is_file()
    checks["badcase_script_exists"] = (ROOT / "eval/build_badcases.py").is_file()

    models = {role: verify_model(role, spec) for role, spec in config["models"].items()}
    checks["all_model_identities_pass"] = all(item["status"] == "PASS" for item in models.values())

    inspected_outputs = []
    new_score_files = []
    for spec in config["models"].values():
        for key in ("formal_output", "smoke_output"):
            output = resolve(spec[key])
            present = sorted(name for name in SCORE_BEARING if (output / name).exists())
            inspected_outputs.append({"path": str(output.resolve()), "score_bearing_files": present})
            new_score_files.extend(str(output / name) for name in present)
    checks["no_new_scores_before_freeze"] = not new_score_files

    script_paths = [ROOT / "eval/compare_experiments.py", ROOT / "eval/build_badcases.py"]
    result = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "experiment_id": config["experiment_id"],
        "statement": "Frozen before any Vision-OPD or Cached R3 smoke/formal predictions or scores.",
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "r3_config": str(r3.resolve()),
        "r3_config_sha256": sha256(r3),
        "base_run": str(base.resolve()),
        "base_run_manifest_sha256": sha256(base / "run_manifest.json"),
        "base_validation_sha256": sha256(base / "validation.json"),
        "models": models,
        "output_schema": config["output_schema"],
        "badcase_sampling": config["badcase_sampling"],
        "implementation": {
            str(path.relative_to(ROOT)): sha256(path) for path in script_paths if path.is_file()
        },
        "inspected_outputs": inspected_outputs,
        "new_score_files_found": new_score_files,
        "checks": checks,
    }
    target = resolve(config["outputs"]["freeze_receipt"])
    if target.exists():
        raise RuntimeError(f"refusing to overwrite existing freeze receipt: {target}")
    write_json(target, result)
    print(f"DAY15_EVAL_DESIGN_FREEZE={result['status']}")
    if result["status"] != "PASS":
        raise RuntimeError(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
