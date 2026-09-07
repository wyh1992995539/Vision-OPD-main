#!/usr/bin/env python3
"""Promote the Day 13 Cached Prefix candidate without starting training."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "artifacts/runs/E-D13-6K-CACHED-PILOT-001"
FORMAL = ROOT / "configs/cached_prefix_6241.yaml"
POLICY = ROOT / "configs/cached_prefix_6241_abort_policy.yaml"
OPERATIONS = ROOT / "configs/day14_cached_operations.yaml"
ELIGIBILITY = RUN_ROOT / "evidence/formal_eligibility_receipt.json"
COMPLETION = RUN_ROOT / "evidence/completion_receipt.json"
POSTFLIGHT = RUN_ROOT / "evidence/postflight.json"
PROMOTION_DIR = RUN_ROOT / "formal_promotion_v1"
RECEIPT = PROMOTION_DIR / "promotion_receipt.json"
CANDIDATE_COPY = PROMOTION_DIR / "candidate_cached_prefix_6241.yaml"
ELIGIBILITY_COPY = PROMOTION_DIR / "eligibility_receipt.json"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def promoted_config(candidate: dict[str, Any], receipt_path: Path = RECEIPT) -> dict[str, Any]:
    value = copy.deepcopy(candidate)
    value["status"] = "ready_after_day13_gate"
    value["paper_alignment"]["pending_gates"] = []
    value["promotion"] = {
        "receipt": relative(receipt_path),
        "source_candidate": relative(CANDIDATE_COPY),
        "eligibility_receipt": relative(ELIGIBILITY_COPY),
        "validated_workload": "cached_64x8",
        "validated_steps": 8,
        "strict_prefix_source_ablation": "PASS",
        "formal_training_authorized": True,
    }
    return value


def verify_eligibility(value: dict[str, Any], candidate_path: Path) -> None:
    require(value.get("status") == "PASS", "Eligibility status is not PASS")
    require(value.get("eligible_for_formal_promotion") is True, "Formal eligibility is false")
    require(value.get("formal_training_authorized") is False, "Eligibility receipt self-authorized")
    require(value.get("strict_prefix_source_ablation") == "PASS", "Strict ablation did not pass")
    require(not value.get("failed_checks"), "Eligibility has failed checks")
    require(bool(value.get("checks")) and all(value["checks"].values()), "Eligibility checks incomplete")
    for name, entry in value.get("sources", {}).items():
        expected = str(entry.get("sha256", ""))
        if name == "formal_config":
            path = candidate_path
        else:
            path = Path(str(entry.get("path", "")))
        require(path.is_file() and sha256_file(path) == expected, f"Stale eligibility source: {name}")


def build_receipt(formal_path: Path, candidate_path: Path, eligibility_path: Path) -> dict[str, Any]:
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    formal = yaml.safe_load(formal_path.read_text(encoding="utf-8"))
    eligibility = load_json(eligibility_path)
    verify_eligibility(eligibility, candidate_path)
    require(formal == promoted_config(candidate), "Formal config differs from promoted candidate")

    completion = load_json(COMPLETION)
    postflight = load_json(POSTFLIGHT)
    require(completion.get("status") == "PASS", "Pilot completion did not pass")
    require(postflight.get("status") == "PASS", "Pilot postflight did not pass")
    require(postflight.get("strict_prefix_source_ablation") == "PASS", "Postflight ablation failed")
    require(postflight.get("checks", {}).get("online_generation_calls_zero_each_step") is True,
            "Pilot used online generation")
    require(postflight.get("checks", {}).get("cached_records_resolved_total_64") is True,
            "Pilot cache coverage failed")

    sources = {
        "candidate_config": candidate_path,
        "eligibility_receipt": eligibility_path,
        "completion_receipt": COMPLETION,
        "pilot_postflight": POSTFLIGHT,
        "formal_policy": POLICY,
        "operations": OPERATIONS,
        "promotion_builder": Path(__file__).resolve(),
        "guarded_launcher": ROOT / "scripts/run_day14_cached.py",
        "training_preflight": ROOT / "scripts/vopd_training_preflight.py",
        "runtime_monitor": ROOT / "scripts/monitor_vopd_training.py",
        "training_shell": ROOT / "scripts/run_vopd_2gpu.sh",
        "cached_shell": ROOT / "scripts/run_cached_prefix_2gpu.sh",
    }
    require(all(path.is_file() for path in sources.values()), "Promotion source missing")
    config_hash = sha256_file(formal_path)
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    checks = {
        "eligibility_pass_and_current": True,
        "pilot_completion_pass": True,
        "pilot_postflight_pass": True,
        "strict_prefix_source_ablation_pass": True,
        "online_generation_zero_in_pilot": True,
        "cached_64_records_resolved": True,
        "formal_config_exactly_derived": True,
        "formal_config_status_ready": formal.get("status") == "ready_after_day13_gate",
        "formal_config_authorized": formal.get("promotion", {}).get("formal_training_authorized") is True,
        "formal_run_starts_from_base": formal.get("paths", {}).get("model") == "/root/autodl-tmp/models/Qwen3.5-4B",
        "drop_last_6241_to_6240": (
            formal["training"]["source_samples"] == 6241
            and formal["training"]["expected_samples"] == 6240
            and formal["training"]["padding_rows"] == 0
            and formal["training"]["dropped_rows"] == 1
            and formal["training"]["total_optimizer_steps"] == 780
        ),
        "formal_policy_identity": policy.get("experiment_id") == "E-D14-6K-CACHED-001",
        "formal_policy_cached_runtime_guard": (
            policy.get("metrics", {}).get("cached_prefix_required") is True
            and policy["metrics"].get("cached_expected_records_per_step") == 8
            and policy["metrics"].get("cached_online_generation_calls_max") == 0
        ),
        "estimate_only_accounting": (
            policy.get("budget", {}).get("hourly_dual_gpu_rate_cny") == 14.0
            and policy["budget"].get("require_billing_observation") is False
            and policy["budget"].get("enforce_cumulative_cost_gate") is False
        ),
    }
    require(all(checks.values()), "Promotion checks incomplete")
    return {
        "schema_version": 1,
        "promotion_id": "E-D14-6K-CACHED-PROMOTION-001",
        "created_at_utc": utc_now(),
        "status": "PASS_FORMAL_CONFIG_PROMOTED",
        "artifact_status": "COMPLETE",
        "formal_training_authorized": True,
        "training_started": False,
        "checks": checks,
        "source_candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
        "promoted_formal_config": {
            "path": str(formal_path), "sha256": config_hash, "status": formal["status"]
        },
        "formal_policy": {"path": str(POLICY), "sha256": sha256_file(POLICY)},
        "eligibility": {
            "path": str(eligibility_path),
            "sha256": sha256_file(eligibility_path),
            "strict_prefix_source_ablation": "PASS",
        },
        "projection_780": postflight.get("projection_780"),
        "sources": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
            for name, path in sources.items()
        },
        "limits": [
            "Promotion is metadata-only and does not start training.",
            "The guarded launcher must rerun static and live resource checks immediately before training.",
            "Any change to the promoted config, policy, launcher, monitor, preflight or training shell invalidates this receipt.",
            "The formal run starts from the frozen Base and never resumes the Pilot checkpoint.",
        ],
    }


def verify_receipt(receipt_path: Path = RECEIPT, formal_path: Path = FORMAL) -> bool:
    try:
        receipt_path = Path(receipt_path).resolve()
        formal_path = Path(formal_path).resolve()
        value = load_json(receipt_path)
        require(value.get("status") == "PASS_FORMAL_CONFIG_PROMOTED", "Wrong promotion status")
        require(value.get("artifact_status") == "COMPLETE", "Incomplete promotion")
        require(value.get("formal_training_authorized") is True, "Promotion is not authorized")
        require(value.get("training_started") is False, "Promotion claims training started")
        require(bool(value.get("checks")) and all(value["checks"].values()), "Promotion check failed")
        for entry in value.get("sources", {}).values():
            path = Path(str(entry.get("path", "")))
            require(path.is_file() and sha256_file(path) == entry.get("sha256"),
                    f"Stale promotion source: {path}")
        formal_entry = value["promoted_formal_config"]
        require(Path(formal_entry["path"]).resolve() == formal_path, "Wrong formal config")
        require(formal_entry["sha256"] == sha256_file(formal_path), "Formal config hash changed")
        candidate_path = Path(value["source_candidate"]["path"])
        require(candidate_path.is_file(), "Candidate snapshot missing")
        candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
        formal = yaml.safe_load(formal_path.read_text(encoding="utf-8"))
        require(formal == promoted_config(candidate, receipt_path), "Promoted config semantics changed")
        eligibility_path = Path(value["eligibility"]["path"])
        verify_eligibility(load_json(eligibility_path), candidate_path)
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", type=Path, default=FORMAL)
    parser.add_argument("--eligibility", type=Path, default=ELIGIBILITY)
    parser.add_argument("--output", type=Path, default=RECEIPT)
    args = parser.parse_args()
    formal_path = args.formal.resolve()
    eligibility_path = args.eligibility.resolve()
    output = args.output.resolve()
    require(output == RECEIPT.resolve(), "Use the frozen Day 14 promotion path")
    require(not output.exists(), "Preserve existing promotion receipt")
    require(not CANDIDATE_COPY.exists() and not ELIGIBILITY_COPY.exists(), "Promotion snapshot already exists")

    candidate = yaml.safe_load(formal_path.read_text(encoding="utf-8"))
    require(candidate.get("status") == "implementation_ready_requires_cached_pilot",
            "Cached config is not in the pre-promotion state")
    require(candidate.get("promotion", {}).get("formal_training_authorized") is False,
            "Pre-promotion config self-authorized")
    eligibility = load_json(eligibility_path)
    verify_eligibility(eligibility, formal_path)

    PROMOTION_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(formal_path, CANDIDATE_COPY)
    shutil.copy2(eligibility_path, ELIGIBILITY_COPY)
    old_text = formal_path.read_text(encoding="utf-8")
    try:
        promoted = promoted_config(candidate, output)
        write_atomic(formal_path, yaml.safe_dump(promoted, sort_keys=False, allow_unicode=True))
        receipt = build_receipt(formal_path, CANDIDATE_COPY, ELIGIBILITY_COPY)
        write_atomic(output, json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        hash_path = output.parent / "promotion_receipt_sha256.txt"
        write_atomic(hash_path, f"{sha256_file(output)}  {output.name}\n")
        require(verify_receipt(output, formal_path), "Written promotion receipt failed verification")
    except BaseException:
        write_atomic(formal_path, old_text)
        for path in (output, output.parent / "promotion_receipt_sha256.txt", CANDIDATE_COPY, ELIGIBILITY_COPY):
            path.unlink(missing_ok=True)
        raise
    print(f"STATUS={receipt['status']}")
    print("FORMAL_TRAINING_AUTHORIZED=true")
    print(f"PROMOTED_CONFIG_SHA256={receipt['promoted_formal_config']['sha256']}")
    print(f"RECEIPT={output}")
    print("TRAINING_STARTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
