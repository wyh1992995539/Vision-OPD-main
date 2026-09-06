#!/usr/bin/env python3
"""Promote the source-bound 6241 candidate with an immutable receipt.

Promotion is a CPU-only metadata/configuration operation.  It does not perform
live resource checks and never starts training; the guarded launcher owns those
checks immediately before ``--run``.
"""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_formal_candidate_resources import verify as verify_candidate_freeze
from scripts.freeze_formal_cpu import verify as verify_cpu_freeze

RUN_ROOT = ROOT / "artifacts/runs/E-D11-6K-GATE-001"
CANDIDATE = ROOT / "configs/vopd_6241_candidate.yaml"
FORMAL = ROOT / "configs/vopd_6241.yaml"
POLICY = ROOT / "configs/vopd_6241_abort_policy.yaml"
CANDIDATE_FREEZE = RUN_ROOT / "formal_candidate_validation_v1/formal_gate_freeze.json"
CPU_FREEZE = RUN_ROOT / "resource_refreeze_v1/cpu_freeze.json"
PRE_PROMOTION_GATE = RUN_ROOT / "preflight.json"
RECEIPT = RUN_ROOT / "formal_promotion_v1/promotion_receipt.json"


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
        raise ValueError(f"{path}: expected a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def relative_or_absolute(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_reference(value: str, *, root: Path = ROOT) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def promoted_config(candidate: dict[str, Any], *, receipt_reference: str) -> dict[str, Any]:
    """Return the only allowed formal form of the validated candidate."""
    value = copy.deepcopy(candidate)
    value["status"] = "ready_after_day11_gate"
    value["paper_alignment"]["pending_gates"] = []
    value["resources"]["memory_profile"] = "offload_3way_graph4_deferred_formal_v1"
    value.pop("candidate", None)
    value["promotion"] = {
        "receipt": receipt_reference,
        "source_candidate": "configs/vopd_6241_candidate.yaml",
        "source_candidate_gate_freeze": (
            "artifacts/runs/E-D11-6K-GATE-001/formal_candidate_validation_v1/formal_gate_freeze.json"
        ),
        "validated_workload": "natural_eos_128x16",
        "validated_steps": 16,
        "normal_eos_validated": True,
        "formal_training_authorized": True,
    }
    return value


def verify_hash_mapping(mapping: dict[str, str]) -> bool:
    return bool(mapping) and all(
        Path(path).is_file() and sha256_file(Path(path)) == digest
        for path, digest in mapping.items()
    )


def validate_pre_promotion_gate(
    gate: dict[str, Any], *, previous_formal_copy: Path | None = None,
) -> None:
    require(gate.get("status") == "READY_TO_UNBLOCK_FORMAL_CONFIG", "Gate is not promotion-ready")
    require(gate.get("artifact_status") == "COMPLETE", "Gate artifact is incomplete")
    require(gate.get("formal_training_authorized") is False, "Pre-promotion Gate self-authorized")
    require(gate.get("ready_to_unblock_formal_config") is True, "Candidate checks are incomplete")
    require(gate.get("blocking_gates") == ["formal_config_released"], "Unexpected Gate blockers")
    require(gate.get("checks", {}).get("formal_candidate_validation_bound") is True,
            "Candidate validation is not bound")
    require(gate.get("checks", {}).get("candidate_checkpoint_validation_bound") is True,
            "Candidate checkpoint is not bound")
    require(gate.get("checks", {}).get("formal_config_released") is False,
            "Pre-promotion Gate release bit is invalid")
    require(all(gate.get("evidence_checks", {}).values()), "Gate evidence checks are incomplete")
    require(all(gate.get("runtime_checks", {}).values()), "Gate resource snapshot did not pass")
    require(all(gate.get("safety_checks", {}).values()), "Gate safety checks are incomplete")
    require(verify_hash_mapping(gate.get("gate_implementation", {})),
            "Gate implementation hashes are stale")
    for name, entry in gate.get("sources", {}).items():
        path = Path(str(entry.get("path", "")))
        if name == "formal_config" and previous_formal_copy is not None:
            previous_formal_copy = previous_formal_copy.resolve()
            require(previous_formal_copy.is_file()
                    and entry.get("sha256") == sha256_file(previous_formal_copy),
                    "Pre-promotion formal config backup does not match the Gate")
            continue
        require(path.is_file() and entry.get("sha256") == sha256_file(path),
                f"Stale Gate source: {path}")


def build_receipt(
    *, candidate_path: Path, formal_path: Path, policy_path: Path,
    candidate_freeze_path: Path, cpu_freeze_path: Path,
    pre_promotion_gate_copy: Path, previous_formal_copy: Path,
    receipt_path: Path, generated_at: str | None = None,
) -> dict[str, Any]:
    candidate_path = candidate_path.resolve()
    formal_path = formal_path.resolve()
    policy_path = policy_path.resolve()
    candidate_freeze_path = candidate_freeze_path.resolve()
    cpu_freeze_path = cpu_freeze_path.resolve()
    pre_promotion_gate_copy = pre_promotion_gate_copy.resolve()
    previous_formal_copy = previous_formal_copy.resolve()
    receipt_path = receipt_path.resolve()

    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    formal = yaml.safe_load(formal_path.read_text(encoding="utf-8"))
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    candidate_freeze = load_json(candidate_freeze_path)
    cpu_freeze = load_json(cpu_freeze_path)
    gate = load_json(pre_promotion_gate_copy)
    receipt_reference = relative_or_absolute(receipt_path)
    expected = promoted_config(candidate, receipt_reference=receipt_reference)

    validate_pre_promotion_gate(gate, previous_formal_copy=previous_formal_copy)
    require(formal == expected, "Promoted formal config has unapproved semantic differences")
    require(candidate_freeze.get("status") == "PASS_CANDIDATE_GATE_FREEZE",
            "Candidate resource freeze did not pass")
    require(candidate_freeze.get("candidate_validation", {}).get("validated_candidate_sha256")
            == sha256_file(candidate_path), "Candidate hash differs from validated workload")
    require(cpu_freeze.get("status") == "PASS_CPU_FLOOR_FROZEN", "CPU floor freeze did not pass")
    require(int(policy["memory"]["prelaunch_cgroup_minimum_bytes"]) == 240 * 1024**3,
            "Formal CPU floor is not 240 GiB")
    require(int(formal["resources"]["prelaunch_cgroup_minimum_bytes"])
            == int(policy["memory"]["prelaunch_cgroup_minimum_bytes"]),
            "Formal config and policy CPU floors differ")
    require(formal["rollout"]["ignore_eos"] is False, "Promotion changed normal EOS behavior")
    require(formal["actor"]["defer_optimizer_state_load"] is True,
            "Promotion lost deferred optimizer loading")

    sources = {
        "candidate": candidate_path,
        "candidate_gate_freeze": candidate_freeze_path,
        "cpu_freeze": cpu_freeze_path,
        "formal_policy": policy_path,
        "pre_promotion_gate": pre_promotion_gate_copy,
        "previous_formal_config": previous_formal_copy,
        "promotion_builder": Path(__file__).resolve(),
        "formal_launcher": ROOT / "scripts/run_vopd_6241_guarded.py",
        "aggregate_gate_builder": ROOT / "scripts/finalize_day11_candidate_gate.py",
        "aggregate_gate_entrypoint": ROOT / "scripts/finalize_day11_preflight.py",
        "training_preflight": ROOT / "scripts/vopd_training_preflight.py",
        "training_shell": ROOT / "scripts/run_vopd_2gpu.sh",
    }
    require(all(path.is_file() for path in sources.values()), "A promotion source is missing")
    checks = {
        "pre_promotion_gate_ready": True,
        "only_release_gate_was_pending": True,
        "candidate_freeze_pass_and_current": True,
        "candidate_hash_matches_validation": True,
        "candidate_checkpoint_bound": True,
        "cpu_freeze_verified_before_promotion": True,
        "formal_config_exactly_derived_from_candidate": True,
        "normal_eos_preserved": True,
        "deferred_optimizer_load_preserved": True,
        "drop_last_6241_to_6240_preserved": (
            formal["training"]["source_samples"] == 6241
            and formal["training"]["expected_samples"] == 6240
            and formal["training"]["dropped_rows"] == 1
            and formal["training"]["total_optimizer_steps"] == 780
        ),
        "formal_policy_cpu_floor_matches": True,
    }
    require(all(checks.values()), "Promotion checks are incomplete")
    return {
        "schema_version": 1,
        "promotion_id": "E-D11-6K-FORMAL-PROMOTION-001",
        "generated_at_utc": generated_at or utc_now(),
        "status": "PASS_FORMAL_CONFIG_PROMOTED",
        "artifact_status": "COMPLETE",
        "formal_training_authorized": True,
        "training_started": False,
        "checks": checks,
        "source_candidate": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
        },
        "promoted_formal_config": {
            "path": str(formal_path),
            "sha256": sha256_file(formal_path),
            "status": formal["status"],
        },
        "candidate_gate_freeze": {
            "path": str(candidate_freeze_path),
            "sha256": sha256_file(candidate_freeze_path),
        },
        "pre_promotion_gate": {
            "path": str(pre_promotion_gate_copy),
            "sha256": sha256_file(pre_promotion_gate_copy),
        },
        "budget": candidate_freeze["budget"],
        "disk": candidate_freeze["disk"],
        "sources": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}
            for name, path in sources.items()
        },
        "limits": [
            "Promotion authorizes only the static formal configuration; it does not start training.",
            "The guarded launcher must freshly verify billing, disk, cgroup, GPUs, Git and output collisions.",
            "Any change to the candidate, formal config, policy, Gate, launcher or bound source invalidates this receipt.",
            "Formal training starts from the frozen Base; the candidate checkpoint is validation evidence, not a resume source.",
        ],
    }


def verify_receipt(
    receipt_path: Path = RECEIPT, *, formal_path: Path = FORMAL,
    candidate_path: Path = CANDIDATE,
) -> bool:
    try:
        receipt_path = Path(receipt_path).resolve()
        formal_path = Path(formal_path).resolve()
        candidate_path = Path(candidate_path).resolve()
        value = load_json(receipt_path)
        require(value.get("status") == "PASS_FORMAL_CONFIG_PROMOTED", "Wrong promotion status")
        require(value.get("artifact_status") == "COMPLETE", "Incomplete promotion receipt")
        require(value.get("formal_training_authorized") is True, "Promotion did not authorize config")
        require(value.get("training_started") is False, "Promotion receipt claims training started")
        require(bool(value.get("checks")) and all(value["checks"].values()), "Promotion check failed")
        for entry in value.get("sources", {}).values():
            path = Path(str(entry.get("path", "")))
            require(path.is_file() and entry.get("sha256") == sha256_file(path),
                    f"Stale promotion source: {path}")
        formal_entry = value["promoted_formal_config"]
        require(Path(formal_entry["path"]).resolve() == formal_path, "Wrong promoted config path")
        require(formal_entry["sha256"] == sha256_file(formal_path), "Promoted config hash changed")
        candidate_entry = value["source_candidate"]
        require(Path(candidate_entry["path"]).resolve() == candidate_path, "Wrong candidate path")
        require(candidate_entry["sha256"] == sha256_file(candidate_path), "Candidate hash changed")
        candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
        formal = yaml.safe_load(formal_path.read_text(encoding="utf-8"))
        expected = promoted_config(candidate, receipt_reference=relative_or_absolute(receipt_path))
        require(formal == expected, "Live formal config is not the exact promoted candidate")
        require(resolve_reference(formal["promotion"]["receipt"]) == receipt_path,
                "Formal config points to a different promotion receipt")
        require(verify_candidate_freeze(Path(value["candidate_gate_freeze"]["path"])),
                "Candidate freeze no longer verifies")
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
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--formal", type=Path, default=FORMAL)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--candidate-freeze", type=Path, default=CANDIDATE_FREEZE)
    parser.add_argument("--cpu-freeze", type=Path, default=CPU_FREEZE)
    parser.add_argument("--pre-promotion-gate", type=Path, default=PRE_PROMOTION_GATE)
    parser.add_argument("--output", type=Path, default=RECEIPT)
    args = parser.parse_args()

    candidate_path = args.candidate.resolve()
    formal_path = args.formal.resolve()
    policy_path = args.policy.resolve()
    candidate_freeze_path = args.candidate_freeze.resolve()
    cpu_freeze_path = args.cpu_freeze.resolve()
    gate_path = args.pre_promotion_gate.resolve()
    output = args.output.resolve()
    require(not output.exists() and not output.with_suffix(".sha256").exists(),
            "Preserve the promotion receipt; choose a new output")
    require(verify_candidate_freeze(candidate_freeze_path), "Candidate freeze verification failed")
    require(verify_cpu_freeze(cpu_freeze_path), "CPU freeze must verify before promotion")
    gate = load_json(gate_path)
    validate_pre_promotion_gate(gate)

    output.parent.mkdir(parents=True, exist_ok=True)
    previous_formal = output.parent / "previous_vopd_6241.yaml"
    pre_gate_copy = output.parent / "pre_promotion_gate.json"
    for destination in (previous_formal, pre_gate_copy):
        require(not destination.exists(), f"Preserve existing promotion input: {destination}")
    shutil.copy2(formal_path, previous_formal)
    shutil.copy2(gate_path, pre_gate_copy)
    for suffix in (".md", ".sha256"):
        source = gate_path.with_suffix(suffix)
        if source.is_file():
            shutil.copy2(source, output.parent / f"pre_promotion_gate{suffix}")

    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    promoted = promoted_config(candidate, receipt_reference=relative_or_absolute(output))
    previous_text = formal_path.read_text(encoding="utf-8")
    try:
        write_atomic(formal_path, yaml.safe_dump(promoted, sort_keys=False, allow_unicode=True))
        value = build_receipt(
            candidate_path=candidate_path,
            formal_path=formal_path,
            policy_path=policy_path,
            candidate_freeze_path=candidate_freeze_path,
            cpu_freeze_path=cpu_freeze_path,
            pre_promotion_gate_copy=pre_gate_copy,
            previous_formal_copy=previous_formal,
            receipt_path=output,
        )
        with output.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        with output.with_suffix(".sha256").open("x", encoding="utf-8") as stream:
            stream.write(f"{sha256_file(output)}  {output}\n")
        require(verify_receipt(output, formal_path=formal_path, candidate_path=candidate_path),
                "Written promotion receipt did not verify")
    except BaseException:
        write_atomic(formal_path, previous_text)
        raise

    print("FORMAL_PROMOTION=PASS_FORMAL_CONFIG_PROMOTED")
    print("TRAINING_STARTED=false")
    print(f"FORMAL_CONFIG={formal_path}")
    print(f"RECEIPT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
