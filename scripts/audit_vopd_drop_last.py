#!/usr/bin/env python3
"""Generate an independent, fail-closed audit of the 6K native drop-last contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the Vision-OPD native drop-last contract.")
    parser.add_argument("--project-config", type=Path, default=Path("configs/project_6241.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/vopd_6241.yaml"))
    parser.add_argument(
        "--abort-policy", type=Path, default=Path("configs/vopd_6241_abort_policy.yaml")
    )
    parser.add_argument(
        "--trainer-source", type=Path, default=Path("verl/trainer/ppo/ray_trainer.py")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/E-D11-6K-GATE-001/drop_last"),
    )
    return parser.parse_args()


def resolve(path: Path, root: Path = PROJECT_ROOT) -> Path:
    return path if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def inspect_native_drop_last(path: Path) -> dict[str, Any]:
    """Find drop_last=True on the StatefulDataLoader assigned to train_dataloader."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    evidence: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or _call_name(value) != "StatefulDataLoader":
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        is_train_loader = any(
            isinstance(target, ast.Attribute) and target.attr == "train_dataloader"
            for target in targets
        )
        if not is_train_loader:
            continue
        keyword = next((item for item in value.keywords if item.arg == "drop_last"), None)
        literal = keyword.value.value if keyword and isinstance(keyword.value, ast.Constant) else None
        evidence.append(
            {
                "assignment_line": node.lineno,
                "drop_last_line": keyword.value.lineno if keyword else None,
                "drop_last_literal": literal,
            }
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "train_loader_assignments": evidence,
        "drop_last_true": len(evidence) == 1 and evidence[0]["drop_last_literal"] is True,
    }


def build_audit(
    project_config_path: Path,
    training_config_path: Path,
    abort_policy_path: Path,
    trainer_source_path: Path,
) -> dict[str, Any]:
    project_config_path = project_config_path.resolve()
    training_config_path = training_config_path.resolve()
    abort_policy_path = abort_policy_path.resolve()
    trainer_source_path = trainer_source_path.resolve()

    project = load_yaml(project_config_path)
    training = load_yaml(training_config_path)
    policy = load_yaml(abort_policy_path)

    project_contract = project["training_contract"]
    data_contract = training["data"]
    run_contract = training["training"]
    coverage_contract = policy["coverage"]

    train_file = resolve(Path(training["paths"]["train_file"]), training_config_path.parent.parent)
    parquet_rows = pq.ParquetFile(train_file).metadata.num_rows
    source_rows = int(project_contract["active_train_rows"])
    batch_size = int(project_contract["global_batch_size"])
    complete_batches, remainder = divmod(source_rows, batch_size)
    calculated_effective = complete_batches * batch_size
    calculated_dropped = remainder
    configured_full_coverage = bool(project_contract["require_full_coverage_sampler"])
    vopd_padding = data_contract.get("full_coverage_padding", {})
    native_evidence = inspect_native_drop_last(trainer_source_path)

    project_seed = int(project["reproducibility"]["dataloader_seed"])
    training_seed = int(training["experiment"]["seed"])
    checks = {
        "parquet_rows_match_project_source": parquet_rows == source_rows,
        "training_source_rows_match_project": int(run_contract["source_samples"]) == source_rows,
        "batch_size_matches_training": int(data_contract["train_batch_size"]) == batch_size,
        "one_epoch_frozen": int(run_contract["total_epochs"]) == 1,
        "shuffle_enabled": data_contract["shuffle"] is True,
        "dataloader_seed_matches_training_seed": project_seed == training_seed,
        "tail_is_nonempty": remainder > 0,
        "project_optimizer_steps_match_floor_division": (
            int(project_contract["optimizer_steps"]) == complete_batches
        ),
        "project_effective_samples_match": (
            int(project_contract["effective_train_samples"]) == calculated_effective
        ),
        "project_padding_is_zero": int(project_contract["padding_rows"]) == 0,
        "project_dropped_rows_match_remainder": (
            int(project_contract["dropped_rows"]) == calculated_dropped
        ),
        "project_full_coverage_sampler_disabled": configured_full_coverage is False,
        "project_tail_policy_is_native_drop_last": (
            project_contract["tail_policy"] == "native_drop_last"
        ),
        "training_optimizer_steps_match": (
            int(run_contract["total_optimizer_steps"]) == complete_batches
        ),
        "training_effective_samples_match": (
            int(run_contract["expected_samples"]) == calculated_effective
            and int(run_contract["padded_samples"]) == calculated_effective
        ),
        "training_padding_is_zero": int(run_contract["padding_rows"]) == 0,
        "training_dropped_rows_match": int(run_contract["dropped_rows"]) == calculated_dropped,
        "training_tail_policy_is_native_drop_last": (
            data_contract["tail_policy"] == "native_drop_last"
        ),
        "training_full_coverage_padding_disabled": not bool(vopd_padding.get("enabled", False)),
        "training_does_not_require_full_epoch_coverage": (
            run_contract["require_full_epoch"] is False
        ),
        "abort_policy_matches_contract": (
            coverage_contract["mode"] == "native_drop_last"
            and int(coverage_contract["source_rows"]) == source_rows
            and int(coverage_contract["expected_unique_source_seen"]) == calculated_effective
            and int(coverage_contract["expected_effective_train_samples"]) == calculated_effective
            and int(coverage_contract["expected_padding_rows"]) == 0
            and int(coverage_contract["expected_dropped_rows"]) == calculated_dropped
        ),
        "native_trainer_sets_drop_last_true": native_evidence["drop_last_true"],
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": 1,
        "audit_id": "E-D11-6K-DROP-LAST-001",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS" if not failed_checks else "FAIL",
        "inputs": {
            "project_config": {"path": str(project_config_path), "sha256": sha256_file(project_config_path)},
            "training_config": {"path": str(training_config_path), "sha256": sha256_file(training_config_path)},
            "abort_policy": {"path": str(abort_policy_path), "sha256": sha256_file(abort_policy_path)},
            "train_parquet": {"path": str(train_file), "sha256": sha256_file(train_file)},
        },
        "observed_contract": {
            "source_rows": source_rows,
            "parquet_rows": parquet_rows,
            "global_batch_size": batch_size,
            "complete_batches_per_epoch": complete_batches,
            "optimizer_steps": complete_batches,
            "effective_train_samples": calculated_effective,
            "tail_remainder": remainder,
            "padding_rows": 0,
            "dropped_rows_per_epoch": calculated_dropped,
            "epochs": int(run_contract["total_epochs"]),
            "shuffle": bool(data_contract["shuffle"]),
            "seed": training_seed,
            "tail_policy": "native_drop_last",
        },
        "native_trainer_evidence": native_evidence,
        "checks": checks,
        "failed_checks": failed_checks,
        "runtime_receipt_requirement": {
            "required_for_final_training_acceptance": True,
            "expected_unique_source_seen": calculated_effective,
            "expected_dropped_rows": calculated_dropped,
            "exact_dropped_sample_id_known_statically": False,
            "reason": (
                "shuffle=true; the dropped item is the seeded sampler tail, not necessarily "
                "Parquet physical row 6240. Record the exact identity from Pilot/runtime evidence."
            ),
        },
    }


def write_artifacts(audit: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "drop_last_audit.json"
    report_path = output_dir / "drop_last_audit.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    observed = audit["observed_contract"]
    check_rows = "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |" for name, passed in audit["checks"].items()
    )
    report = f"""# Day 11 Drop-last 独立审计

> 状态：**{audit['status']}**  
> 审计 ID：`{audit['audit_id']}`

## 结论

训练集 {observed['source_rows']} 条，global batch={observed['global_batch_size']}，1 epoch 使用 verl 原生
`drop_last=True`，因此形成 {observed['complete_batches_per_epoch']} 个完整 batch、
{observed['effective_train_samples']} 条有效训练记录、0 条 padding，并丢弃打乱后尾部
{observed['dropped_rows_per_epoch']} 条。

由于 `shuffle=true`，静态审计不能把 Parquet 物理末行认定为被丢弃样本；准确 sample ID
必须由 Pilot/正式运行回执记录。

## 检查

| 检查项 | 状态 |
|---|---|
{check_rows}

## 原生训练器证据

- 文件：`{audit['native_trainer_evidence']['path']}`
- SHA256：`{audit['native_trainer_evidence']['sha256']}`
- `drop_last=True`：{audit['native_trainer_evidence']['drop_last_true']}
"""
    report_path.write_text(report, encoding="utf-8")
    hash_lines = [
        f"{sha256_file(json_path)}  {json_path.name}",
        f"{sha256_file(report_path)}  {report_path.name}",
    ]
    (output_dir / "sha256.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit = build_audit(
        resolve(args.project_config),
        resolve(args.training_config),
        resolve(args.abort_policy),
        resolve(args.trainer_source),
    )
    output_dir = resolve(args.output_dir)
    write_artifacts(audit, output_dir)
    print(f"DROP_LAST_AUDIT={audit['status']}")
    print(f"OUTPUT_DIR={output_dir}")
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
