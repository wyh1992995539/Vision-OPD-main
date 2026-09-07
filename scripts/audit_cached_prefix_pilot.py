#!/usr/bin/env python3
"""Postflight audit for the 64-row Cached Prefix contract Pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from scripts.audit_vopd_6241_pilot import audit as audit_vopd
from scripts.monitor_vopd_training import ANSI_RE, METRIC_RE, STEP_RE, parse_scalar, write_json
from scripts.run_vopd_6241_pilot_guarded import resolve, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs/cached_prefix_6241_pilot_abort_policy.yaml"


def cached_runtime_rows(log_path: Path) -> list[dict]:
    rows = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        clean = ANSI_RE.sub("", line).replace("\r", "")
        match = STEP_RE.search(clean)
        if match is None:
            continue
        raw = {key: parse_scalar(value) for key, value in METRIC_RE.findall(clean)}
        if "cached_prefix/online_generation_calls" not in raw:
            continue
        step = int(match.group(1))
        rows.setdefault(step, {
            "step": step,
            "online_generation_calls": raw.get("cached_prefix/online_generation_calls"),
            "resolved_records": raw.get("cached_prefix/resolved_records"),
            "generation_seconds": raw.get("timing_s/gen"),
        })
    return [rows[step] for step in sorted(rows)]


def render(report: dict) -> str:
    checks = "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in report["checks"].items()
    )
    projection = report.get("projection_780") or {}
    return f"""# Cached Prefix 6,241 Pilot Postflight

- 状态：**{report['status']}**
- 训练链路通过：`{str(report['training_gate_pass']).lower()}`
- 冷加载与阶段 Gate 通过：`{str(report['stage_gate_pass']).lower()}`
- 严格 prefix-source 消融：`{report['strict_prefix_source_ablation']}`
- 正式训练授权：`false`

| 检查 | 结果 |
| --- | --- |
{checks}

780-step 外推：`{json.dumps(projection, ensure_ascii=False)}`

该报告只在 8-step 训练、Cached 零在线生成、Teacher EMA、checkpoint 和 5 条冷加载全部通过后，才把消融状态标记为 PASS。
"""


def audit(policy_path: Path, reload_report: Path | None) -> dict:
    report = audit_vopd("64", policy_path, reload_report)
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    contract = policy["pilot"]["stage_contracts"]["64"]
    config_path = resolve(contract["config"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = resolve(contract["output_dir"])
    log_path = output_dir / "logs/train.log"
    runtime_rows = cached_runtime_rows(log_path) if log_path.is_file() else []
    expected_steps = int(contract["expected_optimizer_steps"])
    batch_size = int(config["data"]["train_batch_size"])
    cache_path = Path(config["cached_prefix"]["output_parquet"]).resolve()
    contract_audit_path = resolve(policy["pilot"]["static_gate"])
    contract_audit = (
        json.loads(contract_audit_path.read_text(encoding="utf-8"))
        if contract_audit_path.is_file() else {}
    )
    cached_checks = {
        "cached_prefix_config_active": config["experiment"]["prefix_source"] == "cached",
        "cached_runtime_metrics_all_steps": [row["step"] for row in runtime_rows]
        == list(range(1, expected_steps + 1)),
        "online_generation_calls_zero_each_step": len(runtime_rows) == expected_steps
        and all(float(row["online_generation_calls"]) == 0 for row in runtime_rows),
        "cached_records_resolved_full_batch_each_step": len(runtime_rows) == expected_steps
        and all(float(row["resolved_records"]) == batch_size for row in runtime_rows),
        "cached_records_resolved_total_64": sum(
            float(row["resolved_records"] or 0) for row in runtime_rows
        ) == int(config["data"]["expected_train_rows"]),
        "cached_parquet_sha256_current": cache_path.is_file()
        and sha256_file(cache_path) == config["cached_prefix"]["output_sha256"],
        "static_contract_audit_pass_and_current": contract_audit.get("status")
        == "STATIC_CONTRACT_PASS"
        and bool(contract_audit.get("inputs"))
        and all(
            Path(entry["path"]).is_file()
            and sha256_file(Path(entry["path"])) == entry["sha256"]
            for entry in contract_audit.get("inputs", {}).values()
        ),
    }
    report.setdefault("checks", {}).update(cached_checks)
    report["failed_checks"] = sorted(name for name, passed in report["checks"].items() if not passed)
    report["training_gate_pass"] = all(report["checks"].values())
    reload_pass = (
        not report.get("reload_required")
        or (
            report.get("reload", {}).get("status") == "PASS"
            and report.get("reload", {}).get("source_checkpoint_unchanged") is True
            and report.get("reload", {}).get("verification", {}).get("status") == "PASS"
        )
    )
    report["stage_gate_pass"] = report["training_gate_pass"] and reload_pass
    if report["stage_gate_pass"]:
        report["status"] = "PASS"
        report["strict_prefix_source_ablation"] = "PASS"
    elif report["training_gate_pass"] and report.get("reload_required") and reload_report is None:
        report["status"] = "PASS_TRAINING_PENDING_RELOAD"
        report["strict_prefix_source_ablation"] = "PENDING_COLD_RELOAD"
    else:
        report["status"] = "FAIL"
        report["strict_prefix_source_ablation"] = "NOT_PROVEN"
    report["cached_runtime"] = {
        "rows": runtime_rows,
        "observed_steps": len(runtime_rows),
        "resolved_records_total": sum(float(row["resolved_records"] or 0) for row in runtime_rows),
    }
    report["formal_training_authorized"] = False
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("64",), default="64")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--reload-report", type=Path)
    args = parser.parse_args()
    policy_path = resolve(args.policy)
    reload_report = resolve(args.reload_report) if args.reload_report else None
    report = audit(policy_path, reload_report)
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    output_dir = resolve(policy["pilot"]["stage_contracts"]["64"]["output_dir"])
    evidence = output_dir / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    json_path = evidence / "postflight.json"
    markdown_path = ROOT / "artifacts/reports/cached_6241_pilot.md"
    write_json(json_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render(report), encoding="utf-8")
    (evidence / "postflight_sha256.txt").write_text(
        f"{sha256_file(json_path)}  {json_path.resolve()}\n"
        f"{sha256_file(markdown_path)}  {markdown_path.resolve()}\n",
        encoding="utf-8",
    )
    print(f"CACHED_PILOT_POSTFLIGHT={report['status']}")
    print(f"STRICT_PREFIX_SOURCE_ABLATION={report['strict_prefix_source_ablation']}")
    print(f"OUTPUT={json_path.resolve()}")
    return 0 if report["stage_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
