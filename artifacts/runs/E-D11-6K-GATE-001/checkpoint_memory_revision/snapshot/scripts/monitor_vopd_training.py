#!/usr/bin/env python3
"""Fail-closed telemetry and abort controls for the E-D10-001 training run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"(?:^|\s)step:(\d+)\s+-")
METRIC_RE = re.compile(r"(?:^|\s)-\s+([^:\s]+):([^\s]+)")
FATAL_LOG_RULES = {
    "dataloader_worker_killed": re.compile(r"DataLoader worker .*killed by signal", re.IGNORECASE),
    "checkpoint_save_failure": re.compile(
        r"(?:failed|error|exception).{0,80}(?:save|saving).{0,30}checkpoint|"
        r"checkpoint.{0,80}(?:failed|error|exception)|No space left on device",
        re.IGNORECASE,
    ),
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, dict) or policy.get("schema_version") != 1:
        raise ValueError("abort policy must be a schema_version=1 mapping")
    required = {"telemetry", "runtime", "memory", "disk", "metrics", "checkpoint"}
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"abort policy missing sections: {', '.join(missing)}")
    disk = policy["disk"]
    formula_required = 2 * int(disk["checkpoint_estimate_bytes"]) + int(disk["reserve_bytes"])
    minimum_free = int(disk.get("minimum_free_bytes", formula_required))
    expected = max(formula_required, minimum_free)
    if int(disk.get("formula_required_bytes", formula_required)) != formula_required:
        raise ValueError("formula_required_bytes must equal 2 * checkpoint estimate + reserve")
    if int(disk["prelaunch_required_bytes"]) != expected:
        raise ValueError("prelaunch_required_bytes must equal max(formula requirement, minimum free)")
    soft = int(disk["checkpoint_estimate_bytes"]) + int(disk["reserve_bytes"])
    if int(disk["runtime_soft_floor_bytes"]) != soft:
        raise ValueError("runtime_soft_floor_bytes must equal checkpoint estimate + reserve")
    for key in ("gpu_used_ratio_abort", "cgroup_used_ratio_abort"):
        value = float(policy["memory"][key])
        if not 0 < value <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    prelaunch_cgroup_minimum = policy["memory"].get("prelaunch_cgroup_minimum_bytes")
    if prelaunch_cgroup_minimum is not None and int(prelaunch_cgroup_minimum) <= 0:
        raise ValueError("prelaunch_cgroup_minimum_bytes must be positive")
    return policy


def load_policy(path: Path) -> dict[str, Any]:
    policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    return validate_policy(policy)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_scalar(text: str) -> Any:
    cleaned = text.strip().rstrip(",")
    if cleaned.lower() in {"nan", "+nan", "-nan"}:
        return float("nan")
    if cleaned.lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if cleaned.lower() in {"-inf", "-infinity"}:
        return float("-inf")
    try:
        return float(cleaned)
    except ValueError:
        return cleaned


def parse_training_metric_line(line: str) -> dict[str, Any] | None:
    clean = ANSI_RE.sub("", line).replace("\r", "")
    match = STEP_RE.search(clean)
    if not match:
        return None
    raw = {key: parse_scalar(value) for key, value in METRIC_RE.findall(clean)}
    aliases = {
        "loss": raw.get("actor/vopd_loss", raw.get("actor/pg_loss")),
        "grad_norm": raw.get("actor/grad_norm"),
        "learning_rate": raw.get("actor/lr"),
        "student_optimizer_delta": raw.get("evidence/student_param_probe_max_delta_after_optimizer"),
        "teacher_optimizer_delta": raw.get("evidence/teacher_param_probe_max_delta_after_optimizer"),
        "teacher_grad_non_none_count": raw.get("evidence/teacher_grad_non_none_count"),
        "teacher_ema_delta": raw.get("evidence/teacher_param_probe_max_delta_after_ema"),
        "ema_update_applied": raw.get("evidence/ema_update_applied"),
        "aborted_ratio": raw.get("response/aborted_ratio"),
    }
    return {"step": int(match.group(1)), **aliases, "raw_metric_count": len(raw)}


def scan_fatal_log_line(line: str) -> list[str]:
    clean = ANSI_RE.sub("", line)
    return [name for name, pattern in FATAL_LOG_RULES.items() if pattern.search(clean)]


def is_nonfinite(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and not math.isfinite(float(value))


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


@dataclass
class RuleEvaluator:
    policy: dict[str, Any]
    counters: dict[str, int] = field(default_factory=dict)
    initial_cgroup_events: dict[str, int] | None = None

    def _consecutive(self, name: str, active: bool, limit: int) -> bool:
        self.counters[name] = self.counters.get(name, 0) + 1 if active else 0
        return self.counters[name] >= limit

    @staticmethod
    def issue(rule: str, detail: str, immediate: bool = False) -> dict[str, Any]:
        return {"rule": rule, "detail": detail, "immediate": immediate}

    def evaluate_metric(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        limits = self.policy["metrics"]
        warmup_aware = bool(
            limits.get("student_update_required_only_when_lr_positive", False)
        )
        inspected = {
            key: row.get(key)
            for key in (
                "loss",
                "grad_norm",
                "student_optimizer_delta",
                "teacher_optimizer_delta",
                "teacher_grad_non_none_count",
                "teacher_ema_delta",
                "aborted_ratio",
            )
        }
        if warmup_aware:
            inspected["learning_rate"] = row.get("learning_rate")
        bad = [key for key, value in inspected.items() if is_nonfinite(value)]
        if bad:
            issues.append(self.issue("nonfinite_metric", f"step={row.get('step')} fields={bad}", True))
        learning_rate = row.get("learning_rate")
        if warmup_aware and not finite_number(learning_rate) and not is_nonfinite(learning_rate):
            issues.append(
                self.issue(
                    "learning_rate_missing_or_invalid",
                    f"step={row.get('step')} value={learning_rate}",
                    True,
                )
            )
        teacher_grad = row.get("teacher_grad_non_none_count")
        if teacher_grad is not None and float(teacher_grad) > 0:
            issues.append(self.issue("teacher_direct_gradient", f"step={row.get('step')} value={teacher_grad}", True))
        teacher_delta = row.get("teacher_optimizer_delta")
        if teacher_delta is not None and float(teacher_delta) != 0:
            issues.append(
                self.issue("teacher_optimizer_changed", f"step={row.get('step')} value={teacher_delta}", True)
            )

        ema = row.get("teacher_ema_delta")
        if ema is not None and self._consecutive(
            "teacher_ema_nonpositive", float(ema) <= 0, int(limits["teacher_ema_nonpositive_consecutive_steps"])
        ):
            issues.append(self.issue("teacher_ema_not_updating", f"step={row.get('step')} value={ema}"))
        student = row.get("student_optimizer_delta")
        student_update_expected = (
            not warmup_aware
            or (finite_number(learning_rate) and float(learning_rate) > 0)
        )
        if warmup_aware and finite_number(learning_rate) and float(learning_rate) < 0:
            issues.append(
                self.issue(
                    "invalid_learning_rate",
                    f"step={row.get('step')} value={learning_rate}",
                    True,
                )
            )
        if student is not None and self._consecutive(
            "student_delta_nonpositive",
            student_update_expected and float(student) <= 0,
            int(limits["student_delta_nonpositive_consecutive_steps"]),
        ):
            issues.append(self.issue("student_optimizer_not_updating", f"step={row.get('step')} value={student}"))
        aborted = row.get("aborted_ratio")
        if aborted is not None and self._consecutive(
            "generation_error",
            float(aborted) > 0,
            int(limits["generation_error_consecutive_steps"]),
        ):
            issues.append(
                self.issue("consecutive_generation_errors", f"step={row.get('step')} aborted_ratio={aborted}")
            )
        return issues

    def evaluate_telemetry(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        memory = self.policy["memory"]
        consecutive = int(memory["consecutive_samples"])
        gpu_over = any(
            float(row["memory_used_bytes"]) / float(row["memory_total_bytes"])
            >= float(memory["gpu_used_ratio_abort"])
            for row in sample.get("gpus", [])
            if row.get("memory_total_bytes")
        )
        if self._consecutive("gpu_memory", gpu_over, consecutive):
            issues.append(self.issue("gpu_memory_pressure", f"threshold={memory['gpu_used_ratio_abort']}"))

        cgroup = sample.get("cgroup") or {}
        current, maximum = cgroup.get("memory_current_bytes"), cgroup.get("memory_max_bytes")
        cgroup_over = (
            maximum not in (None, "max", 0)
            and current is not None
            and float(current) / float(maximum) >= float(memory["cgroup_used_ratio_abort"])
        )
        if self._consecutive("cgroup_memory", cgroup_over, consecutive):
            issues.append(self.issue("cgroup_memory_pressure", f"current={current} max={maximum}"))

        events = cgroup.get("memory_events") or {}
        if self.initial_cgroup_events is None and events:
            self.initial_cgroup_events = {key: int(value) for key, value in events.items()}
        elif events and self.initial_cgroup_events is not None:
            for key in ("oom", "oom_kill"):
                if int(events.get(key, 0)) > int(self.initial_cgroup_events.get(key, 0)):
                    issues.append(self.issue("cgroup_oom_event", f"{key} increased to {events[key]}", True))

        free = int(sample["disk_free_bytes"])
        disk = self.policy["disk"]
        if free < int(disk["runtime_hard_floor_bytes"]):
            issues.append(self.issue("disk_hard_floor", f"free={free}", True))
        elif self._consecutive(
            "disk_soft_floor",
            free < int(disk["runtime_soft_floor_bytes"]),
            int(disk["soft_floor_consecutive_samples"]),
        ):
            issues.append(self.issue("disk_checkpoint_reserve", f"free={free}"))
        return issues

    def evaluate_elapsed(self, elapsed_seconds: float) -> list[dict[str, Any]]:
        maximum = float(self.policy["runtime"]["max_wall_time_hours"]) * 3600
        if elapsed_seconds >= maximum:
            return [self.issue("wall_time_limit", f"elapsed_seconds={elapsed_seconds}")]
        return []


def list_process_tree(root_pid: int) -> list[int]:
    parent_to_children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            after_name = stat.rsplit(")", 1)[1].split()
            ppid = int(after_name[1])
        except (OSError, ValueError, IndexError):
            continue
        parent_to_children.setdefault(ppid, []).append(int(entry.name))
    result, pending = [], [root_pid]
    while pending:
        pid = pending.pop()
        if pid in result:
            continue
        result.append(pid)
        pending.extend(parent_to_children.get(pid, []))
    return result


def process_memory(root_pid: int) -> dict[str, Any]:
    rows, rss_total, vms_total = [], 0, 0
    for pid in list_process_tree(root_pid):
        try:
            values = {}
            for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
                if line.startswith(("Name:", "VmRSS:", "VmSize:")):
                    key, value = line.split(":", 1)
                    values[key] = value.strip()
            rss = int(values.get("VmRSS", "0 kB").split()[0]) * 1024
            vms = int(values.get("VmSize", "0 kB").split()[0]) * 1024
            rows.append({"pid": pid, "name": values.get("Name", "unknown"), "rss_bytes": rss, "vms_bytes": vms})
            rss_total += rss
            vms_total += vms
        except (OSError, ValueError):
            continue
    return {
        "root_pid": root_pid,
        "process_count": len(rows),
        "rss_bytes": rss_total,
        "vms_bytes": vms_total,
        "processes": rows,
    }


def read_cgroup(pid: int) -> dict[str, Any]:
    lines = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines()
    unified = next((line.split("::", 1)[1] for line in lines if line.startswith("0::")), None)
    if unified is None:
        return {"supported": False, "reason": "cgroup_v2_not_detected"}
    root = Path("/sys/fs/cgroup") / unified.lstrip("/")
    current = int((root / "memory.current").read_text(encoding="utf-8").strip())
    maximum_text = (root / "memory.max").read_text(encoding="utf-8").strip()
    events = {}
    for line in (root / "memory.events").read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        events[key] = int(value)
    return {
        "supported": True,
        "path": str(root),
        "memory_current_bytes": current,
        "memory_max_bytes": maximum_text if maximum_text == "max" else int(maximum_text),
        "memory_events": events,
    }


def cgroup_has_minimum_capacity(cgroup: dict[str, Any], required_bytes: int) -> bool:
    if not cgroup.get("supported") or required_bytes <= 0:
        return False
    maximum = cgroup.get("memory_max_bytes")
    return maximum == "max" or (isinstance(maximum, int) and maximum >= required_bytes)


def query_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid, used_mib, total_mib, utilization = [item.strip() for item in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "memory_used_bytes": int(used_mib) * 1024 * 1024,
                "memory_total_bytes": int(total_mib) * 1024 * 1024,
                "utilization_percent": int(utilization),
            }
        )
    return rows


def collect_sample(root_pid: int, output_dir: Path, started_monotonic: float) -> dict[str, Any]:
    timestamp = utc_now()
    processes = process_memory(root_pid)
    cgroup = read_cgroup(root_pid)
    gpus = query_gpus()
    disk = shutil.disk_usage(output_dir)
    common = {"timestamp_utc": timestamp, "elapsed_seconds": time.monotonic() - started_monotonic}
    append_jsonl(output_dir / "evidence/telemetry/gpu.jsonl", {**common, "gpus": gpus})
    append_jsonl(output_dir / "evidence/telemetry/process_rss.jsonl", {**common, **processes})
    append_jsonl(output_dir / "evidence/telemetry/cgroup_memory.jsonl", {**common, **cgroup})
    append_jsonl(output_dir / "evidence/telemetry/disk.jsonl", {**common, "free_bytes": disk.free})
    return {**common, "gpus": gpus, "processes": processes, "cgroup": cgroup, "disk_free_bytes": disk.free}


class IncrementalLogReader:
    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.partial = ""
        self.last_growth_monotonic: float | None = None

    def read_lines(self, now_monotonic: float) -> list[str]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self.partial = ""
        if size == self.offset:
            return []
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        self.last_growth_monotonic = now_monotonic
        combined = self.partial + chunk
        parts = combined.split("\n")
        self.partial = parts.pop()
        return parts


def terminate_process_group(process: subprocess.Popen[Any], grace_seconds: float) -> dict[str, Any]:
    receipt = {"term_sent_at_utc": utc_now(), "kill_sent_at_utc": None}
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return receipt
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        receipt["kill_sent_at_utc"] = utc_now()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    return receipt


def validate_checkpoint(output_dir: Path, policy: dict[str, Any]) -> dict[str, Any]:
    checkpoint = policy["checkpoint"]
    expected = int(checkpoint["expected_final_step"])
    root = output_dir / "checkpoints"
    marker = root / checkpoint["marker"]
    marker_value = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    step_dir = root / f"global_step_{expected}"
    missing, empty = [], []
    for relative in checkpoint["required_relative_files"]:
        path = step_dir / relative
        if not path.is_file():
            missing.append(relative)
        elif path.stat().st_size <= 0:
            empty.append(relative)
    checks = {
        "marker_exists": marker.is_file(),
        "marker_matches_expected_step": marker_value == str(expected),
        "step_directory_exists": step_dir.is_dir(),
        "required_files_present": not missing,
        "required_files_nonempty": not empty,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "expected_step": expected,
        "marker_value": marker_value,
        "step_directory": str(step_dir),
        "checks": checks,
        "missing_files": missing,
        "empty_files": empty,
    }


def replay(metrics_path: Path, log_path: Path, policy: dict[str, Any]) -> dict[str, Any]:
    evaluator = RuleEvaluator(policy)
    metric_issues = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            metric_issues.extend(evaluator.evaluate_metric(json.loads(line)))
    log_issues = []
    for number, line in enumerate(log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        for rule in scan_fatal_log_line(line):
            log_issues.append({"rule": rule, "line_number": number})
    return {
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
        "metric_rows": sum(1 for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()),
        "metric_issues": metric_issues,
        "fatal_log_issues": log_issues,
    }


def monitor_process(
    process: subprocess.Popen[Any], output_dir: Path, policy: dict[str, Any], train_log: Path
) -> tuple[int, dict[str, Any]]:
    started = time.monotonic()
    evaluator = RuleEvaluator(policy)
    reader = IncrementalLogReader(train_log)
    events_path = output_dir / "evidence/guard_events.jsonl"
    latest_step = None
    collection_failures = 0
    trigger = None
    termination = None
    interval = float(policy["telemetry"]["sample_interval_seconds"])
    while process.poll() is None:
        now = time.monotonic()
        issues = evaluator.evaluate_elapsed(now - started)
        try:
            sample = collect_sample(process.pid, output_dir, started)
            collection_failures = 0
            issues.extend(evaluator.evaluate_telemetry(sample))
        except Exception as exc:  # fail closed after the frozen consecutive limit
            collection_failures += 1
            append_jsonl(events_path, {"timestamp_utc": utc_now(), "event": "telemetry_error", "detail": repr(exc)})
            if collection_failures >= int(policy["telemetry"]["max_consecutive_collection_failures"]):
                issues.append(RuleEvaluator.issue("telemetry_unavailable", repr(exc)))

        for line in reader.read_lines(now):
            metric = parse_training_metric_line(line)
            if metric:
                latest_step = metric["step"]
                issues.extend(evaluator.evaluate_metric(metric))
                append_jsonl(output_dir / "evidence/runtime_metrics.jsonl", {"timestamp_utc": utc_now(), **metric})
            for rule in scan_fatal_log_line(line):
                issues.append(RuleEvaluator.issue(rule, ANSI_RE.sub("", line)[-500:], True))

        telemetry_policy = policy["telemetry"]
        grace = float(telemetry_policy["startup_grace_seconds"])
        heartbeat = float(telemetry_policy["log_heartbeat_timeout_seconds"])
        if now - started >= grace:
            last = reader.last_growth_monotonic or started
            if now - last >= heartbeat:
                issues.append(RuleEvaluator.issue("log_heartbeat_timeout", f"silent_seconds={now - last:.1f}"))
        if issues:
            trigger = {"timestamp_utc": utc_now(), "latest_step": latest_step, "issues": issues}
            append_jsonl(events_path, {"event": "abort_triggered", **trigger})
            termination = terminate_process_group(process, float(policy["runtime"]["terminate_grace_seconds"]))
            break
        time.sleep(interval)

    return_code = process.wait()
    checkpoint = validate_checkpoint(output_dir, policy) if trigger is None and return_code == 0 else None
    passed = trigger is None and return_code == 0 and checkpoint and checkpoint["status"] == "PASS"
    status = "PASS" if passed else "FAIL"
    summary = {
        "schema_version": 1,
        "finished_at_utc": utc_now(),
        "status": status,
        "return_code": return_code,
        "latest_step": latest_step,
        "trigger": trigger,
        "termination": termination,
        "checkpoint": checkpoint,
    }
    write_json(output_dir / "evidence/guard_summary.json", summary)
    if trigger:
        return 40, summary
    if return_code != 0:
        return return_code if 0 < return_code < 126 else 40, summary
    if not checkpoint or checkpoint["status"] != "PASS":
        return 42, summary
    return 0, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--replay-metrics", type=Path)
    parser.add_argument("--replay-log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    policy = load_policy(args.policy)
    if args.replay_metrics and args.replay_log:
        result = replay(args.replay_metrics, args.replay_log, policy)
        if args.output:
            write_json(args.output, result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    parser.error("standalone mode currently requires --replay-metrics and --replay-log")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
