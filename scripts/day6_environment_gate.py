#!/usr/bin/env python3
"""Run the frozen Day 6 server import and pytest gate from vision-opd."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    output = repo / "artifacts/runs/E-D6-001/preflight"
    output.mkdir(parents=True, exist_ok=True)
    imports = {}
    for module_name in ("openai", "yaml", "torch"):
        try:
            module = importlib.import_module(module_name)
            package_name = "pyyaml" if module_name == "yaml" else module_name
            imports[module_name] = {
                "status": "pass",
                "version": importlib.metadata.version(package_name),
                "module_path": str(Path(module.__file__).resolve()),
            }
        except Exception as exc:
            imports[module_name] = {
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
            }
    command = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(command, cwd=repo, text=True, capture_output=True, check=False)
    log = (
        f"generated_at_utc: {now()}\n"
        f"required_entry: conda run -n vision-opd python -m pytest -q\n"
        f"actual_command: {' '.join(command)}\n"
        f"cwd: {repo}\n"
        f"exit_code: {completed.returncode}\n"
        "--- stdout ---\n"
        f"{completed.stdout}"
        "--- stderr ---\n"
        f"{completed.stderr}"
    )
    (output / "pytest.log").write_text(log, encoding="utf-8")
    summary = {
        "schema_version": 1,
        "generated_at_utc": now(),
        "required_entry": "conda run -n vision-opd python -m pytest -q",
        "actual_python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "imports": imports,
        "pytest": {
            "command": command,
            "exit_code": completed.returncode,
            "log": "artifacts/runs/E-D6-001/preflight/pytest.log",
        },
        "status": (
            "pass"
            if all(item["status"] == "pass" for item in imports.values())
            and completed.returncode == 0
            else "fail"
        ),
    }
    (output / "environment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
