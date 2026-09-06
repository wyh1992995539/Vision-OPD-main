import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import yaml

from scripts import run_day12_vopd as launcher


class Day12OperationsTest(TestCase):
    def setUp(self):
        self.config = yaml.safe_load(launcher.CONFIG.read_text())
        self.policy = launcher.frozen.load_policy(launcher.POLICY)
        self.operations = yaml.safe_load(launcher.OPERATIONS.read_text())
        self.result = {
            "status": "PASS", "checks": {"day11_gate_pass_and_authorized": True},
            "disk_required_bytes": 120 * 1024**3,
            "accounting": {"hourly_dual_gpu_rate_cny": 14.0},
            "operational_sources": {},
        }

    def test_price_is_for_both_gpus_and_uses_unrounded_duration(self):
        result = launcher.cost_estimate({
            "planning_hours": 7.043446635950425,
            "reservation_hours": 8.074135051401228,
            "hard_abort_ceiling_hours": 38,
        }, self.operations)
        self.assertEqual(round(result["planning_incremental_cost_cny"], 2), 98.61)
        self.assertEqual(round(result["reservation_incremental_cost_cny"], 2), 113.04)
        self.assertEqual(result["hard_abort_ceiling_incremental_cost_cny"], 532)
        self.assertFalse(result["is_provider_bill"])

    def test_invalid_rate_rejected(self):
        selected = dict(planning_hours=7, reservation_hours=8, hard_abort_ceiling_hours=38)
        for rate in (0, -1, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                launcher.cost_estimate(selected, {**self.operations, "hourly_dual_gpu_rate_cny": rate})

    def live(self, directory, *, free=150, capacity=240, used=0, gpu_count=2, dirty=False):
        with patch.object(launcher.shutil, "disk_usage", return_value=SimpleNamespace(free=free * 1024**3)), \
             patch.object(launcher.frozen, "read_cgroup", return_value={
                 "supported": True, "memory_max_bytes": capacity * 1024**3}), \
             patch.object(launcher.frozen, "query_gpus", return_value=[{
                 "memory_total_bytes": 100, "memory_used_bytes": used,
             }] * gpu_count), \
             patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(stdout=" M file" if dirty else "")):
            return launcher.live_gate(copy.deepcopy(self.result), self.config, self.policy, directory)

    def test_no_bill_needed_and_resource_failures_still_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.live(root)["status"], "PASS")
            for kwargs, check in (({"free": 119}, "storage_pass"),
                                  ({"capacity": 239}, "cgroup_capacity_pass"),
                                  ({"used": 11}, "gpus_idle"),
                                  ({"gpu_count": 1}, "expected_gpu_count"),
                                  ({"dirty": True}, "git_clean")):
                with self.subTest(check=check):
                    result = self.live(root, **kwargs)
                    self.assertEqual(result["status"], "FAIL")
                    self.assertFalse(result["live_checks"][check])
            (root / "evidence").mkdir()
            (root / "evidence/exit_receipt.json").write_text("{}")
            self.assertFalse(self.live(root)["live_checks"]["no_output_collision"])

    def test_stale_day11_evidence_never_launches(self):
        result = {**self.result, "status": "FAIL"}
        with patch.object(launcher, "preflight", return_value=result), \
             patch.object(launcher.frozen, "write_json"), \
             patch.object(launcher, "live_gate") as live, \
             patch.object(launcher.subprocess, "Popen") as popen:
            self.assertEqual(launcher.main(["--run"]), 41)
            live.assert_not_called()
            popen.assert_not_called()

    def test_run_without_billing_uses_frozen_monitor_and_records_estimate(self):
        process = SimpleNamespace(poll=lambda: 0)
        with patch.object(launcher, "preflight", return_value=self.result), \
             patch.object(launcher, "live_gate", return_value=self.result), \
             patch.object(launcher.frozen, "write_json") as write, \
             patch.object(launcher.subprocess, "Popen", return_value=process) as popen, \
             patch.object(launcher.frozen, "monitor_process", return_value=(0, {"status": "PASS"})) as monitor, \
             patch.object(launcher.time, "monotonic", side_effect=[100, 3700]):
            self.assertEqual(launcher.main(["--run"]), 0)
            self.assertEqual(monitor.call_args.args[2], self.policy)
            self.assertEqual(popen.call_args.args[0][-3:], ["--config", str(launcher.CONFIG), "--run"])
            receipt = write.call_args.args[1]
            self.assertEqual(receipt["accounting"]["estimated_incremental_cost_cny"], 14)
            self.assertFalse(receipt["accounting"]["is_provider_bill"])

    def test_failed_resource_gate_never_launches(self):
        with patch.object(launcher, "preflight", return_value=self.result), \
             patch.object(launcher, "live_gate", return_value={
                 **self.result, "status": "FAIL", "live_checks": {"storage_pass": False}}), \
             patch.object(launcher.frozen, "write_json"), \
             patch.object(launcher.subprocess, "Popen") as popen:
            self.assertEqual(launcher.main(["--run"]), 41)
            popen.assert_not_called()
