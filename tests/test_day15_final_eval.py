import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/runs/E-D15-6K-FINAL-EVAL-001"
CONFIG = ROOT / "configs/day15_final_eval_freeze.yaml"


def load(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pre_score_design_was_frozen_before_new_outputs():
    receipt = load(RUN / "preflight/eval_design_freeze.json")
    assert receipt["status"] == "PASS"
    assert all(receipt["checks"].values())
    assert receipt["new_score_files_found"] == []
    assert receipt["config_sha256"] == sha256(CONFIG)
    assert receipt["statement"].startswith("Frozen before any Vision-OPD or Cached")


def test_cached_final_model_cold_reload_passed():
    receipt = load(RUN / "cached_prefix/cold_reload/reload_validation_summary.json")
    assert receipt["status"] == "PASS"
    assert receipt["verification"]["status"] == "PASS"
    assert receipt["verification"]["prediction_count"] == 5
    assert receipt["verification"]["nonempty_response_count"] == 5
    assert receipt["verification"]["inference_error_count"] == 0
    assert receipt["source_checkpoint_unchanged"] is True
    assert receipt["merged_model_unchanged"] is True


def test_smoke_and_formal_r3_gates_pass():
    for role in ("vision_opd", "cached_prefix"):
        smoke = load(RUN / "smoke" / role / "validation.json")
        formal = load(RUN / role / "validation.json")
        assert smoke["status"] == smoke["run"]["status"] == "pass"
        assert smoke["run"]["prediction_count"] == smoke["run"]["score_count"] == 12
        assert formal["status"] == formal["run"]["status"] == "pass"
        assert formal["run"]["prediction_count"] == formal["run"]["score_count"] == 2536
        assert formal["run"]["completed_judge_count"] == formal["run"]["required_judge_count"]


def test_comparison_uses_frozen_denominators_and_three_roles():
    config = yaml.safe_load(CONFIG.read_text())
    comparison = load(RUN / "comparison.json")
    assert comparison["status"] == "PASS"
    assert comparison["freeze_config_sha256"] == sha256(CONFIG)
    assert {row["benchmark"] for row in comparison["rows"]} == {
        "zoombench", "mmstar", "vstar", "overall_micro"
    }
    totals = {row["benchmark"]: row["total"] for row in comparison["rows"]}
    assert totals == {"zoombench": 845, "mmstar": 1500, "vstar": 191, "overall_micro": 2536}
    assert set(comparison["runs"]) == {"base", "vision_opd", "cached_prefix"}
    assert config["badcase_sampling"]["score_mutation_forbidden"] is True


def test_badcase_selection_obeys_frozen_strata_and_hash():
    config = yaml.safe_load(CONFIG.read_text())
    rows = [
        json.loads(line)
        for line in (RUN / "badcases.jsonl").read_text().splitlines()
        if line.strip()
    ]
    summary = load(RUN / "badcases_summary.json")
    counts = Counter(row["stratum"] for row in rows)
    assert len(rows) == 18
    assert set(counts) == set(config["badcase_sampling"]["strata_priority"])
    assert all(value == 3 for value in counts.values())
    assert summary["selected_counts"] == dict(counts)
    assert summary["output_sha256"] == sha256(RUN / "badcases.jsonl")
    assert summary["score_mutation_performed"] is False
    assert summary["checkpoint_reselection_performed"] is False
