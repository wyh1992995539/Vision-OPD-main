#!/usr/bin/env python3
"""Build the Day 16 paper-gap diagnostic dataset and portable report artifact.

The script reads frozen Day 15 scores without mutating them.  Its parser
sensitivity analysis only removes records where the frozen first-letter rule
accepted an explicit answer option that contradicts the reference option.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/runs/E-D15-6K-FINAL-EVAL-001"
OUTPUT = ROOT / "artifacts/reports/vision_opd_gap_diagnostic"
SCORE_PATHS = {
    "Base": ROOT / "artifacts/runs/E-PAPER-BASEJUDGE-001/base/scores.jsonl",
    "Vision-OPD": RUN / "vision_opd/scores.jsonl",
    "Cached Prefix": RUN / "cached_prefix/scores.jsonl",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def leading_option(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^Answer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.lstrip(" *([`#")
    match = re.match(r"([A-F])(?=$|[\s*`\])\).:\-])", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def is_explicit_first_letter_false_positive(row: dict) -> bool:
    if row.get("rule_source") != "first_letter":
        return False
    expected = leading_option(row.get("reference_answer"))
    predicted = leading_option(row.get("parsed_answer"))
    return bool(expected and predicted and expected != predicted)


def adjusted_correct(row: dict) -> bool:
    return bool(row.get("final_is_correct")) and not is_explicit_first_letter_false_positive(row)


def pct(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    comparison = read_json(RUN / "comparison.json")
    scores = {role: read_jsonl(path) for role, path in SCORE_PATHS.items()}
    for role, rows in scores.items():
        assert len(rows) == 2536, (role, len(rows))

    paper = {
        "V* Bench": {"Base": 0.8429, "Vision-OPD": 0.9215},
        "ZoomBench": {"Base": 0.4769, "Vision-OPD": 0.5976},
        "MMStar": {"Base": 0.7853, "Vision-OPD": 0.7960},
    }
    benchmark_labels = {"vstar": "V* Bench", "zoombench": "ZoomBench", "mmstar": "MMStar"}

    official_rows = {row["benchmark"]: row for row in comparison["rows"] if row["benchmark"] != "overall_micro"}
    gain_rows = []
    benchmark_long = []
    for benchmark in ["V* Bench", "ZoomBench", "MMStar"]:
        key = {value: name for name, value in benchmark_labels.items()}[benchmark]
        local = official_rows[key]
        gain_rows.extend(
            [
                {
                    "benchmark": benchmark,
                    "setup": "论文 Table 2",
                    "delta_vs_base": paper[benchmark]["Vision-OPD"] - paper[benchmark]["Base"],
                    "base_accuracy": paper[benchmark]["Base"],
                    "vision_accuracy": paper[benchmark]["Vision-OPD"],
                    "protocol": "paper_reported",
                },
                {
                    "benchmark": benchmark,
                    "setup": "本项目冻结 R3",
                    "delta_vs_base": local["vision_opd_delta_vs_base"],
                    "base_accuracy": local["base_accuracy"],
                    "vision_accuracy": local["vision_opd_accuracy"],
                    "protocol": "local_r3_frozen",
                },
            ]
        )
        for role, field in [("Base", "base_accuracy"), ("Vision-OPD", "vision_opd_accuracy"), ("Cached Prefix", "cached_prefix_accuracy")]:
            benchmark_long.append(
                {
                    "benchmark": benchmark,
                    "model": role,
                    "accuracy": local[field],
                    "total": local["total"],
                    "protocol": "local_r3_frozen",
                }
            )

    sensitivity_rows = []
    parser_summary = []
    adjusted_by_role_benchmark = {}
    for role, rows in scores.items():
        first_letter_rows = [row for row in rows if row.get("rule_source") == "first_letter"]
        explicit_false_positives = [row for row in first_letter_rows if is_explicit_first_letter_false_positive(row)]
        parser_summary.append(
            {
                "model": role,
                "first_letter_accepted": len(first_letter_rows),
                "explicit_mismatch_false_positives": len(explicit_false_positives),
                "affected_share": pct(len(explicit_false_positives), len(first_letter_rows)),
            }
        )
        for benchmark_key in ["zoombench", "mmstar", "vstar"]:
            subset = [row for row in rows if row["benchmark"] == benchmark_key]
            official_correct = sum(bool(row["final_is_correct"]) for row in subset)
            adjusted_count = sum(adjusted_correct(row) for row in subset)
            false_positive_count = sum(is_explicit_first_letter_false_positive(row) for row in subset)
            adjusted_by_role_benchmark[(role, benchmark_key)] = adjusted_count
            sensitivity_rows.append(
                {
                    "model": role,
                    "benchmark": benchmark_labels[benchmark_key],
                    "total": len(subset),
                    "frozen_r3_correct": official_correct,
                    "frozen_r3_accuracy": pct(official_correct, len(subset)),
                    "explicit_false_positives": false_positive_count,
                    "explicit_mismatch_adjusted_correct": adjusted_count,
                    "explicit_mismatch_adjusted_accuracy": pct(adjusted_count, len(subset)),
                }
            )
        official_correct = sum(bool(row["final_is_correct"]) for row in rows)
        adjusted_count = sum(adjusted_correct(row) for row in rows)
        sensitivity_rows.append(
            {
                "model": role,
                "benchmark": "Micro 汇总",
                "total": len(rows),
                "frozen_r3_correct": official_correct,
                "frozen_r3_accuracy": pct(official_correct, len(rows)),
                "explicit_false_positives": official_correct - adjusted_count,
                "explicit_mismatch_adjusted_correct": adjusted_count,
                "explicit_mismatch_adjusted_accuracy": pct(adjusted_count, len(rows)),
            }
        )

    category = defaultdict(dict)
    for role, rows in scores.items():
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["benchmark"], row.get("official_category", "unknown"))].append(row)
        for key, subset in grouped.items():
            category[key][role] = {
                "total": len(subset),
                "correct": sum(adjusted_correct(row) for row in subset),
            }
    category_rows = []
    for (benchmark_key, category_name), role_values in sorted(category.items()):
        if "Base" not in role_values or "Vision-OPD" not in role_values:
            continue
        total = role_values["Base"]["total"]
        base_acc = pct(role_values["Base"]["correct"], total)
        vision_acc = pct(role_values["Vision-OPD"]["correct"], total)
        cached_acc = pct(role_values["Cached Prefix"]["correct"], total)
        category_rows.append(
            {
                "benchmark": benchmark_labels[benchmark_key],
                "category": category_name,
                "total": total,
                "base_accuracy": base_acc,
                "vision_accuracy": vision_acc,
                "vision_delta_vs_base": vision_acc - base_acc,
                "cached_accuracy": cached_acc,
                "cached_delta_vs_base": cached_acc - base_acc,
            }
        )
    category_rows.sort(key=lambda row: row["vision_delta_vs_base"])

    overlap_ids = {
        row["benchmark_sample_uid"]
        for row in read_jsonl(ROOT / "artifacts/runs/E-D10-6K-DATA-001/overlap/overlap_candidates.jsonl")
        if row.get("review_status") == "confirmed_overlap"
    }
    overlap_rows = []
    for role, rows in scores.items():
        vstar = [row for row in rows if row["benchmark"] == "vstar"]
        for split, subset in [
            ("21 张训练重合图像", [row for row in vstar if row["sample_uid"] in overlap_ids]),
            ("其余 170 条", [row for row in vstar if row["sample_uid"] not in overlap_ids]),
        ]:
            correct = sum(adjusted_correct(row) for row in subset)
            overlap_rows.append(
                {"model": role, "split": split, "total": len(subset), "correct": correct, "accuracy": pct(correct, len(subset))}
            )

    transition_rows = []
    transition_labels = {
        "base_to_vision_opd": "Base → Vision-OPD",
        "base_to_cached_prefix": "Base → Cached",
        "vision_opd_to_cached_prefix": "Vision-OPD → Cached",
    }
    for key, values in comparison["paired_transitions"].items():
        net = values["incorrect_to_correct"] - values["correct_to_incorrect"]
        transition_rows.extend(
            [
                {"comparison": transition_labels[key], "transition": "错→对", "count": values["incorrect_to_correct"], "net_change": net},
                {"comparison": transition_labels[key], "transition": "对→错", "count": values["correct_to_incorrect"], "net_change": net},
            ]
        )

    config_rows = [
        {"parameter": "GPU", "author_entry": "8", "current_run": "2", "ratio_or_effect": "资源缩放"},
        {"parameter": "global / PPO batch", "author_entry": "96 / 96", "current_run": "8 / 8", "ratio_or_effect": "当前为 1/12"},
        {"parameter": "rollout n", "author_entry": "8", "current_run": "1", "ratio_or_effect": "当前为 1/8"},
        {"parameter": "每个 epoch 的 rollout 序列", "author_entry": "49,920", "current_run": "6,240", "ratio_or_effect": "当前为 1/8"},
        {"parameter": "trainer steps / EMA events", "author_entry": "约 65", "current_run": "780", "ratio_or_effect": "当前为 12×"},
        {"parameter": "Student Adam updates（估算）", "author_entry": "约 520", "current_run": "780", "ratio_or_effect": "当前约为 1.5×；mini-batch 为 1/12"},
        {"parameter": "EMA 初始教师权重系数", "author_entry": f"{math.pow(0.95, 65):.4f}", "current_run": f"{math.pow(0.95, 780):.2e}", "ratio_or_effect": "同一 0.05 系数处于不同时间尺度"},
        {"parameter": "learning rate", "author_entry": "2e-6", "current_run": "2e-6", "ratio_or_effect": "未随 batch / step 缩放"},
        {"parameter": "response / epoch / Top-K / JSD / EMA", "author_entry": "1024 / 1 / 100 / 0.5 / 0.05", "current_run": "相同", "ratio_or_effect": "核心参数对齐"},
    ]

    diagnostic = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "status": "share_with_caveats",
        "question": "为什么当前 Vision-OPD 与 Cached Prefix 的效果明显低于论文？",
        "frozen_r3": comparison,
        "paper_table_2": paper,
        "gain_comparison": gain_rows,
        "parser_sensitivity": sensitivity_rows,
        "parser_summary": parser_summary,
        "category_sensitivity": category_rows,
        "vstar_overlap_sensitivity": overlap_rows,
        "configuration_comparison": config_rows,
        "paired_transitions": transition_rows,
        "assessment": {
            "verified": [
                "Day 12 and Day 14 training completed without NaN, OOM, abort, or checkpoint failure.",
                "The selected run changed batch 96 to 8, rollout n 8 to 1, and trainer-step/EMA events from about 65 to 780 while retaining LR 2e-6 and EMA rate 0.05; estimated Student Adam updates changed from about 520 to 780.",
                "The frozen first-letter matcher can accept explicit wrong options after an Answer: prefix; at least 86 Base, 87 Vision-OPD, and 48 Cached records are explicit mismatches.",
                "Removing only those explicit false positives leaves Base-to-Vision micro delta at about -4.18 percentage points.",
                "Vision-OPD improves ZoomBench in frozen R3 but loses on MMStar and V*; losses exceed gains in paired transitions.",
            ],
            "likely": [
                "The batch, rollout, optimizer-step, and EMA-time-scale changes materially altered the optimization regime and are the leading explanation for the paper gap.",
                "Cached Prefix removes the paper method's on-policy prefix refresh and therefore should be treated as an ablation rather than the paper's primary method.",
            ],
            "unresolved": [
                "The causal share of each training change cannot be separated without controlled reruns.",
                "The effect of replacing the paper's GPT-OSS-120B judge with the frozen local Qwen3.5-4B Base Judge is not quantified.",
                "A full corrected R4 needs a repaired parser and fresh judging for newly routed records; this report does not mutate frozen R3 scores.",
            ],
        },
    }
    diagnostic_path = OUTPUT / "diagnostic_data.json"
    diagnostic_path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sources = [
        {
            "id": "day15_r3",
            "label": "Day 15 frozen R3 comparison",
            "path": "artifacts/runs/E-D15-6K-FINAL-EVAL-001/comparison.json",
            "query": {
                "engine": "duckdb",
                "sql": "SELECT * FROM read_json_auto('artifacts/runs/E-D15-6K-FINAL-EVAL-001/comparison.json')",
                "description": "Loads the frozen Day 15 benchmark rows and paired transitions.",
                "tables_used": ["artifacts/runs/E-D15-6K-FINAL-EVAL-001/comparison.json"],
                "filters": ["status = PASS", "all three model roles present"],
                "metric_definitions": {
                    "accuracy": "correct / frozen benchmark denominator",
                    "delta_vs_base": "trained-model accuracy minus Base accuracy",
                },
            },
        },
        {"id": "paper_pdf", "label": "Vision-OPD paper, Tables 1–6", "path": "docs/Vision-OPD.pdf"},
        {"id": "author_launcher", "label": "Author training launcher", "path": "scripts/run_vision_opd.sh"},
        {"id": "selected_config", "label": "Selected two-GPU training config", "path": "configs/vopd_6241.yaml"},
        {"id": "score_code", "label": "Frozen R3 answer parsing and scoring code", "path": "eval/paper_aligned_common.py"},
        {
            "id": "diagnostic_data",
            "label": "Reproducible Day 16 diagnostic data",
            "path": "artifacts/reports/vision_opd_gap_diagnostic/diagnostic_data.json",
            "query": {
                "engine": "duckdb",
                "sql": "SELECT * FROM read_json_auto('artifacts/reports/vision_opd_gap_diagnostic/diagnostic_data.json')",
                "description": "Loads the reproducible parser sensitivity, category, overlap, paper-gap, and configuration datasets.",
                "tables_used": ["artifacts/reports/vision_opd_gap_diagnostic/diagnostic_data.json"],
                "filters": ["frozen Day 15 scores only", "explicit first-letter mismatches only in sensitivity adjustment"],
                "metric_definitions": {
                    "explicit_false_positive": "frozen first-letter rule accepted while a leading explicit A-F answer differs from the reference",
                    "explicit_mismatch_adjusted_accuracy": "(frozen correct minus explicit false positives) / frozen denominator",
                },
            },
        },
        {"id": "overlap_audit", "label": "Train-to-benchmark overlap audit", "path": "artifacts/runs/E-D10-6K-DATA-001/overlap/overlap_report.json"},
    ]
    headline = next(row for row in comparison["rows"] if row["benchmark"] == "overall_micro")
    headline_dataset = [{
        "scope": "frozen_r3",
        "vision_micro_delta": headline["vision_opd_delta_vs_base"],
        "zoom_delta": official_rows["zoombench"]["vision_opd_delta_vs_base"],
        "mmstar_delta": official_rows["mmstar"]["vision_opd_delta_vs_base"],
        "vstar_delta": official_rows["vstar"]["vision_opd_delta_vs_base"],
        "total": 2536,
    }]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "为什么当前 Vision-OPD 效果低于论文",
            "description": "Day 16 对训练缩放、评测逻辑和能力回归的可复核诊断。",
            "generatedAt": generated_at,
            "cards": [
                {"id": "micro_delta", "description": "冻结 R3 中 Vision-OPD 相对 Base 的 2,536 条样本加权变化。", "dataset": "headline", "sourceId": "day15_r3", "metrics": [{"label": "Micro 变化", "field": "vision_micro_delta", "format": "percent", "signed": True}]},
                {"id": "zoom_delta", "description": "冻结 R3 ZoomBench，N=845。", "dataset": "headline", "sourceId": "day15_r3", "metrics": [{"label": "ZoomBench 变化", "field": "zoom_delta", "format": "percent", "signed": True}]},
                {"id": "mmstar_delta", "description": "冻结 R3 MMStar，N=1,500。", "dataset": "headline", "sourceId": "day15_r3", "metrics": [{"label": "MMStar 变化", "field": "mmstar_delta", "format": "percent", "signed": True}]},
                {"id": "vstar_delta", "description": "冻结 R3 V* Bench，N=191。", "dataset": "headline", "sourceId": "day15_r3", "metrics": [{"label": "V* Bench 变化", "field": "vstar_delta", "format": "percent", "signed": True}]},
            ],
            "charts": [
                {
                    "id": "paper_vs_local_gain",
                    "title": "相对 Base 的准确率变化",
                    "subtitle": "论文三项均提升；本项目仅 ZoomBench 提升，MMStar 与 V* 下降。两套协议不可视为同一次复现。",
                    "type": "bar",
                    "dataset": "gain_comparison",
                    "sourceId": "diagnostic_data",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "benchmark", "type": "nominal", "label": "Benchmark"},
                        "y": {"field": "delta_vs_base", "type": "quantitative", "label": "准确率变化"},
                        "color": {"field": "setup", "type": "nominal", "label": "结果来源"},
                        "tooltip": [
                            {"field": "base_accuracy", "type": "quantitative", "label": "Base", "format": "percent"},
                            {"field": "vision_accuracy", "type": "quantitative", "label": "Vision-OPD", "format": "percent"},
                        ],
                    },
                },
                {
                    "id": "paired_transitions",
                    "title": "冻结 R3 的逐题正确性迁移",
                    "subtitle": "Vision-OPD 修正 306 条 Base 错题，同时把 411 条 Base 对题变错，净减少 105 条。",
                    "type": "bar",
                    "dataset": "paired_transitions",
                    "sourceId": "day15_r3",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "comparison", "type": "nominal", "label": "模型比较"},
                        "y": {"field": "count", "type": "quantitative", "label": "样本数"},
                        "color": {"field": "transition", "type": "nominal", "label": "变化方向"},
                        "tooltip": [{"field": "net_change", "type": "quantitative", "label": "净变化", "format": "number"}],
                    },
                },
            ],
            "tables": [
                {
                    "id": "config_table",
                    "title": "作者入口与本次正式训练的关键差异",
                    "subtitle": "核心 loss 参数相同，但优化批量、rollout 数和 EMA 更新频率不同。",
                    "dataset": "configuration_comparison",
                    "sourceId": "diagnostic_data",
                    "columns": [
                        {"field": "parameter", "label": "参数", "type": "text"},
                        {"field": "author_entry", "label": "作者入口", "type": "text"},
                        {"field": "current_run", "label": "本次运行", "type": "text"},
                        {"field": "ratio_or_effect", "label": "差异", "type": "text"},
                    ],
                },
                {
                    "id": "sensitivity_table",
                    "title": "评分器明确误报剔除敏感性",
                    "subtitle": "只剔除输出明确选择错误选项、却因 Answer 首字母 A 被判对的记录；不是完整 R4 重评分。",
                    "dataset": "parser_sensitivity",
                    "sourceId": "diagnostic_data",
                    "columns": [
                        {"field": "model", "label": "模型", "type": "text"},
                        {"field": "benchmark", "label": "Benchmark", "type": "text"},
                        {"field": "total", "label": "N", "format": "number"},
                        {"field": "frozen_r3_accuracy", "label": "冻结 R3", "format": "percent"},
                        {"field": "explicit_false_positives", "label": "明确误报", "format": "number"},
                        {"field": "explicit_mismatch_adjusted_accuracy", "label": "剔除后", "format": "percent"},
                    ],
                },
                {
                    "id": "category_table",
                    "title": "能力回归集中的类别",
                    "subtitle": "按明确误报剔除口径排序；V* 相对位置和 MMStar 逻辑推理降幅最大。",
                    "dataset": "category_sensitivity",
                    "sourceId": "diagnostic_data",
                    "defaultSort": {"field": "vision_delta_vs_base", "direction": "asc"},
                    "columns": [
                        {"field": "benchmark", "label": "Benchmark", "type": "text"},
                        {"field": "category", "label": "类别", "type": "text"},
                        {"field": "total", "label": "N", "format": "number"},
                        {"field": "base_accuracy", "label": "Base", "format": "percent"},
                        {"field": "vision_accuracy", "label": "Vision-OPD", "format": "percent"},
                        {"field": "vision_delta_vs_base", "label": "变化", "format": "percent"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# 为什么当前 Vision-OPD 效果低于论文"},
                {"id": "executive_summary", "type": "markdown", "body": "## Executive Summary / 执行摘要\n\n当前结果差并非一次明显的工程故障：两次训练均完整结束，未出现 OOM、NaN、abort 或 checkpoint 损坏。模型也并非完全没有学习，Vision-OPD 在 ZoomBench 提升 1.89 个百分点。\n\n证据最强的解释是：本次运行改变了论文方法的优化时间尺度。作者入口使用 batch 96、rollout n=8、约 65 个 trainer step/EMA 事件；本次使用 batch 8、rollout n=1、780 个 trainer step/EMA 事件，同时保持 LR 2e-6 和 EMA 0.05。每个 epoch 的 on-policy 序列减少到 1/8，EMA 事件增加到 12 倍；考虑 rollout 展开后的 PPO mini-batch，Student Adam 更新约从 520 次变为 780 次，但单次 mini-batch 从 96 降为 8。Cached Prefix 还移除了方法最关键的实时 on-policy prefix。\n\nR3 评分器另有一个已确认缺陷，会把 `Answer` 的 A 当作选项 A。它改变绝对分数，但剔除能明确确认的误报后，Base→Vision 的 micro 差值仍约为 −4.18 个百分点，因此训练差距仍然存在。结论可用于决定下一步，但精确分数需要修复评分器后建立 R4。"},
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["micro_delta", "zoom_delta", "mmstar_delta", "vstar_delta"]},
                {"id": "measured_results", "type": "markdown", "body": "## 实测结果与论文方向相反\n\n论文 Table 2 的 Qwen3.5-4B 从 Base 到 Vision-OPD，在 V*、ZoomBench、MMStar 分别变化 +7.86、+12.07、+1.07 个百分点。本项目冻结 R3 分别为 −7.85、+1.89、−7.07 个百分点。模型在细节计数和颜色识别上确有新增正确样本，但新错误更多，属于收益与遗忘同时发生。"},
                {"id": "gain_chart", "type": "chart", "chartId": "paper_vs_local_gain"},
                {"id": "verified_causes", "type": "markdown", "body": "## 已证实的差异\n\n1. **训练不是作者原配置复现。** 模型、6.2K 数据规模、JSD β=0.5、Top-K=100、EMA=0.05、response 1024 和 1 epoch 对齐，但 GPU、batch、rollout、Student mini-batch 和 EMA 事件次数没有对齐。\n2. **EMA 的时间尺度发生了数量级变化。** 实现每个 `update_policy` / trainer step 在全部 PPO mini-batch 完成后执行一次 `teacher = 0.95 × teacher + 0.05 × student`。仅看初始教师权重的理论系数，65 次后约剩 3.56%，780 次后约为 4.3e−18。该值不是性能预测，但证明相同 EMA 系数在两种 step 粒度下并不等价。\n3. **Cached Prefix 不是论文主方法。** 它在 780 steps 中 online generation calls 为 0，无法保留论文强调的“沿当前 Student rollout 蒸馏”机制。"},
                {"id": "config", "type": "table", "tableId": "config_table"},
                {"id": "behavior", "type": "markdown", "body": "## 回归不只来自一个 Benchmark\n\n冻结 R3 的逐题迁移显示，Vision-OPD 产生 306 个错→对，也产生 411 个对→错。明确误报剔除后，最大分项退化出现在 V* relative_position（约 −19.74 个百分点）、MMStar logical reasoning（−12.80）和 fine-grained perception（−9.60）。这更像训练后能力分布发生迁移，而不是单次推理失败。"},
                {"id": "transition_chart", "type": "chart", "chartId": "paired_transitions"},
                {"id": "category", "type": "table", "tableId": "category_table"},
                {"id": "evaluation_issue", "type": "markdown", "body": "## 评测器缺陷改变绝对值，但没有消除主结论\n\n`extract_answer_official()` 保留 `Answer:` 前缀，随后 `extract_first_option()` 在无法先识别显式选项时会抓取第一个大写字母。例如参考答案 A、模型输出 `Answer: D` 可被判为正确。现有数据中至少有 Base 86 条、Vision-OPD 87 条、Cached 48 条属于可明确确认的这种误报。\n\n该问题意味着 Day 15 的完整性 Gate 证明了记录齐全和协议一致，却没有证明评分逻辑正确。下表只做保守敏感性分析，冻结 R3 文件未被修改。"},
                {"id": "sensitivity", "type": "table", "tableId": "sensitivity_table"},
                {"id": "other_factors", "type": "markdown", "body": "## 次要但必须保留的限制\n\n- 论文 Judge 记录为 GPT-OSS-120B，本项目因不可用改用冻结 Qwen3.5-4B Base Judge；该替代的偏差尚未量化。\n- V* 有 21/191 条测试图像与训练图像确认重合。剔除明确误报后，Vision-OPD 在重合 21 条为 18/21，在其余 170 条为 125/170；两部分都没有支持论文式增益，重合不能解释当前低分。\n- 本项目 Base 在 V* 和 ZoomBench 接近论文量级，但 MMStar 在保守敏感性口径下明显低于论文 Base。这说明 MMStar 的绝对数值尤其不能直接横向对齐。"},
                {"id": "next_steps", "type": "markdown", "body": "## 下一步行动\n\n1. **先修评分，不立刻重训。** 修复 Answer 前缀解析并新增回归用例；建立独立 R4，重新路由所有 Base、Vision-OPD、Cached 样本，必要时补跑 Judge。保留 R3 作为历史结果。\n2. **用最小对照验证训练缩放假设。** 优先做 64–128 prompt Pilot，对比当前 batch 8 / n=1 与尽可能接近 batch 96 / n=8 的配置；至少把 EMA 系数按 optimizer-step 粒度重新标定。\n3. **确认 Pilot 的配对趋势后再花费完整训练成本。** Gate 应同时要求 ZoomBench 增益、MMStar/V* 不出现显著回归，并查看错→对与对→错。\n4. **不要仅延长当前训练。** 当前 loss 已从 0.02727 降至 0.000778，继续相同轨迹没有证据能恢复外部能力，反而可能加深漂移。"},
                {"id": "validation", "type": "markdown", "body": "## 验证结论与限制\n\n**总体评估：Share with caveats。** 训练完成、分母、逐题配对和配置差异均已复核；核心结论“当前缩规模配置未复现论文收益”可靠。精确准确率存在高严重度评分缺陷，必须在 R4 后才能作为最终对外指标。\n\n最可能原因是 batch / rollout / update / EMA 时间尺度共同变化，但现有单次运行不能分离各因素的因果贡献。当前证据不支持把问题归因于 GPU 故障、checkpoint 损坏、训练未完成或单纯训练时间不足。"},
                {"id": "methods", "type": "markdown", "body": "## 数据、定义与方法\n\n数据截至 2026-09-08 UTC。冻结 R3 包含 ZoomBench 845、MMStar 1,500、V* 191，共 2,536 条；所有 inference failure 均保留在分母。`明确误报剔除` 仅指：旧 first-letter 规则判对，同时模型回答开头可明确解析为与参考不同的 A–F 选项。该分析不会推断无法明确解析的回答，也不会修改原始 score。\n\n可继续核查的问题：R4 重评分后各 Benchmark 差值是否稳定；更接近作者 batch / rollout 的 Pilot 是否恢复 V* relative_position；EMA 按样本量或 step 数重新标定后是否减少遗忘。"},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline_dataset,
                "gain_comparison": gain_rows,
                "benchmark_long": benchmark_long,
                "paired_transitions": transition_rows,
                "configuration_comparison": config_rows,
                "parser_sensitivity": sensitivity_rows,
                "parser_summary": parser_summary,
                "category_sensitivity": category_rows,
                "vstar_overlap_sensitivity": overlap_rows,
            },
            "accessIssues": [],
        },
        "sources": sources,
        "package_info": {"originUrl": "artifact://vision-opd-gap-diagnostic", "controls": {"edit": False, "refresh": False}},
    }
    artifact_path = OUTPUT / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "diagnostic": str(diagnostic_path), "artifact": str(artifact_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
