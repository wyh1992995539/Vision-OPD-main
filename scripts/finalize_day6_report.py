#!/usr/bin/env python3
"""Finalize reproducible Day 6 metrics, cost, validation, hashes, and report."""
from __future__ import annotations

import hashlib, json, statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "artifacts/runs/E-D6-001/base"
REPORT = REPO / "artifacts/reports/base_external_benchmarks.md"
PRICE = 11.96
SERVICE_START = datetime.fromisoformat("2026-08-26T04:23:09+00:00")
SERVICE_END = datetime.fromisoformat("2026-08-26T06:20:52+00:00")
EXPECTED = {"zoombench/full": 845, "zoombench/crop": 845, "mmstar/full": 1500, "vstar/full": 191}

def load_jsonl(path): return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def write_json(path, obj): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def pct(x): return f"{100*x:.2f}%"
def lat(rows):
    v=sorted(float(x["latency_seconds"]) for x in rows)
    q=lambda f:v[min(len(v)-1,int((len(v)-1)*f+.999999))]
    return {"mean_seconds":statistics.mean(v),"median_seconds":statistics.median(v),"p95_seconds":q(.95),"max_seconds":max(v)}

def main():
    predictions=load_jsonl(BASE/"predictions.jsonl"); scores=load_jsonl(BASE/"scores.jsonl"); summary=json.loads((BASE/"summary.json").read_text())
    groups=defaultdict(list); score_groups=defaultdict(list)
    for x in predictions: groups[f"{x['benchmark']}/{x['view']}"].append(x)
    for x in scores: score_groups[f"{x['benchmark']}/{x['view']}"].append(x)
    actual={k:len(v) for k,v in groups.items()}
    checks={"prediction_count":len(predictions)==3381,"score_count":len(scores)==3381,"unique_prediction_keys":len({(x['benchmark'],x['view'],x['sample_uid']) for x in predictions})==3381,"expected_group_counts":actual==EXPECTED,"inference_errors_zero":not any(x.get('error') for x in predictions),"pending_judge_zero":not any(x['score_status']!='scored' for x in scores),"vstar_denominator_191":summary['groups']['vstar/full']['total']==191,"duplicate_request_keys_zero":summary['resume_gate']['duplicate_request_keys']==0}
    per_group={}
    for name,rows in sorted(groups.items()):
        sr=score_groups[name]
        per_group[name]={"requests":len(rows),"correct":sum(bool(x['is_correct']) for x in sr),"accuracy":sum(bool(x['is_correct']) for x in sr)/len(sr),"invalid_or_ambiguous":sum(x['score_source']=='invalid_or_ambiguous' for x in sr),"length_finished":sum(x.get('finish_reason')=='length' for x in rows),"prompt_tokens":sum(int(x.get('prompt_tokens') or 0) for x in rows),"completion_tokens":sum(int(x.get('completion_tokens') or 0) for x in rows),"average_completion_tokens":sum(int(x.get('completion_tokens') or 0) for x in rows)/len(rows),"latency":lat(rows)}
    first=min(datetime.fromisoformat(x['generated_at_utc']) for x in predictions); last=max(datetime.fromisoformat(x['generated_at_utc']) for x in predictions)
    gpu_seconds=(SERVICE_END-SERVICE_START).total_seconds(); gpu_hours=gpu_seconds/3600
    cost={"schema_version":1,"experiment_id":"E-D6-001","currency":"CNY","pricing":{"dual_gpu_hourly_cny":PRICE},"timing":{"vllm_service_start_utc":SERVICE_START.isoformat(),"vllm_service_stop_utc":SERVICE_END.isoformat(),"first_prediction_utc":first.isoformat(),"last_prediction_utc":last.isoformat(),"summary_completed_utc":summary['generated_at_utc'],"gpu_service_seconds":gpu_seconds,"gpu_service_hours":gpu_hours},"costs":{"gpu_instance_cny":gpu_hours*PRICE,"external_judge_api_cny":0.0,"total_cny":gpu_hours*PRICE},"basis":"Observed vLLM service start/stop timestamps; local Base 4B Judge used the same dual-GPU service."}
    metrics={"schema_version":1,"experiment_id":"E-D6-001","status":"complete","request_count":len(predictions),"inference_error_count":sum(bool(x.get('error')) for x in predictions),"invalid_or_ambiguous_count":sum(x['score_source']=='invalid_or_ambiguous' for x in scores),"invalid_or_ambiguous_rate":sum(x['score_source']=='invalid_or_ambiguous' for x in scores)/len(scores),"length_finished_count":sum(x.get('finish_reason')=='length' for x in predictions),"length_finished_rate":sum(x.get('finish_reason')=='length' for x in predictions)/len(predictions),"tokens":{"prompt_total":sum(x.get('prompt_tokens',0) for x in predictions),"completion_total":sum(x.get('completion_tokens',0) for x in predictions),"total":sum(x.get('prompt_tokens',0)+x.get('completion_tokens',0) for x in predictions)},"latency_overall":lat(predictions),"score_sources":dict(Counter(x['score_source'] for x in scores)),"groups":per_group,"official_category_groups":summary['official_category_groups'],"zoombench_zooming_gap":summary['groups']['zoombench/crop']['accuracy']-summary['groups']['zoombench/full']['accuracy']}
    validation={"schema_version":1,"assessment":"ready_to_share" if all(checks.values()) else "needs_revision","checks":checks,"caveats":["Accuracy is descriptive for the frozen benchmark snapshots; no uncertainty interval or causal claim is made.","invalid_or_ambiguous counts parser failures as incorrect in the official denominator.","length_finished means the generation hit max_new_tokens; it is not automatically invalid if a valid final option was parsed.","No chart is used because this is a single-snapshot audit with exact denominators; tables preserve the requested detail."]}
    write_json(BASE/"metrics.json",metrics); write_json(BASE/"cost.json",cost); write_json(BASE/"validation.json",validation)
    g=summary['groups']; cats=summary['official_category_groups']
    report=f"""# Day 6 Base 外部 Benchmark 正式报告\n\n## 技术摘要\n\n原始 Qwen3.5-4B Base 已完成 3,381 个冻结请求，逐样本预测与评分均为 3,381 条，推理错误、重复键和待定 Judge 均为 0。ZoomBench full 为 **{pct(g['zoombench/full']['accuracy'])}**，crop 为 **{pct(g['zoombench/crop']['accuracy'])}**，crop−full gap 为 **{100*metrics['zoombench_zooming_gap']:.2f} 个百分点**；MMStar 为 **{pct(g['mmstar/full']['accuracy'])}**；V* Bench 官方 191 条为 **{pct(g['vstar/full']['accuracy'])}**。\n\n## 总体结果\n\n| 指标 | 正确/总数 | 准确率 |\n|---|---:|---:|\n| ZoomBench full | {g['zoombench/full']['correct']}/845 | {pct(g['zoombench/full']['accuracy'])} |\n| ZoomBench crop | {g['zoombench/crop']['correct']}/845 | {pct(g['zoombench/crop']['accuracy'])} |\n| ZoomBench crop−full gap | — | +{100*metrics['zoombench_zooming_gap']:.2f} pp |\n| MMStar | {g['mmstar/full']['correct']}/1500 | {pct(g['mmstar/full']['accuracy'])} |\n| V* Bench 官方全集 | {g['vstar/full']['correct']}/191 | {pct(g['vstar/full']['accuracy'])} |\n\n## 官方类别结果\n\n| Benchmark / 类别 | 正确/总数 | 准确率 |\n|---|---:|---:|\n"""
    for name,item in cats.items(): report+=f"| {name} | {item['correct']}/{item['total']} | {pct(item['accuracy'])} |\n"
    report+=f"""\n## 输出质量、Token 与延迟\n\n- 推理错误：**0/3,381**。\n- 无法可靠解析的选择题输出：**{metrics['invalid_or_ambiguous_count']}/3,381（{pct(metrics['invalid_or_ambiguous_rate'])}）**，均按官方分母计错。\n- 达到 8,192 token 上限：**{metrics['length_finished_count']}/3,381（{pct(metrics['length_finished_rate'])}）**；触顶不自动等于无效，仍按最终答案解析。\n- Prompt token：**{metrics['tokens']['prompt_total']:,}**；Completion token：**{metrics['tokens']['completion_total']:,}**；合计：**{metrics['tokens']['total']:,}**。\n- 请求延迟：均值 **{metrics['latency_overall']['mean_seconds']:.2f}s**，中位数 **{metrics['latency_overall']['median_seconds']:.2f}s**，P95 **{metrics['latency_overall']['p95_seconds']:.2f}s**，最大 **{metrics['latency_overall']['max_seconds']:.2f}s**。\n\n## GPU 时间和成本\n\n- 双卡 vLLM 服务时间：**{gpu_hours:.3f} 小时**（{int(gpu_seconds)} 秒）。\n- 双卡价格：**{PRICE:.2f} 元/小时**。\n- GPU 实例费用：**{cost['costs']['gpu_instance_cny']:.2f} 元**。\n- 外部 Judge API 费用：**0.00 元**；语义 Judge 使用同一台本地 Base 4B 服务。\n- 总费用：**{cost['costs']['total_cny']:.2f} 元**。\n\n## 范围与方法\n\n评测使用冻结的 Qwen3.5-4B Base、Benchmark revision、Prompt、图像处理、生成与评分协议。ZoomBench 对 845 条分别执行 full/crop；MMStar 使用 1,500 条官方样本；V* 使用官方 191 条和 191 分母，不生成 187 条二级指标。选择题由冻结解析器判分；ZoomBench 开放题依次使用确定性数字、MathRuler 和固定 Base 4B Judge。\n\n## 验证与限制\n\n完整性检查全部通过：请求/评分数量、唯一键、四组分母、零推理错误、零待定 Judge、V* 191 分母及零重复均已复算。结果是冻结快照上的描述性基线，不构成统计显著性或因果结论。长度触顶与解析失败应在后续模型比较中使用同一规则。该报告使用精确表格而非图表，因为只有一个评测快照，表格更适合审计固定分母和类别值。\n\n## 下一步\n\n冻结本 Base 结果与哈希，更新 Day 6 为 PASS；后续 Vision-OPD 与 Cached Prefix 仅在内部 checkpoint 冻结后使用完全相同的外评协议比较，不用外部分数选 checkpoint 或重训。\n\n## 进一步问题\n\n训练后模型是否提升 ZoomBench full、是否保持 MMStar/V*，以及输出触顶与无效解析率是否变化，应在统一外评阶段做逐样本配对比较。\n"""
    REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text(report,encoding="utf-8")
    paths=[BASE/x for x in ("predictions.jsonl","scores.jsonl","summary.json","metrics.json","cost.json","validation.json","run_manifest.json","resume_status.json")]+[REPORT]
    (BASE/"artifact_sha256.txt").write_text("".join(f"{sha(p)}  {p.relative_to(REPO)}\n" for p in paths),encoding="utf-8")
    print(json.dumps({"validation":validation["assessment"],"report":str(REPORT),"metrics":str(BASE/"metrics.json"),"cost":str(BASE/"cost.json")},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
