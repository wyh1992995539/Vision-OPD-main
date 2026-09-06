#!/usr/bin/env python3
"""Reconstruct save-window evidence without loading checkpoint tensors."""

import argparse
import hashlib
import json
from pathlib import Path

GIB = 1024**3


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(pilot):
    telemetry = pilot / "evidence/telemetry"
    cpu = read_rows(telemetry / "cgroup_memory.jsonl")
    rss = read_rows(telemetry / "process_rss.jsonl")
    disk = read_rows(telemetry / "disk.jsonl")
    postflight = json.loads((pilot / "evidence/postflight.json").read_text())
    duration = postflight["steps"][-1]["checkpoint_save_seconds"]
    # Console metrics lack a timestamp for save entry. Estimate backwards from
    # last telemetry, explicitly retaining polling/teardown uncertainty.
    window_end = cpu[-1]["elapsed_seconds"]
    window_start = window_end - duration
    baseline = max((r for r in cpu if r["elapsed_seconds"] <= window_start),
                   key=lambda r: r["elapsed_seconds"])
    peak = max(cpu, key=lambda r: r["memory_current_bytes"])
    timeline = []
    for row in cpu:
        if row["elapsed_seconds"] < baseline["elapsed_seconds"]:
            continue
        process = min(rss, key=lambda r: abs(r["elapsed_seconds"] - row["elapsed_seconds"]))
        storage = min(disk, key=lambda r: abs(r["elapsed_seconds"] - row["elapsed_seconds"]))
        timeline.append({"timestamp_utc": row["timestamp_utc"], "elapsed_seconds": row["elapsed_seconds"],
                         "cgroup_bytes": row["memory_current_bytes"], "tree_rss_sum_bytes": process["rss_bytes"],
                         "disk_free_bytes": storage["free_bytes"], "memory_stat": row.get("memory_stat")})
    shards = [{"path": str(p), "bytes": p.stat().st_size}
              for p in sorted((pilot / "checkpoints/global_step_2/actor").glob("*.pt"))]
    sources = [telemetry / n for n in ("cgroup_memory.jsonl", "process_rss.jsonl", "disk.jsonl")]
    sources += [pilot / "evidence/postflight.json", pilot / "logs/train.log"]
    return {"schema_version": 1, "status": "ANALYZED_WITH_HISTORICAL_BREAKDOWN_GAP",
            "gpu_used": False, "checkpoint_tensors_loaded": False,
            "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
            "checkpoint_save_seconds": duration, "estimated_save_window_elapsed_seconds": [window_start, window_end],
            "window_caveat": "Estimated from final telemetry minus save duration; not an exact save-entry timestamp.",
            "baseline": baseline, "peak": peak,
            "growth_from_baseline_gib": (peak["memory_current_bytes"] - baseline["memory_current_bytes"]) / GIB,
            "historical_memory_stat_available": all(r.get("memory_stat") is not None for r in cpu),
            "timeline": timeline, "shard_sizes": shards,
            "evidence_limits": [
                "No historical memory.stat: cannot quantify anon/file/shmem at the peak.",
                "RSS sums can double-count shared mappings and omit processes outside the tree.",
                "cgroup minus summed RSS is NOT a valid page-cache estimate.",
                "Shard disk size is NOT identical to temporary in-memory tensor allocation.",
                "Stable RSS with rising cgroup usage during writes supports, but does not prove, page-cache growth.",
            ],
            "code_findings": [
                "Old save_checkpoint retains model_state_dict while constructing/saving optimizer_state_dict.",
                "Old per-rank torch.save has no explicit fsync/file-cache advice between shards.",
                "Current revision releases temporary dict references before the next shard; never mutates live tensors.",
                "Optional per-shard fsync then DONTNEED only advises eviction of the newly saved file cache.",
            ],
            "validation_required": "New GPU Pilot-16 plus checkpoint cold reload; peak reduction is not measured yet.",
            "resource_thresholds_lowered": False, "pilot_64_authorized_by_this_report": False}


def markdown(report):
    lines = ["# Pilot-16 checkpoint 内存分析", "", "结论：保存阶段是主要峰值窗口；文件缓存增长有间接证据，但历史分类遥测不足以精确归因。", "",
             f"- 保存耗时：{report['checkpoint_save_seconds']:.2f} 秒。",
             f"- 窗口前最近采样：{report['baseline']['memory_current_bytes']/GIB:.2f} GiB。",
             f"- cgroup 峰值：{report['peak']['memory_current_bytes']/GIB:.2f} GiB。",
             "- 旧遥测没有 memory.stat，不能精确给出模型/优化器、匿名内存、文件缓存的比例。", "",
             "| elapsed 秒 | cgroup GiB | 进程树 RSS 合计 GiB | 磁盘可用 GiB |",
             "| --- | --- | --- | --- |"]
    for r in report["timeline"]:
        lines.append(f"| {r['elapsed_seconds']:.1f} | {r['cgroup_bytes']/GIB:.2f} | {r['tree_rss_sum_bytes']/GIB:.2f} | {r['disk_free_bytes']/GIB:.2f} |")
    lines += ["", "RSS 包含共享映射重复计数，不能从 cgroup 总量减去 RSS 来估计缓存。保存窗口起点是推算值，不是精确埋点。", "",
              "## 修改与边界", "",
              "1. 模型 state_dict 保存返回后即释放临时引用，再创建优化器 state_dict；不清空张量，不改变训练状态。",
              "2. 双卡启动器显式启用分片 fsync → POSIX_FADV_DONTNEED。仅请求回收本次保存文件的干净缓存，不清理全机缓存。",
              "3. 每个采样新增原始 memory.stat（anon/file/shmem/dirty/writeback 等）；守护仍使用 memory.current，不能扣掉缓存。", "",
              "缓存建议可能不被内核执行，且只能在单个分片完成后生效；不能消除分片生成/写入中的峰值。fsync 也可能增加保存时间。",
              "没有调整 batch、n、序列长度、学习率、offload、checkpoint 内容/频率或任何资源门槛。没有加载/删除旧 checkpoint。", "",
              "## 必须后续验证", "",
              "CPU 测试仅验证恢复语义、顺序、错误路径及临时引用释放，不证明 FSDP 双卡峰值下降。",
              "下一次需要新的 Pilot-16、分类遥测和冷重载检查；旧 Pilot-16 PASS 不等于新保存实现已实测。",
              "Pilot-64 224 GiB 下限和 95% 连续三次保护保留，220 GiB 尚未重新获得安全结论。", "",
              "参考：[Python 文件同步/缓存建议接口](https://docs.python.org/3/library/os.html#os.posix_fadvise)，",
              "[Linux cgroup memory.stat 定义](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html)。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("artifacts/runs/E-D11-6K-GATE-001/pilot/16"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/E-D11-6K-GATE-001/checkpoint_memory_revision"))
    args = parser.parse_args()
    report = analyze(args.pilot)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "analysis.md").write_text(markdown(report))
    print(json.dumps({k: report[k] for k in ("status", "checkpoint_save_seconds", "historical_memory_stat_available")}))


if __name__ == "__main__":
    main()
