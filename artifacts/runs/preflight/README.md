# Day 1 Preflight

> 创建日期：2026-08-20  
> 总体状态：CONDITIONAL PASS

本目录保存 Day 1 原始审计证据。数据获取方案已经确认；双卡验证因当前无实例条件延期，但在任何 GPU Smoke 或正式训练前仍是硬门槛。

| 检查项 | 状态 | 证据 |
|---|---|---|
| Git 基线与工作树 | PASS | git_state.txt、working_tree.patch |
| Python 与核心依赖 | PASS WITH WARNINGS | env.txt、pip_check.txt |
| Qwen3.5-4B 文件清单与哈希 | PASS | model_manifest.txt、model_sha256.txt |
| 当前磁盘快照 | PASS | disk.txt |
| 双卡 RTX PRO 6000 可见性 | DEFERRED | hardware.txt；首次 GPU Smoke 前必须完成 |
| 本地抽取冻结子集后上传 | PASS | storage_decision.md |

服务器当前数据盘不满足完整原始数据下载安全线，因此禁止下载完整数据，只允许上传冻结后的 1024/128/64 子集。本目录不得保存 Token、认证文件或完整环境变量。
