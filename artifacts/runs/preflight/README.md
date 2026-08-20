# Day 1 Preflight

> 创建日期：2026-08-20  
> 总体状态：IN PROGRESS

本目录保存 Day 1 原始审计证据。实际验证项记录为 PASS；尚未在双卡实例验证或尚未执行的外部操作记录为 PENDING。

| 检查项 | 状态 | 证据 |
|---|---|---|
| Git 基线与工作树 | PASS | git_state.txt、working_tree.patch |
| Python 与核心依赖 | PASS WITH WARNINGS | env.txt、pip_check.txt |
| Qwen3.5-4B 文件清单与哈希 | PASS | model_manifest.txt、model_sha256.txt |
| 当前磁盘快照 | PASS | disk.txt |
| 双卡 RTX PRO 6000 可见性 | PENDING | hardware.txt |
| 数据盘扩容或异机抽取决策 | PENDING | storage_decision.md |

当前数据盘不满足完整原始数据下载安全线；存储项 PASS 前禁止下载完整数据。本目录不得保存 Token、认证文件或完整环境变量。
