# E-D10-001 Task 5 已完成：允许进入 Day 10 动态启动检查

> 生成时间：2026-09-02T06:55:17.608888+00:00
> 决策：**PASS_TO_DAY10**
> GPU 使用：**false**

## 结论

Task 5 的可执行中止策略、逐卡 GPU/进程树 RSS/cgroup/磁盘观测、训练指标解析、进程组终止、
最终 checkpoint 后置校验和防绕过入口均已落地。任务五完成，可以进入 Day10；这不等于可以跳过
启动时动态 Gate，AutoDL 累计费用必须在启动前 15 分钟内刷新。

正式训练配置 SHA256 仍为 `5977d0b7adda448287d7410431c9461a6f6f53c04792390b9b13d9529a00b30c`，没有因 Task 5 改动训练数学合同。
中止策略 SHA256 为 `6fbea2890817a08baaaeb911e7a491d5c1003dc2bc08c8f77057ea5311f29174`。

## Gate

| Gate | 状态 |
|---|---|
| task4_and_static_contract | PASS |
| formal_config_hash_unchanged | PASS |
| dataloader_workers_zero | PASS |
| policy_schema_and_formulas | PASS |
| unit_and_replay_tests | PASS |
| day8_metric_contract_clean | PASS |
| day8_killed_event_detected | PASS |
| direct_run_blocked | PASS |
| cpu_only_task5 | PASS |
| source_worktree_clean_before_report | PASS |

## 冻结控制

| 控制 | 值 |
|---|---:|
| 观测周期 | 10 秒 |
| 最长墙钟时间 | 38 小时 |
| GPU 显存中止比例 | 95%，连续 3 次 |
| cgroup 内存中止比例 | 95%，连续 3 次 |
| 启动磁盘要求 | 119438631082 bytes |
| 运行期磁盘软下限 | 62403670101 bytes |
| 运行期磁盘硬下限 | 5368709120 bytes |
| 最终 checkpoint | step 128，13 个必需文件 |

NaN/Inf、Teacher 直接梯度、Teacher optimizer 改变、cgroup OOM、checkpoint 保存错误和磁盘硬下限
属于立即中止条件；EMA/Student 不更新、连续生成错误、内存压力、磁盘软下限和日志心跳使用冻结的
连续阈值。中止先发送 `SIGTERM`，60 秒后必要时升级为 `SIGKILL`。

## Day8 回放

- 结构化指标：8 步，合同异常 0 项。
- 日志命中：dataloader_worker_killed。
- 这证明守护器能保留 Day8 的真实 caveat，而不是将其改写为已解决。

## Day10 启动前仍需执行

1. 在 AutoDL 控制台读取最新累计费用，并记录带时区的 ISO-8601 时间。
2. 保持 Git 工作区 clean；重新检查输出冲突、磁盘、两张 GPU 和 cgroup v2。
3. 仅使用以下入口：

```bash
python scripts/run_vopd_guarded.py --current-autodl-cost-cny <LATEST> --billing-observed-at-utc <ISO-8601> --run
```

直接执行 `bash scripts/run_vopd_2gpu.sh --run` 已被拒绝。完整操作见
`docs/day9_task5_abort_runbook.md`。
