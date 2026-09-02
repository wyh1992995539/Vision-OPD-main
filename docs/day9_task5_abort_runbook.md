# Day 9 Task 5：E-D10-001 正式训练守护手册

## 状态和边界

Task 5 只冻结中止策略、监控实现和启动流程，不启动正式 GPU 训练。正式配置仍为
`configs/vopd_1024.yaml`，SHA256 必须保持
`5977d0b7adda448287d7410431c9461a6f6f53c04792390b9b13d9529a00b30c`。

中止策略位于 `configs/vopd_abort_policy.yaml`。正式训练只能由
`scripts/run_vopd_guarded.py` 启动；直接执行 `run_vopd_2gpu.sh --run` 会以退出码 2 拒绝。

## CPU-only 复核

```bash
python scripts/run_vopd_guarded.py --preflight-only
python -m unittest tests.test_vopd_abort_guard -v
python scripts/monitor_vopd_training.py \
  --policy configs/vopd_abort_policy.yaml \
  --replay-metrics artifacts/runs/E-D8-001/metrics.jsonl \
  --replay-log artifacts/runs/E-D8-001/logs/train.log
```

上述命令不会启动 GPU 训练。Day8 回放的预期结果是：8 个结构化指标步没有违反
Student/Teacher 合同，但日志规则识别出一次 `dataloader_worker_killed`。

## Day 10 正式启动

先在 AutoDL 控制台读取最新累计费用，记录带时区的观测时间；观测值有效期为 15 分钟。

```bash
python scripts/run_vopd_guarded.py \
  --config configs/vopd_1024.yaml \
  --policy configs/vopd_abort_policy.yaml \
  --current-autodl-cost-cny <最新累计费用> \
  --billing-observed-at-utc <ISO-8601时间> \
  --run
```

启动前会再次检查 Task4、正式配置哈希、`dataloader_num_workers=0`、128 步合同、
Git clean、输出冲突、磁盘、两张 GPU、cgroup v2 和预算。任一 Gate 失败均不启动训练。

## 证据和退出码

运行期每 10 秒将逐卡显存、进程树 RSS、cgroup 内存事件和磁盘空间写入
`artifacts/runs/E-D10-001/evidence/telemetry/`。解析出的训练指标、触发规则、终止信号、
最终 checkpoint 校验分别写入 `runtime_metrics.jsonl`、`guard_events.jsonl`、
`guard_summary.json` 和 `exit_receipt.json`。

| 退出码 | 含义 |
|---:|---|
| 0 | 训练正常退出且 step 128 checkpoint 完整 |
| 2 | 直接绕过守护器或参数错误 |
| 40 | 中止规则触发或无法保留原始训练退出码 |
| 41 | Day 10 动态启动 Gate 失败，训练未启动 |
| 42 | 训练退出为 0，但最终 checkpoint 后置校验失败 |

规则触发后先向整个训练进程组发送 `SIGTERM`，等待 60 秒，再在需要时发送 `SIGKILL`。
守护器不会自动删除 Day8、Task5 或正式训练证据。

