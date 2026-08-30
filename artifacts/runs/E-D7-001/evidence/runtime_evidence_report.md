# E-D7-001 Day 7 运行时证据

## 结论

状态：`PASS_WITH_CAVEAT`

Vision-OPD 双卡 Smoke 已完成 2 个真实 optimizer steps 和 16 条在线 rollout。Student optimizer 更新、Teacher optimizer 隔离、Teacher 无梯度以及 optimizer 后 EMA 更新均有运行时数值证据。

## 运行时验收

| 指标 | Step 1 | Step 2 | 验收 |
|---|---:|---:|---|
| `actor/vopd_loss` | 0.04209464 | 0.02820932 | 有限且大于 0 |
| `actor/grad_norm` | 5.213622 | 6.766190 | 有限且大于 0 |
| `evidence/runtime_probe_enabled` | 1 | 1 | PASS |
| `evidence/parameter_probe_elements` | 512 | 512 | PASS |
| Student optimizer 后最大参数变化 | 2.0266e-6 | 2.0018e-6 | PASS，Student 被更新 |
| Teacher optimizer 后最大参数变化 | 0 | 0 | PASS，optimizer 未更新 Teacher |
| Teacher 非空梯度数 | 0 | 0 | PASS，Teacher 无梯度 |
| Teacher EMA 后最大参数变化 | 2.3842e-7 | 4.7684e-7 | PASS，EMA 改变 Teacher |
| `evidence/ema_update_applied` | 1 | 1 | PASS |
| Prompt clip ratio | 0 | 0 | PASS，无静默截断 |

## 产物完整性

- 训练进度：`2/2`。
- Rollout：Step 1 为 8 条，Step 2 为 8 条，共 16 条。
- Checkpoint：`global_step_2`，包含两个 rank 的 model、optimizer 和 extra-state 分片。
- 训练日志 SHA256：`b470672347a65d9c90a9ad5b6b687a4e1a3756eb9914213aaf40753552053845`。
- 完整 1024 条 Token 审计：Student max 7880，Teacher max 2213，超长数和处理错误数均为 0。

## 非阻断警告

训练达到 100% 且 checkpoint 保存完成后，一个 DataLoader worker 在进程清理阶段被系统 `Killed`。这没有破坏本次两步 Smoke 的指标、rollout 或 checkpoint 产物，因此不阻断 Day 7。

日志中的 `perf/cpu_memory_used_gb≈193.9` 来自 `psutil.virtual_memory().used`，是宿主机整体已用内存的瞬时值，不是训练进程 RSS，也不是容器峰值。当前 cgroup 记录为 `oom=0`、`oom_kill=0`，没有证据表明该清理警告由容器 OOM 引起。Day 8 应额外记录 cgroup memory 或进程 RSS，并执行 checkpoint 重载验证；不能仅根据 193.9 GiB 这一指标要求降低 worker 数。

详细机器可读结果见 `runtime_evidence_summary.json`。
