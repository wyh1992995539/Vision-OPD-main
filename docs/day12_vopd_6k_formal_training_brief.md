# Day 12 Vision-OPD 6K 正式训练工作简报

> 执行日期：2026-09-07（UTC）  
> 实验 ID：`E-D12-6K-VOPD-001`  
> 最终状态：**PASS**  
> 正式训练：**780/780 optimizer steps 完成**  
> 最终 checkpoint：`artifacts/runs/E-D12-6K-VOPD-001/checkpoints/global_step_780`

## 技术摘要

Day 12 从冻结的原始 Qwen3.5-4B Base 冷启动，使用 train-6241、online Student prefix、Crop Teacher、Top-K JSD 和 EMA Teacher 完成 Vision-OPD 6K 正式训练。数据按原生 `drop_last=True` 执行：读取 6,241 条源记录，有效训练 6,240 条，padding 0，丢弃 shuffle 后尾部 1 条，global batch 8、1 epoch，共完成 780 个 optimizer steps。

训练通过 `scripts/run_day12_vopd.py` 的受保护入口启动。启动前重新检查配置与数据哈希、两张 GPU、240 GiB cgroup、磁盘、输出冲突和冻结 Gate；训练中每 10 秒采集 GPU、进程树 RSS、cgroup 和磁盘遥测，并监控数值、Student 更新、Teacher 梯度、EMA、心跳、OOM 和 checkpoint。运行从 `2026-09-07T03:39:48Z` 持续至 `09:39:44Z`，总墙钟约 6.00 小时，退出码 0；按双卡合计 14 元/小时估算本次费用为 83.99 元。

最终 `global_step_780` checkpoint 的 marker、目录和 13 个必需文件全部存在且非空。训练日志和 780 行运行指标连续完整，没有 NaN/Inf、OOM、Teacher 直接梯度、Teacher optimizer 更新、EMA 漏更新或 rollout abort。该结论证明正式训练与 checkpoint 完成，不包含模型能力评测；模型合并、哈希与冷加载属于 Day 13。

## 一、实际完成的工作

| 顺序 | 实际工作 | 结果 | 状态 |
|---:|---|---|---|
| 1 | 执行实时启动 Gate | 两张 RTX PRO 6000 空闲；cgroup 上限 240 GiB；配置、数据、Git、磁盘与输出路径检查通过 | PASS |
| 2 | 从冻结 Base 启动正式训练 | 使用 online prefix、Student 原图、Teacher crop、response 1024、LR `2e-6`、Top-K 100、EMA 0.05 | PASS |
| 3 | 核对最初 3 steps | loss 有限；step 2 起学习率和 Student delta 为正；Teacher optimizer delta 与直接梯度均为 0；EMA 生效 | PASS |
| 4 | 运行自动守护与遥测 | 2,129 组 GPU/cgroup/磁盘/进程采样落盘；无中止事件、OOM 或心跳失败 | PASS |
| 5 | 保存中期恢复点 | step 390 checkpoint 成功；最终 checkpoint 完成后按 `max_actor_ckpt_to_keep=1` 淘汰中间分片 | PASS |
| 6 | 完成最终 checkpoint | marker=`780`，`global_step_780` 目录和 13 个必需文件完整、非空 | PASS |
| 7 | 写入退出与费用凭据 | guard status=PASS、return code=0、latest step=780；墙钟和估算费用落盘 | PASS |

## 二、实际训练合同

| 项目 | 实际值 |
|---|---:|
| 源数据记录 | 6,241 |
| 有效训练记录 | 6,240 |
| padding / dropped | 0 / 1 |
| global batch | 8 |
| epoch | 1 |
| optimizer steps | 780 |
| prefix source | `online` |
| prompt / response 上限 | 8,192 / 1,024 tokens |
| learning rate | `2e-6`，前 10 steps warmup |
| rollout n | 1 |
| VOPD Top-K / JSD alpha | 100 / 0.5 |
| EMA update rate | 0.05 |
| checkpoint | step 390、最终 step 780；最终只保留最新恢复点 |
| 初始化模型 | 冻结原始 Qwen3.5-4B Base |

`drop_last=True` 丢弃的是 seed 42 的 shuffle 序列尾部记录，并非固定的 Parquet 物理末行。Day 12 没有使用 Cached Prefix，也没有从此前 Pilot 或其他训练后模型继续训练。

## 三、训练指标与参数更新合同

运行指标文件包含 step 1～780 共 780 行，step 连续且无重复。

| Step | VOPD loss | LR | Student delta | Teacher optimizer delta | Teacher direct gradients | Teacher EMA delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.027270 | 0 | 0 | 0 | 0 | `2.38e-7` |
| 2 | 0.019213 | `2e-7` | `2.38e-7` | 0 | 0 | `2.38e-7` |
| 3 | 0.018499 | `4e-7` | `4.77e-7` | 0 | 0 | `1.19e-7` |
| 390 | 0.001416 | `2e-6` | `1.97e-6` | 0 | 0 | `8.05e-7` |
| 780 | 0.000778 | `2e-6` | `8.08e-7` | 0 | 0 | `5.10e-7` |

全程检查结果：

- loss 范围约为 `0.000214～0.027270`，780 个值全部有限。
- 所有正学习率 step 都观测到正 Student 参数 delta。
- 780/780 steps 的 Teacher optimizer delta 为 0，直接梯度计数为 0。
- 780/780 steps 都执行 Teacher EMA 更新。
- 780/780 steps 的 aborted ratio 为 0。
- 日志没有未解释的 traceback、CUDA OOM、cgroup OOM 或 checkpoint 保存失败。

loss 下降是训练过程信号，不等同于外部准确率提高；模型能力只能由后续冻结 R3 外部评测判断。

## 四、资源与运行结果

| 指标 | 实测结果 | 运行边界 |
|---|---:|---:|
| GPU 0 峰值显存 | 86,669,000,704 bytes，约 84.44% | 低于运行中止线 |
| GPU 1 峰值显存 | 82,229,329,920 bytes，约 80.11% | 低于运行中止线 |
| cgroup 峰值内存 | 237,306,273,792 / 257,698,037,760 bytes，约 92.09% | 低于连续 95% 中止线 |
| 磁盘最低可用 | 329,954,197,504 bytes，约 307.29 GiB | 足以完成最终 checkpoint |
| 训练墙钟 | 21,596.86 秒，约 6.00 小时 | launcher 全流程 |
| 估算费用 | 83.99 元 | 双卡 14 元/小时；不是平台账单 |

CPU/cgroup 峰值已经接近但未达到 95% 保护线，说明 240 GiB 配额在这次运行中足够，但不应据此下调正式任务的内存门槛。

## 五、checkpoint 与训练产物

最终恢复 checkpoint：

```text
artifacts/runs/E-D12-6K-VOPD-001/checkpoints/global_step_780/
```

主要产物：

- `logs/train.log`：完整训练控制台与 step 指标。
- `evidence/runtime_metrics.jsonl`：780 个 step 的关键训练合同指标。
- `evidence/telemetry/`：GPU、进程树、cgroup 和磁盘遥测。
- `evidence/guard_summary.json`：guard 最终 PASS 与 checkpoint 检查。
- `evidence/exit_receipt.json`：退出码、墙钟、估算费用和关键源码绑定。
- `rollouts/`：按配置保存的 rollout 证据，包括最终 step 780。
- `checkpoints/latest_checkpointed_iteration.txt`：值为 `780`。

Day 12 只负责生成并验证恢复 checkpoint。checkpoint 的逐文件 SHA256、FSDP 合并、独立冷加载和最终模型冻结在 Day 13 完成。

## 六、阶段结论与边界

Day 12 的计划任务全部完成：正式训练 780/780 steps、6241→6240 drop-last 合同、Student/Teacher/EMA 合同、自动守护、遥测和最终 checkpoint 均通过验收。

本次训练是论文核心算法参数对齐下的双卡资源缩放实现，不是论文原始 8 卡、global batch 96、rollout n=8 的完整硬件配置复现。Day 12 没有运行历史 internal eval，也没有打开外部 Benchmark，不能仅凭训练 loss 宣称模型能力提升。

## 七、关键证据索引

- 启动实时 Gate：`artifacts/runs/E-D12-6K-VOPD-001/preflight/day12_live_launch_gate.json`
- 运行调用快照：`artifacts/runs/E-D12-6K-VOPD-001/preflight/run_invocation.json`
- 完整训练日志：`artifacts/runs/E-D12-6K-VOPD-001/logs/train.log`
- 运行指标：`artifacts/runs/E-D12-6K-VOPD-001/evidence/runtime_metrics.jsonl`
- 自动守护结论：`artifacts/runs/E-D12-6K-VOPD-001/evidence/guard_summary.json`
- 最终退出凭据：`artifacts/runs/E-D12-6K-VOPD-001/evidence/exit_receipt.json`
- 最终 checkpoint：`artifacts/runs/E-D12-6K-VOPD-001/checkpoints/global_step_780/`
