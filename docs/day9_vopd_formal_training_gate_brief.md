# Day 9 Vision-OPD 正式训练 Gate、配置冻结与中止控制工作简报

> 执行日期：2026-09-01～2026-09-02（UTC）
> 对应实验：`E-D10-001`
> Day 9 最终状态：**PASS_TO_DAY10**
> GPU 使用：**false**

## 技术摘要

Day 9 已完成 Vision-OPD 1024 条正式训练前的完整工程 Gate。工作从 Day 8 的 64 条稳定性训练证据出发，依次完成耗时与费用预算、正式输入与路径审计、训练配置冻结、正式 preflight 报告，以及可执行的运行期中止与观测控制。

正式训练合同冻结为：1024 条样本、global batch 8、128 个 optimizer steps、完整 1 epoch，从原始 Qwen3.5-4B Base 冷启动。正式配置 SHA256 为 `5977d0b7adda448287d7410431c9461a6f6f53c04792390b9b13d9529a00b30c`。

Day 8 checkpoint 保存后出现的 DataLoader worker `Killed` 没有被忽略或改写为已解决。Day 9 将 `dataloader_num_workers` 从 4 调整为 0，移除训练 DataLoader 子进程，并新增逐卡 GPU 显存、训练进程树 RSS、cgroup 内存事件、磁盘和训练指标监控。Task 5 使用 Day 8 日志回放验证：8 个结构化训练步没有违反 Student/Teacher/EMA 合同，同时准确识别到 `dataloader_worker_killed`。

Day 9 全程没有启动正式 GPU 训练。最终机器报告为 `PASS_TO_DAY10`，表示可以进入 Day 10 的动态启动检查，不表示 E-D10-001 已经训练完成或模型效果已有提升。

## 一、Day 9 范围和任务完成情况

| 任务 | 目标 | 主要结果 | 状态 |
|---|---|---|---|
| Task 1 | 用 Day 8 实测吞吐估算 1024 条正式训练时间与费用 | 规划 1.02 双卡小时/¥12.25；保守预留 1.74 双卡小时/¥20.84 | PASS |
| Task 2 | 审计数据、Base、配置、Git、输出目录、磁盘和日志路径 | 8 项 readiness Gate 全部通过，无阻塞项 | PASS |
| Task 3 | 冻结正式训练参数和训练合同 | 1024 条、batch 8、128 steps、1 epoch；配置哈希冻结 | PASS |
| Task 4 | 生成正式训练 preflight 报告 | 基础 Gate 完整归档，结论由 `PASS_TO_TASK5` 推进到 Task 5 | PASS |
| Task 5 | 设置可执行的中止条件和运行期观测 | 守护启动器、策略、日志回放和 14 项专项测试全部通过 | PASS |

最终准入结论见 `artifacts/runs/E-D10-001/preflight.md` 和 `artifacts/runs/E-D10-001/preflight/task5_abort_controls.json`。

## 二、Task 1：正式训练时间与费用预算

### 1. 计算口径

正式训练的 1024 条样本在 global batch 8 下对应 128 个 optimizer steps。预算没有直接把 64 条运行时间乘以 16，而是拆分为：

```text
启动固定开销 + 首步预热 + 127 × 稳态 step + 最终 checkpoint 保存
```

稳态统计来自 Day 8 除首步外的 7 个 step。费用按 `11.96 CNY/双卡小时` 计算；均值场景用于规划，稳态最大值用于资源预留。

| 场景 | 稳态 step | 预计双卡小时 | 预计费用 | 用途 |
|---|---:|---:|---:|---|
| 中位稳态 | 21.86 秒 | 0.8912 | ¥10.66 | 参考下界 |
| 均值稳态 | 25.64 秒 | 1.0246 | ¥12.25 | 正式规划值 |
| 稳态最大值 | 46.00 秒 | 1.7427 | ¥20.84 | 启动预留值 |

### 2. 预算 Gate

仓库内能够直接找到的历史成本文件只能形成 ¥27.68 的不完整工作负载小计，缺少部分启动、空闲、失败尝试和实验记录，不能冒充 AutoDL 平台账单。用户在 Day 9 报告的 AutoDL 控制台累计费用为 ¥200.00；加入 ¥20.84 保守预留后的投影为 ¥220.84，低于 ¥2000 项目上限，因此 Budget Gate 为 PASS。

用户在 Day 9 完成后明确决定费用无需持续监控。当前控制含义是：训练期遥测不轮询费用；费用不属于每 10 秒采集的运行指标。仓库仍保留 Day 9 的预算快照和计算证据，以说明当时的资源判断。

主要证据：

- `artifacts/runs/E-D10-001/preflight/budget_projection.json`
- `artifacts/runs/E-D10-001/preflight/budget_evidence.md`
- `artifacts/runs/E-D10-001/preflight/autodl_billing_input.json`

## 三、Task 2：正式输入、环境与路径审计

Task 2 对正式训练开始前的静态条件进行了逐项审计。

| Gate | 审计结果 | 关键证据 |
|---|---|---|
| data | PASS | 冻结训练集 1024 行，样本 ID 唯一，必需字段和图像均存在，SHA256 匹配 |
| Base | PASS | 原始 Qwen3.5-4B Base 目录和必需权重分片存在，冻结哈希匹配 |
| config identity | PASS | 实验为 E-D10-001，训练合同为 1024/8/128/1 epoch |
| Git | PASS | 审计提交存在，工作区 clean，`git diff --check` 通过 |
| output directory | PASS | 输出位于预期 runs 根目录，无训练产物碰撞，目录可写 |
| log path | PASS | `artifacts/runs/E-D10-001/logs/train.log` 无冲突 |
| storage | PASS | 要求 119438631082 bytes，可用 165980688384 bytes，缺口为 0 |
| budget | PASS | Day 9 控制值 ¥200，加入保守预留后 ¥220.84 |

磁盘要求按以下公式冻结：

```text
2 × Day 8 checkpoint 实测大小 + 5 GiB
= 2 × 57034960981 + 5368709120
= 119438631082 bytes
```

审计时高于最低要求 46542057302 bytes，约 43.35 GiB。Task 2 没有删除、移动或覆盖任何历史训练产物，也没有启动 GPU。

主要证据：

- `artifacts/runs/E-D10-001/preflight/task2_readiness.json`
- `artifacts/runs/E-D10-001/preflight/task2_readiness.md`
- `artifacts/runs/E-D10-001/preflight/data_manifest.json`
- `artifacts/runs/E-D10-001/preflight/base_model_manifest.json`
- `artifacts/runs/E-D10-001/preflight/storage_gate.json`
- `artifacts/runs/E-D10-001/preflight/output_path_gate.json`

## 四、Task 3：正式训练配置冻结

正式配置位于 `configs/vopd_1024.yaml`，冻结哈希为：

```text
5977d0b7adda448287d7410431c9461a6f6f53c04792390b9b13d9529a00b30c
```

### 1. 关键参数

| 类别 | 冻结值 |
|---|---|
| 实验 | `E-D10-001`，seed 42，online prefix |
| 数据 | 1024 条，shuffle=true，batch 8，`dataloader_num_workers=0` |
| 长度 | prompt 8192，response 256，truncation=error |
| Actor | LR `2e-6`，mini batch 8，dynamic batch，gradient checkpointing |
| Rollout | n=1，TP=1，GPU memory utilization=0.45，agent workers=2 |
| VOPD | Top-K 100，alpha 0.5，importance sampling clip 2.0 |
| Teacher | always-on，legacy source，EMA rate 0.05，不由 optimizer 直接更新 |
| 训练 | 128 steps，1 epoch，从 Base 冷启动，`resume_mode=disable` |
| 保存 | 仅保存最终 checkpoint，最多保留 1 个 actor checkpoint |
| 资源 | 1 node，2 GPUs |

### 2. 相对 Day 8 的有意变化

- 训练数据由固定 64 条子集切换为冻结的 1024 条正式集。
- optimizer steps 从 8 增加到 128。
- Day 8 为保证子集顺序可审计而关闭 shuffle；正式训练恢复 seeded shuffle。
- `dataloader_num_workers` 从 4 降为 0，以移除发生过 `Killed` 的 DataLoader 子进程路径。
- 实验 ID、输出目录和名称切换为 E-D10-001。
- 学习率、batch、长度、Top-K、JSD 权重、EMA 和 rollout 核心数学合同保持不变。

`dataloader_num_workers=0` 只改变数据加载进程结构，不改变模型前向、loss、梯度或优化器公式；代价是数据读取与预处理无法由 DataLoader 子进程并行，可能影响吞吐，因此仍需用正式训练遥测观察实际 step 时间。

主要证据：

- `artifacts/runs/E-D10-001/preflight/task3_config_freeze.json`
- `artifacts/runs/E-D10-001/preflight/task3_config_freeze.md`

## 五、Task 4：正式训练 preflight 报告

Task 4 将前述预算、数据、Base、配置、Git、路径、磁盘和 Day 8 冷重载结果汇总为统一报告。生成时所有基础 Gate 均为 PASS，正式配置在 Task 3 冻结结果和 launcher preflight 中的 SHA256 完全一致；Day 8 冷重载为 5/5 输出非空、0 个推理错误。

Task 4 当时的状态为 `PASS_TO_TASK5`，原因是可执行中止条件尚未落盘，而不是基础输入失败。Task 5 完成后，`artifacts/runs/E-D10-001/preflight.md` 已更新为最终 `PASS_TO_DAY10` 报告；原始 Task 4 机器证据仍保存在 `task4_preflight_report.json`，没有被删除。

Task 4 采用两阶段提交：先提交报告工具，再在 clean commit 上生成和提交正式报告，避免报告声称审计了一个并不存在或仍在变动的工作区。

主要证据：

- `artifacts/runs/E-D10-001/preflight/task4_preflight_report.json`
- `artifacts/runs/E-D10-001/preflight.md`
- `scripts/finalize_day9_preflight_report.py`

## 六、Task 5：训练中止条件与观测控制

### 1. 实现结构

- `configs/vopd_abort_policy.yaml`：机器可读的冻结策略。
- `scripts/monitor_vopd_training.py`：指标解析、资源采样、规则判断、进程组终止和 checkpoint 校验。
- `scripts/run_vopd_guarded.py`：正式训练受保护入口和启动 Gate。
- `scripts/run_vopd_2gpu.sh`：保留训练命令，但直接 `--run` 会被拒绝，必须由 guarded launcher 持有。
- `tests/test_vopd_abort_guard.py`：规则、回放和终止动作测试。
- `docs/day9_task5_abort_runbook.md`：正式操作手册。

策略文件 SHA256 为：

```text
6fbea2890817a08baaaeb911e7a491d5c1003dc2bc08c8f77057ea5311f29174
```

### 2. 冻结中止条件

| 条件 | 阈值 | 动作 |
|---|---:|---|
| NaN/Inf | 任意一次 | 立即中止 |
| Teacher 出现直接梯度 | 任意一步 | 立即中止 |
| Teacher optimizer delta 非零 | 任意一步 | 立即中止 |
| Teacher EMA 不更新 | 连续 2 个有效步 | 中止 |
| Student optimizer 不更新 | 连续 2 个有效步 | 中止 |
| generation/rollout abort | 连续 3 步 | 中止 |
| cgroup OOM/oom_kill 计数增加 | 任意一次 | 立即中止 |
| 任一 GPU 显存占比达到 95% | 连续 3 次采样 | 中止 |
| cgroup 内存占比达到 95% | 连续 3 次采样 | 中止 |
| 遥测采集失败 | 连续 3 次 | 中止 |
| 日志无心跳 | 启动宽限后连续 15 分钟 | 中止 |
| 训练墙钟时间 | 达到 38 小时 | 中止 |
| 磁盘低于 checkpoint+5 GiB | 连续 2 次采样 | 中止 |
| 磁盘低于 5 GiB | 任意一次 | 立即中止 |
| checkpoint 保存错误 | 任意一次 | 立即中止 |

守护器每 10 秒采集逐卡 `nvidia-smi`、完整训练进程树 RSS/VMS、cgroup v2 `memory.current`/`memory.max`/`memory.events` 和磁盘可用空间。即使 `dataloader_num_workers=0`，Ray 和 vLLM 仍可能创建子进程，因此内存口径覆盖整个进程树，而不是只看 launcher PID。

规则触发时先向整个训练进程组发送 `SIGTERM`，等待 60 秒；仍未退出时升级为 `SIGKILL`。守护器自身异常也会进入 fail-closed 路径，避免训练脱离监控继续运行。

训练正常退出后还必须满足：`latest_checkpointed_iteration.txt=128`、`global_step_128` 存在，并且 13 个必需 checkpoint 文件均存在且非空，否则退出码为 42，不允许把运行标记为成功。

### 3. 测试与回放

- Task 5 专项测试：14/14 PASS。
- 既有 Day 9 preflight 回归：10/10 PASS。
- Python 语法、Shell 语法和 `git diff --check`：PASS。
- 直接运行原始 `run_vopd_2gpu.sh --run`：退出码 2，训练未启动。
- Day 8 回放：8 个指标步的数值、Student、Teacher 和 EMA 合同异常为 0；日志识别到一次 `dataloader_worker_killed`。
- Task 5 全过程：`gpu_used=false`。

主要证据：

- `artifacts/runs/E-D10-001/preflight/task5_abort_controls.json`
- `artifacts/runs/E-D10-001/preflight/task5_abort_policy.json`
- `artifacts/runs/E-D10-001/preflight/task5_test_report.json`
- `artifacts/runs/E-D10-001/preflight/task5_guarded_launcher_preflight.json`

## 七、遇到的问题与解决方法

| 问题 | 分析或影响 | 解决方法 | 验证结果 |
|---|---|---|---|
| Day 8 checkpoint 后 DataLoader worker 被 `Killed` | checkpoint 有效，但缺少同时期 RSS/cgroup 证据，无法确定是子进程内存、系统压力还是其他原因 | 正式配置将 DataLoader workers 从 4 降为 0；Task 5 增加进程树 RSS、cgroup 事件和 fail-closed 中止 | 配置不变量 PASS；Day 8 回放能识别原异常 |
| Day 8 训练器显存值超过冻结的 96 GB/卡物理口径 | 训练器聚合日志不能解释为可信的逐卡物理峰值 | 禁止用原值做容量结论；正式训练旁路采集每张卡的 `nvidia-smi` used/total/utilization | 守护策略和遥测输出路径已冻结，专项测试通过 |
| 仓库历史成本小计与 AutoDL 累计费用不一致 | 仓库记录缺少部分实验、空闲和失败窗口，不能作为平台账单 | 把“可见小计”“用户报告的平台累计费用”和“E-D10 预留”分开计算，不互相替代 | Task 1 Budget Gate PASS；限制已写入证据 |
| 直接按 Day 8 配置扩大样本容易引入隐性漂移 | 实验 ID、shuffle、步数、保存策略和输出路径需要有意变化，其余数学参数应保持 | 生成独立正式 YAML，逐字段比较 Day 8，冻结 SHA256，并让 launcher preflight 再次校验 | Task 3 config Gate PASS；Task 4 两份配置哈希一致 |
| Day 8 checkpoint 约 57 GB，保存时可能耗尽磁盘 | 只预留一个 checkpoint 容量不足以覆盖临时写入和最终保留 | 启动 Gate 使用 `2 × checkpoint + 5 GiB`；运行期设置 62.40 GB 软下限和 5 GiB 硬下限 | 审计时有约 43.35 GiB 额外余量；磁盘规则测试通过 |
| 原始 `--run` 可绕过监控 | 人工误用旧命令会让正式训练失去中止控制和证据 | 将旧入口设为内部入口；只有 guarded launcher 设置授权环境后才能运行 | 未授权命令退出码 2，`gpu_training_started=false` |
| 基础 Python 环境缺少 `pyarrow` | 首轮既有 Day 9 回归在导入阶段失败，并非断言失败 | 切换到项目 `vision-opd` Conda 环境执行需要 Parquet 的回归 | 3+4+3，共 10 项既有回归通过 |
| Conda 环境中模块式 unittest 被同名 `tests` 包遮蔽 | `tests.test_*` 无法按模块名导入 | 改用 `unittest discover -s tests -p <文件>` 精确发现本地测试 | 三组既有测试均成功运行 |
| 报告生成会使工作区从 clean 变为 dirty | 如果边生成边声称审计当前提交，报告来源可能自相矛盾 | 采用“工具实现提交 → clean 状态生成报告 → 报告证据提交”的两阶段方式 | Task 4、Task 5 均保留 audited source commit，最终工作区 clean |

## 八、提交与产物归档

Day 9 关键提交：

| 提交 | 内容 |
|---|---|
| `a1e261a` | Task 1～3：预算、readiness、正式配置冻结及测试 |
| `f47e994` | Task 4 preflight 工具和测试 |
| `f3937d0` | Task 4 正式报告 |
| `043a6be` | Task 5 可执行中止控制、guarded launcher、runbook 和测试 |
| `5aecae9` | Task 5 最终报告，状态推进到 `PASS_TO_DAY10` |

所有正式 preflight 机器证据位于：

```text
artifacts/runs/E-D10-001/preflight/
```

最终人读报告位于：

```text
artifacts/runs/E-D10-001/preflight.md
```

## 九、结论边界与 Day 10 交接

### 可以确认

- Day 8 实测数据已转换为可复算的 1024 条时间和费用预算。
- 1024 条正式数据、原始 Base、配置、Git、输出路径、日志路径和磁盘 Gate 已通过。
- 正式训练配置及 SHA256 已冻结，Student/Teacher/EMA 核心合同没有被 Day 9 改写。
- DataLoader 子进程风险已通过 workers=0 降低，并补上正式训练期资源证据链。
- 中止规则、信号升级、checkpoint 后置校验和防绕过入口已经实现并测试。
- Day 9 没有消耗 GPU 训练资源；最终结论为 `PASS_TO_DAY10`。

### 不能确认

- Day 9 没有执行 E-D10-001，因此不能确认 128 步一定完成。
- 预算是基于 7 个稳态 step 的工程外推，不是训练时长 SLA。
- workers=0 降低子进程风险，但只有正式训练遥测才能证明是否仍存在宿主机或 cgroup 内存压力。
- Day 8 的 `Killed` 根因仍无法从既有证据中精确归因。
- Day 9 没有产生模型效果结果，不能声称 internal eval 或外部 benchmark 提升。

### Day 10 正式入口

当前代码中的正式入口为：

```bash
python scripts/run_vopd_guarded.py \
  --current-autodl-cost-cny <累计费用快照> \
  --billing-observed-at-utc <ISO-8601时间> \
  --run
```

费用只在启动 Gate 中作为一次性输入，不在训练期间持续采集。训练开始后重点观察最初 3 个 optimizer steps，并核对样本、在线 response、Teacher crop、loss、Student/Teacher/EMA、逐卡显存、进程树 RSS、cgroup 和输出目录。
