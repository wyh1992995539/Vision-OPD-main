# Day 11 Vision-OPD 6241 正式训练准入工作简报

> 执行窗口：2026-09-05 至 2026-09-06（UTC）
> 对应 Gate：`E-D11-6K-GATE-001`
> Day 11 最终状态：**PASS**
> 正式训练授权：`formal_training_authorized=true`
> 正式训练是否已经启动：**否**
> 冻结提交：`078d403 day11: finalize formal training gate and validation evidence`

## 技术摘要

Day 11 已完成 Vision-OPD 6,241 条正式训练之前的全部准入工作。工作从论文和作者入口参数核对开始，先冻结双卡缩规模方案、训练算术、checkpoint 和中止策略；随后对 6,241 条训练数据执行真实 Processor 下的全量 Prompt 长度审计，完成 train-6241 与 ZoomBench、MMStar、V* Bench 的 overlap 检查，并将训练合同固定为原生 `drop_last=True`：源数据 6,241 条、有效训练 6,240 条、padding 0、每个 epoch 丢弃 1 条、共 780 optimizer steps。

Cached Prefix 已完成 16 条 smoke 和 6,241 条全量生成，记录数、sample ID、错误、空响应和哈希检查全部通过。训练侧先后完成 Pilot-16、Pilot-64 和 Student checkpoint 冷重载；这些实验暴露了双卡资源布局、checkpoint 保存期 CPU 峰值、GPU 显存口径和 warmup 覆盖问题，随后补齐 guarded launcher、abort policy、postflight、warmup-aware 审计和分阶段内存遥测。

显存优化采用 optimizer deferred load，并经过非固定负载 A/B、固定 actor 输入对照、参数透传修复和 128×16 长回复压力验证。压力验证中 128 条回复全部达到 1,024 tokens，完成 16 steps、warmup 后更新和 checkpoint，CUDA 同步物理显存峰值约 90.59%，CPU 峰值约占 240 GiB 配额的 76.34%。之后又使用正常 EOS 执行 128×16 正式候选验证，自然生成最大回复为 471 tokens，GPU 峰值约 89.14%，CPU 峰值约 76.19%，最终状态为 `PASS_CANDIDATE_VALIDATION`。

依据自然生成候选的真实吞吐和 checkpoint 尺寸，正式 CPU 门槛冻结为 240 GiB，磁盘门槛冻结为 120 GiB，规划预算约 7.04 双卡小时/84.24 元，保守预留约 8.07 小时/96.57 元。正式配置已晋升为 `ready_after_day11_gate`，Day 11 汇总 Gate 为 `PASS`、阻塞项为 0，Day 12 guarded launcher 静态预检通过。该结论表示可以进入正式训练，不表示 780-step 正式训练已经完成，也不表示模型能力已经提升。

## 一、实际执行顺序

| 顺序 | 实际完成内容 | 为什么要做 | 主要结果 | 状态 |
|---:|---|---|---|---|
| 1 | 核对论文、官方脚本与当前配置 | 区分算法合同和双卡资源缩放，避免把当前方案误称为官方原样复现 | 选择双卡缩规模主方案；另存算法对齐双卡参考配置；补齐两个官方 vLLM 开关 | PASS |
| 2 | 冻结训练参数、checkpoint 和初始中止策略 | 让后续实验围绕明确变量进行，防止边跑边改合同 | LR、Top-K、JSD、EMA、长度、warmup、clip、offload、step 390/780 保存计划进入静态配置 | PASS |
| 3 | 执行 6,241 条全量 Prompt 长度审计 | 关闭 1,024 条抽样看不到的 6K 长尾、Processor 错误和静默截断风险 | 6,241/6,241 可处理，error=0，over-8192=0；Student max=7,880，Teacher max=2,809 | PASS |
| 4 | 修正 `project_6241.yaml` 并冻结 drop-last | 按项目负责人的决定丢弃最后一条，不再用全覆盖 sampler 补齐 | 6,241→6,240，padding=0，dropped=1，global batch 8 下共 780 steps | PASS |
| 5 | 运行 train-6241 与三个外部 Benchmark 的全量 overlap | 识别数据泄漏风险，同时避免静默删除官方评测样本 | 2,536 个 benchmark 样本完成检查；确认 21 对 overlap，均来自 V*；unresolved=0 | PASS_WITH_CONFIRMED_OVERLAP |
| 6 | 补齐 Cached Prefix 哈希和测试，运行 16 条 smoke 与 6,241 条全量生成 | 为后续 Cached 分支建立完整、按 sample ID 可追溯的 Base 输出 | records=6,241、unique IDs=6,241；missing/duplicate/error/empty=0；max=1,024；不重采样 | PASS |
| 7 | 建立 Pilot 配置、静态 Gate、guarded launcher、abort policy 和 postflight | 使启动、异常中止、证据保存和结束判断自动化且 fail-closed | GPU/CPU/磁盘/账单、Student/Teacher/EMA、drop-last 和 checkpoint 均进入守护检查 | PASS |
| 8 | 多轮修复并完成 Pilot-16 | 用最小真实训练闭环验证 online rollout、Teacher crop、JSD、Student backward、EMA 和保存 | 最终 16 条、2 steps、checkpoint、遥测和 postflight 全部通过 | PASS |
| 9 | 执行 Pilot-64 并完成 Student 冷重载 | 扩大训练覆盖，验证保存后的 Student 至少可以新进程加载和生成 | 64 条、8 steps、checkpoint、postflight 和 Student 冷重载通过 | PASS |
| 10 | 分析 CPU 峰值并修正 GPU 显存计数口径 | 区分模型/优化器、文件缓存、checkpoint 临时副本、allocator 与整卡显存 | 正式 CPU 门槛最终冻结为 240 GiB；GPU 中止线为 98%；新增分阶段同步显存证据 | PASS |
| 11 | 运行 baseline/deferred 显存 A/B | 验证 optimizer deferred load 是否有机会降低危险阶段显存 | 两组单独训练均通过；非固定负载结果显示下降，但因生成负载不同，不作完整因果结论 | 完成，结论受限 |
| 12 | 准备并执行固定 actor 输入对照 | 消除自然生成长度和微批次数量不同造成的 A/B 混杂 | capture、fixed baseline、fixed deferred 完成；相同 actor 输入和源码绑定通过 | PASS_MATCHED_ACTOR_INPUTS_PENDING_MEMORY_REVIEW |
| 13 | 启动首轮长回复压力验证并主动停止 | 检查近 1,024-token 回复、warmup 后更新与保存峰值 | 发现 `ignore_eos` 未传入真实采样器；主动 SIGTERM；cgroup oom/oom_kill=0；没有误写成 PASS | FAIL 被正确保留 |
| 14 | 修复参数透传并补端到端参数测试、首批长度 Gate | 确保压力实验真正生成计划中的 1,000～1,024 token 回复 | CPU 定向测试和隔离 launcher preflight 通过，旧失败记录不覆盖 | PASS_CPU_REPAIR |
| 15 | 运行 128×16 长回复压力验证 | 验证最坏长度附近的显存、CPU、warmup 后训练和 checkpoint 安全性 | 128/128 回复均为 1,024 tokens；16 steps 和 checkpoint 完成；GPU 同步峰值 90.59%，CPU 76.34% | PASS_PRESSURE_DIAGNOSTIC |
| 16 | 整理正式候选并运行 128×16 正常 EOS 验证 | 压力测试是人为负载，仍需验证正式候选的自然生成行为 | 自然最大回复 471 tokens；GPU 89.14%，CPU 76.19%；checkpoint 和 postflight 通过 | PASS_CANDIDATE_VALIDATION |
| 17 | 清理旧分片并重新冻结 CPU、磁盘和预算 | 避免沿用过时 Pilot 外推，并确保正式保存具备双副本写入空间 | 白名单删除 8 个旧 A/B 分片；CPU=240 GiB；磁盘门槛=120 GiB；预算重算通过 | PASS |
| 18 | 晋升正式配置、生成最终 Gate、静态验证 Day 12 launcher 并提交版本 | 防止仅修改 YAML 状态绕过证据验收，形成可审计的正式训练入口 | Gate=PASS，authorized=true，blocking=[]；launcher preflight=PASS；commit `078d403` 已同步 | PASS_TO_DAY12 |

## 二、冻结的数据与训练合同

| 项目 | 最终值 |
|---|---:|
| 源训练记录 | 6,241 |
| 有效训练记录 | 6,240 |
| padding rows | 0 |
| dropped rows per epoch | 1 |
| global batch | 8 |
| epoch | 1 |
| optimizer steps | 780 |
| shuffle / seed | `true` / `42` |
| prompt 最大长度 | 8,192 tokens |
| response 最大长度 | 1,024 tokens |
| LR | `2e-6` |
| Top-K / JSD β / EMA | `100` / `0.5` / `0.05` |
| warmup steps | 10 |
| checkpoint | step 390、最终 step 780 |
| 正式启动来源 | 冻结 Qwen3.5-4B Base 冷启动 |

`drop_last=True` 丢弃的是 seed 42 下 shuffle 序列末尾的一条记录，而不是静态保证丢弃 Parquet 物理末行。运行回执仍须验证实际 unique source seen=6,240 和 dropped rows=1。

## 三、数据级 Gate 已关闭主要长尾与泄漏风险

### 1. Prompt 长度

| 视图 | P50 | P95 | P99 | Max | over-8192 | error |
|---|---:|---:|---:|---:|---:|---:|
| Student total tokens | 3,366 | 3,974 | 4,445 | 7,880 | 0 | 0 |
| Teacher total tokens | 289 | 1,545 | 1,899 | 2,809 | 0 | 0 |

审计使用训练时相同的 Processor、Chat Template 和图像输入。`truncation=error` 保持不变，没有通过删除样本或静默截断让 Gate 通过。

### 2. 外部 Benchmark overlap

| Benchmark | 官方样本数 | 确认 overlap | 处理方式 |
|---|---:|---:|---|
| ZoomBench | 845 | 0 | 1 个文本候选经人工证据排除 |
| MMStar | 1,500 | 0 | 无候选 |
| V* Bench | 191 | 21 | 全部保留并在报告中分层披露 |
| 合计 | 2,536 | 21 | unresolved=0 |

`PASS_WITH_CONFIRMED_OVERLAP` 表示审计执行和人工复核闭环通过，不表示不存在 overlap。V* 主结果仍固定官方 191 分母；overlap 分层或去重只能作为同一批预测上的次级诊断。

### 3. Cached Prefix

| 检查项 | 结果 |
|---|---:|
| expected / actual records | 6,241 / 6,241 |
| unique sample IDs | 6,241 |
| duplicate IDs | 0 |
| inference errors | 0 |
| empty responses / token IDs | 0 / 0 |
| maximum response tokens | 1,024 |
| resampled records | 0 |

Cached Prefix 为 Day 13～14 的 cached 分支准备；Day 12 Vision-OPD 正式训练仍使用 online Student prefix，不会误用缓存响应。

## 四、Pilot 和显存验证形成了逐级证据链

| 验证 | 样本 | Steps | 生成行为 | 回答的问题 | 结果 |
|---|---:|---:|---|---|---|
| Pilot-16 | 16 | 2 | 正常采样 | 最小端到端训练、EMA、保存是否工作 | PASS |
| Pilot-64 | 64 | 8 | 正常采样 | 扩大训练覆盖，checkpoint 是否可加载 | PASS + Student 冷重载 PASS |
| 固定负载 baseline/deferred | 64 | 8 | 完整 actor 输入回放 | deferred 在相同 actor 输入下是否降低危险阶段内存 | 输入匹配，内存结论保守保留 |
| Pressure v2 | 128 | 16 | 128/128 强制 1,024 tokens | 最坏长度、warmup 后、保存阶段是否安全 | PASS_PRESSURE_DIAGNOSTIC |
| 正式候选 | 128 | 16 | 正常 EOS，max 471 | deferred + 正常 EOS 的正式候选是否可训练 | PASS_CANDIDATE_VALIDATION |

### 失败和修复没有被覆盖

- 早期 Pilot 因双卡资源布局和内存条件失败，之后才冻结 reference/offload 与资源方案。
- checkpoint 保存阶段暴露 CPU 峰值后，新增文件缓存和 `memory.stat` 审计；没有简单降低启动门槛。
- baseline 原始结束审计因 allocator/物理显存口径混用失败，原记录保留，随后使用修订口径离线重审。
- 第一轮 pressure 因 `ignore_eos` 未真正传入采样器而主动停止；`oom=0`、`oom_kill=0`，不是训练 OOM，也没有被改写成成功。
- 参数透传修复后，先通过 CPU 端到端测试和首批长度 Gate，再执行新的 Pressure v2。

这些记录说明 Day 11 的 PASS 来自问题被发现、修复并重新验证，而不是删除失败证据。

## 五、正式资源、预算和中止策略

### 1. CPU

- 正式 cgroup 内存门槛：至少 **240 GiB**。
- 运行期 CPU 中止线：连续达到 **95%**。
- 冻结依据是多次真实训练的 cgroup 原始遥测，而不是仅采用最新或最低的一次峰值。
- 240 GiB 是已经成功使用并复核的配置，不代表 224/220 GiB 已经验证安全。

### 2. GPU

- 逐卡物理显存中止线：**98%**。
- 同时保留 NVML 整卡采样和 CUDA 同步阶段峰值。
- Pressure v2 的同步物理显存峰值约 **90.59%**；自然候选 GPU 峰值约 **89.14%**。
- 上述结果低于中止线，但不能保证 780-step 正式训练永远不出现更高峰值。

### 3. 磁盘与 checkpoint

正式启动磁盘公式：

```text
required = max(120 GiB, 2 × checkpoint_budget + 5 GiB reserve)
```

真实单个 checkpoint payload 约 53.12 GiB，公式计算需求约 111.24 GiB，向上冻结为 120 GiB。候选 Gate 快照可用约 148.59 GiB，余量约 28.59 GiB。该快照只证明当时通过；Day 12 启动器仍须重新读取实时磁盘。

### 4. 预算

| 场景 | 双卡小时 | 预计增量费用 |
|---|---:|---:|
| 平均规划 | 7.04 h | 84.24 元 |
| 保守预留 | 8.07 h | 96.57 元 |
| 硬中止上限 | 38.00 h | 454.48 元 |

预算按自然生成正式候选的 startup、first step、step 2～16 稳态时间和两次 checkpoint 保存时间重新计算。此前 Pilot-64 的旧预算只保留为历史证据。

`configs/vopd_6241_abort_policy.yaml` 的 `budget` 段继续保留旧 256-token 参考值和“Day 12 前必须重估”的历史约束；它不是当前正式预算数值的权威来源。当前预算权威来源是 `formal_candidate_validation_v1/formal_gate_freeze.json`，最终 Gate 和 promotion receipt 已核验自然生成候选重估状态为 `PASS_BUDGET_REFROZEN_FROM_NATURAL_CANDIDATE`。正式 launcher 读取并绑定这份候选冻结证据，同时仍执行 38 小时、项目总额和账单新鲜度保护。

## 六、正式候选配置与最终 Gate

最终 `configs/vopd_6241.yaml` 已晋升为：

```text
status: ready_after_day11_gate
prefix source: online
normal EOS: true
optimizer deferred load: true
actor / optimizer / reference offload: enabled
GPU abort ratio: 0.98
minimum cgroup memory: 240 GiB
minimum disk free: 120 GiB
```

正式候选晋升不是单纯修改状态字段。promotion receipt 同时绑定候选配置、旧正式配置、CPU freeze、预算、磁盘、候选 checkpoint、Gate builder、正式 launcher 和关键源码 SHA256。缺失或篡改任一绑定文件，最终 Gate 都应重新阻塞。

最终汇总结果：

```text
status=PASS
formal_training_authorized=true
blocking_gates=[]
formal_config_status=ready_after_day11_gate
guarded_launcher_static_preflight=PASS
training_started=false
```

关联回归测试为 `84 passed, 5 subtests passed`。冻结提交 `078d403` 已同步 `origin/main`，提交后工作树干净。

## 七、主要证据

- `configs/vopd_6241.yaml`
- `configs/vopd_6241_abort_policy.yaml`
- `configs/vopd_6241_algorithm_aligned_2gpu.reference.yaml`
- `artifacts/runs/E-D11-6K-GATE-001/prompt_length/prompt_length_summary.json`
- `artifacts/runs/E-D11-6K-GATE-001/drop_last/drop_last_audit.json`
- `artifacts/runs/E-D10-6K-DATA-001/overlap/overlap_validation.json`
- `artifacts/runs/E-D11-6K-GATE-001/cached_prefix/report.json`
- `artifacts/runs/E-D11-6K-GATE-001/pilot/16/evidence/postflight.json`
- `artifacts/runs/E-D11-6K-GATE-001/pilot/64/evidence/postflight.json`
- `artifacts/runs/E-D11-6K-GATE-001/memory_optimization/fixed_validation_v1/pressure_v2/run/evidence/postflight.json`
- `artifacts/runs/E-D11-6K-GATE-001/formal_candidate_validation_v1/formal_gate_freeze.json`
- `artifacts/runs/E-D11-6K-GATE-001/formal_promotion_v1/promotion_receipt.json`
- `artifacts/runs/E-D11-6K-GATE-001/preflight.json`
- `artifacts/runs/E-D12-6K-VOPD-001/preflight/guarded_launcher_preflight.json`

## 八、结论边界

### 可以确认

- 6,241 条数据在真实训练 Processor 下均可处理，没有 over-8192 或 Processor error。
- 6241→6240 原生 drop-last 合同、780-step 算术和运行回执要求已冻结。
- Cached Prefix 6,241/6,241 完整且可核验。
- Pilot-16、Pilot-64、Student 冷重载、长回复压力和正常 EOS 正式候选均有可核验 PASS 证据。
- 当前正式配置、资源门槛、预算、磁盘、中止策略和 launcher 已通过 Day 11 静态准入。
- 可以进入 Day 12 的正式 6,241 数据训练。

### 不能扩大表述

- Day 11 没有完成 780-step 正式训练，也没有产生最终 Vision-OPD checkpoint。
- Pilot 和候选验证证明可行性与安全边界，不代表最终模型能力提升。
- 强制 1,024-token 压力验证不是论文自然采样结果。
- Student 冷重载不等于 optimizer、Teacher EMA 和 RNG 精确续训已验证。
- 固定负载 A/B 支持 deferred 降低显存风险，但诊断同步、有限步数和运行环境使其不能升级为全程严格因果结论。
- 静态 Gate 的 CPU、GPU、磁盘和账单快照不能代替 Day 12 启动时的实时检查。

## 九、Day 12 交接

Day 12 将执行 `E-D12-6K-VOPD-001` 正式训练。启动前必须重新满足：

1. 两张目标 GPU 均可见且空闲；
2. cgroup 内存至少 240 GiB；
3. 实时可用磁盘至少 120 GiB；
4. Git 工作树干净且配置/源码/证据哈希匹配；
5. 正式输出目录不存在冲突；
6. AutoDL 累计费用和 UTC 观测时间不超过 15 分钟；
7. guarded launcher 实时 preflight 全部 PASS。

正式训练从冻结 Qwen3.5-4B Base 冷启动。启动后人工重点观察前 3 个 optimizer steps，确认 sample ID、online response、Teacher crop、有限 loss、Student 更新、Teacher 无直接梯度、EMA、GPU、CPU 和磁盘正常；随后由守护器持续监控。step 390 保存唯一周期恢复点，step 780 完成后再验证最终 checkpoint 和成功 receipt。
