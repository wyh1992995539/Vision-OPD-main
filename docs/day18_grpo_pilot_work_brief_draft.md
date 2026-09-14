# Day 18 GRPO 32/64 Prompt Pilot 工作简报（待定草案）

> 文档状态：**DRAFT / NOT EXECUTED**  
> 目标实验：`E-D18-GRPO-PILOT-001`  
> 适用前提：Day17 数据、Reward、配置静态 Gate 保持通过  
> 说明：以下“预期通过”内容是目标情景，不是已经发生的训练结果；所有 `[待填]` 均为运行后才能取得的数值，并须由真实产物回填。

## 技术摘要

若 Day18 正常运行并达到目标，本日应证明 `vision_opd_mcq_grpo_v1` 已经完成真实多模态 group rollout、规则 Reward、组内优势和策略梯度更新。32-prompt Pilot 应先证明优化信号成立；64-prompt Pilot 再证明运行稳定、checkpoint 可保存恢复、最终 actor 可合并并在新进程冷加载。

Pilot 权重只用于机制和资源验证，不接续为 Day19 正式训练。32、64 和正式训练均从同一个冻结 Qwen3.5-4B Base 独立初始化。

## 一、实际完成的工作（待执行成功后确认）

| 顺序 | 具体操作 | 主要结果与证据 | 状态 |
|---:|---|---|---|
| 1 | 从 GRPO 6,241 行 Parquet 按 gold、prompt token 长度和图像尺寸分层固定 32 条；先冻结成员、顺序、源/输出 SHA256 | `selection.json`、32 行 Pilot Parquet、四类 gold/长度/图像分层统计 | 预期 PASS |
| 2 | 复制正式候选配置生成 32 条 Pilot 配置，只修改数据路径、实验 ID、总步数、保存频率和 Pilot 授权 | 配置 diff 证明 Reward、模型、batch、n、KL 与正式候选一致 | 预期 PASS |
| 3 | 执行 `--preflight-only`，重新检查 Base/Data/Reward 哈希、两张空闲 GPU、cgroup、磁盘、端口和输出目录冲突 | preflight、resolved Hydra、run invocation、资源快照 | 预期 PASS |
| 4 | 从原始 Base 启动 32-prompt Pilot，运行 4 个外层迭代；逐步保存 4-way rollout、Reward、advantage、梯度和参数 delta | 4 steps、128 trajectories、连续 runtime metrics | 预期 PASS |
| 5 | 离线按 group UID 重算 Reward 和 advantage，与训练 tensor 逐条比较；检查 response padding 不进入 loss | advantage audit、mask audit、group alignment report | 预期 PASS |
| 6 | 按冻结规则人工复核至少 20 条输出，重点查看正奖励、冲突答案、截断和 Reward 投机 | 人工审计表与异常处置记录 | 预期 PASS |
| 7 | 32 条通过后，从 Base 独立初始化 64 条 Pilot，运行 8 steps 并在中间保存 checkpoint | 8 steps、256 trajectories、中间 checkpoint | 预期 PASS |
| 8 | 中断后按原配置恢复，核对 global step、optimizer/scheduler、sampler/RNG 与样本顺序；完成最终 checkpoint | restore receipt、恢复前后计数对齐报告 | 预期 PASS |
| 9 | 合并 64 条最终 actor，在新进程固定冷加载 5 条；测量保存、恢复、合并各阶段资源峰值 | merged manifest、5/5 冷加载和资源报告 | 预期 PASS |
| 10 | 根据两轮真实运行冻结 response length、资源阈值、checkpoint 策略、正式墙钟和中止条件 | Day19 promotion/freeze receipt | 预期 PASS |

执行时先创建独立的 `32/` 和 `64/` 实验目录，禁止复用已有 logs、rollouts 或 checkpoints。每次启动都保存 Git commit、dirty patch、配置/Reward/数据哈希和完整 argv；失败重试必须创建新的 attempt 目录，不能覆盖失败现场。

## 二、32-prompt 真实更新 Pilot

| 项目 | 冻结目标 | 正常完成情景 |
|---|---:|---:|
| 固定 prompt | 32 | 32 |
| prompt batch | 8 | 8 |
| rollout n | 4 | 4 |
| 外层迭代 | 4 | 4 |
| rollout 轨迹 | 128 | 128 |
| optimizer step | 4 | 4 |
| mixed-reward groups | 至少 1 | `[待填]` |
| parse-invalid | 完整披露 | `[待填]` |
| length 截断 | 完整披露 | `[待填]` |
| NaN/Inf | 0 | 0 |
| 非零有限 PG 梯度 | 必须成立 | 成立（待真实梯度凭据） |
| actor 参数变化 | 必须成立 | 成立（待参数差异凭据） |

应保存每条 `sample_id`、`group_uid`、`rollout_index`、response、finish reason、解析选项、gold、Reward、reason、advantage 和 policy version。每个 prompt 必须恰好对应 4 条 rollout，经过分布式重排后图像、gold、Reward 和 group UID 仍须一一对应。

组内优势需要离线复算并与训练 tensor 对齐。对于奖励 `[1,0,0,0]`，在本地样本标准差实现下预期优势约为 `[1.5,-0.5,-0.5,-0.5]`；全 0 或全 1 组优势应为 0。策略梯度必须来自正确性 Reward，而不能只由 KL 或 weight decay 造成参数变化。

## 三、64-prompt 稳定性与恢复 Pilot

| 项目 | 冻结目标 | 正常完成情景 |
|---|---:|---:|
| 固定 prompt | 64 | 64 |
| prompt batch | 8 | 8 |
| rollout n | 4 | 4 |
| 外层迭代 | 8 | 8 |
| rollout 轨迹 | 256 | 256 |
| optimizer step | 8 | 8 |
| checkpoint 保存 | 成功 | PASS（待凭据） |
| checkpoint 恢复 | 成功 | PASS（待凭据） |
| merged actor | 成功 | PASS（待凭据） |
| 新进程冷加载 | 5/5 非空 | 5/5（待凭据） |
| NaN/Inf/OOM | 0 | 0 |

64 条选择应固定覆盖四类 gold、最长 prompt、较大图像和常规样本。保存恢复路径需要证明 optimizer、scheduler、global step、dataloader/sampler 位置和 RNG 状态可恢复；不要求跨 GPU 推理逐 token 位级一致。

## 四、Reward 与真实输出人工审计

至少人工检查 20 条真实 rollout，并覆盖：

- 正奖励且语义正确；
- 明确错误选项；
- 空回答和无法解析；
- 多个冲突答案；
- `finish_reason=length`；
- 重复输出或格式投机；
- 高 Reward 长回答；
- A/B/C/D 输出偏置。

| 审计项目 | 目标 | 实际结果 |
|---|---|---|
| 抽查数量 | ≥20 | `[待填]` |
| 已知 Reward 漏洞 | 0 | 0（成功情景） |
| 高 Reward 误判 | 0 | 0（成功情景） |
| 需要升级 Reward 版本 | 否 | 否（以人工审计为准） |
| 冻结 response length | 由截断证据决定 | `[待填]` tokens |

如果发现 Reward 漏洞，必须创建新 Reward 版本并从 32-prompt Pilot 重新验证，不能直接沿用受影响 checkpoint。

## 五、资源与正式参数冻结

| 指标 | 32 Pilot | 64 Pilot |
|---|---:|---:|
| GPU 型号/数量 | 2 卡（型号待运行凭据） | 2 卡（型号待运行凭据） |
| 峰值显存/GPU | `[待填]` | `[待填]` |
| 峰值 CPU RSS/cgroup | `[待填]` | `[待填]` |
| 稳态 step 中位数 | `[待填]` | `[待填]` |
| 最慢 step | `[待填]` | `[待填]` |
| completion tokens | `[待填]` | `[待填]` |
| GPU 小时 | `[待填]` | `[待填]` |
| 估算费用 | `[待填]` | `[待填]` |

只有在 32 和 64 Pilot 都通过后，才能把 response length、显存阈值、cgroup 下限、checkpoint 频率、正式墙钟、费用和中止策略冻结到 Day19。

## 六、可能问题与处置

| 可能问题 | 识别信号 | 具体处置 |
|---|---|---|
| 当前进程仍只有 2 GiB cgroup 或看不到 GPU | preflight 中 cgroup/GPU Gate 失败 | 不启动训练；切换到真实训练会话后重跑 preflight，不通过则保持 BLOCKED |
| Qwen Processor 或 RLHFDataset 加载 OOM | 进程被 SIGKILL、cgroup OOM 增长 | 先单进程加载 1/32 条确认内存，再按 Pilot 实测调整 worker/offload；不静默跳过样本 |
| vLLM rollout OOM | CUDA OOM、KV cache 分配失败 | 降低 vLLM memory utilization、限制 CUDA graph capture 或启用已验证 offload；保持 batch=8、n=4 的有效合同，参数变化另建 attempt |
| 每组 Reward 全 0 或全 1 | mixed-group fraction 为 0，advantage 全零 | 先查 sampling 是否真实开启、四条输出是否重复、解析失败率和题目难度；不得人为改 Reward 或筛选有利样本 |
| rollout 与 gold 串组 | group UID 内 prompt/image/gold 不一致 | 立即停止；检查 repeat/interleave、负载均衡与 non-tensor batch 重排，修复后从 Base 重跑 32 条 |
| Advantage 与离线复算不一致 | 同组均值/std、mask 或张量顺序不匹配 | 保存原 tensor，核对样本标准差、epsilon、padding mask 和 group 顺序；未解释前不得晋级 |
| actor 参数变化只来自 KL | PG 梯度为 0但总参数 delta 非零 | 分离记录 PG、KL、weight decay 梯度；必须找到至少一个 mixed group 对应的非零有限 PG 更新 |
| 128-token 截断偏高 | `finish_reason=length` 集中、答案区缺失 | 审计完整答案率后预先改为 256 并从 32 条重跑；不能在同一实验中临时改变上限 |
| Reward hacking | 高 Reward 输出只靠格式、冲突标签或引用 gold | 修复并版本化 Reward、补回归测试、更新哈希，从 Base 重新执行受影响 Pilot |
| checkpoint 恢复错位 | step、LR、sampler位置或样本顺序跳变 | 保留失败 checkpoint 和日志，修复恢复合同后重新做中断/恢复对照，不将其用于正式训练 |
| 日志尾行或指标缺失 | 进程成功退出但最后 step 未落盘 | 在 post-exit final drain 后重放完整日志；保留原凭据与修正凭据，禁止无记录覆盖 |

## 七、阶段结论与边界（待真实结果确认）

> Day18 GRPO Pilot 预期 **PASS**。32-prompt Pilot 完成 4 个真实 optimizer steps 和 128 条 rollout，mixed-reward group 为 `[待填]`，离线 advantage 与训练 tensor 一致，非零有限策略梯度与 actor 参数变化成立。64-prompt Pilot 完成 8 steps、256 条 rollout，checkpoint 保存/恢复、合并及 5/5 新进程冷加载全部通过；正式配置据此完成晋级冻结。

该结论只证明 GRPO 训练链路、更新信号、恢复机制和资源参数满足进入正式训练的条件。Pilot 权重不作为最终模型，不与 Base、Vision-OPD 或 Cached 做能力比较，也不能替代 Day20 的冻结 R4 外部评测。

## 八、关键证据索引

- `artifacts/runs/E-D18-GRPO-PILOT-001/32/`
- `artifacts/runs/E-D18-GRPO-PILOT-001/64/`
- `artifacts/reports/grpo_64_stability.md`
- 固定样本选择清单及 SHA256
- resolved Hydra 配置与启动命令
- rollout/Reward/advantage 逐条记录
- 梯度和参数差异凭据
- checkpoint 保存恢复凭据
- 资源遥测与成本外推
