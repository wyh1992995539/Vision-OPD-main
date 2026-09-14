# Vision-OPD GRPO 项目总结（含待完成工作）

> 文档状态：**PARTIAL / TRAINING NOT EXECUTED**  
> 当前进度：数据、Reward、候选配置和静态预检已完成；真实 GPU Pilot、正式训练、模型定版和外部评测待完成  
> 模型起点：原始 Qwen3.5-4B Base  
> 数据版本：`vision_opd_mcq_grpo_v1`  
> Reward 版本：`vision_opd_mcq_v1`  
> 说明：文中的固定规模属于已冻结合同；所有 `[待填]` 均为真实训练或评测后才能获得的数值。

## 技术摘要

本项目在已有 Base、Vision-OPD 和 Cached Prefix 三条路线之后，新增一条面向视觉选择题的 GRPO 强化学习路线。目标是让 Qwen3.5-4B 在观察完整红框图后，通过在线采样多个候选回答、确定性选择题 Reward 和组内相对优势，直接优化答对 A–D 选择题的概率，并在统一 R4 协议下检验是否改善 ZoomBench、MMStar 和 V* Bench。

当前已经完成训练前闭环：6,241 条 Vision-OPD 数据完成 GRPO 转换；确定性 Reward 已实现并通过单元、格式和全量路由验证；GRPO 候选参数已通过 Hydra 真实合并；受保护的启动入口能够校验模型、数据、Reward、图像、batch、KL、资源和授权，并在条件不满足时拒绝启动。

真实 GPU 训练仍未发生。当前工具会话只有 2 GiB cgroup，且不能验证 `nvidia-smi`；因此 `training_started=false`、`gpu_used=false`。下一步是双卡环境中的 32-prompt Pilot，而不是直接启动 6K 正式训练。

## 一、项目目标

项目需要回答四个问题：

1. 现有视觉选择题数据能否转换成 verl/GRPO 可直接消费、可审计、可稳定判分的数据格式？
2. 不部署神经 Reward Model 的情况下，能否用确定性规则准确判断 A–D 最终答案并避免解析投机？
3. 在 Qwen3.5-4B 多模态模型上，GRPO 能否产生来自正确性 Reward 的真实策略梯度，并稳定完成双卡训练、保存、恢复和合并？
4. 冻结后的 GRPO 模型在相同 R4 数据、请求、输出上限、解析器、Judge 和分母下，相对 Base、Vision-OPD、Cached 有何收益和退化？

## 二、项目采用的方法

### 2.1 数据工程：从 Vision-OPD 数据转换为 GRPO 数据

源数据原本服务于 Vision-OPD 蒸馏训练，包含完整红框图、Teacher crop、bbox、多个答案表示和训练辅助字段。GRPO 只需要策略输入、可判分 gold 和追踪信息，因此采用非破坏性转换：

- 保留完整红框图，模型仍直接观察原始视觉上下文；
- 保留问题和顺序固定的 A–D 选项；
- 将唯一正确选项写入 `reward_model.ground_truth`；
- 使用独立的 `data_source=vision_opd_mcq_grpo_v1`；
- 使用独立的 `reward_route=vision_opd_mcq_v1`；
- 保留 `sample_id`、来源身份和审计字段；
- 移除 `bbox_images`、Teacher crop、重复答案字段和 Vision-OPD 专用蒸馏字段；
- 输出固定 Schema：`data_source`、`prompt`、`images`、`ability`、`reward_model`、`extra_info`。

转换过程不覆盖源 Parquet，通过 SHA256 将源数据、转换脚本、输出数据和 Reward 路由绑定。

### 2.2 Reward：确定性二元选择题规则

项目没有训练或部署神经 Reward Model，也不在训练过程中调用外部 Base Judge。Reward 是确定性、可复算的二元规则：

- 模型给出唯一、合法、明确且等于 gold 的最终选项：Reward=1；
- 选项错误、回答为空、无法解析、存在冲突选项、标签不合法或选项越界：Reward=0；
- 先独立解析模型回答，再与 gold 比较，gold 不参与预测提取；
- 支持裸选项、`Answer`/`Final answer`、中文答案标记、合法 `<answer>`、括号、大小写和 `\boxed{}` 等有限格式；
- 不扫描任意英文大写字母，不从选项全文、普通推理文本或英文单词片段猜测答案；
- 数据合同错误直接抛出异常；模型输出解析失败则保留诊断原因并给 0 分。

每次判分保存 `prediction`、`parse_valid`、`parse_status`、`parse_evidence`、`candidate_options`、`truncated`、`reward_route` 和 `reward_source`，便于识别 Reward hacking、截断和解析回归。

### 2.3 GRPO：在线多候选、组内相对优势

每个 prompt 在线采样 4 条 rollout。对于同题的 4 个 Reward，在组内计算均值和标准差并标准化为 advantage，然后执行策略更新。该方法具有以下特点：

- `rollout.n=4`，每题形成一个四候选比较组；
- `adv_estimator=grpo`，开启组内标准化；
- 不训练 Critic，关闭 value loss；
- 全 0 或全 1 组的组内 advantage 为 0，不产生正确性策略梯度；
- mixed-reward group 才提供有方向的相对学习信号；
- 训练时必须离线重算 Reward 和 advantage，并与训练 tensor、group UID 和 mask 对齐。

例如 Reward 为 `[1,0,0,0]` 时，按当前样本标准差实现，预期 advantage 约为 `[1.5,-0.5,-0.5,-0.5]`。这一示例只用于验证计算路径，不代表实际训练分布。

### 2.4 策略优化与正则

候选配置采用：

| 参数 | 候选值 |
|---|---:|
| 初始化模型 | 原始 Qwen3.5-4B Base |
| seed | 42 |
| prompt batch | 8 |
| rollout n | 4 |
| 每次外层迭代轨迹 | 32 |
| policy loss | vanilla clipped policy gradient |
| loss aggregation | token-mean |
| PPO mini-batch | 8 |
| PPO epochs | 1 |
| learning rate | 1e-6 |
| warmup | 10 steps |
| clip low/high | 0.2 / 0.2 |
| entropy coefficient | 0 |
| actor KL coefficient | 0.001 |
| KL 类型 | `low_var_kl` |
| KL in Reward | false |
| 初始 response length | 128，待 Pilot 决定是否升至 256 |

KL 只位于 actor loss，Reward 中不再加入 KL，避免双重惩罚。训练时需要分别记录 policy-gradient、KL 和 weight-decay 对参数变化的贡献，防止把只有 KL 引起的参数变化误判为 Reward 学习成功。

### 2.5 分布式训练与资源优化

候选方案使用 1 节点 2 GPU，并启用：

- gradient checkpointing；
- actor parameter offload；
- optimizer offload；
- reference parameter offload；
- vLLM 在线 rollout；
- dynamic batch size；
- 每 GPU log-prob micro-batch=1；
- tensor parallel size=1，双卡主要用于数据并行。

顶层 `ppo_mini_batch_size=8` 会按 `rollout.n=4` 展开为全局 32 条轨迹，再按双卡数据并行拆分为每卡 16 条。正式运行前要求约 240 GiB cgroup，并以至少 120 GiB 空闲磁盘作为安全基线。

### 2.6 分级 Pilot 与 fail-closed Gate

项目不直接运行 6K 正式训练，而是采用两级 Pilot：

1. 32-prompt Pilot 验证真实 rollout、mixed group、advantage、非零有限策略梯度、optimizer step 和 actor 参数变化；
2. 64-prompt Pilot 验证更长运行、checkpoint 保存恢复、最终 actor 合并、5 条冷加载和资源预算。

启动入口采用 fail-closed 设计。它检查 Base/Data/Reward 哈希、图像路径、gold 隔离、batch 算术、KL 路径、禁止组件、GPU、cgroup、磁盘、端口、输出目录碰撞和训练授权。任何必要条件失败都会拒绝运行并生成阻断凭据。

### 2.7 冻结外部评测与配对分析

最终模型采用固定 1,024-token R4 协议：

- ZoomBench：845 条；
- MMStar：1,500 条；
- V* Bench：191 条；
- 合计：2,536 条。

四模型使用相同输入、图像、prompt、生成上限、解析器、Base Judge、请求键、错误规则和分母。明确答案由保守解析器直接评分，歧义回答路由至冻结 Base Judge；inference/Judge error 保留在分母中计错。

最终分析不只比较准确率，还计算：

- GRPO 相对 Base、Vision-OPD、Cached 的百分点变化；
- 样本级错→对、对→错和净变化；
- paired bootstrap 区间；
- train-overlap 与 non-overlap 分层；
- 类别、解析失败、长度截断和 A/B/C/D 输出分布；
- 固定类型、seed 和 SHA256 排序的 Bad Case。

## 三、与已有训练路线的关系

| 路线 | 轨迹来源 | 主要监督/目标 | Teacher/reference | 关键机制 |
|---|---|---|---|---|
| Base | 无新增训练 | 原始模型 | 无 | 评测基线 |
| Vision-OPD | 在线 Student 轨迹 | Teacher 引导的视觉蒸馏 | crop Teacher | JSD、EMA、在线视觉轨迹 |
| Cached Prefix | 预缓存 Base 轨迹 | 缓存前缀监督 | 预缓存 Base | 降低在线生成成本 |
| GRPO | 每题 4 条在线 rollout | 确定性正确性 Reward | 冻结 Base reference | 组内相对优势、clipped PG、KL |

这些路线的轨迹来源、监督信号和训练机制都不同，因此最终四模型比较属于工程路线比较，不是只改变一个 objective 的严格单变量实验。

## 四、已经完成的工作

### 4.1 数据盘检查与清理

在转换前完成数据盘检查，并按授权清理两批不再需要的内容。Base、Vision-OPD merged、Cached merged 和已冻结评测产物继续保留。清理后约有 509 GiB 可用空间。

旧 FSDP 中间 checkpoint 已不再作为精确恢复来源。Vision-OPD 和 Cached 仍可从 Base 重新调参训练，但无法从已删除的旧优化器状态原位续训。

### 4.2 GRPO 数据转换

| 项目 | 已完成结果 |
|---|---:|
| 源数据行数 | 6,241 |
| 输出数据行数 | 6,241 |
| 唯一 sample_id | 6,241 |
| 可判分样本 | 6,241 |
| 不可判分样本 | 0 |
| 每条策略图像 | 1 张完整红框图 |
| Gold A | 1,740 |
| Gold B | 1,540 |
| Gold C | 1,461 |
| Gold D | 1,500 |

数据哈希：

- 源 Parquet：`142c972f182cc0bf90b2ab44a2255896643c5983e0eaf8c7c58f5a488a89031e`
- GRPO Parquet：`9550fac714fe030e6604e38b2e78db9970268d22464808a36f2a885ff1229206`

### 4.3 Reward 实现与测试

| 项目 | 已完成结果 |
|---|---:|
| Reward 单元测试 | 12 passed |
| 格式子用例 | 125 passed |
| 数据转换测试 | 3 passed |
| 全量正确模板 | 6,241/6,241 得 1 |
| 全量错误模板 | 6,241/6,241 得 0 |
| 全量 Reward 试验 | 12,482 |
| 全量验证失败 | 0 |
| 动态文件路径加载 | PASS |

Reward SHA256：`27508c95922f7717c3b5ecad6d71a8c6d7c38a616518accb033d52d633676c20`

### 4.4 候选配置与启动入口

已完成：

- `configs/grpo_6241.yaml`；
- `scripts/grpo_training_preflight.py`；
- `scripts/run_grpo_guarded.py`；
- `scripts/run_grpo_2gpu.sh`；
- `tests/test_grpo_training_preflight.py`。

静态预检实际检查了 Base 文件、6,241 行和唯一 ID、全部图像路径、Reward 路由、Data/Reward 哈希、gold 输入隔离、无 bbox crop、batch 算术、单一 KL 路径、禁止组件和 Hydra 可合并性。

最终回归为 18 passed、125 个 Reward 子用例通过。`--preflight-only` 已生成 `preflight.json`、`hydra_overrides.json`、`resolved_hydra_config.yaml` 和 `command.txt`。当前 Pilot 与正式训练授权均为 false，`--run` 会按设计返回 `RUN BLOCKED`。

### 4.5 执行与交付方案

已完成 Day18–21 的工作简报草案，明确了：

- 每个阶段的固定合同、具体操作、验收 Gate 和证据目录；
- 失败 attempt 不覆盖、输出目录不碰撞和 checkpoint 原子晋级；
- Reward、gradient、advantage、coverage、资源和费用的记录方法；
- R4 pre-score freeze、Smoke、正式评测和四模型比较；
- 常见故障的识别信号与处理方式；
- 单 seed、本地替代 Judge 和方法差异造成的结论边界。

## 五、尚未完成的工作

### 5.1 Day18：32-prompt 真实更新 Pilot

固定 32 条分层样本，从原始 Base 独立启动：

- 运行 4 个外层迭代；
- 生成 128 条 rollout；
- 保存逐条 response、gold、Reward、advantage、group UID 和 policy version；
- 验证至少存在 mixed-reward group；
- 离线复算 Reward/advantage；
- 验证 response padding 不进入 loss；
- 证明存在非零有限 policy-gradient；
- 证明 actor 参数发生由正确性 Reward 驱动的变化；
- 人工复核至少 20 条真实输出并检查 Reward hacking；
- 根据 `finish_reason=length` 冻结 response length。

### 5.2 Day18：64-prompt 稳定性与恢复 Pilot

32 条 Gate 通过后，从 Base 重新独立初始化：

- 运行 8 steps、256 条 rollout；
- 保存中间 checkpoint；
- 做一次受控中断和恢复；
- 核对 optimizer、scheduler、global step、sampler、RNG 和样本顺序；
- 合并最终 actor；
- 在新进程完成 5/5 非空冷加载；
- 测量显存、CPU/cgroup、step 时间、磁盘、GPU 小时和费用；
- 生成 Day19 promotion/freeze receipt。

### 5.3 Day19：6K 正式训练

正式训练从冻结 Base 独立开始，不能接续 Pilot、Vision-OPD 或 Cached checkpoint。

| 项目 | 冻结合同 |
|---|---:|
| 源 prompt | 6,241 |
| epoch | 1 |
| 有效 prompt | 6,240 |
| dropped prompt | 1 |
| padding | 0 |
| 外层迭代 | 780 |
| 有效 rollout | 24,960 |
| optimizer updates | 780 |
| 中段 checkpoint | global_step_390 |
| 最终 checkpoint | global_step_780 |

训练过程中需要持续记录 Reward 均值/方差、mixed-group fraction、全对/全错组、parse-invalid、A–D 分布、length rate、KL、entropy、clip fraction、grad norm、吞吐和资源。训练结束后生成 coverage receipt，证明 6,240 个有效样本、唯一 dropped ID、24,960 条有效轨迹和 780 次更新完整一致。

### 5.4 Day20：模型定版与 R4

- 冻结 `global_step_780` 文件清单和 SHA256；
- 在 `.inprogress` 中将 FSDP 合并为 Hugging Face 模型；
- 校验 tensor、dtype、配置差异和 tokenizer/processor/template；
- 原子晋级 merged 模型；
- 在新进程完成 5/5 冷加载；
- 在产生 GRPO 分数前冻结模型身份、R4 配置、请求清单、Judge、分母和错误规则；
- 每个 Benchmark 抽取 4 条，共运行 12 条 Smoke；
- Smoke Gate 通过后完成 2,536 条正式预测和评分；
- 保存逐样本原始回答、解析结果、Judge 输出、finish reason 和 token 数。

### 5.5 Day21：四模型比较与交付

- 校验 Base、Vision-OPD、Cached、GRPO 的 2,536 个请求键完全一致；
- 生成逐 Benchmark 准确率、正确数和百分点变化；
- 计算错→对、对→错、净变化和 paired bootstrap；
- 按 overlap、类别、invalid、length 和 Judge 路由分层；
- 在查看案例前冻结 Bad Case 类型、seed、SHA256 排序和每类上限；
- 汇总训练方法、轨迹来源、损失、资源、墙钟和费用；
- 更新最终报告、README、面试问答和证据索引；
- 对所有数字执行来源和 SHA256 回溯检查。

## 六、已经取得的成果

目前能够确认的是工程和验证成果，不能确认模型能力提升。

| 成果类别 | 已确认成果 |
|---|---|
| 数据 | 6,241/6,241 行成功转换；唯一且全部可判分 |
| 图像输入 | 每条保留 1 张完整红框图；移除 Teacher crop 和 bbox 图像 |
| Reward | 确定性二元规则已实现；无需神经 Reward Model |
| Reward 测试 | 12 tests、125 格式子用例、12,482 次全量验证，失败 0 |
| 训练合同 | batch、rollout、advantage、policy loss、KL、资源与覆盖规模已冻结 |
| 静态预检 | 模型、数据、图像、哈希、路由、Hydra 和禁止组件检查通过 |
| 安全入口 | 无授权或资源不满足时拒绝启动；训练和 GPU 未被误触发 |
| 可复现性 | 数据、Reward、配置、命令和预检产物均有 SHA256 或冻结凭据 |
| 执行方案 | Day18–21 的 Pilot、正式训练、R4 和交付 Gate 已建立 |

## 七、训练完成后需要回填的成果数字

### 7.1 Pilot 与正式训练

| 指标 | 结果 |
|---|---:|
| 32-prompt mixed-reward groups | `[待填]` |
| 32-prompt parse-invalid | `[待填]` |
| 32-prompt length 截断 | `[待填]` |
| 人工 Reward 审计数量 | `[待填]` |
| 正式 response length | `[待填]` tokens |
| 正式 mean Reward：初始/中段/最终 | `[待填] / [待填] / [待填]` |
| mixed-group fraction：初始/中段/最终 | `[待填] / [待填] / [待填]` |
| all-zero group rate | `[待填]` |
| all-one group rate | `[待填]` |
| parse-invalid rate | `[待填]` |
| length rate | `[待填]` |
| KL | `[待填]` |
| entropy | `[待填]` |
| clip fraction | `[待填]` |
| grad norm | `[待填]` |
| A/B/C/D 输出分布 | `[待填]` |

### 7.2 资源与成本

| 指标 | 32 Pilot | 64 Pilot | 正式训练 |
|---|---:|---:|---:|
| GPU 型号/数量 | `[待填]` | `[待填]` | `[待填]` |
| 峰值显存/GPU | `[待填]` | `[待填]` | `[待填]` |
| 峰值 CPU RSS/cgroup | `[待填]` | `[待填]` | `[待填]` |
| 稳态 step 中位数/P95 | `[待填]` | `[待填]` | `[待填]` |
| completion tokens | `[待填]` | `[待填]` | `[待填]` |
| 墙钟 | `[待填]` | `[待填]` | `[待填]` |
| GPU 小时 | `[待填]` | `[待填]` | `[待填]` |
| 峰值磁盘 | `[待填]` | `[待填]` | `[待填]` |
| 估算费用 | `[待填]` | `[待填]` | `[待填]` |

### 7.3 固定 R4 四模型结果

| Benchmark | N | Base | Vision-OPD | Cached | GRPO | GRPO−Base | GRPO−VOPD | GRPO−Cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ZoomBench | 845 | 50.65% | 52.54% | 51.36% | `[待填]` | `[待填]` pp | `[待填]` pp | `[待填]` pp |
| MMStar | 1,500 | 68.00% | 59.20% | 56.33% | `[待填]` | `[待填]` pp | `[待填]` pp | `[待填]` pp |
| V* Bench | 191 | 82.72% | 71.73% | 73.30% | `[待填]` | `[待填]` pp | `[待填]` pp | `[待填]` pp |
| Micro | 2,536 | 63.33% | 57.93% | 55.95% | `[待填]` | `[待填]` pp | `[待填]` pp | `[待填]` pp |

Base MMStar 的 72.73% 属于自适应输出上限诊断，不能替换固定 1,024-token R4 主表中的 68.00%。

### 7.4 配对变化与失败分析

| 比较 | 错→对 | 对→错 | 净变化 | paired bootstrap |
|---|---:|---:|---:|---:|
| Base → GRPO | `[待填]` | `[待填]` | `[待填]` | `[待填]` |
| Vision-OPD → GRPO | `[待填]` | `[待填]` | `[待填]` | `[待填]` |
| Cached → GRPO | `[待填]` | `[待填]` | `[待填]` | `[待填]` |

Bad Case 还需回填候选数、固定抽样数、人工复核数、Reward 投机案例数和解析/长度失败案例数。

## 八、可能遇到的问题与处理方法

| 问题 | 识别信号 | 处理方法 |
|---|---|---|
| GPU 或 cgroup 不满足 | 看不到 GPU、cgroup 仍为 2 GiB | 不启动训练；切换真实双卡会话并重跑 preflight |
| Qwen Processor/RLHFDataset OOM | SIGKILL 或 cgroup OOM 增长 | 先单进程加载小样本，检查 worker/offload，不跳过数据 |
| vLLM rollout OOM | CUDA OOM、KV cache 分配失败 | 调整已允许的 memory utilization、capture 或 offload，另建 attempt |
| Reward 全 0 或全 1 | mixed-group fraction 为 0 | 检查采样是否重复、解析失败率和题目难度，不人为改 Reward |
| rollout 与 gold 串组 | group UID 内图片、问题或 gold 不一致 | 立即停止，修复 repeat/interleave 和分布式重排后从 Base 重跑 |
| advantage 对不上 | 离线均值/std 与训练 tensor 不一致 | 核对样本标准差、epsilon、padding mask 和 group 顺序 |
| 参数变化只来自 KL | policy-gradient 为 0但 actor delta 非零 | 分离 PG、KL 和 weight decay 梯度，要求 mixed group 产生非零 PG |
| response 截断 | `finish_reason=length` 集中 | 在正式训练前冻结更长上限，并从 32-prompt Pilot 重跑 |
| Reward hacking | 高 Reward 依赖格式、冲突标签或泄漏 | 升级 Reward 版本和哈希，补测试并重走 Pilot |
| checkpoint 恢复错位 | step、LR、sampler 或 RNG 跳变 | 保留失败现场，修复恢复合同，不晋级该 checkpoint |
| 正式训练 NaN/Inf | loss、KL、entropy 或 grad norm 非有限 | 受控停止，保存故障 batch，新建 attempt 排查 |
| coverage 重复或遗漏 | sample/rollout 数与合同不符 | 区分失败重试和有效轨迹，修复 sampler 后重跑受影响区间 |
| 模型合并 OOM | merger SIGKILL、磁盘急降 | 使用 `.inprogress` 原子合并，失败目录不晋级 |
| 冷加载失败 | 5 条中出现空输出或启动错误 | 检查权重、tokenizer、processor 和 template；禁止进入外评 |
| R4 parser 回归 | 普通大写字母被误判为选项 | 使用 Day16 冻结回归集阻断，不增加有利回退 |
| Judge 失败 | 非 Yes/No 或请求失败 | 保留原始输出并按冻结规则计错，不人工改判 |
| 外评分数不理想 | 某项低于前三模型 | 完成全量评测并如实分析，不重选 checkpoint 或修改协议 |
| 单项退化被 Micro 掩盖 | Micro 提升但某 Benchmark 下降 | 三项 Benchmark 分开报告，Micro 只作附加汇总 |
| 单 seed 被过度解读 | 无重复训练却声称稳定提升 | 明确探索性限制，bootstrap 只描述固定测试样本不确定性 |

## 九、成功情景下的项目结论模板

> GRPO 项目完成了 6,241 行视觉选择题数据转换和确定性 Reward 构建，通过 12 个 Reward 单元测试、125 个格式子用例及 12,482 次全量验证。训练从原始 Qwen3.5-4B Base 独立初始化，使用每题 4 条在线 rollout、组内标准化 advantage、clipped policy gradient 和 reference KL，在 1 epoch 内完成 780 次 optimizer update、覆盖 6,240 个有效 prompt 和 24,960 条有效轨迹。最终模型在固定 1,024-token R4 上的 ZoomBench、MMStar、V* 和 Micro 分别为 `[待填]`、`[待填]`、`[待填]`、`[待填]`，相对 Base 分别变化 `[待填]`、`[待填]`、`[待填]`、`[待填]` 个百分点。训练总墙钟为 `[待填]`，GPU 小时为 `[待填]`，估算费用为 `[待填]`。

只有在 Day18–21 的真实凭据完整后，才能使用上述结论。训练 Reward 提升不能替代外部 R4 accuracy；Pilot 冷加载不能作为能力指标；单 seed 和本地 Qwen3.5-4B Base Judge 也不能支持跨 seed 稳定性或论文级完整复现声明。

## 十、项目边界

- GRPO 是在看到 Base、Vision-OPD、Cached 结果后设计的探索性扩展；
- 当前只计划单训练 seed；
- Reward 是选择题规则函数，不是 learned Reward Model；
- 本地 R4 Judge 是 Qwen3.5-4B Base，未复现论文 GPT-OSS-120B Judge；
- Base、Vision-OPD、Cached、GRPO 的训练信号和轨迹来源不同；
- 固定 1,024-token R4 是主比较协议，自适应输出上限只能作为诊断；
- 外评只能使用训练前冻结的请求清单和最终 checkpoint；
- 不因结果不理想修改 Reward、重选中间 checkpoint、改变分母或删除失败请求。

## 十一、关键证据索引

- Day17 工作简报：`docs/day17_grpo_data_reward_config_work_brief.md`
- Day17–21 执行计划：`docs/day17_day21_grpo_execution_plan.md`
- Day18 Pilot 草案：`docs/day18_grpo_pilot_work_brief_draft.md`
- Day19 正式训练草案：`docs/day19_grpo_formal_training_work_brief_draft.md`
- Day20 定版与 R4 草案：`docs/day20_grpo_model_finalization_and_r4_eval_work_brief_draft.md`
- Day21 四模型交付草案：`docs/day21_grpo_four_model_delivery_work_brief_draft.md`
- 数据转换：`scripts/prepare_grpo_data.py`
- Reward：`verl/utils/reward_score/vision_opd_mcq_grpo.py`
- Reward 验证：`scripts/validate_grpo_reward.py`
- 候选配置：`configs/grpo_6241.yaml`
- 静态预检：`scripts/grpo_training_preflight.py`
- guarded launcher：`scripts/run_grpo_guarded.py`
- 双卡入口：`scripts/run_grpo_2gpu.sh`
- 数据与 Reward 凭据：`artifacts/runs/E-D17-GRPO-DATA-001/`
- 配置预检凭据：`artifacts/runs/E-D17-GRPO-CONFIG-001/preflight/`

## 十二、可直接投递的简历表述

**多模态GRPO后训练（进行中）：**基于Qwen3.5-4B Base构建视觉选择题分支，完成6,241条数据转换与Reward审计，`scorable=6,241/6,241`。每题在线生成4个回答，以组内标准化优势和clipped PG更新策略，并用Base KL约束偏移；结合动态微批次、CPU offload、资源门禁、异常中止与断点恢复保障双卡训练。固定R4四模型对比；完成`[待填]`步，Micro`[待填]%`，较Base/Vision-OPD提升`[待填]/[待填]pp`。
