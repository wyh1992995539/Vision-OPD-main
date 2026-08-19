# Vision-OPD 可执行项目规划

> Day 1～20 完成可投递版本；Day 21～30 补齐 GRPO。
> 本文档替代原“21 天学习计划”。已完成的论文阅读、代码理解、环境配置、Qwen3.5-4B 下载与普通多模态推理不再重复，也不安排与项目无关的 Tensor/玩具实验。

## 1. 最终目标

围绕同一个 Qwen3.5-4B、同一批 1024 条训练数据和统一评测集，完成以下实验：

| 实验 | 训练信号 | 必须完成时间 | 定位 |
|---|---|---:|---|
| Vanilla | 不训练 | Day 4 | 统一训练前基线 |
| SFT | 参考答案 Token CE | Day 6 | 后训练基线 |
| Vision-OPD | Student 在线轨迹上的 Crop Teacher Top-K JSD | Day 12 | 项目主实验 |
| Cached Prefix | 预先缓存的 Base Student 轨迹上的同一 JSD | Day 18 | 唯一主消融 |
| GRPO | 可验证答案 Reward + 组内相对优势 | Day 30 | 求职向扩展 |

术语边界必须始终保持准确：

- Vision-OPD 是 on-policy 自蒸馏，不是 GRPO/RLVR。
- Cached Prefix 是“前缀来源”消融，不是另一种完整算法。
- SFT、Vision-OPD、GRPO 是三条独立训练分支，均从同一个 Base checkpoint 启动，不串行继承。
- 所有结果都限定为“4B、1024 条数据的小规模复现与受控比较”，不宣称复现论文完整 6.2K 结果。

## 2. 已完成进度与正式起点

以下内容视为已完成，不再占用计划天数：

- 已阅读论文并理解 Vision-OPD 的 Student、EMA Teacher、双视图、在线轨迹和 Top-K JSD。
- 已理解仓库训练主链路。
- 已配置训练环境。
- 已下载并能使用 Qwen3.5-4B。
- 已完成普通图像推理。

Day 1 从“冻结项目、数据和评测协议”开始。工作区现有个人修改必须保留；开始前只做审计和提交，不执行 reset 或覆盖。

## 3. 冻结约束

### 3.1 硬件、预算与总工时

| 项目 | 冻结值 |
|---|---|
| GPU | 单机双卡 RTX PRO 6000 96GB |
| 计费 | 5.98 元/卡/小时，即双卡 11.96 元/小时 |
| 模型 | Qwen3.5-4B |
| 模型路径 | `/root/autodl-tmp/models/Qwen3.5-4B` |
| 数据盘 | 50GB；不得同时保留多个完整 checkpoint |
| 完整项目预算 | 建议 1500～1850 元，硬上限 2000 元 |
| 双卡时长 | 建议 110～154 小时，硬上限约 167 小时 |
| Day 1～20 主动工时 | 约 90～100 小时，平均 4.5～5.5 小时/天 |
| 高强度日 | Smoke、改代码和评测日 6～8 小时 |

每次启动训练前都计算：

```text
本次预计费用 = 预计双卡小时 × 11.96 元
累计预计费用 = 已发生费用 + 本次预计费用
```

若累计预计费用将超过 2000 元，停止扩规模，优先完成评测、报告和证据归档。

### 3.2 数据划分

只使用一次固定划分，后续所有训练不得重新抽样：

| Split | 数量 | 用途 |
|---|---:|---|
| train | 1024 | SFT、Vision-OPD、Cached Prefix、GRPO |
| eval | 128 | 统一主评测，训练期间不可使用 |
| retention | 64 | 通用能力/格式保持检查 |

要求：

- 以原始问题或图像 ID 分组切分，禁止同图泄漏到 train/eval。
- 每条数据有稳定 `sample_id`。
- Vision-OPD 使用 `完整红框图 + 裁剪图 + 问题`。
- SFT 使用 `完整红框图 + 问题 → 参考答案`，只对 Assistant 答案 Token 计算 CE。
- Cached Prefix 使用 Day 4 基于 Base 模型预生成的回答，不能使用 Vision-OPD 训练后的模型生成。
- GRPO 只保留可以被规则可靠判分的封闭式样本；若不足 1024 条，以实际可验证数量为准并如实记录。

### 3.3 建议训练配置

以下是起始配置，Smoke 证据优先于纸面参数。

#### Vision-OPD 与 Cached Prefix

| 参数 | 值 |
|---|---|
| epoch | 1 |
| global batch | 8 |
| 预计 optimizer steps | 128 |
| rollout n | 1 |
| max prompt length | 先统计 P99，默认上限 4096 |
| max response length | 256 |
| learning rate | 2e-6 |
| Top-K | 100 |
| JSD alpha/beta | 0.5 |
| EMA update rate | 0.05 |
| GPU / node | 2 / 1 |

两组只能改变一项：

```text
Vision-OPD: prefix_source=online
Cached:     prefix_source=cached
```

模型、数据、batch、步数、学习率、Teacher 图像、JSD、EMA、评测协议全部相同。

#### SFT

| 参数 | 值 |
|---|---|
| epoch | 1 |
| global batch | 8 |
| micro batch | 1～2，按显存实测 |
| 预计 optimizer steps | 128 |
| learning rate | 2e-6～5e-6，由 Smoke 冻结 |
| precision | BF16 |
| gradient checkpointing | 开启 |
| 更新方式 | 全参数，若无法稳定运行再降级为 LoRA-SFT |
| 保存 | 只保留最终可加载模型 |

#### GRPO

| 参数 | 值 |
|---|---|
| epoch | 1 |
| prompts | 最多 1024 条可验证样本 |
| rollout n | 4 |
| max response length | 先 128，必要时再到 256 |
| global prompt batch | 约 8，以 Smoke 为准 |
| reward | 规则判分，不额外部署 Reward Model |
| GPU | 2 |

### 3.4 存储与安全规则

官方完整数据约 37.5GB，50GB 数据盘无法安全地同时容纳完整下载、4B 模型、缓存和 checkpoint。必须采用以下方案之一：

1. 推荐：数据准备阶段临时扩容到 80～100GB，抽取 1024+128+64 后删除原始大包并恢复容量。
2. 或在其他有空间的机器完成子集抽取，只上传冻结后的子集。

强制规则：

- 原始数据不得盲目下载到只剩不足 45GB 的盘。
- 训练时只保留当前实验 checkpoint；完成合并、加载测试和 SHA256 后，才删除分片。
- 不保存 optimizer state 和大量 step checkpoint，除非中断恢复确有需要。
- 每天结束记录 `df -h`、模型/数据/checkpoint 大小和累计费用。
- 不得为了省空间删除尚未通过加载测试的唯一模型。

## 4. 统一目录与证据规范

计划创建以下项目文件；文档中的命令是目标接口，脚本需在对应日期实现后再执行：

```text
configs/
  project_1024.yaml
  sft_1024.yaml
  vopd_1024.yaml
  cached_prefix_1024.yaml
  grpo_1024.yaml
scripts/
  prepare_project_subset.py
  validate_project_data.py
  prepare_sft_data.py
  run_sft_2gpu.sh
  run_vopd_2gpu.sh
  generate_cached_prefix.py
  run_cached_prefix_2gpu.sh
  prepare_grpo_data.py
  run_grpo_2gpu.sh
  archive_experiment.py
eval/
  run_internal_eval.py
  compare_experiments.py
  build_badcases.py
tests/
  test_project_dataset.py
  test_sft_loss_mask.py
  test_cached_prefix_contract.py
  test_reward_rules.py
  test_grpo_parquet.py
artifacts/
  data/
  eval/
  runs/
  reports/
docs/
  project_freeze.md
  final_report.md
  interview_qa.md
```

每个实验使用唯一 ID：

| ID | 实验 |
|---|---|
| E-D4-001 | Vanilla 统一评测 |
| E-D5-001 | SFT Smoke |
| E-D6-001 | SFT 1024 正式训练 |
| E-D7-001 | Vision-OPD Smoke |
| E-D8-001 | Vision-OPD 64 条稳定性训练 |
| E-D10-001 | Vision-OPD 1024 正式训练 |
| E-D14-001 | Cached Prefix 契约测试 |
| E-D15-001 | Cached Prefix 64 条稳定性训练 |
| E-D16-001 | Cached Prefix 1024 正式训练 |
| E-D23-001 | GRPO 32 prompt Pilot |
| E-D24-001 | GRPO 64 prompt Pilot |
| E-D25-001 | GRPO 正式训练 |

每个实验目录至少包含：

```text
config.yaml
command.txt
git_commit.txt
env.txt
train.log
metrics.jsonl
cost.json
checkpoint_sha256.txt
eval/
  predictions.jsonl
  summary.json
```

只有同时满足“训练完成、模型可加载、固定评测完成、日志与配置齐全”才能标记为实验完成。

## 5. Day 1～20：可投递版本

### Day 1：冻结项目状态与资源边界（4 小时）

任务：

1. 记录本地与服务器仓库的 commit、branch、remote、`git status` 和现有 diff。
2. 对现有个人改动做独立提交或补丁备份；不得 reset。
3. 创建 `docs/project_freeze.md` 和 `configs/project_1024.yaml`。
4. 在服务器记录双卡型号、显存、CUDA、PyTorch、Python、磁盘和模型路径。
5. 确定数据获取方案：临时扩容，或异机抽取后上传。

建议命令：

```bash
git rev-parse HEAD
git status --short
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
df -h
```

产物：

- `docs/project_freeze.md`
- `configs/project_1024.yaml`
- `artifacts/runs/preflight/`

验收：

- 模型、硬件、数据量、评测集、预算和实验矩阵均已写死。
- 所有现有修改有可恢复副本。
- 数据盘方案不会让磁盘在下载中耗尽。

### Day 2：获取数据并实现确定性抽样（5 小时）

任务：

1. 获取原始元数据，先看字段与 ID，再处理图像。
2. 实现 `scripts/prepare_project_subset.py`，固定 seed，按图像/问题组切分。
3. 生成 train 1024、eval 128、retention 64 的候选清单。
4. 为每条样本生成稳定 `sample_id`，保留原始来源 ID。
5. 记录缺图、重复图、空答案和无裁剪图数量。

产物：

- `artifacts/data/candidate_manifest.jsonl`
- `artifacts/data/split_manifest.json`
- `scripts/prepare_project_subset.py`

验收：

- 重复运行得到相同 sample_id 与划分。
- train/eval/retention 的原始图像或问题组无交叉。
- 未开始训练。

### Day 3：冻结 1024 数据与多模态 QA（5 小时）

任务：

1. 抽取并保存 1024+128+64 的完整图、裁剪图、问题和答案。
2. 实现 `scripts/validate_project_data.py` 与数据测试。
3. 检查图像可解码、bbox crop 非空、路径可读、Parquet 字段完整。
4. 统计题型、答案长度、图像尺寸、prompt Token 长度分布。
5. 人工检查至少 30 条：红框内容、裁剪内容、问题和答案是否一致。

建议命令：

```bash
python scripts/validate_project_data.py --config configs/project_1024.yaml
pytest -q tests/test_project_dataset.py
```

产物：

- 冻结 Parquet 与图像子集
- `artifacts/data/data_stats.json`
- `artifacts/data/manual_qa_30.jsonl`
- 数据文件 SHA256

验收：

- 1024/128/64 数量准确。
- 自动校验全通过，人工抽查没有系统性错配。
- 数据总大小与剩余磁盘满足后续训练。

### Day 4：冻结评测器、跑 Vanilla、生成 Cached Prefix（5 小时主动 + 4～8 双卡小时）

任务：

1. 实现 `eval/run_internal_eval.py`，按题型使用确定性规则评测。
2. 规则至少覆盖：多选、短字符串、数字、颜色/方向等封闭答案。
3. 使用固定 generation 参数评测 Base 在 eval 128 上的表现。
4. 保存逐样本预测，不只保存总准确率。
5. 使用同一 Base checkpoint 为 train 1024 预生成 Cached Prefix，并保存生成参数、Token IDs、文本与哈希。

建议命令：

```bash
python eval/run_internal_eval.py --model /root/autodl-tmp/models/Qwen3.5-4B --split eval
python scripts/generate_cached_prefix.py --config configs/project_1024.yaml
```

产物：

- `artifacts/runs/E-D4-001/`
- `artifacts/data/cached_prefix_base_1024.parquet`
- `artifacts/eval/evaluator_version.json`

验收：

- 同一预测重复评分结果一致。
- 128 条预测齐全，无法可靠评分的题目单独标记，不硬判。
- Cached Prefix 恰好对应 1024 个 train sample_id，来自训练前 Base。

### Day 5：SFT 数据适配与真实 Smoke（5 小时主动 + 1～2 双卡小时）

任务：

1. 实现 `scripts/prepare_sft_data.py`：完整图+问题为输入，参考答案为 Assistant。
2. 验证 chat template 和 loss mask，User/Image/Pad Token 必须为 `-100`。
3. 统计序列长度并设置 max length。
4. 编写 `configs/sft_1024.yaml` 和 `scripts/run_sft_2gpu.sh`。
5. 用真实模型、真实 8～16 条项目数据完成 forward/loss/backward/optimizer step。

产物：

- SFT Parquet
- `tests/test_sft_loss_mask.py`
- `artifacts/runs/E-D5-001/`

验收：

- 图像张量确实进入模型。
- 只有 Assistant 答案 Token 贡献 CE。
- loss 有限且至少完成 2 个 optimizer steps。
- 双卡显存、吞吐和预计正式训练时长已记录。

### Day 6：SFT 1024 正式训练与评测（4 小时主动 + 6～12 双卡小时）

任务：

1. 根据 Day 5 冻结 batch、学习率和 max length。
2. 从 Base 启动 1024 条、1 epoch SFT。
3. 监控 loss、grad norm、显存、吞吐和异常样本。
4. 合并/保存模型并执行重新加载测试。
5. 在 eval 128 上按统一协议评测，生成初步对照。

产物：

- `artifacts/runs/E-D6-001/`
- SFT 最终 checkpoint 与 SHA256
- SFT `predictions.jsonl`、`summary.json`

验收：

- 训练按计划结束，模型可在新进程加载。
- 评测覆盖完整 128 条。
- 若当日仍不能得到可加载模型，SFT 降为次要支线，Day 7 起优先保证 Vision-OPD。

### Day 7：Vision-OPD 双卡入口与真实 Smoke（6 小时主动 + 2～4 双卡小时）

任务：

1. 编写 `configs/vopd_1024.yaml` 和 `scripts/run_vopd_2gpu.sh`。
2. 将官方默认 8 卡、大 batch、长 response 改为双卡小规模配置。
3. 用真实 8～16 条项目数据跑完整训练链：
   - Student 看完整图并在线生成；
   - Teacher 看裁剪图；
   - 两者沿用相同 Student prefix；
   - 计算 Top-K JSD；
   - 只对 Student backward；
   - optimizer step 后 EMA 更新 Teacher。
4. 记录一次训练前后 Student/Teacher 参数差异与 Teacher 无梯度证据。

产物：

- `artifacts/runs/E-D7-001/`
- 双卡训练脚本与冻结配置草案

验收：

- 至少完成 2 个真实 optimizer steps。
- 日志中出现有效 `vopd_loss`、response length、Student grad、EMA update。
- Teacher 没有被 optimizer 直接更新。

### Day 8：Vision-OPD 64 条稳定性训练（6～8 小时主动 + 4～8 双卡小时）

任务：

1. 用固定 64 条数据跑完整 1 epoch 或至少 8 个 optimizer steps。
2. 处理 OOM、Ray/vLLM、FSDP、图像 shape、response mask 和 checkpoint 问题。
3. 记录每步耗时、峰值显存、生成耗时占比、loss 曲线。
4. 保存 checkpoint，关闭进程后重新加载并推理 5 条样本。

产物：

- `artifacts/runs/E-D8-001/`
- `artifacts/reports/vopd_64_stability.md`

验收：

- 连续稳定运行，无 NaN/Inf。
- 64 条 checkpoint 可重新加载。
- 已有可信的 1024 条耗时与费用外推。

### Day 9：正式训练 Gate 与配置冻结（5 小时）

任务：

1. 用 Day 8 实测吞吐估算 1024 条训练时长与成本。
2. 检查数据、Base、config、commit、输出目录、磁盘和日志路径。
3. 固定 seed、batch、response length、Top-K、JSD、EMA 和保存策略。
4. 生成正式训练 preflight 报告。
5. 设置训练中止条件。

中止条件：

- 预计单次训练超过 38 双卡小时且没有明确原因。
- 可用磁盘低于“最终 checkpoint 预计大小 × 2 + 5GB”。
- 64 条阶段仍存在 NaN、checkpoint 不可加载或 Teacher 直接有梯度。

产物：

- `artifacts/runs/E-D10-001/preflight.md`
- 最终 `configs/vopd_1024.yaml`

验收：

- 正式训练命令可复制执行。
- 预计费用纳入 2000 元总预算。
- 所有 Gate 为 PASS 才进入 Day 10。

### Day 10：启动 Vision-OPD 1024 正式训练（2～3 小时主动，其余为机器时间）

任务：

1. 从原始 Base checkpoint 启动，不继承 SFT。
2. 观察最初 3 个 optimizer steps。
3. 核对样本 ID、在线 response、Teacher crop、loss、grad、EMA、显存和保存目录。
4. 记录实际开始时间、计费和预计结束时间。

产物：

- `artifacts/runs/E-D10-001/train.log`
- 首 3 步健康检查记录

验收：

- 首 3 步指标稳定。
- 没有误用 cached response。
- 可安全无人值守继续运行。

### Day 11：监控 Vision-OPD 正式训练（2～3 小时主动）

任务：

1. 每 1～2 小时查看一次 loss、吞吐、GPU 利用率、磁盘和进程状态。
2. 异常时先保存日志和报错，再决定恢复或重跑；禁止无记录地反复试参。
3. 同时准备 checkpoint 合并和评测命令。
4. 起草 Cached Prefix 的单变量设计。

产物：

- 更新后的 `metrics.jsonl`
- 训练事件记录和实际费用

验收：

- 训练完成，或有可恢复 checkpoint 和明确恢复命令。
- 没有超出预算仍持续空跑。

### Day 12：Vision-OPD 模型合并与统一评测（5 小时主动 + 2～4 双卡小时）

任务：

1. 合并/导出最终 Student。
2. 在新进程完成模型加载与 5 条推理测试。
3. 在 eval 128 上评测并保存逐样本预测。
4. 若 retention 64 已就绪，同步做能力保持评测。
5. 计算相对 Vanilla 的 corrected、regressed、unchanged 样本。

产物：

- Vision-OPD 最终 checkpoint 与 SHA256
- `artifacts/runs/E-D10-001/eval/`
- `artifacts/reports/vopd_vs_vanilla.md`

验收：

- checkpoint 可复现加载。
- 128 条预测齐全，评测器版本与 Vanilla 相同。
- 即使整体指标没有提升，也保留真实结果和失败分析。

### Day 13：Vision-OPD 审计与 Cached 设计冻结（5 小时）

任务：

1. 归档 Vision-OPD config、命令、commit、环境、日志、模型哈希和费用。
2. 检查是否存在数据泄漏、评测参数变化或训练后重生成 Baseline。
3. 明确 Cached Prefix 的实现位置和数据契约。
4. 设计 `prefix_source: online|cached`，默认保持 `online`。
5. 编写 Cached 契约测试用例。

产物：

- `artifacts/reports/vopd_audit.md`
- `configs/cached_prefix_1024.yaml`
- `tests/test_cached_prefix_contract.py`

验收：

- Vision-OPD 实验可从记录中重建。
- Cached 对照除 prefix 来源外无其他变量变化。

### Day 14：实现 Cached Prefix 分支（6 小时）

任务：

1. 在数据/Trainer 中加入配置开关：

```yaml
data:
  prefix_source: online  # online | cached
```

2. online 分支保持官方 Student 当前策略生成逻辑。
3. cached 分支读取 Day 4 保存的 Base response，构造同样的 response IDs/mask。
4. cached 分支必须绕过在线生成，Teacher 仍看 crop，Student 仍看 full image。
5. 用真实 4～8 条数据执行契约测试和至少 1 个 optimizer step。

产物：

- Cached Prefix 实现代码
- `artifacts/runs/E-D14-001/`

验收：

- 默认 online 行为不变。
- cached 模式没有调用 rollout generation。
- sample_id 与 cached response 一一对应。
- 训练 loss、梯度和 EMA 均有效。

### Day 15：Cached Prefix 64 条稳定性训练（6～8 小时主动 + 4～8 双卡小时）

任务：

1. 从同一个 Base checkpoint 启动 64 条 Cached 训练。
2. 使用与 Vision-OPD 64 条相同的数据顺序、参数、步数和 seed。
3. 记录吞吐、显存、loss、response length 和费用。
4. 保存、重载 checkpoint 并推理 5 条。
5. 估算正式 1024 条时间，目标不超过 24 双卡小时。

产物：

- `artifacts/runs/E-D15-001/`
- `artifacts/reports/cached_64_stability.md`

验收：

- 连续稳定，无 NaN/Inf。
- 与 online 配置 diff 只有 prefix_source 和 cached 文件路径。
- checkpoint 可加载。

### Day 16：启动 Cached Prefix 1024 正式训练（2～3 小时主动）

任务：

1. 从原始 Base checkpoint 启动。
2. 检查最初 3 个 optimizer steps。
3. 核对 cached response 哈希与 Day 4 产物一致。
4. 记录开始时间、预计结束时间和成本。

产物：

- `artifacts/runs/E-D16-001/`

验收：

- 前 3 步稳定。
- 确认没有在线 rollout。
- 模型、数据和超参均与 E-D10-001 匹配。

### Day 17：监控 Cached Prefix 正式训练（2～3 小时主动）

任务：

1. 监控 GPU、磁盘、loss、吞吐和进程。
2. 记录所有恢复、重启和异常。
3. 准备统一评测和对比脚本。

验收：

- 正式训练完成，或有明确可恢复状态。
- 费用记录完整。

### Day 18：Cached 评测与四组统一对比（6 小时主动 + 2～4 双卡小时）

任务：

1. 合并/导出 Cached 最终模型并重新加载。
2. 在 eval 128 和 retention 64 上评测。
3. 实现 `eval/compare_experiments.py`。
4. 统一对比 Vanilla、SFT、Vision-OPD、Cached：
   - 总体与题型准确率；
   - corrected/regressed；
   - 输出长度与格式；
   - 训练时长、峰值显存和费用。
5. 对 Vision-OPD vs Cached 做 paired sample 分析。

产物：

- Cached checkpoint 与 SHA256
- 四组结果表
- `artifacts/reports/prefix_ablation.md`

验收：

- 所有实验使用同一个 eval manifest、evaluator version 和 generation config。
- 对照结论只归因于可验证证据，不用单次小样本波动夸大效果。

### Day 19：Bad Case、报告与面试证据（6 小时）

任务：

1. 人工分析 12～20 条代表性样本，覆盖：
   - Vision-OPD 修正、退化；
   - Cached 优于/劣于 online；
   - SFT 修正、退化；
   - 评测规则无法判断。
2. 实现 `eval/build_badcases.py`。
3. 完成 `docs/final_report.md`：
   - 问题定义；
   - 方法与代码改动；
   - 数据与评测；
   - 实验矩阵；
   - 结果、成本、限制；
   - 失败案例与下一步。
4. 更新 README 的复现命令和目录说明。
5. 编写 `docs/interview_qa.md`，准备 10 个高频追问。

产物：

- Bad Case 表
- 最终实验报告初稿
- README 复现说明
- 面试问答

验收：

- 每个表格数字都能回溯到 `predictions.jsonl` 或训练日志。
- 清楚区分论文结论、本人复现结果和推断。

### Day 20：投递版本验收与简历落地（4～6 小时）

任务：

1. 按最终验收清单逐项补缺。
2. 运行所有轻量测试和 Markdown/Git 检查。
3. 对配置、日志、结果、哈希、成本和命令做最终归档。
4. 写一页项目简介和 3～4 条简历 bullet。
5. 提交代码并打里程碑 tag；开始投递。

简历表述模板：

> 基于 Qwen3.5-4B 与双卡 RTX PRO 6000，构建 1024 条细粒度视觉问答训练集及固定评测集，完成 Vision-OPD 小规模复现；实现在线 Student Prefix 与 Base Cached Prefix 的单变量消融，并与 SFT、Vanilla 统一比较，沉淀训练/评测流水线、逐样本预测、Bad Case 与成本分析。

Day 20 不得写：

- “复现论文完整结果”；
- “完成 GRPO”，除非 Day 21～30 已实际完成；
- “显著提升”，除非统一评测有足够证据。

验收：

- Vanilla、SFT、Vision-OPD、Cached 均有训练或基线证据与统一评测。
- 项目可以写入简历并经得住 checkpoint、日志和代码追问。

## 6. Day 21～30：GRPO 扩展

### Day 21：GRPO 数据转换与可验证样本筛选（5 小时）

任务：

1. 实现 `scripts/prepare_grpo_data.py`。
2. 从冻结 train 1024 中筛选规则可验证样本。
3. 保留完整图、问题、ground truth、题型、sample_id。
4. 将 `data_source` 路由到实际 Reward 函数支持的名称，例如 `zoom-bench`。
5. 禁止给 GRPO 使用 Teacher crop；它只看完整图。

产物：

- GRPO Parquet
- `tests/test_grpo_parquet.py`
- 可验证样本题型统计

验收：

- 每条样本可由对应规则判分。
- 数据量若不足 1024，报告真实数量，不用开放题凑数。

### Day 22：Reward、配置与双卡入口（5 小时）

任务：

1. 实现并测试多选、数字、短答案规范化与 Reward。
2. 对歧义答案返回 unsupported，不强行给 0/1。
3. 编写 `configs/grpo_1024.yaml` 和 `scripts/run_grpo_2gpu.sh`。
4. 设置 rollout n=4、response=128、规则 Reward、无额外 Critic/Reward Model。
5. 记录 group reward、advantage、KL、entropy、response length、actor loss。

产物：

- `tests/test_reward_rules.py`
- GRPO 配置和启动脚本

验收：

- 手工构造的正例、负例、格式变体和歧义例全部通过测试。
- 明确看到 Reward 路由到正确函数。

### Day 23：GRPO 32 Prompt 真实 Pilot（6 小时主动 + 4～6 双卡小时）

任务：

1. 使用 32 prompt、每题 4 rollouts，生成约 128 条轨迹。
2. 完成至少 1～3 个 optimizer steps。
3. 验证同一 prompt 的四条回答正确分组。
4. 检查 Reward 分布、组内相对优势和 actor 参数更新。
5. 人工检查至少 20 条 rollout，排查 reward hacking。

产物：

- `artifacts/runs/E-D23-001/`
- Pilot rollout 样例和 Reward 审计

验收：

- 这是真实 policy-gradient 更新，不只是生成或计算 Reward。
- Reward 非全 0/全 1，组内存在有效差异。

### Day 24：GRPO 64 Prompt 稳定性与正式 Gate（6 小时主动 + 4～8 双卡小时）

任务：

1. 修复 Pilot 问题并跑 64 prompt 稳定性训练。
2. 记录 reward mean/std、advantage、KL、entropy、长度和显存。
3. 保存、重载 checkpoint 并推理。
4. 估算正式训练时间和成本，目标不超过 40 双卡小时。
5. 冻结正式配置。

产物：

- `artifacts/runs/E-D24-001/`
- `artifacts/reports/grpo_64_stability.md`

验收：

- 无 NaN、Reward 崩溃和明显格式投机。
- checkpoint 可加载。
- 正式训练成本仍在总预算内。

### Day 25：启动 GRPO 正式训练（3 小时主动）

任务：

1. 从原始 Base checkpoint 启动，不继承 SFT 或 Vision-OPD。
2. 检查最初 3 个 optimizer steps。
3. 核对 prompt grouping、rollout n、Reward 和 actor update。
4. 记录预计结束时间与费用。

产物：

- `artifacts/runs/E-D25-001/`

验收：

- 前 3 步健康。
- 只有 actor 被预期目标更新。

### Day 26：监控 GRPO（3 小时主动）

任务：

1. 监控 Reward、advantage、KL、entropy、response length 和 GPU。
2. 抽查高 Reward 输出是否真正正确。
3. 记录异常、恢复与实际费用。

验收：

- Reward 上升不伴随明显长度爆炸或格式钻空子。
- 没有无证据改 Reward 规则。

### Day 27：完成训练与 checkpoint 审计（3 小时主动）

任务：

1. 完成训练或从明确 checkpoint 恢复。
2. 合并/导出模型并在新进程加载。
3. 固定 SHA256、配置、命令、日志和费用。
4. 删除分片前确认最终模型可加载。

验收：

- GRPO 模型可复现加载。
- 实验记录完整。

### Day 28：GRPO 统一评测（5 小时主动 + 2～4 双卡小时）

任务：

1. 在 eval 128 上执行与其他实验相同的 generation 与评分。
2. 在 retention 64 上做能力保持评测。
3. 保存逐样本预测并计算 corrected/regressed。
4. 审查高 Reward 但主评测错误的样本。

产物：

- GRPO `predictions.jsonl`、`summary.json`
- `artifacts/reports/grpo_eval.md`

验收：

- 评测器版本未变。
- Reward 指标和外部固定评测指标分开报告。

### Day 29：五组方法统一比较（6 小时）

任务：

1. 比较 Vanilla、SFT、Vision-OPD、Cached、GRPO。
2. 分别说明每种方法：
   - 谁生成轨迹；
   - 监督来自哪里；
   - 使用什么 loss；
   - 哪些参数被更新。
3. 汇总效果、训练时间、峰值显存、GPU 小时、费用和实现工作量。
4. 更新 Bad Case 与实验结论。

产物：

- 五组统一结果表
- 方法关系图
- 完整成本表

验收：

- 不把 JSD loss、Reward 和 accuracy 混为同一指标。
- 不因训练方法更多而掩盖数据量和评测限制。

### Day 30：GRPO 版本项目收尾（4～6 小时）

任务：

1. 更新 README、`docs/final_report.md` 和 `docs/interview_qa.md`。
2. 将简历项目升级为“多模态后训练方法受控比较”。
3. 归档模型哈希、配置、命令、日志、逐样本预测和费用。
4. 运行测试，提交代码并打最终 tag。

简历升级模板：

> 围绕 Qwen3.5-4B 搭建多模态后训练实验矩阵，在同一 1024 条数据与固定评测协议下完成 SFT、Vision-OPD 在线自蒸馏、Cached Prefix 消融及可验证 Reward GRPO；实现双卡训练、规则评测、逐样本配对分析和训练成本审计，分析密集 Token 分布监督与稀疏结果奖励的效果及工程权衡。

验收：

- GRPO 有真实 Reward、relative advantage、policy-gradient 更新、checkpoint 和统一评测证据。
- 所有简历数字能回溯到仓库产物。

## 7. 决策 Gate 与降级顺序

不得为了“训练方法数量”牺牲主线闭环。发生阻塞时按以下顺序降级：

1. 保住 Vanilla、Vision-OPD、Cached Prefix 与统一评测。
2. SFT 全参数 OOM 时降为 LoRA-SFT，但必须改名并记录可训练参数量。
3. eval 题目无法规则判分时标记 unsupported，不引入临时变化的 LLM Judge。
4. GRPO 可验证样本不足时缩小数据量，不混入高噪声开放题。
5. GRPO 正式训练超预算时，保留 32/64 prompt 的真实 Pilot，明确写成“机制与工程验证”，不冒充完整训练。
6. 任何方法若只有脚本启动、没有 checkpoint 与评测，只能写“跑通 Smoke”，不能写“完成训练”。

## 8. 最终验收清单

### Day 20 投递版

- [ ] 固定 1024 train、128 eval、64 retention，且无组级泄漏。
- [ ] Vanilla 128 条预测和评测结果完整。
- [ ] SFT 有真实 Smoke；正式模型若完成，则有统一评测。
- [ ] Vision-OPD 真实链路包含在线生成、Crop Teacher、Top-K JSD、Student backward、EMA。
- [ ] Vision-OPD 1024 checkpoint 可加载并完成 eval 128。
- [ ] Cached Prefix 只改变 prefix 来源。
- [ ] Cached 1024 checkpoint 可加载并完成 eval 128。
- [ ] 有逐样本 paired comparison 和 12～20 条 Bad Case。
- [ ] 每个实验有 config、命令、commit、日志、费用和哈希。
- [ ] README、最终报告、面试问答和简历 bullet 已完成。

### Day 30 完整版

- [ ] GRPO 数据只含规则可验证题目。
- [ ] Reward 单元测试覆盖正例、负例、格式变体和歧义例。
- [ ] GRPO Pilot 证明 group rollout、relative advantage 和 actor update。
- [ ] GRPO checkpoint 可加载并完成固定评测。
- [ ] 五组方法使用统一评测协议。
- [ ] 总费用不超过 2000 元，或对超支原因有事前批准和完整记录。
- [ ] 项目边界、失败结果和未复现内容均如实写明。
