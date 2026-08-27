# Vision-OPD 可执行项目规划

> Day 1～20 完成可投递版本；Day 21～30 补齐 GRPO。
> 本文档替代原“21 天学习计划”。已完成的论文阅读、代码理解、环境配置、Qwen3.5-4B 下载与普通多模态推理不再重复，也不安排与项目无关的 Tensor/玩具实验。
>
> 计划修订：Day 4 完成后删除 SFT 分支，改为在训练开发阶段使用内部 `eval-128` 与 `retention-64`，在模型最终定版后统一运行 ZoomBench、MMStar、V* Bench。Day 1～4 已完成的数据、评测与 Cached Prefix 证据继续有效，不重复执行；仅更新冻结文档中的实验范围和术语。
>
> 后续 Gate 修订（2026-08-25）：Day 1～5 的任务正文、结果、失败轮次和哈希证据不回写；从 Day 6 起新增外部结果防泄漏、Judge 校准、多模态长度、Cached Prefix 契约和跨平台哈希 Gate。
>
> 论文对齐评测修订（2026-08-26）：`E-D6-001` 已按 Day 5 冻结的旧项目协议完成并永久保留，定位为旧协议工程诊断基线，不覆盖、不删除，也不与后续训练模型的论文对齐主结果直接比较。后续外部主比较统一使用 `E-PAPER-BASEJUDGE-001`：按公开论文/官方仓库尽量对齐非思考推理参数，唯一预先声明的核心替代是使用固定原始 Qwen3.5-4B Base Judge 代替不可获得的 GPT-OSS-120B Judge。机器可读协议见 `artifacts/runs/E-PAPER-BASEJUDGE-001/preflight/paper_aligned_evaluation_amendment.yaml`。
>
> 论文对齐 R2 历史修订（2026-08-26）：R2 曾移除三项可控实现差异：服务使用 TP=1、显存利用率 0.75 和 GDN Triton；ZoomBench/MMStar 保留源图字节；V* 请求图无条件 RGB→PNG。其 amendment 与 Smoke 继续作为历史审计证据；R1/R2 可执行 YAML 已于 2026-08-27 按负责人决定删除。
>
> 单卡计费 R3 修订（2026-08-27）：R3 已成为 Base、Vision-OPD、Cached Prefix、GRPO 的唯一现行外部 Benchmark 标准。唯一可执行配置为 `configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml`；除被测 checkpoint 与独立输出目录外不得改变任何评测条件。机器可读 amendment 见 `artifacts/runs/E-PAPER-BASEJUDGE-001/preflight/paper_alignment_r3_single_gpu_cost_amendment.yaml`，规范见 `docs/benchmark_protocol.md`。

## 1. 最终目标

围绕同一个 Qwen3.5-4B、同一批 1024 条训练数据和两层冻结评测协议，完成以下实验：

| 实验 | 训练信号 | 必须完成时间 | 定位 |
|---|---|---:|---|
| Vanilla / Base | 不训练 | Day 6 | 内部与外部统一训练前基线 |
| Vision-OPD | Student 在线轨迹上的 Crop Teacher Top-K JSD | Day 12 | 项目主实验 |
| Cached Prefix | 预先缓存的 Base Student 轨迹上的同一 JSD | Day 18 | 唯一主消融 |
| GRPO | 可验证答案 Reward + 组内相对优势 | Day 30 | 求职向扩展 |

两层评测协议：

| 阶段 | 数据 | 用途 |
|---|---|---|
| 训练开发 | internal `eval-128`、`retention-64` | 快速回归、配对分析、格式与同分布能力保持检查 |
| 最终定版 | ZoomBench、MMStar、V* Bench | 细粒度能力、通用多模态保持和论文方向交叉验证 |

术语边界必须始终保持准确：

- Vision-OPD 是 on-policy 自蒸馏，不是 GRPO/RLVR。
- Cached Prefix 的目标是“前缀来源”消融；只有 Day 14 严格契约 Gate 通过后才能称为单变量。若无法证明文本重编码与训练 rollout 契约等价，必须降级表述为“Base 离线文本重编码 Prefix 实现消融”。
- Base checkpoint 是未经本项目 Vision-OPD 训练的原始 Qwen3.5-4B；Vanilla 是直接评测该 Base，不产生新 checkpoint。
- Vision-OPD、Cached Prefix、GRPO 是三条独立训练分支，均从同一个 Base checkpoint 启动，不串行继承。
- 论文作者发布的官方 Vision-OPD-4B 只可作为可选参考行，不能替代本项目 Base，也不能冒充本人训练结果。
- 外部 Benchmark 只用于冻结后的最终模型和一次 Base 基线，不用于反复挑 checkpoint 或调超参数。
- 所有结果都限定为“4B、1024 条数据的小规模复现与受控比较”，不宣称复现论文完整 6.2K 结果。

当前执行状态（2026-08-27 同步）：

| 范围 | 状态 | 边界 |
|---|---|---|
| Day 1～3 | PASS | 项目冻结、确定性 1024/128/64 划分、图片/QA/服务器 Parquet Gate 已完成 |
| Day 4 | PASS | Base internal `67/128`；Cached Prefix `1024/1024`，保留 54 条固定上限截断 |
| Day 5 | PASS | 三项 Benchmark 协议、数据、overlap、64 次 Smoke 请求与 Day 6 预算 Gate 已完成 |
| Day 6 | PASS（旧协议） | `E-D6-001` 的逐样本预测、评分、汇总、资源指标和成本已归档；仅作为旧协议诊断证据 |
| 论文对齐 Base 重评 | PASS（R3） | 2536/2536 完整；ZoomBench 50.65%、MMStar 75.07%、V* 83.77%，作为后续统一基线 |
| Day 7～30 | 未开始 | Vision-OPD、Cached、GRPO 定版后均只使用 R3 单卡协议评测 |

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
| 完整项目预算 | 训练与评测合计硬上限 2000 元；Day 5 Benchmark Smoke 后冻结外部评测预算 |
| 双卡时长 | 删除 SFT 后重新按 Vision-OPD、Cached、GRPO 实测吞吐和 Benchmark Smoke 估算 |
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
| train | 1024 | Vision-OPD、Cached Prefix、GRPO |
| eval | 128 | 内部主评测，训练期间不可用于优化 |
| retention | 64 | 内部格式与同分布能力保持检查，不冒充外部通用能力评测 |

要求：

- 以原始问题或图像 ID 分组切分，禁止同图泄漏到 train/eval。
- 每条数据有稳定 `sample_id`。
- Vision-OPD 使用 `完整红框图 + 裁剪图 + 问题`。
- Cached Prefix 使用 Day 4 基于 Base 模型预生成的回答，不能使用 Vision-OPD 训练后的模型生成。
- GRPO 只保留可以被规则可靠判分的封闭式样本；若不足 1024 条，以实际可验证数量为准并如实记录。
- ZoomBench、MMStar、V* Bench 使用各自官方划分；下载后检查与 train/eval/retention 的精确哈希和感知哈希重叠，不能静默删除官方测试样本或隐瞒重叠。

### 3.3 建议训练配置

以下是起始配置，Smoke 证据优先于纸面参数。

#### Vision-OPD 与 Cached Prefix

| 参数 | 值 |
|---|---|
| epoch | 1 |
| global batch | 8 |
| 预计 optimizer steps | 128 |
| rollout n | 1 |
| max prompt length | Day 7 用实际 Processor 统计 P99/max；根据 Day 4 已见 4,212～6,511 Token 输入，起始候选为 8192，禁止继续默认 4096 |
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
  benchmark_eval.yaml  # E-D5 历史协议
  benchmark_eval_paper_basejudge_r3_single_gpu.yaml  # 唯一现行外评协议
  vopd_1024.yaml
  cached_prefix_1024.yaml
  grpo_1024.yaml
scripts/
  prepare_project_subset.py
  validate_project_data.py
  check_benchmark_overlap.py
  run_vopd_2gpu.sh
  generate_cached_prefix.py
  run_cached_prefix_2gpu.sh
  prepare_grpo_data.py
  run_grpo_2gpu.sh
  archive_experiment.py
eval/
  run_internal_eval.py
  run_eval.sh
  compare_experiments.py
  build_badcases.py
tests/
  test_project_dataset.py
  test_benchmark_protocol.py
  test_benchmark_overlap.py
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
  benchmark_protocol.md
  final_report.md
  interview_qa.md
```

每个实验使用唯一 ID：

| ID | 实验 |
|---|---|
| E-D4-001 | Base / Vanilla 内部 eval-128 与 Cached Prefix 生成 |
| E-D5-001 | ZoomBench、MMStar、V* Bench 协议与 Smoke |
| E-D6-001 | Base / Vanilla 三项外部 Benchmark 旧协议诊断基线（已完成，永久保留） |
| E-PAPER-BASEJUDGE-001 | Base、Vision-OPD、Cached、GRPO 论文对齐外部主评测（固定 Base Judge） |
| E-D7-001 | Vision-OPD Smoke |
| E-D8-001 | Vision-OPD 64 条稳定性训练 |
| E-D10-001 | Vision-OPD 1024 正式训练 |
| E-D12-001 | Vision-OPD 内部定版 |
| E-D14-001 | Cached Prefix 契约测试 |
| E-D15-001 | Cached Prefix 64 条稳定性训练 |
| E-D16-001 | Cached Prefix 1024 正式训练 |
| E-D18-001 | Cached Prefix 内部定版 |
| E-D18-002 | Vision-OPD/Cached 统一外部评测 |
| E-D23-001 | GRPO 32 prompt Pilot |
| E-D24-001 | GRPO 64 prompt Pilot |
| E-D25-001 | GRPO 正式训练 |
| E-D28-001 | GRPO 最终内部与外部评测 |

每个训练实验目录至少包含：

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

评测实验至少保存协议、数据 revision/hash、模型 hash、命令、环境、逐样本预测、评分详情、汇总和成本。只有同时满足“模型身份可核验、固定评测完成、逐样本结果与配置齐全”才能标记为评测完成。

## 5. Day 1～20：可投递版本

### Day 1：冻结项目状态与资源边界（4 小时）

任务：

1. 记录本地与服务器仓库的 commit、branch、remote、`git status` 和现有 diff。
2. 对现有个人改动做独立提交或补丁备份；不得 reset。
3. 创建 `docs/project_freeze.md` 和 `configs/project_1024.yaml`。
4. 在服务器记录双卡型号、显存、CUDA、PyTorch、Python、磁盘和模型路径。
5. 确定数据获取方案：临时扩容，或异机抽取后上传。
6. 若后续调整实验范围，在冻结文档追加带日期、原因和影响范围的 amendment，不伪造原始记录，也不因文字修订重跑已通过的环境 Gate。

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

- 模型、硬件、数据量、两层评测协议、预算和实验矩阵均已冻结；删除 SFT、加入外部 Benchmark 的后续变更有 amendment 记录。
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

### Day 4：冻结内部评测器、跑 Vanilla、生成 Cached Prefix（5 小时主动 + 4～8 双卡小时）

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
- 本日结果只定义内部 Base / Vanilla 基线；三项外部 Benchmark 的 Base 基线在 Day 6 新增，不覆盖或重算本日结果。

### Day 5：外部 Benchmark 协议、数据与 Smoke（5～6 小时主动 + 以 Smoke 实测为准）

任务：

1. 冻结 ZoomBench、MMStar、V* Bench 的官方数据来源、revision、许可、样本数、Prompt、图像预处理、生成参数、评分规则和 Judge 配置。
2. 审计仓库现有 `eval/prepare_data.py`、`eval/infer.py`、`eval/judge_qwenlm.py`、`eval/cal_acc.py` 与 `eval/run_eval.sh`，记录哪些题目规则判分、哪些需要固定 Judge。
3. 下载并准备三个 Benchmark；保存原始与转换后数据 hash，不静默改动官方测试集。
4. 实现训练集与 Benchmark 的文件 SHA256、问题文本和感知哈希重叠检查；若发现重叠，同时报告官方全量分数和去重诊断，不把有重叠的结果称为完全独立测试。
5. 使用同一个 Base checkpoint，每个 Benchmark 固定抽取 16 条做端到端 Smoke，验证图片、Prompt、推理、断点恢复、答案解析、Judge 和汇总。
6. 根据 Smoke 的吞吐、输出长度和 Judge 调用量，冻结完整外部评测的时间与费用预算；不得根据 Smoke 准确率选择或更换 Benchmark。

产物：

- `docs/benchmark_protocol.md`
- `configs/benchmark_eval.yaml`
- Benchmark 数据 revision/hash 与 overlap 报告
- `tests/test_benchmark_protocol.py`
- `tests/test_benchmark_overlap.py`
- `artifacts/runs/E-D5-001/`

验收：

- 三个 Benchmark 各 16 条预测齐全，无图片错位、空响应或未记录错误。
- 规则评分可复现；需要 Judge 的任务已冻结同一 Judge 模型、版本、Prompt 和参数。
- overlap 报告能区分精确重复、疑似感知重复和未确认项。
- 完整评测的 GPU/Judge 预算不突破 2000 元项目硬上限。

### Day 6 启动前：后续实验可比性 Gate（新增，不回写 Day 1～5）

任务：

1. 生成 `artifacts/runs/E-D6-001/preflight/training_design_lock.yaml`，在查看 Day 6 完整外部分数前冻结 Vision-OPD/Cached 共享的 Base、数据、seed、Chat Template、Student/Teacher 视图、Top-K/JSD/EMA、epoch、内部评测器和 checkpoint 选择规则。OOM/吞吐参数后续只能根据训练 Smoke 修订，不得引用外部准确率。
2. 建立外部结果防泄漏规则：Day 6 可执行 Base 全量评测，但在 `training_design_lock.yaml` 及其 SHA256 落盘前不得打开汇总分数；Day 12 只做 Vision-OPD 内部定版，Vision-OPD 与 Cached 的外部结果统一延后到 Day 18，避免影响 Cached 设计。
3. 对 ZoomBench 开放题 Judge 做至少 32 对的人工校准，覆盖确定性数字、MathRuler 可解、语义等价、明确错误和边界表达；保存人工标签、Judge 结果、一致率和错误类型。一致率低于 90% 或存在系统性 Base 表达偏置时，LLM Judge 不得作为自动主评分，未决样本改为人工复核或单独报告。
4. 固定服务器轻量验证入口为 `conda run -n vision-opd python -m pytest -q`，先校验 `openai`、`yaml`、`torch` 等必需 import；不再把 Windows base Conda 或独立 `pytest.exe` 的失败冒充服务器项目环境结果。
5. 对 Git 跟踪的 Markdown/YAML/JSON/JSONL/TXT 证据使用 canonical LF SHA256，大型模型、图片、Parquet 和其他二进制文件仍按原始 bytes 计算 SHA256；哈希清单必须标记算法与换行规则。

产物：

- `artifacts/runs/E-D6-001/preflight/training_design_lock.yaml` 及 SHA256
- `artifacts/runs/E-D6-001/preflight/judge_calibration.jsonl`
- `artifacts/runs/E-D6-001/preflight/judge_calibration_summary.json`
- 服务器环境与轻量测试日志

验收：

- 设计锁定、Judge 校准、环境和 canonical hash 四个 Gate 全部 PASS，才能启动 E-D6-001。
- 任何 Gate 失败都只修正后续执行协议，不重跑、删除或改写 Day 1～5 证据。

### Day 6：Base / Vanilla 三项外部 Benchmark 基线（4～6 小时主动 + 机器时间按 Day 5 实测）

状态说明（2026-08-26）：本日 `E-D6-001` 已按当时冻结协议完成。其 thinking/system prompt/8192-token/Zoom crop/严格 MCQ 解析等设置与后续论文对齐协议不同，因此结果保留为旧协议工程诊断，不作为后续跨 checkpoint 主比较基线。

任务：

1. 再次校验 Day 4 使用的原始 Qwen3.5-4B Base checkpoint hash，不加载官方 Vision-OPD-4B 或任何训练后权重。
2. 确认“Day 6 启动前 Gate”全部 PASS，再按 Day 5 冻结协议完整运行 ZoomBench、MMStar、V* Bench。
3. 保存每个 Benchmark 的逐样本输入标识、原始输出、解析结果、Judge 来源、正确性、错误和延迟。
4. 汇总总体及官方子类指标、无效输出率、平均输出长度、GPU 小时和 Judge 成本。
5. 可选评测官方 Vision-OPD-4B 作为参考行，但必须使用独立实验 ID，并明确它不是本项目 Base 或本人训练结果。

产物：

- `artifacts/runs/E-D6-001/`
- Base 的 ZoomBench、MMStar、V* Bench `predictions.jsonl`、评分详情和 `summary.json`
- `artifacts/reports/base_external_benchmarks.md`

验收：

- 三个 Benchmark 均覆盖官方协议要求的全部样本，或对缺失/失败样本逐条解释，禁止只按成功样本计算准确率。
- 模型 hash、Chat Template、thinking 开关、图像预处理、生成参数和评分协议全部可回溯。
- 本日是新增外部 Base 基线，不重跑或覆盖 Day 4 的 `67/128` 内部结果。

### Day 6 补充：论文对齐协议与 Base 重评（新增，完成后进入 Day 7）

任务：

1. 冻结 `E-PAPER-BASEJUDGE-001` 的 R1、R2、R3 amendment；保留 `E-D6-001`、R1/R2 Smoke 和历史哈希作为审计证据。
2. 将 `configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml` 固定为唯一现行外评配置；R1/R2 YAML 已按负责人决定删除，后续不得恢复为可执行入口。
3. 实现可供 Base、Vision-OPD、Cached、GRPO 共用的推理与评分入口：无 system prompt，`enable_thinking=false`，`temperature=0`，`max_tokens=1024`，不传 `top_p`、`top_k`、presence/repetition penalty；并发 16、失败最多重试 3 次。
4. 外部主指标只运行 ZoomBench full 845、MMStar 1500、V* Bench 官方 191，共 2536 个视觉推理请求。V* 保留全部官方样本且汇总分母固定为 191；已确认的 4 条训练集重叠不删除，按“不重复假设”处理。ZoomBench crop 仅可作为另行标注的诊断，不进入论文主表。
5. 评分链统一为 MathRuler、选择题首字母匹配、对仍未判对/未解决样本调用固定 Judge。Judge 固定为未经项目训练的原始 Qwen3.5-4B Base，使用官方 Judge prompt、无 system prompt、`enable_thinking=false`、`temperature=0`、`max_tokens=2048`；不得让被训练后的 checkpoint 给自身评分。
6. 被测模型推理与 Base Judge 分阶段执行并分别支持断点恢复；保存逐样本原始输出、解析结果、规则判分、Judge 输入/输出与来源、最终正确性、token、延迟、错误和重试次数。
7. R3 Smoke、正式 Base 与自动 Gate 已完成；Base 结果和配置哈希已经冻结。Day 18/28/30 的最终模型只更换 checkpoint，并分别写入 `vision_opd`、`cached_prefix`、`grpo`。

产物：

- `artifacts/runs/E-PAPER-BASEJUDGE-001/preflight/paper_aligned_evaluation_amendment.yaml`
- `artifacts/runs/E-PAPER-BASEJUDGE-001/preflight/paper_alignment_r2_amendment.yaml`（历史）
- `artifacts/runs/E-PAPER-BASEJUDGE-001/preflight/paper_alignment_r3_single_gpu_cost_amendment.yaml`
- `configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml`（唯一现行配置）
- `docs/benchmark_protocol.md`（统一评测标准）
- `docs/benchmark_introduction_and_usage.md`（三个 Benchmark 介绍与运行指南）
- `docs/day6_external_benchmark_brief.md`（Day 6 工作简报）
- `artifacts/runs/E-PAPER-BASEJUDGE-001/base/`
- 新协议的推理、Judge、评分、验证和报告脚本/测试

验收：

- 正式视觉请求恰好为 2536，三个 Benchmark 无缺失、重复或被静默排除的样本。
- V* `total=191`；失败、空输出和无法判定结果仍留在官方分母并按冻结失败策略计错。
- `predictions.jsonl`、`judge_results.jsonl`、`scores.jsonl`、`summary.json`、`resume_status.json`、运行清单、哈希、资源与成本记录齐全，且可由逐样本记录重新生成汇总。
- Base、Vision-OPD、Cached、GRPO 之间只允许被测 checkpoint 身份不同；数据、Prompt、图像、生成、解析、Judge 和报告协议完全相同。
- 结果表述为“论文对齐推理 + 固定本地 Base Judge”，不得宣称使用 GPT-OSS-120B 或精确复现论文 Table 2。

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
5. 用训练时实际 Processor 对 train 1024 统计 text/image/total prompt Token 的 P50/P95/P99/max；起始候选用 8192，若仍有超长样本则在 Smoke 前明确扩容或冻结排除规则，不得静默截断。

产物：

- `artifacts/runs/E-D7-001/`
- 双卡训练脚本与冻结配置草案

验收：

- 至少完成 2 个真实 optimizer steps。
- 日志中出现有效 `vopd_loss`、response length、Student grad、EMA update。
- Teacher 没有被 optimizer 直接更新。
- 真实 8～16 条 Smoke 的 prompt 超长/静默截断数为 0，并已保存 train 1024 长度统计。

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

1. 从原始 Base checkpoint 启动，不继承任何训练后分支。
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

### Day 12：Vision-OPD 模型合并与内部定版（5～6 小时主动）

任务：

1. 合并/导出最终 Student。
2. 在新进程完成模型加载与 5 条推理测试。
3. 在 internal eval-128 与 retention-64 上评测并保存逐样本预测。
4. 计算内部样本相对 Vanilla 的 corrected、regressed、unchanged，冻结 Vision-OPD checkpoint、配置、内部结果与 SHA256。
5. 本日不打开 Vision-OPD 外部结果；三项外部 Benchmark 统一延后到 Day 18，待 Cached 实现和 checkpoint 也冻结后再运行。

产物：

- Vision-OPD 最终 checkpoint 与 SHA256
- `artifacts/runs/E-D12-001/`
- `artifacts/reports/vopd_vs_vanilla.md`

验收：

- checkpoint 可复现加载。
- internal 128/64 结果完整；checkpoint、配置和内部结果不再因 Day 18 外部分数更改。
- 即使整体指标没有提升，也保留真实结果和失败分析。

### Day 13：Vision-OPD 审计与 Cached 设计冻结（5 小时）

任务：

1. 归档 Vision-OPD config、命令、commit、环境、日志、模型哈希和费用。
2. 检查是否存在数据泄漏、评测参数变化或训练后重生成 Baseline。
3. 明确 Cached Prefix 的实现位置和数据契约。
4. 设计 `prefix_source: online|cached`，默认保持 `online`。
5. 编写 Cached 契约测试用例。
6. 将 Day 4 缓存的来源明确冻结为 `base_tokenizer_reencoded_openai_response_text`，不冒充 vLLM 原始 sampled token IDs；冻结严格契约的 PASS/降级命名条件。

产物：

- `artifacts/reports/vopd_audit.md`
- `configs/cached_prefix_1024.yaml`
- `tests/test_cached_prefix_contract.py`

验收：

- Vision-OPD 实验可从记录中重建。
- Cached 对照除 prefix 来源外无其他变量变化。
- 在看到 Vision-OPD 外部结果前，Cached 配置、契约和降级命名规则已冻结。

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
6. 契约测试必须比较 online/cached 的 Base 身份、Chat Template 哈希、Processor、图像输入、sampling 参数、EOS/停止策略、prompt IDs、response decode→encode 往返、response mask 和 padding；不要求两次随机采样的 response 逐 Token 相等。

产物：

- Cached Prefix 实现代码
- `artifacts/runs/E-D14-001/`

验收：

- 默认 online 行为不变。
- cached 模式没有调用 rollout generation。
- sample_id 与 cached response 一一对应。
- 训练 loss、梯度和 EMA 均有效。
- 只有所有非 prefix-source 契约项都相同才记为 `STRICT_PREFIX_SOURCE_ABLATION=PASS`；否则停止“单变量”表述，记为 `IMPLEMENTATION_ABLATION` 并列出额外差异。

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

### Day 18：Cached 内部定版与三组统一外部对比（6 小时主动 + 机器时间按 Day 5 实测）

任务：

1. 合并/导出 Cached 最终模型并重新加载。
2. 先在 internal eval-128、retention-64 上按冻结协议评测，并冻结 Cached checkpoint 与内部结果。
3. 实现 `eval/compare_experiments.py`。
4. 统一对比 Vanilla / Base、Vision-OPD、Cached：
   - 总体与题型准确率；
   - corrected/regressed；
   - 输出长度与格式；
   - 训练时长、峰值显存和费用。
5. 对 Vision-OPD vs Cached 做 paired sample 分析。
6. 在 Vision-OPD、Cached 两个 checkpoint、内部结果和比较脚本均冻结后，使用 `E-PAPER-BASEJUDGE-001` 统一运行两个模型的 ZoomBench full、MMStar、V* Bench，并只与 `E-PAPER-BASEJUDGE-001/base` 比较；不得把旧协议 `E-D6-001` 混入主表，外部分数不得触发重选 checkpoint 或重训。

产物：

- Cached checkpoint 与 SHA256
- 三组内部与外部结果表
- `artifacts/reports/prefix_ablation.md`

验收：

- 所有实验分别使用同一个内部 manifest，以及同一版本的三项外部 Benchmark、模型输入协议和评分配置。
- 对照结论只归因于可验证证据，不用单次小样本波动夸大效果。

### Day 19：Bad Case、报告与面试证据（6 小时）

任务：

1. 人工分析 12～20 条代表性样本，覆盖：
   - Vision-OPD 修正、退化；
   - Cached 优于/劣于 online；
   - ZoomBench、MMStar、V* Bench 的一致和冲突趋势；
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

> 基于 Qwen3.5-4B 与双卡 RTX PRO 6000，构建 1024 条细粒度视觉问答训练集，完成 Vision-OPD 小规模复现；实现在线 Student Prefix 与 Base Cached Prefix 的单变量消融，在固定 internal eval/retention 及 ZoomBench、MMStar、V* Bench 上统一比较 Base 与训练后模型，沉淀逐样本预测、Bad Case 和成本分析。

Day 20 不得写：

- “复现论文完整结果”；
- “完成 GRPO”，除非 Day 21～30 已实际完成；
- “显著提升”，除非统一评测有足够证据。

验收：

- Vanilla / Base、Vision-OPD、Cached 均有可核验模型身份与统一内部/外部评测证据。
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

1. 从原始 Base checkpoint 启动，不继承 Vision-OPD 或 Cached Prefix。
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

### Day 28：GRPO 最终评测（5～6 小时主动 + 机器时间按 Day 5 实测）

任务：

1. 在 eval 128 上执行与其他实验相同的 generation 与评分。
2. 在 retention 64 上做能力保持评测。
3. checkpoint 与内部结果冻结后，按 `E-PAPER-BASEJUDGE-001` 运行 ZoomBench full、MMStar、V* Bench，并与同协议 Base、Vision-OPD、Cached 结果比较。
4. 保存逐样本预测并计算 corrected/regressed。
5. 审查高 Reward 但内部或外部固定评测错误的样本。

产物：

- GRPO 内部与三项外部 Benchmark 的 `predictions.jsonl`、评分详情和 `summary.json`
- `artifacts/reports/grpo_eval.md`

验收：

- 评测器版本未变。
- Reward、内部固定评测和外部 Benchmark 指标分开报告。

### Day 29：四组方法统一比较（6 小时）

任务：

1. 比较 Vanilla / Base、Vision-OPD、Cached、GRPO。
2. 分别说明每种方法：
   - 谁生成轨迹；
   - 监督来自哪里；
   - 使用什么 loss；
   - 哪些参数被更新。
3. 汇总效果、训练时间、峰值显存、GPU 小时、费用和实现工作量。
4. 更新 Bad Case 与实验结论。

产物：

- 四组统一结果表
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

> 围绕 Qwen3.5-4B 搭建多模态后训练实验矩阵，在同一 1024 条数据与固定内部/外部评测协议下完成 Vision-OPD 在线自蒸馏、Cached Prefix 消融及可验证 Reward GRPO；实现双卡训练、规则与固定 Judge 评测、逐样本配对分析和成本审计，分析密集 Token 分布监督与稀疏结果奖励的效果及工程权衡。

验收：

- GRPO 有真实 Reward、relative advantage、policy-gradient 更新、checkpoint 和统一评测证据。
- 所有简历数字能回溯到仓库产物。

## 7. 决策 Gate 与降级顺序

不得为了“训练方法数量”牺牲主线闭环。发生阻塞时按以下顺序降级：

1. 保住 Vanilla、Vision-OPD、Cached Prefix 与统一评测。
2. 外部 Benchmark 费用超出 Day 5 预算时，先保留 ZoomBench、MMStar，再延后 V* Bench；不得只保留分数更好看的集合。
3. 外部主评测需要 Judge 时，必须使用 `E-PAPER-BASEJUDGE-001` 冻结的原始 Qwen3.5-4B Base、官方 Judge Prompt 和统一参数；Judge 失败按错并保留逐样本错误，不得临时换 Judge 或让训练后模型自评。
4. GRPO 可验证样本不足时缩小数据量，不混入高噪声开放题。
5. GRPO 正式训练超预算时，保留 32/64 prompt 的真实 Pilot，明确写成“机制与工程验证”，不冒充完整训练。
6. 任何方法若只有脚本启动、没有 checkpoint 与评测，只能写“跑通 Smoke”，不能写“完成训练”。

## 8. 最终验收清单

### Day 20 投递版

- [x] 固定 1024 train、128 eval、64 retention，且无组级泄漏。
- [x] Vanilla 128 条预测和评测结果完整。
- [x] 旧协议 `E-D6-001` 的三项外部 Benchmark 逐样本预测、评分、汇总、overlap 说明和成本已归档。
- [x] 论文对齐 `E-PAPER-BASEJUDGE-001` amendment 已冻结，明确固定 Base Judge 替代 GPT-OSS-120B。
- [ ] `E-PAPER-BASEJUDGE-001/base` 在 ZoomBench full、MMStar、V* 191 上的逐样本预测、Judge 记录与汇总完整。
- [ ] Vision-OPD 真实链路包含在线生成、Crop Teacher、Top-K JSD、Student backward、EMA。
- [ ] Vision-OPD 1024 checkpoint 可加载并完成 internal 128/64 与三项外部评测。
- [ ] Cached Prefix 只改变 prefix 来源。
- [ ] Cached 1024 checkpoint 可加载并完成 internal 128/64 与三项外部评测。
- [ ] 有逐样本 paired comparison 和 12～20 条 Bad Case。
- [ ] 每个实验有 config、命令、commit、日志、费用和哈希。
- [ ] README、最终报告、面试问答和简历 bullet 已完成。

### Day 30 完整版

- [ ] GRPO 数据只含规则可验证题目。
- [ ] Reward 单元测试覆盖正例、负例、格式变体和歧义例。
- [ ] GRPO Pilot 证明 group rollout、relative advantage 和 actor update。
- [ ] GRPO checkpoint 可加载并完成 internal 128/64 与三项外部评测。
- [ ] Base、Vision-OPD、Cached、GRPO 四组方法均使用 `E-PAPER-BASEJUDGE-001`，且固定原始 Base Judge，不让训练后模型自评。
- [ ] 总费用不超过 2000 元，或对超支原因有事前批准和完整记录。
- [ ] 项目边界、失败结果和未复现内容均如实写明。
