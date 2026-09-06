# Vision-OPD 可执行项目规划

> Day 1～9 保留已完成证据；Day 10～16 完成 6K 数据版 Vision-OPD/Cached 可投递版本；Day 17～21 完成 GRPO 扩展。

> Day12 运行修订（2026-09-07）：双卡合计 14 元/小时；Day12 使用 `scripts/run_day12_vopd.py`，
> 费用只作估算记录，不再要求每次填写累计费用/账单时间或以累计费用门禁阻塞启动。
> 资源与训练异常守护仍启用。旧计费数值为历史口径，详见 [运行修订](day12_operations_amendment.md)。
> 本文档替代原“21 天学习计划”。已完成的论文阅读、代码理解、环境配置、Qwen3.5-4B 下载与普通多模态推理不再重复，也不安排与项目无关的 Tensor/玩具实验。
>
> 历史计划修订：Day 4 完成后曾删除 SFT 分支，并采用内部 `eval-128` 与 `retention-64`。该条仅保留当时的审计背景；2026-09-02 全量训练修订后，128/64 样本已并入 train-6241，不再用于新模型评测。Day 1～4 已完成的数据、评测与 Cached Prefix 证据继续有效，不重复执行。
>
> 后续 Gate 修订（2026-08-25）：Day 1～5 的任务正文、结果、失败轮次和哈希证据不回写；从 Day 6 起新增外部结果防泄漏、Judge 校准、多模态长度、Cached Prefix 契约和跨平台哈希 Gate。
>
> 论文对齐评测修订（2026-08-26）：`E-D6-001` 已按 Day 5 冻结的旧项目协议完成并永久保留，定位为旧协议工程诊断基线，不覆盖、不删除，也不与后续训练模型的论文对齐主结果直接比较。后续外部主比较统一使用 `E-PAPER-BASEJUDGE-001`：按公开论文/官方仓库尽量对齐非思考推理参数，唯一预先声明的核心替代是使用固定原始 Qwen3.5-4B Base Judge 代替不可获得的 GPT-OSS-120B Judge。机器可读协议见 `artifacts/runs/E-PAPER-BASEJUDGE-001/preflight/paper_aligned_evaluation_amendment.yaml`。
>
> 论文对齐 R2 历史修订（2026-08-26）：R2 曾移除三项可控实现差异：服务使用 TP=1、显存利用率 0.75 和 GDN Triton；ZoomBench/MMStar 保留源图字节；V* 请求图无条件 RGB→PNG。其 amendment 与 Smoke 继续作为历史审计证据；R1/R2 可执行 YAML 已于 2026-08-27 按负责人决定删除。
>
> 单卡计费 R3 修订（2026-08-27）：R3 已成为 Base、Vision-OPD、Cached Prefix、GRPO 的唯一现行外部 Benchmark 标准。唯一可执行配置为 `configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml`；除被测 checkpoint 与独立输出目录外不得改变任何评测条件。机器可读 amendment 见 `artifacts/runs/E-PAPER-BASEJUDGE-001/preflight/paper_alignment_r3_single_gpu_cost_amendment.yaml`，后续治理同步见 `artifacts/runs/E-PAPER-BASEJUDGE-001/preflight/benchmark_governance_sync_amendment.yaml`，规范见 `docs/benchmark_protocol.md`。
>
> 6K 数据训练修订（2026-09-04）：Day 1～9 的正文、结果、失败记录、哈希和 `E-D10-001` 1024 条正式训练前 Gate 全部保留，不回写、不删除。`E-D10-001` 尚未执行，其 1024 条配置只作为已审计的历史准入证据，不再作为现行正式训练入口。现行主线读取冻结 revision 下全部 6,241 条源元数据，不再划分 train/eval/test/retention/QA holdout；既有 eval-128、retention-64 仅保留为 Day 1～9 历史证据，不用于新模型选优或能力结论。按负责人决定取消零权重尾批补齐，沿用 verl 原生 `drop_last=True`：global batch 8、1 epoch 时为 780 个 optimizer steps、有效训练 6,240 条、丢弃打乱后末尾 1 条。数据盘上限为 300GB；复用已冻结元数据和 Schema，只要求全量自动图片 QA、服务器 Parquet、全量 Processor 长度审计和异常/最长样本人工复核，不重新做 30 条完整人工 QA。现行 Day 10～21 见本文“主动执行排期”；旧 Day 10～30 内容仅作历史审计，禁止照旧执行。

## 1. 最终目标

围绕同一个 Qwen3.5-4B、同一冻结 revision 的 6,241 条全量训练数据和冻结的 R3 外部评测协议，完成以下实验：

| 实验 | 训练信号 | 必须完成时间 | 定位 |
|---|---|---:|---|
| Vanilla / Base | 不训练 | Day 6 | 内部与外部统一训练前基线 |
| Vision-OPD | Student 在线轨迹上的 Crop Teacher Top-K JSD | Day 13 | 6,241 条项目主实验 |
| Cached Prefix | 预先缓存的 Base Student 轨迹上的同一 JSD | Day 15 | 6,241 条前缀来源主消融 |
| GRPO | 6,241 条的规则 Reward + 组内相对优势 | Day 21 | 仅在 Reward 可验证覆盖 6,241/6,241 时进入正式训练 |

现行评测协议：

| 阶段 | 数据 | 用途 |
|---|---|---|
| 工程验收 | 5 条冷加载样本、Pilot rollout 与训练日志 | 只检查可加载、非空输出、数值稳定和训练合同，不报告能力分数 |
| 最终定版 | ZoomBench、MMStar、V* Bench | 唯一现行能力评测；同一冻结 checkpoint、R3 协议和固定 Base Judge |

术语边界必须始终保持准确：

- Vision-OPD 是 on-policy 自蒸馏，不是 GRPO/RLVR。
- Cached Prefix 的目标是“前缀来源”消融；只有现行 Day 13 严格契约 Gate 通过后才能称为单变量。若无法证明文本重编码与训练 rollout 契约等价，必须降级表述为“Base 离线文本重编码 Prefix 实现消融”。
- Base checkpoint 是未经本项目 Vision-OPD 训练的原始 Qwen3.5-4B；Vanilla 是直接评测该 Base，不产生新 checkpoint。
- Vision-OPD、Cached Prefix、GRPO 是三条独立训练分支，均从同一个 Base checkpoint 启动，不串行继承。
- 论文作者发布的官方 Vision-OPD-4B 只可作为可选参考行，不能替代本项目 Base，也不能冒充本人训练结果。
- 外部 Benchmark 只用于冻结后的最终模型和一次 Base 基线，不用于反复挑 checkpoint 或调超参数。
- Vision-OPD 与 Cached Prefix 使用同一 6,241 条全量训练数据。GRPO 也必须以同一 6,241 条为正式训练输入；若任一记录无法可靠规则判分，GRPO 正式训练 Gate 记为 BLOCKED，不得静默筛成小子集后冒充“完整 6.2K GRPO”。
- 本项目恢复了 Vision-OPD-6K 源数据规模，并将论文核心 response 上限恢复为 1024；仍使用双卡、batch 8、rollout n=1 等资源缩放配置，不得写成“完整复现论文原始 8 卡训练配置或论文数值”。

当前执行状态（2026-09-02 同步）：

| 范围 | 状态 | 边界 |
|---|---|---|
| Day 1～3 | PASS | 项目冻结、确定性 1024/128/64 划分、图片/QA/服务器 Parquet Gate 已完成 |
| Day 4 | PASS | Base internal `67/128`；Cached Prefix `1024/1024`，保留 54 条固定上限截断 |
| Day 5 | PASS | 三项 Benchmark 协议、数据、overlap、64 次 Smoke 请求与 Day 6 预算 Gate 已完成 |
| Day 6 | PASS（旧协议） | `E-D6-001` 的逐样本预测、评分、汇总、资源指标和成本已归档；仅作为旧协议诊断证据 |
| 论文对齐 Base 重评 | PASS（R3） | 2536/2536 完整；ZoomBench 50.65%、MMStar 75.07%、V* 83.77%，作为后续统一基线 |
| Day 7 | PASS_WITH_CAVEAT | `E-D7-001` 双卡 Smoke 完成；Teacher/Student/EMA 关键证据通过，结束阶段 worker 异常保留为 caveat |
| Day 8 | PASS_WITH_CAVEAT | `E-D8-001` 完成 64 条/8 steps、`global_step_8`、5/5 冷重载和 1024 条成本外推；详见 `artifacts/reports/vopd_64_stability.md` |
| Day 9 | PASS_TO_DAY10 | E-D10-001 预算、readiness、配置冻结、正式 preflight 和中止控制均已通过；详见 `docs/day9_vopd_formal_training_gate_brief.md` |
| 6K full-train amendment | 已决策，待执行 | 现行 train=source=6241，不划分训练/测试集；旧 `E-D10-001` 1024 正式训练不启动 |
| Day 10～16 | 未开始 | 下一步为 300GB 磁盘动态检查、6K 数据自动 Gate、Parquet 和全量长度审计 |
| Day 17～21 | 未开始 | GRPO 数据/Reward、真实 Pilot、正式训练、R3 评测和四组收尾 |

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
| 数据盘 | 300GB；当前基线约已用 86GB，正式训练前建议可用 ≥130GiB，硬下限为重新计算后的 checkpoint Gate 且不得低于 120GiB |
| 完整项目预算 | 训练与评测合计硬上限 2000 元；Day 5 Benchmark Smoke 后冻结外部评测预算 |
| 双卡时长 | 删除 SFT 后重新按 Vision-OPD、Cached、GRPO 实测吞吐和 Benchmark Smoke 估算 |
| Day 10～16 主动工时 | 约 24～36 小时；下载、QA、缓存生成和训练机器时间不等于人工工时 |
| Day 17～21 主动工时 | 约 18～28 小时，以 GRPO Pilot 实测决定正式规模 |

每次启动训练前都计算：

```text
本次预计费用 = 预计双卡小时 × 11.96 元
累计预计费用 = 已发生费用 + 本次预计费用
```

若累计预计费用将超过 2000 元，停止扩规模，优先完成评测、报告和证据归档。

### 3.2 全量训练数据口径

现行主线不再切分训练/测试集，冻结 revision 下 6,241 条有效源记录全部参与训练：

| 集合 | 数量 | 用途 |
|---|---:|---|
| source | 6241 | 冻结 revision 下全部有效元数据 |
| train | 6241 | Vision-OPD、Cached Prefix，以及通过全量 Reward Gate 后的 GRPO |
| active internal eval/test/retention/holdout | 0 | 不从 6,241 条中留出任何记录 |

既有 `eval-128`、`retention-64` 是 Day 1～9 已完成工作的历史产物，其中样本现已并入全量训练。可以保留文件和旧结果用于审计，但不得再用它们评价新模型、选 checkpoint、调参数或声称“未见测试集”结果。

要求：

- 每条数据有稳定 `sample_id`。
- Vision-OPD 使用 `完整红框图 + 裁剪图 + 问题`。
- Day 4 的 Cached Prefix-1024 只作为历史证据；6,241 条 Cached Prefix 必须使用同一个训练前 Base 和同一冻结生成协议重新生成，不能使用 Vision-OPD 训练后的模型。
- GRPO 不允许抽取 1,024 子集冒充正式训练。必须先证明 6,241/6,241 都能按预先冻结规则可靠判分；若做不到，正式 GRPO 停在 Gate，只保留 Pilot 作为机制验证。
- ZoomBench、MMStar、V* Bench 使用各自官方划分；下载后检查与 train-6241 的精确哈希、标准化问题文本和感知哈希重叠，不能静默删除官方测试样本或隐瞒重叠。
- 因为 6,241 不能被 global batch 8 整除，训练数据加载不得沿用会漏样本的 `drop_last=True`。最后一批必须包含最后 1 条真实记录和 7 条带 `sample_weight=0` 的确定性补齐行；补齐权重需贯穿 rollout、loss、metric 和有效样本计数。

### 3.3 建议训练配置

以下是起始配置，Smoke 证据优先于纸面参数。

#### Vision-OPD 与 Cached Prefix

| 参数 | 值 |
|---|---|
| epoch | 1 |
| global batch | 8 |
| 预计 optimizer steps | 780（6241 条源数据，原生 drop-last 后有效 6240 条） |
| rollout n | 1 |
| max prompt length | Day 7 用实际 Processor 统计 P99/max；根据 Day 4 已见 4,212～6,511 Token 输入，起始候选为 8192，禁止继续默认 4096 |
| max response length | 1024（论文核心设置；双卡 Pilot 必须先验证显存、吞吐和预算） |
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
| prompts | 完整 train-6241；Reward 覆盖必须为 6241/6241 |
| rollout n | 4 |
| max response length | 先 128，必要时再到 256 |
| global prompt batch | 约 8，以 Smoke 为准 |
| reward | 规则判分，不额外部署 Reward Model |
| GPU | 2 |

### 3.4 存储与安全规则

数据盘总量固定为 300GB，不能依赖再次扩容。按扩容前约 86GB 已用估算，扩容后约有 214GB 可用；6K 数据准备和正式训练必须分阶段复用空间。

1. 下载、解压、自动 QA 和 Parquet 完成后，将已验证且可重新下载的压缩包转移到外部存储，或经确认后删除；不得在验证前删除唯一原包。
2. 系统盘 `/` 只剩约 5GB 时，不得承载 Hugging Face/Torch/pip/编译缓存或临时解压；统一设置 `HF_HOME`、`TORCH_HOME`、`PIP_CACHE_DIR`、`XDG_CACHE_HOME`、`TMPDIR` 到 `/root/autodl-tmp`。
3. 数据、6241 条 Cached Prefix 和最终 Parquet 准备完成后，正式训练前可用空间建议 ≥130GiB，120GiB 为本计划硬下限；实际 guarded preflight 仍应按 `2 × 最新实测 checkpoint + 5GiB` 重新计算，取两者较大值。
4. 长训练允许半程恢复 checkpoint，但 `max_actor_ckpt_to_keep=1`；最终 checkpoint 写入和验证后自动淘汰半程 checkpoint。
5. Vision-OPD 合并、冷加载、SHA256 和内部评测通过后，只长期保留 merged HF 模型、证据和必要恢复状态；在启动 Cached 前归档或经确认清理大型 FSDP optimizer/actor 分片。

强制规则：

- 原始数据不得在未核对 `df -h / /root/autodl-tmp`、缓存目录和下载/解压峰值时直接开始。
- 训练时只保留当前实验 checkpoint；完成合并、加载测试和 SHA256 后，才删除分片。
- 不长期保存大量 optimizer state 和 step checkpoint；现行 6K 长训练仅允许一个滚动恢复 checkpoint，并由 `max_actor_ckpt_to_keep=1` 控制。
- 每天结束记录 `df -h`、模型/数据/checkpoint 大小和累计费用。
- 不得为了省空间删除尚未通过加载测试的唯一模型。

## 4. 统一目录与证据规范

计划创建以下项目文件；文档中的命令是目标接口，脚本需在对应日期实现后再执行：

```text
configs/
  project_1024.yaml  # 历史 1024 冻结配置
  project_6241.yaml  # 现行 6K 全量训练 scope
  benchmark_eval.yaml  # E-D5 历史协议
  benchmark_eval_paper_basejudge_r3_single_gpu.yaml  # 唯一现行外评协议
  vopd_1024.yaml  # Day 9 历史 E-D10-001，不执行
  vopd_6241.yaml
  cached_prefix_6241.yaml
  grpo_6241.yaml  # 仅在全量 Reward Gate 通过后执行
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
| E-D10-001 | Vision-OPD 1024 正式训练前 Gate（Day 9 已 PASS_TO_DAY10；未执行，已由 6K amendment 取代） |
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
| E-D10-6K-DATA-001 | train-6241 的自动 QA、Parquet、长度与 overlap Gate |
| E-D11-6K-GATE-001 | 6241 尾批覆盖、Cached Prefix、长尾 Pilot、预算、磁盘和 guarded preflight |
| E-D12-6K-VOPD-001 | Vision-OPD 6241 全量正式训练 |
| E-D13-6K-VOPD-FINAL-001 | Vision-OPD 合并、冷加载与 checkpoint 冻结 |
| E-D13-6K-CACHED-PILOT-001 | Cached 契约与新增数据长尾稳定性 Gate |
| E-D14-6K-CACHED-001 | Cached Prefix 6241 全量正式训练 |
| E-D15-6K-FINAL-EVAL-001 | Vision-OPD/Cached R3 外部评测 |
| E-D17-GRPO-DATA-001 | train-6241 全量 Reward 覆盖审计与 GRPO 配置冻结 |
| E-D18-GRPO-PILOT-001 | GRPO 32/64 prompt 真实 Pilot 与稳定性 Gate |
| E-D19-GRPO-TRAIN-001 | GRPO 6241 全量正式训练 |
| E-D20-GRPO-EVAL-001 | GRPO 合并与 R3 外部评测 |

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

## 5. Day 1～9：已完成证据与原始计划正文

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

> 执行状态（2026-08-31 UTC）：**PASS_WITH_CAVEAT / 已收尾**。固定 64 条完成 8/8 steps，`global_step_8` 冷重载 5/5 通过，1024 条规划外推约 1.02 双卡小时、¥12.25，保守上界约 1.75 小时、¥20.84。checkpoint 保存后出现一次 DataLoader worker `Killed`，且训练日志显存值不能作为可信逐卡峰值；两项已写入 `artifacts/reports/vopd_64_stability.md`，并转为 Day 9 观测性 Gate。

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

> 执行状态（2026-09-02 UTC）：**PASS_TO_DAY10 / 已完成**。五项任务及问题处理均已归档，详见 `docs/day9_vopd_formal_training_gate_brief.md`。

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

## 6. Day 10～21 主动执行排期：6K 主实验与 GRPO

> 本节是 2026-09-02 起唯一现行排期。Day 10～16 完成 6K Vision-OPD/Cached 主线；Day 17～21 完成 GRPO。后面的旧 1024/Day 30 排期仅保留历史，不得混用实验 ID、数据路径或验收结论。

### Day 10：6241 全量范围冻结、300GB 存储 Gate 与数据 Gate

目标：把冻结的 6,241 条源元数据全部构建为 train-6241，不划分训练/测试集；复用既有元数据和 Schema 证据，只补齐新下载图片必须具备的自动 Gate。

任务：

1. 在任何下载前记录 `git status --short`、`df -h / /root/autodl-tmp`、缓存目录和当前数据/checkpoint 占用；确认数据盘总量约 300GB、预计可用约 214GB。
2. 将 `HF_HOME`、`TORCH_HOME`、`PIP_CACHE_DIR`、`XDG_CACHE_HOME`、`TMPDIR` 全部指向 `/root/autodl-tmp`，禁止把全量缓存、临时解压或编译文件写入系统盘。
3. 新建 scope amendment 和 `configs/project_6241.yaml`；保留 `configs/project_1024.yaml`、`configs/vopd_1024.yaml`、Day 9 哈希和全部历史证据。
4. 仓库当前缺少 Day 2 所述全量 `candidate_manifest.jsonl`，不得假定其可直接复用。先从同一 source revision 重新取得原始 `train.jsonl`，同时通过 6,241 行、4,566,587 bytes 与冻结 SHA256 三重校验，再用原 stable sample ID 算法重建 candidate/train manifest；历史 1,216 个 sample ID 必须全部可在新 candidate 集合中对账。将 6,241 条全部写入 train，不生成现行 eval/test/retention/holdout。
5. 在新配置/迁移登记中将既有 eval-128、retention-64 标记为 `historical_only=true`；禁止改写旧 JSONL/Parquet 或破坏旧哈希。这些记录同样进入 train-6241，后续不得用旧 split 评价新 checkpoint。
6. 下载并解压全量 Student images 和 Teacher crops。自动检查全部 6,241 条训练记录：路径安全、文件存在、零字节、Pillow 完整解码、尺寸、bbox、full/crop 配对和 sample ID 唯一。
7. 不重新做完整 30 条人工 QA。Day 10 只人工查看图片/路径/配对自动失败项，并从新增 5,025 条中按尺寸与 bbox 面积分层确定性抽取约 10 条正常记录，排查系统性语义错配；Processor 失败项和长度最长的 5～10 条移到 Day 11 长度审计完成后复核。报告按实际阶段写“全量自动图片 QA + 失败项/新增样本分层复核”或“Processor 异常/长尾复核”。
8. 使用项目当前 Schema 在服务器原子生成 `/root/autodl-tmp/data/vision_opd_6241/train_6241.parquet`；不得直接使用缺少 `extra_info.provenance.sample_id` 的官方简化 Parquet。
9. 保存 train-6241 manifest、Parquet SHA256、行数、列、Linux 路径和一次完整 load Gate。
10. 数据验证完成后优先把下载压缩包转移到可恢复归档位置；只有获得明确确认后才可删除。训练前不得让压缩包与 checkpoint 峰值长期共存。

产物：

- `configs/project_6241.yaml`
- `artifacts/data/vision_opd_6241/train_6241.jsonl`
- `artifacts/data/vision_opd_6241/vision_opd_6241_manifest.json`
- `artifacts/data/vision_opd_6241/vision_opd_6241_data_qa.json`
- `artifacts/data/vision_opd_6241/vision_opd_6241_sha256.txt`
- `/root/autodl-tmp/data/vision_opd_6241/train_6241.parquet`
- `artifacts/runs/E-D10-6K-DATA-001/`

验收：

- source=6241、train=6241、现行 eval/test/retention/holdout=0；train sample ID 与 source 完全一致。
- full/crop 缺失、零字节、解码失败、sample ID 重复和路径越界均为 0。
- 6241 行 Parquet 可完整加载，Schema、顺序、sample ID 和 SHA256 已冻结。
- 下载/解压完成后重新记录磁盘；若可用空间低于 120GiB，停止后续训练并先归档压缩包、缓存或已验证的历史大分片。

### Day 11：全量长度、overlap、drop-last、Cached-6241 与正式训练 Gate

目标：关闭 1024 抽样无法覆盖的 6K 长尾风险，显式冻结 6,241 条在 batch 8 下的原生 drop-last 行为，并为两个 6K 分支生成可执行的配置、缓存、预算和守护策略。

任务：

1. 使用训练时相同的 Processor、Chat Template 和图像预处理，对 train-6241 统计 text/image/total prompt Token 的 P50/P95/P99/max、Processor error 和 over-8192 数量。
2. `truncation=error` 保持不变；若存在 over-8192 或 Processor 失败，不得删除或换成留出样本。修复数据/路径/Processor 或事前提高长度上限，直到 6,241/6,241 可处理，并更新全部哈希。
3. 对 train-6241 与 ZoomBench、MMStar、V* Bench 重新执行文件 SHA256、标准化问题文本和 64-bit pHash overlap；旧 train-1024 overlap 只能作历史参考。V* R3 主结果仍固定官方 191 分母；根据新 overlap 报告产生的分层/去重统计仅为同一批预测上的次级诊断，不替换 R3 主结果。
4. 禁用全覆盖补齐，沿用训练器原生 `drop_last=True`；冻结 `source_rows=6241`、`effective_train_samples=6240`、`padding_rows=0`、`dropped_rows=1` 和 780 steps。
5. `shuffle=true`、seed 42 保持不变；预检必须记录丢弃计数，并明确丢弃的是打乱序列末条而非保证为 Parquet 物理末行。全覆盖 sampler 代码和旧测试只保留为未启用能力，不作为当前准入条件。
6. 使用冻结原始 Qwen3.5-4B Base 和论文对齐的 1024-token 生成上限，为 6,241 条训练样本生成 Cached Prefix；保留所有达到 1024-token 上限的记录，不重采样。Day 4 的 256-token Cached Prefix 仅为历史证据。
7. 验证 cached records=6241、unique IDs=6241、missing/extra/duplicate/error/empty=0，保存截断数量、生成配置哈希、Base 身份和 Parquet SHA256；provenance 固定为 `base_tokenizer_reencoded_openai_response_text`。
8. 新建 `configs/vopd_6241.yaml`：global batch 8、780 steps、1 epoch、seed 42、原生 drop-last、online prefix、8192/1024、Top-K 100、alpha 0.5、EMA 0.05、workers=0、从 Base 冷启动；显式冻结 temperature 1.0、top-p 1.0、sampling top-k -1、warmup 10 和 clip 0.2/0.3。
   当前选定并继续执行缩规模双卡方案；同时显式传入作者入口的 vLLM `fuse_allreduce_rms=false` 与 `enable_flashinfer_autotune=false`。Pilot-16 首次实测 GPU 1 峰值达到 95.42% 并被 guard 中止，因此仅开启冻结 Reference 参数 offload；Actor/optimizer offload 仍关闭，0.45 rollout fraction 与 95% 运行期中止线不变；启动前 cgroup 内存下限设为 128 GiB。另保存 `configs/vopd_6241_algorithm_aligned_2gpu.reference.yaml` 作为不可启动参考：batch/PPO mini batch 96、rollout n=8、65 steps、三类 offload 开启，只有独立 Pilot 和明确切换授权后才可采用。
9. 旧 Day 8 的 256-token 外推（均值约 5.68 双卡小时/67.88 元，保守约 10.09 小时/120.64 元）只作为历史下界，不得直接用于 1024-token 正式预算。Pilot 前按 38 小时硬中止上限预留 454.48 元；实测 1024-token Pilot 后重算规划值和保守值。
10. 新建 6K abort policy：运行时间上限、逐卡显存、进程树 RSS、cgroup、磁盘、心跳、Student/Teacher/EMA、generation abort、drop-last 计数和最终 `global_step_780` 校验。保存周期设为 390，只允许 step 390 恢复点和最终 step 780；`max_actor_ckpt_to_keep=1`；正式启动前磁盘取 `max(120GiB, 2 × 最新实测 checkpoint + 5GiB)`，建议可用 ≥130GiB。
11. 从新增记录及 prompt 长度尾部选择 16～64 条真实 Pilot，验证 online rollout、Crop Teacher、JSD、Student backward、Teacher 无直接梯度和 EMA；Pilot 不能冒充正式训练或能力结果。

产物：

- `artifacts/data/train_6241_prompt_lengths.jsonl`
- `artifacts/reports/train_6241_prompt_length_report.md`
- `artifacts/runs/E-D10-6K-DATA-001/overlap/`
- `/root/autodl-tmp/data/vision_opd_6241/cached_prefix_base_6241.parquet`
- `configs/vopd_6241.yaml`
- `configs/vopd_6241_abort_policy.yaml`
- `tests/test_project_6241_config.py`
- `artifacts/runs/E-D11-6K-GATE-001/preflight.md`
- `artifacts/runs/E-D11-6K-GATE-001/pilot/`

验收：

- over-8192=0、Processor error=0、silent truncation=0；任何失败都不得通过丢弃样本解决。
- drop-last 合同为源数据 6241、有效训练 6240、padding 0、dropped 1、总步数 780。
- Cached Prefix 6241/6241 完整，哈希和 Base 身份可核验。
- 新 overlap 报告完成，任何重叠保留并如实解释，不静默删除外部官方样本。
- 新 6K config、budget、storage、guarded launcher、Pilot 和 checkpoint 后置检查全部 PASS，才进入 Day 12。

### Day 12：Vision-OPD 6K 正式训练

目标：完成 `E-D12-6K-VOPD-001`，不再执行旧 `E-D10-001` 1024 正式训练。

任务：

1. 从冻结原始 Base 冷启动，使用 train-6241、online Student prefix、Teacher crop、780 steps、原生 drop-last 和新 guarded launcher。
2. 使用 `scripts/run_day12_vopd.py`，按双卡合计 14 元/小时记录估算，不要求累计费用或账单时间；配置哈希、数据哈希、Git、两张 GPU、cgroup、输出冲突和磁盘任一 Gate 失败均不启动。
3. 观察最初 3 个 optimizer steps，核对 sample ID、在线 response、Teacher crop、有限 loss、Student delta、Teacher gradient/optimizer delta=0、EMA、prompt/response、GPU/RSS/cgroup/磁盘。
4. 正常后允许守护器无人值守；不要求每小时人工查看，但所有自动中止、信号和恢复事件必须落盘。
5. step 390 保存唯一的周期恢复点；最终 step 780 成功后验证 marker、13 个必需文件和非空分片，最终写入成功 receipt。
6. 若触发 NaN、Teacher 直接梯度、Student/EMA 连续不更新、OOM、遥测失败、磁盘、心跳或墙钟上限，先保留证据，再按明确恢复命令决定是否恢复；禁止无记录反复试参。

产物：

- `artifacts/runs/E-D12-6K-VOPD-001/logs/train.log`
- `artifacts/runs/E-D12-6K-VOPD-001/evidence/telemetry/`
- `runtime_metrics.jsonl`、`guard_events.jsonl`、`guard_summary.json`、`exit_receipt.json`
- `checkpoints/global_step_780/`

验收：

- 780/780 steps 完成，`source_rows=6241`、`effective_train_samples=6240`、`padding_rows=0`、`dropped_rows=1`，退出码和预检合同表示成功。
- Student/Teacher/EMA 合同通过，无未解释 NaN/OOM/数据错误。
- checkpoint 完整可审计；仅有启动脚本或中间 step 不得写成“完成 6K 训练”。

### Day 13：Vision-OPD 定版、Cached 契约实现与长尾稳定性

目标：关闭 Vision-OPD 模型交付，并让 Cached-6241 具备正式训练资格。

任务：

1. 对 `global_step_780` 做文件清单和 SHA256；显式传入实际路径完成 FSDP merge，不能使用 `merge_checkpoint.sh` 的旧默认路径。
2. 在新进程加载 merged Student，固定 5 条推理必须 5/5 非空、0 inference error。
3. 冻结 Vision-OPD checkpoint、配置、drop-last 合同和 SHA256；不运行历史 eval-128/retention-64，本日仍不打开外部结果。
4. 实现 `prefix_source: online|cached`；默认 online 行为不变，cached 按 sample ID 加载 6,241 条缓存并完全绕过 online rollout generation，Teacher 仍看 crop，Student 仍看 full image。
5. 比较 Base 身份、Processor、Chat Template、图像输入、sampling、EOS/stop、prompt IDs、decode→encode、response mask、padding、数据顺序、batch、steps、loss 和 EMA。
6. 全部非 prefix-source 契约一致时记 `STRICT_PREFIX_SOURCE_ABLATION=PASS`；否则记 `IMPLEMENTATION_ABLATION` 并列出额外差异，禁止强称单变量。
7. 使用新增样本和 prompt 长尾完成 64～128 条 Cached 稳定性训练，保存、冷加载 5 条，并按实测重新估算 6K 正式耗时和费用。
8. Vision-OPD merged 模型、加载、哈希和 drop-last 合同通过后，归档或经确认清理不再需要的大型 FSDP optimizer/actor 分片，为 Cached 正式训练恢复 ≥130GiB 建议余量；不得删除唯一可加载模型。

产物：

- `artifacts/runs/E-D13-6K-VOPD-FINAL-001/`
- `artifacts/reports/vopd_6241_audit.md`
- `configs/cached_prefix_6241.yaml`
- `tests/test_cached_prefix_contract.py`
- `artifacts/runs/E-D13-6K-CACHED-PILOT-001/`
- `artifacts/reports/cached_6241_pilot.md`

验收：

- Vision-OPD checkpoint 可独立加载，哈希与 6241→6240 drop-last 合同完整；不存在基于内部留出集的重选。
- Cached 默认/分支、sample ID、无在线 generation、Student/Teacher/EMA 和 checkpoint Gate 通过。
- 消融名称已按证据冻结；Cached 正式配置与 Vision-OPD 的差异清单完整。

### Day 14：Cached Prefix 6241 全量正式训练

目标：完成 `E-D14-6K-CACHED-001`。

任务：

1. 从同一原始 Base 冷启动，不继承 Vision-OPD checkpoint。
2. 使用同一 train-6241、原生 drop-last、seed、batch、780 steps、LR、长度、Top-K、JSD、EMA 和保存/守护策略；只允许冻结差异表中的 cached prefix 项。
3. 前 3 步确认 cache SHA256、sample ID、没有 online rollout、有限 loss、Student 更新、Teacher 无直接梯度和 EMA 更新。
4. 运行期保存遥测、中止、恢复、费用、drop-last 计数和滚动 checkpoint；正常结束后验证 `global_step_780`。

产物：

- `artifacts/runs/E-D14-6K-CACHED-001/`
- 最终 Cached FSDP checkpoint、日志、遥测、费用和 SHA256

验收：

- 780/780 steps 完成，有效训练 6240/6241、padding 0、dropped 1，最终 checkpoint 完整。
- 配置 diff 没有未申报变量；若存在则更新为实现消融并保留差异，不能丢弃结果。

### Day 15：Cached 定版与三组统一 R3 外部评测

目标：冻结 Cached 模型，并在所有设计、checkpoint 身份和输出 Schema 锁定后统一打开外部结果。

任务：

1. 合并 Cached 最终 Student，保存清单/SHA256，新进程冷加载 5 条。
2. 不运行历史 internal eval-128/retention-64；在看到外部分数前完成并冻结 `eval/compare_experiments.py`、唯一最终 checkpoint、输出表 Schema 和 Bad Case 抽样规则。
3. 使用唯一现行 `configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml`，对最终 Vision-OPD 和 Cached 各先跑 4×3 Smoke，再跑 ZoomBench full、MMStar、V* Bench 完整 2536 条。
4. 两个模型都使用冻结原始 Qwen3.5-4B Base Judge；分别保存 predictions、Judge、scores、summary、validation、模型哈希和成本，只与冻结 Base R3 比较。
5. 外部分数不得触发重选 checkpoint、修改 Cached 契约或重新训练。

产物：

- `artifacts/runs/E-D15-6K-FINAL-EVAL-001/vision_opd/`
- `artifacts/runs/E-D15-6K-FINAL-EVAL-001/cached_prefix/`
- `artifacts/reports/prefix_ablation_6241.md`

验收：

- 两个模型均为 2536/2536，Judge/评分完整，`validation.json` 为 PASS。
- Base/Vision-OPD/Cached 使用同一 R3 数据、输入、生成、解析、Judge 和 denominator。
- V* 主结果固定官方 191 分母；overlap 分层/去重诊断使用同一批预测并作为次级结果单列，train-6241 overlap、invalid/truncated/API/Judge failure 均单独报告。

### Day 16：6K 主项目 Bad Case、报告与投递验收

目标：完成不含 GRPO 的可投递主版本，并给 Day 17～21 留出独立扩展边界。

任务：

1. 实现/运行 `eval/build_badcases.py`，人工分析 12～20 条代表样本：Vision-OPD 修正/退化、Cached 优/劣、Benchmark 冲突、overlap 和评测规则不确定。
2. 完成 Base/Vision-OPD/Cached 总体、类别、corrected/regressed、invalid、长度、训练时间、GPU 小时、费用和实现工作量主表。
3. 更新 `docs/final_report.md`、README、`docs/interview_qa.md` 和证据索引；每个数字能回溯到 prediction、训练日志或机器报告。
4. 明确项目是“Vision-OPD-6K 冻结数据 6241 条、论文核心算法参数对齐的双卡受控训练”，response 1024 与论文一致，但不是官方 8 卡、batch 96、rollout n=8 的完整资源配置复现。
5. 运行测试、Markdown/Git 检查，归档模型、配置、命令、环境、日志、哈希和成本；提交并打主项目里程碑 tag。

主项目简历模板（必须按真实结果填数）：

> 基于 Qwen3.5-4B 与双卡 RTX PRO 6000，读取 Vision-OPD-6K 冻结数据 6,241 条，按原生 drop-last 有效训练 6,240 条，完成在线 Student Prefix/Crop EMA Teacher 的 Vision-OPD 训练及 Base Cached Prefix 对照；以 ZoomBench、MMStar、V* Bench R3 作为独立外部评测，保存逐样本结果、Bad Case、资源和成本证据。

验收：

- Base、Vision-OPD、Cached 均有可核验身份与统一 R3 外部结果。
- 不写“完整复现论文结果”“严格单变量”或“显著提升”，除非相应证据实际成立。
- 主项目即使 GRPO 后续失败也可独立交付。

### Day 17：GRPO 可验证数据、Reward 与配置冻结

目标：为 train-6241 全部记录冻结可审计 Reward；GRPO 不使用 Teacher crop，也不继承 Vision-OPD/Cached。任何记录不可可靠判分时，正式 GRPO 记为 BLOCKED，不静默缩小训练集。

任务：

1. 实现 `scripts/prepare_grpo_data.py`，保留 train-6241 全部 full-image prompt，不把 bbox crop 传给 GRPO；不得在脚本中按题型过滤、抽样或设置 1,024 上限。
2. 为多选、数字和短答案实现确定性 normalization/Reward；若存在无法可靠规则评分的答案类型，先补充明确且可单测的判分规则，不部署额外 Reward Model。
3. 对 6,241 条逐条输出 `reward_route`、规范化 gold、可判分状态和原因；冻结前必须 `scorable=6241`、`unscorable=0`、`ambiguous=0`，否则正式 GRPO Gate 为 BLOCKED。
4. Reward 测试覆盖正确、错误、格式变体、歧义、空输出、异常答案和每一种 `reward_route`；人工复核全部异常与每类至少 10 条。
5. 冻结 `configs/grpo_6241.yaml`、`scripts/run_grpo_2gpu.sh`、原生 drop-last、guarded launcher、日志/rollout/metrics/checkpoint 路径。
6. 初始候选：rollout n=4、response 128、global prompt batch 8、1 epoch、两张 GPU、780 个 prompt batches；6241 条源数据打乱后丢弃末尾 1 条。正式墙钟与预算由 Pilot 重算。

产物：

- 6241 条 GRPO JSONL/Parquet、逐条 Reward 路由和数据报告
- `tests/test_reward_rules.py`
- `tests/test_grpo_parquet.py`
- `configs/grpo_6241.yaml`
- `artifacts/runs/E-D17-GRPO-DATA-001/`

验收：

- train=6241、scorable=6241、unscorable=0、ambiguous=0，没有题型过滤或抽样。
- Reward 路由、正负例和异常输入测试通过；配置明确从原始 Base 独立启动。

### Day 18：GRPO 32/64 Prompt 真实 Pilot 与稳定性 Gate

目标：证明真实 group rollout、relative advantage 和 policy-gradient 更新，并关闭 reward hacking 与预算风险。

任务：

1. 先跑 32 prompt、rollout n=4，至少完成 1～3 个 optimizer steps。
2. 检查同一 prompt 的 group responses、Reward 分布、relative advantage、actor loss、KL、entropy、response length 和 actor 参数更新。
3. 人工检查至少 20 条 rollout，重点查看高 Reward 是否真实正确、是否只靠格式/长度钻规则漏洞。
4. 32 prompt 通过后扩到 64 prompt 稳定性，保存并冷加载 checkpoint。
5. 根据实测吞吐、显存和 Reward 有效率冻结 6241 条源数据、6240 条有效训练的 780 steps、墙钟、费用、中止条件和 checkpoint 策略；外部 Benchmark 不用于 Pilot 决策，Pilot 不得改变 drop-last 口径。

产物：

- `artifacts/runs/E-D18-GRPO-PILOT-001/32/`
- `artifacts/runs/E-D18-GRPO-PILOT-001/64/`
- `artifacts/reports/grpo_64_stability.md`

验收：

- 真实 policy-gradient 更新成立，Reward 非全 0/全 1且组内有有效差异。
- 无 NaN、Reward 崩溃、明显格式投机或不可加载 checkpoint。
- 只有 Pilot、预算、磁盘和恢复 Gate 全部 PASS 才进入 Day 19。

### Day 19：GRPO 正式训练

目标：从原始 Base 启动 train-6241 全量 GRPO 正式训练。

任务：

1. 仅在 Day 17 的 `scorable=6241/6241` 与 Day 18 Pilot 全部 PASS 后，使用 train-6241、780 prompt batches、rollout n=4、原生 drop-last、冻结 Reward 和 guarded launcher；不得从 Vision-OPD 或 Cached checkpoint 初始化。
2. 前 3 步核对 prompt grouping、group response、Reward、relative advantage、actor update、KL、entropy、长度、GPU/RSS/cgroup/磁盘。
3. 运行期自动监控 Reward collapse、全同优势、KL/长度异常、OOM、心跳、磁盘和 checkpoint；抽查高 Reward 输出。
4. 正常完成后验证最终 checkpoint、日志、rollouts、Reward 记录和费用；训练合同必须为 source prompts 6241、effective prompts 6240、padding 0、dropped 1。

产物：

- `artifacts/runs/E-D19-GRPO-TRAIN-001/`
- GRPO 最终 FSDP checkpoint、训练日志、rollouts、metrics、遥测和成本

验收：

- 780/780 prompt batches 完成，有效训练 6240/6241，checkpoint 完整且可恢复/合并。
- Reward 上升不伴随明显长度爆炸、格式投机或未经记录的 Reward 规则变化。

### Day 20：GRPO 合并、定版与 R3 外部评测

任务：

1. 合并最终 GRPO Student，保存 SHA256，新进程冷加载 5 条。
2. 冻结 checkpoint、配置、Reward 报告、覆盖 receipt 和 SHA256；不运行历史 internal eval-128/retention-64。
3. 使用 R3 对 GRPO 先跑 4×3 Smoke，再跑 ZoomBench full、MMStar、V* Bench 2536 条；使用同一固定原始 Base Judge。
4. 审查训练中高 Reward 但外部评测错误的代表样本，保留 reward hacking 或训练分布偏置证据。

产物：

- `artifacts/runs/E-D20-GRPO-EVAL-001/`
- `artifacts/reports/grpo_eval.md`

验收：

- checkpoint 可加载，外部 2536/2536 完整，`validation.json` 为 PASS。
- GRPO/Vision-OPD/Cached 都以相同 6241 条源记录训练；仍不得把外部差异全部解释为 objective，因为监督、轨迹数、response length 和优化过程不同。

### Day 21：四组方法统一比较、项目收尾与最终 tag

任务：

1. 比较 Base、Vision-OPD-6241、Cached-6241、GRPO-6241；表中强制列出训练样本数、有效覆盖、补齐行、轨迹来源、监督、loss、更新参数、steps、GPU 小时和费用。
2. 更新 paired/Bad Case，分别分析密集 Token 分布监督、固定前缀分布监督和稀疏结果 Reward；不把 JSD、Reward 与 accuracy 混为同一指标。
3. 更新 README、`docs/final_report.md`、`docs/interview_qa.md`、简历 bullet 和最终证据索引。
4. 运行测试、Git/Markdown 检查，归档模型哈希、配置、命令、日志、逐样本预测和费用，提交并打最终 tag。

最终简历模板（必须按真实结果改写）：

> 围绕 Qwen3.5-4B 搭建多模态后训练实验矩阵，在相同 6,241 条冻结源数据上按统一 drop-last 口径完成 Vision-OPD 在线自蒸馏、Cached Prefix 对照及可验证 Reward GRPO；实现固定 Base Judge、逐样本外部评测、Bad Case 和成本审计，并分析不同监督密度与轨迹机制的效果及工程权衡。

验收：

- GRPO 有真实 Reward、relative advantage、policy-gradient、checkpoint 和统一评测证据。
- 四组表明确三条训练分支均读取 6241 条源数据、有效训练 6240 条、丢弃 1 条，并列出其他不可控差异；所有结论、失败和简历数字可追溯。
- 总费用不超过 2000 元；若 GRPO 正式训练未通过 Gate，则保留真实 32/64 Pilot，并明确写为机制与工程验证。

## 6A. 已废止的 Day 10～30 历史排期（仅审计，禁止执行）

> 以下内容保留用于解释原 1024 方案如何演进到 6K 方案。不得使用其中的旧日期、1024 正式训练 ID、磁盘假设或 Day 30 截止时间启动新实验。

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

## 6B. 已废止的 Day 21～30 GRPO 历史排期（仅审计，禁止执行）

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
2. 6K 数据自动 QA、Parquet、长度、6241→6240 drop-last 合同或正式训练前磁盘 Gate 未通过时停止；不得跳过 Gate，也不得把其他未申报子集冒充当前训练口径。
3. 300GB 空间不足时先转移已验证压缩包、缓存和可归档历史大分片；任何唯一 checkpoint 必须在 merge、冷加载、SHA256 和必要备份通过后才能清理。
4. 外部 Benchmark 费用超出 Day 5 预算时，先保留 ZoomBench、MMStar，再延后 V* Bench；不得只保留分数更好看的集合。
5. 外部主评测需要 Judge 时，必须使用 `E-PAPER-BASEJUDGE-001` 冻结的原始 Qwen3.5-4B Base、官方 Judge Prompt 和统一参数；Judge 失败按错并保留逐样本错误，不得临时换 Judge 或让训练后模型自评。
6. GRPO Reward 无法可靠覆盖 6241/6241 时，正式 GRPO 记为 BLOCKED；不得缩小数据量后仍称“6.2K 完整 GRPO”。可以保留 32/64 prompt Pilot，但只能写“机制与工程验证”。
7. GRPO 正式训练超预算时，保留 32/64 prompt 的真实 Pilot，明确写成“机制与工程验证”，不冒充完整训练。
8. 任何方法若只有脚本启动、没有 checkpoint 与评测，只能写“跑通 Smoke”，不能写“完成训练”。

## 8. 最终验收清单

### Day 16：6K 主项目投递版

- [ ] 冻结 source=train=6241、现行 eval/test/retention/holdout=0；旧 128/64 只保留为历史证据。
- [ ] 全量图片自动存在性/解码/full-crop 配对 QA、6241 Parquet、Processor 长度和 train-6241 overlap Gate 通过。
- [x] Vanilla 128 条预测和评测结果完整。
- [x] 旧协议 `E-D6-001` 的三项外部 Benchmark 逐样本预测、评分、汇总、overlap 说明和成本已归档。
- [x] 论文对齐 `E-PAPER-BASEJUDGE-001` amendment 已冻结，明确固定 Base Judge 替代 GPT-OSS-120B。
- [x] `E-PAPER-BASEJUDGE-001/base` 在 ZoomBench full、MMStar、V* 191 上的逐样本预测、Judge 记录与汇总完整；`validation.json` 为 PASS。
- [ ] Vision-OPD 真实链路包含在线生成、Crop Teacher、Top-K JSD、Student backward、EMA。
- [ ] Vision-OPD 读取 6241、有效训练 6240、padding 0、dropped 1，checkpoint 可加载并完成三项外部评测。
- [ ] Cached Prefix 契约完成判定；若非 prefix 项全部一致则标记严格消融，否则明确降级为实现消融并列出额外差异。
- [ ] Cached Prefix 6241/6241 完整；Cached 读取 6241、有效训练 6240、padding 0、dropped 1，checkpoint 可加载并完成三项外部评测。
- [ ] 有逐样本 paired comparison 和 12～20 条 Bad Case。
- [ ] 每个实验有 config、命令、commit、日志、费用和哈希。
- [ ] README、最终报告、面试问答和简历 bullet 已完成。

### Day 21：含 GRPO 的完整版本

- [ ] GRPO 使用完整 6241 条，逐条 Reward 路由审计为 scorable=6241、unscorable=0、ambiguous=0。
- [ ] Reward 单元测试覆盖正例、负例、格式变体和歧义例。
- [ ] GRPO Pilot 证明 group rollout、relative advantage 和 actor update。
- [ ] GRPO 读取 6241、有效训练 6240、padding 0、dropped 1，checkpoint 可加载并完成三项外部评测。
- [ ] Base、Vision-OPD、Cached、GRPO 四组方法均使用 `E-PAPER-BASEJUDGE-001`，且固定原始 Base Judge，不让训练后模型自评。
- [ ] 四组表显式列出 Vision-OPD/Cached/GRPO 的 6241→6240 drop-last 口径，以及轨迹数、response length、监督和优化过程差异。
- [ ] 总费用不超过 2000 元，或对超支原因有事前批准和完整记录。
- [ ] 项目边界、失败结果和未复现内容均如实写明。
