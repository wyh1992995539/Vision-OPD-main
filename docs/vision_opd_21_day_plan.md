# Vision-OPD 21 天核心机制实现与小规模验证计划（每天 5 小时）

## 1. 计划前提

- 周期：21 天。
- 每天主动投入：约 5 小时，共 105 小时。
- 模型训练、批量推理等机器等待时间不计入 5 小时，但启动、监控、排错、分析和记录计入。
- 计算资源：AutoDL 单节点 2 张 RTX PRO 6000 96GB，单卡分配 22 核 Xeon Platinum 8470Q 与 110GB 内存；整机合计 44 核 CPU、220GB 内存和 192GB 显存。
- 存储资源：系统盘 30GB；数据盘默认 50GB，可再扩容 206GB。项目目标容量为 150～200GB，所有 Conda 环境、模型、数据、缓存、rollout 和 checkpoint 均放入数据盘。
- 软件上限：GPU 驱动 580.95.05，支持 CUDA 13.0 及以下；项目按仓库依赖使用 CUDA 12.8 路线，并在安装后以 PyTorch 实测结果为准。
- 主目标：在 Qwen3.5-4B 上完成单卡基线、单节点双卡 FSDP 训练闭环、64～256 条小规模 on-policy 自蒸馏、一个受控消融、固定小型评测、Bad Case 分析和可复现实验报告。
- 非主目标：完整下载和训练全部 Vision-OPD-6K、复现论文绝对指标、9B 训练、全 benchmark 评测、多个大规模 checkpoint 或同时实现多项新算法。

## 2. 21 天结束时的最低交付线

必须留下以下可验证成果：

1. 固定的代码提交、依赖版本、数据 revision 和硬件记录。
2. Qwen3.5-4B 在固定 20～50 条验证样本上的训练前基线；官方 Vision-OPD-4B 对照为推荐项，不作为训练闭环前置条件。
3. 64～256 条微型训练集及其 Student 全图、Teacher 裁剪图、Parquet、路径检查和至少 30 条人工抽查结果。
4. 单卡与双卡环境下 forward、on-policy rollout、Teacher forward、Top-K JSD、backward、optimizer step 和 EMA 更新证据。
5. 使用 64 条数据完成稳定性训练，并在资源允许时完成 128～256 条、1 Epoch 双卡训练；失败时留下可定位证据和降级记录。
6. 至少一个可合并、可部署、可评测的 model-only 自训练结果；完整 optimizer checkpoint 最多保留一个，仅用于恢复验证。
7. 至少一个控制变量明确的消融，优先完成 on-policy prefix vs fixed/cached off-policy prefix。
8. 10～20 个结构化 Bad Case。
9. 一份说明方法、双卡适配、配置、结果、失败和边界的小规模验证报告。

完成核心链路、固定评测和受控消融后，可以称为“Vision-OPD 核心机制实现与小规模验证”。只有机制链路通过而没有可信对照时，应称为“Vision-OPD 核心机制实现与验证”。任何情况下均不称为完整论文复现或论文指标复现。

## 3. 相对原计划的调整

考虑这是第一次复现论文和第一次训练模型，将前 5 天完整用于“边学边验证”，并从 Day 6 开始只保留复现所需任务：

- Day 1～2 保留最小梯度链路、环境安装和普通 VLM 推理，作为后续学习的实物基础。
- Day 3～5 依次学习 VLM 最小知识、OPD/自蒸馏机制、Vision-OPD 论文与核心代码链路。每一天都必须留下可运行或可检查的证据，不做纯理论阅读。
- 删除大范围代码通读。只追踪数据准备、训练入口、Student rollout、Teacher forward、Top-K JSD、Student backward、EMA 和 checkpoint 合并这条主链路。
- 原 Day 4～7 的评测、两模型基线和数据 QA 压缩到 Day 6～8；基础模型与官方模型在同一天、同一协议下完成小型对照。
- 原训练任务整体顺延 1 天：Day 9 先跑单卡，Day 10 再跑双卡，Day 11 完成 64 条稳定性与恢复验证。
- Day 12～14 完成 128～256 条主实验、审计、合并、快评和必要纠错，不保存多个完整 optimizer checkpoint。
- 只要求完成一个可信消融；主消融为 on-policy vs fixed/cached off-policy prefix。若该对照未完成，成果降级为核心机制验证。
- LoRA-SFT、DPO、GRPO、EMA Teacher vs 固定 Teacher、JSD vs forward KL 和 9B 训练均移出 21 天主线。
- Day 18 后不再启动高风险长训练，保证报告、Bad Case 和归档真正完成。

## 4. 105 小时时间分配

| 阶段 | 天数 | 主动时间 | 主要目标 |
|---|---:|---:|---|
| 入门、环境与代码最短路径 | Day 1～5 | 25h | 梯度、VLM、OPD、玩具蒸馏和核心源码映射 |
| 评测与数据 | Day 6～8 | 15h | 可信评测尺子、两模型小型基线和数据 QA |
| 训练链路 | Day 9～11 | 15h | 单卡基线、双卡 FSDP 冒烟、64 条稳定性 |
| 小规模主实验 | Day 12～14 | 15h | 128～256 条训练、审计、恢复、合并和快评 |
| 评测与受控消融 | Day 15～18 | 20h | 固定小型评测、一个受控消融和能力保持检查 |
| 分析与交付 | Day 19～21 | 15h | Bad Case、报告、归档和面试准备 |

### 前 5 天只学什么

必须学：

1. 图像如何经 Processor 和视觉编码器变成语言模型可用的视觉 Token。
2. 自回归生成、logits、response Token、label/response mask 和交叉熵的基本含义。
3. Student、Teacher、stop-gradient、KL/JSD、EMA 和 on-policy prefix。
4. Vision-OPD 的全图 Student、裁剪图 Teacher、同一回答前缀和训练后仅保留 Student。
5. 一条样本如何从 JSON/Parquet 进入 rollout、Teacher/Student forward、loss、backward、EMA、checkpoint。

暂时不学：

- CLIP、BLIP-2、Flamingo、LLaVA 等模型的完整训练历史；
- Qwen3.5、ViT 或语言模型的预训练细节；
- PPO、GRPO、DPO、RLHF/RLVR 的完整理论和实现；
- `verl`、Ray、vLLM、FSDP 的全部内部源码；
- 与本次主链路无关的 SFT、奖励模型、多节点调度和全 benchmark 代码。

## 5. 每天固定工作方式

建议每天 5 小时采用以下结构：

- 0.5h：回顾日志、确认当日唯一主目标。
- 1.0h：学习当日必需理论或阅读关键代码。
- 2.5h：实操、评测、排错或分析。
- 1.0h：整理实验记录、提交配置和写当日结论。

长时间训练开始后，不要一直等待。机器运行期间可以并行完成评测集准备、日志解析、图表脚本、Bad Case 模板和报告初稿。

每个实验至少记录：

```text
实验 ID：
日期：
Git commit：
数据 revision：
模型与 checkpoint：
配置文件：
GPU：
唯一改动：
预期：
结果：
关键指标：
异常：
下一步：
```

---

# 第一阶段：五天聚焦学习、评测与数据

## Day 1：最小梯度链路、项目冻结与双卡服务器检查（5h）

### 时间安排

- 1.0h：运行并解释 `forward → backward → optimizer.step` 最小 Student/Teacher 示例。
- 1.5h：检查 2 张 GPU、CPU、内存、磁盘、网络、Git 和 Conda。
- 1.0h：把 Conda、Hugging Face、Torch、pip 和临时目录规划到数据盘。
- 0.5h：冻结代码 commit 和目录结构。
- 1.0h：建立环境记录与实验登记表。

### 实操

使用小 Tensor 构造 Student/Teacher：

```text
Student logits
Teacher logits（无梯度）
→ softmax
→ 简单分布差异 loss
→ backward
→ optimizer step
```

### AutoDL 数据盘布局

```text
/root/autodl-tmp/vision-opd/       # 项目代码
/root/autodl-tmp/envs/vision-opd/  # Conda 环境
/root/autodl-tmp/hf_cache/         # 模型缓存
/root/autodl-tmp/torch_cache/      # Torch 扩展缓存
/root/autodl-tmp/pip_cache/        # pip 缓存
/root/autodl-tmp/tmp/              # 编译与临时文件
/root/autodl-tmp/checkpoints/      # 唯一恢复 ckpt 与最终模型结果
```

建议设置 `HF_HOME`、`TORCH_HOME`、`PIP_CACHE_DIR` 和 `TMPDIR` 指向上述数据盘目录。扩容后用 `df -h` 确认实际总容量和可用容量，不能只依据控制台标称值。

### 交付物

- `docs/day1_project_freeze.md`
- 环境与硬件记录。
- 实验登记表。
- Student 有梯度、Teacher 无梯度的最小示例。

### 验收

- `torch.cuda.device_count()` 返回 2，且两张卡均为 RTX PRO 6000 96GB。
- CPU 总计约 44 核、内存总计约 220GB；数据盘目标总容量 150～200GB。
- 驱动为 580.95.05，PyTorch 实际 CUDA runtime 与算子导入测试通过。
- 能解释 logits 与最终 Token 的区别。
- 能确认 Student 参数更新、Teacher 参数未被反向传播更新。
- 明确项目代码版本和磁盘位置。

## Day 2：官方环境与普通 VLM 推理（5h）

### 时间安排

- 2.0h：安装并核对官方依赖。
- 2.0h：用 Qwen3.5-4B 完成一次图片问答。
- 0.5h：检查 processor、chat template、输入字段和 shape。
- 0.5h：冻结依赖清单并记录异常。

### 必须理解的数据流

```text
图片 + 问题
→ Processor
→ input_ids + 多模态图像输入
→ 模型生成 logits
→ response tokens
→ 解码文本
```

### 交付物

- 完整依赖版本清单。
- 一次可重复的 VLM 推理记录。
- processor 输出字段与 shape 说明。

### 验收

- PyTorch、Transformers、vLLM、Ray、Flash Attention 均可导入。
- 普通 VLM 能生成非空回答。
- 能描述图像如何进入语言模型生成链路。

## Day 3：VLM 最小知识与 Qwen3.5 多模态数据流（5h）

### 时间安排

- 1.0h：理解 VLM 的最小结构：视觉编码器、投影/适配、语言模型与自回归生成。
- 1.0h：理解 logits、Token、attention mask、response mask、label 和交叉熵。
- 1.5h：复用 Day 2 的真实图片问答，打印 Processor 字段、shape、生成 Token 和解码结果。
- 1.0h：沿一次 `generate` 调用画出数据流，并标明哪些张量来自图像、问题和回答。
- 0.5h：整理 `docs/day3_vlm_minimum.md`，用自己的话回答验收问题。

### 只需掌握的数据流

```text
完整图片 + 问题
→ Processor / chat template
→ 文本 Token + 视觉输入
→ 视觉特征与文本上下文
→ 自回归生成 response Token
→ 解码为答案
```

### 验收

- 能解释“视觉编码器产生视觉特征”和“语言模型逐 Token 生成回答”的分工。
- 能区分 input Token、response Token、logits、最终文本和 loss。
- 能指出 Day 2 真实推理中图像字段、文本字段和输出字段的 shape。
- 不要求理解 Qwen3.5 或 ViT 的完整预训练过程。

## Day 4：OPD、自蒸馏与玩具 JSD/EMA 验证（5h）

### 时间安排

- 1.0h：理解知识蒸馏、Student、Teacher、soft target 和 stop-gradient。
- 1.0h：理解 on-policy prefix、同前缀比较、Token 级 KL/JSD 和 EMA。
- 2.0h：用小 Tensor 实现 masked JSD、Student backward/step 和 EMA Teacher 更新。
- 0.5h：检查 Student 有梯度、Teacher 无反向梯度，且 loss 有限非负。
- 0.5h：保存脚本、日志与一页机制说明。

### 核心链路

```text
当前 Student 生成回答前缀
→ Student 与 Teacher 评价同一前缀的下一个 Token 分布
→ 在有效 response Token 上计算 JSD
→ 只对 Student backward 和 optimizer.step
→ 用 Student 参数 EMA 更新 Teacher
```

### 验收

- 玩具脚本输出有限且非负的 masked JSD。
- Student 参数在一步更新后发生变化，Teacher 不接收反向梯度。
- EMA 后 Teacher 参数向 Student 移动，但不等同于 optimizer 更新。
- 能解释 on-policy 只表示前缀来自当前 Student，不代表一定使用 PPO、GRPO 或奖励。

## Day 5：Vision-OPD 论文机制与核心代码最短路径（5h）

### 时间安排

- 1.0h：重读论文的方法图、训练目标和实验设置，只记录与复现直接相关的变量。
- 1.0h：阅读数据准备与训练入口，确认 `images`、`bbox_images` 和关键启动参数。
- 1.5h：追踪一条样本的 rollout、Teacher/Student forward、Top-K JSD、backward 与 EMA 调用链。
- 0.5h：阅读 checkpoint 合并和评测入口。
- 1.0h：完成“论文概念—配置—代码位置—日志证据”四列表。

### 必读文件，按顺序停止扩散

1. [`scripts/prepare_data.py`](../scripts/prepare_data.py)
2. [`scripts/run_vision_opd.sh`](../scripts/run_vision_opd.sh)
3. `verl/trainer/ppo/ray_trainer.py` 中构造蒸馏 batch、Teacher 图像替换和训练循环的相关方法。
4. `verl/workers/actor/dp_actor.py` 中 Student/Teacher forward、loss、backward 和 EMA 的相关方法。
5. `verl/trainer/ppo/core_algos.py` 中 `compute_self_distillation_loss`。
6. [`scripts/merge_checkpoint.sh`](../scripts/merge_checkpoint.sh) 与实际使用的评测入口。

### 明确跳过

- 不按目录通读整个 `verl`。
- 不深入未触发的 PPO/GRPO、reward model、多节点调度和通用 SFT/DPO 分支。
- Ray、vLLM 和 FSDP 只记录与启动参数、显存和报错直接相关的部分。

### 第一个 Gate：学习与代码链路

- 能完整解释：全图 Student 生成 → 裁剪图 Teacher 对同一前缀打分 → Top-K JSD → Student 更新 → EMA Teacher。
- 能解释 `images` 与 `bbox_images` 的作用以及推理时为什么只保留 Student。
- 能在代码中指出上述每一步的位置，并列出之后必须观察的日志指标。
- 已留下 Day 3 的真实 VLM 数据流证据和 Day 4 的 JSD/梯度/EMA 运行证据。

## Day 6：冻结统一评测协议与解析器测试（5h）

### 时间安排

- 1.0h：只阅读实际使用的评测入口，梳理 `prepare → infer → judge → accuracy`。
- 1.5h：用至少 20 条人工构造答案测试多选、短答案、空回答和异常格式解析。
- 1.0h：固定 chat template、thinking、temperature、随机种子、Judge 和图片预处理。
- 1.0h：冻结 20～50 条主验证样本，并保存样本 ID 与答案。
- 0.5h：编写统一评测协议和逐样本输出 Schema。

### 必查问题

- `Answer: D` 是否可能被错误解析成 `A`。
- 无明确答案时是否误提取正文孤立字母。
- 随机种子是否真正进入推理过程。
- 哪些样本规则判定，哪些样本需要 Judge。
- 是否保留原始回答、解析结果、Judge 原因和人工复核状态。

### 验收

- 至少 20 条答案格式单元测试通过。
- 每个 accuracy 都能追溯到逐样本输出。
- 基础模型、官方模型和自训练模型的统一评测协议已冻结。

## Day 7：基础模型与官方 Vision-OPD-4B 小型对照（5h）

### 时间安排

- 0.5h：验证 Qwen3.5-4B 基础模型和官方 Vision-OPD-4B 均可加载。
- 0.5h：各跑少量 pilot，排除空回答、图片未加载和格式错误。
- 2.0h：用完全相同的配置评测冻结的 20～50 条样本。
- 1.0h：抽查原始回答、解析器和 Judge。
- 1.0h：建立基础模型、官方模型和论文报告值的分栏对照表。

### 固定条件

- `ENABLE_THINKING=False`。
- 固定 chat template、图像处理、temperature、seed、Judge 和答案解析器。
- 保存原始回答，不只保存最终分数。

### Go/No-Go

如果官方模型结果明显不合理，不据此判断方法无效；优先排查 thinking、chat template、模型/tokenizer、图像预处理、Judge 和解析器。官方模型因磁盘或下载时间无法完成时，记录原因，但不阻塞微型训练链路。

### 验收

- 两个模型使用同一组样本和同一评测协议。
- 得到可追溯的训练前基线，并整理至少 10 个候选错误样本。
- 论文值、官方 checkpoint 本地值和基础模型本地值明确分开。

## Day 8：微型 Vision-OPD 数据集构建与 QA（5h）

### 时间安排

- 1.0h：构建 64 条起步、目标 128～256 条的微型数据集，不默认下载完整 37.5GB 数据。
- 1.0h：检查 Parquet 行数、字段、Token 长度和文件路径。
- 1.5h：人工可视化至少 30 条全图/裁剪图配对。
- 0.5h：统计损坏图、空裁剪、路径失效、问答缺失和异常比例。
- 1.0h：编写数据质量报告并冻结数据 revision。

### 必查字段

- `images`：Student 完整红框图。
- `bbox_images`：Teacher 局部裁剪图。
- `prompt`：问题。
- `extra_info` 与保留答案字段。

### 第二个 Gate：进入训练

只有同时满足以下条件才能进入训练：

- Day 1～5 的环境、VLM、玩具蒸馏和代码映射验收通过；
- 统一评测协议通过，至少有基础模型基线；
- 微型训练集至少 64 条，目标 128～256 条；
- 图片路径有效，至少 30 条人工抽查未发现系统性错配。

---

# 第二阶段：单卡、双卡与小规模主实验

## Day 9：单卡端到端冒烟基线（5h）

### 时间安排

- 1.0h：复制官方脚本，建立独立 `smoke_1gpu` 配置，不覆盖官方脚本。
- 2.0h：仅暴露 GPU 0，运行 8～16 条数据、2～3 step。
- 1.0h：检查 Student/Teacher 输入、response mask、Top-K JSD、loss 和梯度。
- 1.0h：记录单步时间、峰值显存、CPU 内存、rollout 吞吐和日志。

### 单卡 smoke 起始配置

```text
TRAINER_N_GPUS_PER_NODE=1
TRAINER_NNODES=1
TRAIN_BATCH_SIZE=8
PPO_MIMI_BATCH_SIZE=8
ROLLOUT_N=1
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=256
ROLLOUT_GPU_MEMORY_UTILIZATION=0.50
PARAM/OPTIMIZER/REF_OFFLOAD=True
```

### 验收

- Student 使用完整图，Teacher 使用裁剪图，且两者评价相同 response Token。
- `teacher_image_swap_fraction=1`，`num_distill_tokens>0`。
- JSD、loss、grad norm 均有限，Student 与 EMA Teacher 参数按预期更新。
- 形成单卡基线，供 Day 10 双卡公平比较。

## Day 10：双卡 FSDP 冒烟与扩展效率对比（5h）

### 时间安排

- 0.5h：确认两张 GPU 可见，记录 `nvidia-smi topo -m` 和卡间拓扑。
- 1.0h：冻结与 Day 9 相同的数据、随机种子、global batch 和序列长度。
- 2.0h：运行单节点双卡 FSDP 2～3 step。
- 1.0h：比较单卡/双卡的单步时间、吞吐、单卡峰值显存和 CPU 内存。
- 0.5h：记录 NCCL、Ray、FSDP 分片问题及最终修复。

### 双卡配置唯一必要改动

```text
TRAINER_N_GPUS_PER_NODE=2
TRAINER_NNODES=1
FSDP_SIZE=2（或保留 -1 自动使用当前 world size，并保存 resolved config）
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
```

双卡 smoke 不同时放大 batch、rollout 或 response length。扩展效率必须在相同 global batch 下比较；不能预设接近 2 倍加速。

### 验收

- 两个 rank 正常初始化，无 NCCL 超时、死锁或持续 OOM。
- 两张卡均有合理显存和计算利用率。
- 得到可追溯的单卡/双卡效率表，并解释通信、offload 和小 batch 对扩展效率的影响。

## Day 11：64 条稳定性、恢复与合并测试（5h）

### 时间安排

- 0.5h：固定 64 条数据、8 条 global batch 和双卡配置。
- 1.5h：完成约 8 个 global step 的小训练。
- 1.0h：分析 JSD、grad norm、有效 Token、response length 和 EMA。
- 1.0h：仅保存一个包含 optimizer 的恢复 checkpoint，并验证恢复 1 step。
- 1.0h：合并 model 权重并完成一次推理。

### 第三个 Gate：主实验前置条件

必须全部通过：

- 连续训练无 NaN/Inf，无持续 OOM。
- 单卡和双卡损失量级合理，无明显 rank 数据错位。
- 一个 checkpoint 能保存、恢复和合并，合并模型能推理。
- 能确认 Student rollout、Teacher dense token supervision 与梯度流符合方法定义。
- 保存 checkpoint 前数据盘至少留出 70GB；恢复验证和备份完成后不再保留多个 optimizer checkpoint。

## Day 12：启动 128～256 条双卡主实验（5h 主动工作）

### 时间安排

- 1.0h：冻结主实验配置、环境、数据 revision 和固定验证集。
- 0.5h：检查数据盘、日志目录、缓存目录和 checkpoint 策略。
- 0.5h：优先启动 128 条、1 Epoch；只有 128 条稳定后才扩到 256 条。
- 2.0h：检查前几个 step 的关键指标、rollout 和异常。
- 1.0h：准备训练前后固定样本对比清单。

### 主实验起始参数

```text
TRAIN_BATCH_SIZE=8
PPO_MIMI_BATCH_SIZE=8
ROLLOUT_N=1
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=256
LR=2e-6
TOPK=100
ALPHA=0.5
EMA_RATE=0.05
EPOCHS=1
TRAINER_N_GPUS_PER_NODE=2
TRAINER_NNODES=1
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
```

128 条预计为 `128 / 8 = 16` 个 global step；256 条预计为 32 个 global step。若 OOM，依次降低 batch、prompt length、response length 和 vLLM 显存比例，不同时修改多项变量。

### 存储策略

- Conda、Hugging Face、Torch、pip、临时文件和输出全部放在数据盘。
- 主实验默认不周期性保存 optimizer；最多保留一个可恢复 checkpoint。
- 最终保留一个 model-only 结果。中间 model-only 快照最多一个，用于方向性比较。
- 任何清理都在确认合并模型可推理且重要文件已备份后执行。

## Day 13：监控、完成与训练审计（5h 主动工作）

### 时间安排

- 1.0h：检查 JSD、grad norm、有效 Token 和 response length。
- 1.0h：检查 Teacher 图像替换、policy fallback、EMA 更新和各 rank 样本。
- 1.0h：检查双卡利用率、CPU 内存、数据盘余量和单步耗时。
- 1.0h：有中断则定位并恢复；无中断则审计 rollout 样本。
- 1.0h：更新训练问题清单、单/双卡对照表和报告草稿。

任何参数修改必须记录“修改前、修改原因、修改后影响”，不能无记录地同时改变多项关键参数。

## Day 14：合并结果、固定样本快评与机动纠错（5h）

### 时间安排

- 0.5h：检查并合并最终 model-only 结果；若存在一个中间快照则一并合并。
- 1.5h：用固定 20～50 条验证样本运行训练前后快评。
- 1.0h：比较准确率、regional-to-global gap、格式错误、输出长度和分布距离。
- 1.5h：按最小范围处理训练、合并或评测异常，并用少量样本验证修复。
- 0.5h：冻结自训练结果和后续消融所用的数据、步数与评测协议。

### 验收

- 至少一个自训练结果可部署、可评测。
- 所有结果明确标注数据量、step、随机种子和配置，不凭训练 loss 宣称模型能力提升。
- 如果准确率未提升，仍保留机制、稳定性和失败分析证据，不包装为论文效果复现。
- 如果训练失败，只修复阻断主链路的问题；Day 14 不引入新算法。
- 主实验、合并或快评仍未通过时，Day 15 优先完成闭环，消融相应降级为选做。

---

# 第三阶段：小型评测、诊断与受控消融

## Day 15：自训练模型固定小型评测（5h）

### 时间安排

- 0.5h：确认评测配置与 Day 6～7 一致。
- 2.0h：评测训练前基础模型和自训练最终结果。
- 1.0h：抽查 Judge 与答案解析。
- 1.0h：建立基础全图、基础裁剪图、官方模型（可选）和自训练模型结果表。
- 0.5h：记录与论文和官方 checkpoint 的差距。

### 比较对象

1. Qwen3.5-4B 全图基线。
2. Qwen3.5-4B 裁剪图诊断。
3. 官方 Vision-OPD-4B（可选对照）。
4. 自训练最终结果。

主结论只基于冻结的 20～50 条验证样本；完整 V*、ZoomBench、HR-Bench 评测均移入未来工作。

## Day 16：必做消融——on-policy vs fixed/cached off-policy prefix（5h）

### 时间安排

- 1.0h：设计控制变量和固定子集。
- 0.5h：检查两组配置唯一差异。
- 0.5h：启动实验。
- 1.5h：分析训练曲线与样本输出。
- 1.5h：运行固定验证集并整理对照表。

### 控制变量

- 相同数据子集。
- 相同初始化。
- 相同 batch、rollout、学习率、步数和评测集。
- 只改变回答前缀来源：当前 Student 在线采样，或固定/预生成前缀。

### 指标

- global accuracy；
- crop accuracy；
- regional-to-global gap；
- JSD；
- response length；
- 峰值显存；
- 运行时间；
- 是否发生训练崩塌。

## Day 17：补齐主消融与双卡工程证据（5h）

### 时间安排

- 0.5h：根据主结果和 Day 16 消融完成度决定当天路径。
- 1.0h：设计控制变量并审查配置差异。
- 0.5h：启动主消融补实验或工程证据补跑。
- 2.0h：分析输出、曲线和固定验证集结果。
- 1.0h：整理结论，决定纳入正式结果还是标记为未来工作。

优先级：

1. 补齐 Day 16 的公平对照。
2. 补齐固定小型评测缺失项。
3. 整理 Day 9～10 单卡/双卡显存、吞吐和稳定性对比。
4. 补齐训练配置、原始输出和逐样本结果索引；不再开启第二消融。

### 停止规则

- 如果主消融还不可信，只修复该消融，不启动第二消融。
- on/off-policy 主对照最多额外投入一天。
- 当天无法通过单步 backward 和控制变量检查，就记录为未来工作，不进入正式结果表。

## Day 18：结果复核与能力保持评测（5h）

### 时间安排

- 1.5h：复跑冻结的 20～50 条主验证样本。
- 1.0h：用少量通用视觉样本检查明显能力退化。
- 1.0h：核对所有表格与原始输出。
- 1.0h：绘制训练曲线、主结果和消融图。
- 0.5h：冻结最终实验结果。

Day 18 后停止新训练。完整 6K、9B、全 benchmark、第二消融或新算法全部进入“未来工作”。

---

# 第四阶段：Bad Case、报告和归档

## Day 19：Bad Case 分析（5h）

### 时间安排

- 1.0h：筛选候选样本。
- 2.0h：完成 10～20 条人工分析。
- 1.0h：建立错误分类统计。
- 1.0h：选取报告中的代表案例。

### 建议组成

- 3～6 条基础模型错误、自训练模型修正或置信度改善。
- 3～5 条全部模型都错误。
- 2～4 条自训练模型退化。
- 2～4 条数据、Judge 或解析问题。

### 错误类型

- OCR；
- 小目标识别；
- 计数；
- 属性；
- 空间关系；
- 裁剪缺少上下文；
- 红框偏移；
- 幻觉；
- 回答格式；
- Judge 误判。

## Day 20：撰写核心机制与小规模验证报告（5h）

### 时间安排

- 1.0h：方法和代码链路。
- 1.0h：环境、微型数据和单节点双卡 FSDP 适配。
- 1.0h：基线、小规模训练和结果选择。
- 1.0h：消融、Bad Case 和差距分析。
- 1.0h：复现命令、边界和未来工作。

### 报告结构

1. 项目背景与 regional-to-global gap。
2. Student、Crop Teacher、on-policy prefix、JSD、EMA。
3. 论文概念与代码实现映射。
4. 环境、微型数据和双卡 FSDP 适配。
5. 评测协议。
6. 基线与官方 checkpoint。
7. 自训练 checkpoint。
8. 消融实验。
9. Bad Case。
10. 与论文差距。
11. 已确认事实、合理推测和未确认问题。
12. 复现限制与未来工作。

## Day 21：终验、归档与面试准备（5h）

### 时间安排

- 1.5h：按照 README 从头检查关键命令。
- 1.0h：检查 checkpoint 合并、部署和评测复跑。
- 1.0h：归档配置、日志、输出和图表索引。
- 1.0h：准备 30 秒、2 分钟和 5 分钟项目介绍。
- 0.5h：完成最终验收清单。

### 最终验收

- 环境能重建。
- 数据能重新转换并通过检查。
- 至少一个 model-only 结果能合并和部署；如执行过恢复实验，则其 checkpoint 能恢复。
- 评测命令能复跑。
- 每个表格数字都能追溯到逐样本输出。
- 论文值、官方模型值和自训练值明确分开。
- 失败实验和未确认问题没有被包装成成果。
- Day 21 不启动任何新训练。

## 6. 项目成功等级

### A 档：可信小规模验证

- 双卡 FSDP 下完成 128～256 条 on-policy 主实验。
- 完成训练前基线、固定小型评测和 on/off-policy 受控消融。
- 至少一个结果可合并、部署和评测。
- 完成单卡/双卡效率表、10～20 个 Bad Case 和可复现实验归档。

### B 档：核心机制实现与验证

- 8～64 条数据训练成功。
- Student rollout、Crop Teacher、Top-K JSD、stop-gradient、backward 和 EMA 链路经过验证。
- 双卡 FSDP 冒烟成功，但尚未形成可信的效果对照。

### C 档：最小工程验证

- Tensor 级蒸馏和单步真实模型链路通过。
- 双卡训练或小型评测尚未闭环。

只有达到 A 档，才使用“核心机制实现与小规模验证”；B 档使用“核心机制实现与验证”；任何等级都不使用“完整论文复现”或“论文指标复现”。

## 7. 风险优先级与降级策略

| 风险 | 处理方式 |
|---|---|
| 环境或 NCCL 不稳定 | 不进入训练，先完成单卡和多卡最小测试 |
| 官方模型评测不合理 | 修复评测尺子，不用错误指标指导训练 |
| 数据路径或裁剪错配 | 停止训练，先完成数据 QA |
| 双卡参数 OOM | 依次调低 global batch、prompt/response length、vLLM 显存比例和并发，记录所有改动 |
| 30GB 系统盘被写满 | 所有环境、缓存和临时文件迁入数据盘，安装前后检查占用 |
| 数据盘不足 | 不下载完整 6K；限制为 64～256 条；最多保留一个 optimizer checkpoint 和一个最终 model-only 结果 |
| 小规模训练中断 | 从唯一恢复 checkpoint 继续；保留完整错误日志 |
| 自训练结果低于官方 | 先查配置、数据、checkpoint 和评测，再决定是否重跑 |
| 消融来不及 | 保留一个可信消融，删除第二消融 |
| 报告时间被挤占 | Day 18 后停止高风险实验 |
| 9B 或完整 6K 占用主线资源 | 移到未来工作 |

## 8. 一句话执行原则

先证明评测可信，再证明单卡链路正确；随后完成双卡 FSDP 与小规模训练，再做唯一主消融；先保留可追溯证据，再追求更多数据或更高指标。
