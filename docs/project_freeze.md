# Vision-OPD 项目冻结文档

> 冻结日期：2026-08-20  
> 对应计划：正式 Day 1——冻结项目状态与资源边界  
> 范围修订日期：2026-08-24
> 当前状态：PASS WITH AMENDMENT；Day 1～4 已完成证据继续有效，活跃实验范围删除 SFT，并加入最终外部 Benchmark。

## 0. 范围修订记录

2026-08-24 在不修改 Day 1～4 原始执行证据的前提下，对后续实验范围作如下修订：

- 删除尚未开始、未生成专属数据、checkpoint 或运行目录的 SFT 分支；
- 保留原始 Qwen3.5-4B、1024/128/64 固定划分、Day 4 Vanilla-128 结果与 Cached Prefix-1024；
- 训练开发阶段继续使用 internal `eval-128` 与 `retention-64`；
- 最终定版的 Base、Vision-OPD、Cached Prefix 和后续 GRPO 统一运行 ZoomBench、MMStar、V* Bench；
- 外部 Benchmark 只用于冻结后的最终 checkpoint 和一次 Base 基线，不用于反复调参或挑选 checkpoint；
- Day 1～4 历史简报与 preflight 快照保持原样，作为当时状态的不可回写证据。

修订原因：优先回答 Vision-OPD 相对 Base 是否有效、on-policy 是否优于 Cached Prefix，以及专项提升是否伴随外部通用能力退化；不再为非核心 SFT 分支消耗训练预算。

## 1. 项目定位

本项目基于 Qwen3.5-4B，在固定的 1024 条训练数据和两层冻结评测协议上完成 Vision-OPD 小规模复现与受控比较。

项目结果限定为“4B、1024 条数据的小规模复现”，不宣称复现论文完整 6.2K 训练结果。

必须保持以下边界：

- Vision-OPD 是 on-policy 自蒸馏，不是 GRPO 或 RLVR。
- Cached Prefix 是 Student 前缀来源的单变量消融，不是另一套完整算法。
- Base / Vanilla 是未经本项目 Vision-OPD 训练的原始 Qwen3.5-4B；Vanilla 只是直接评测，不产生新 checkpoint。
- Vision-OPD、Cached Prefix 和 GRPO 均从同一个 Base checkpoint 独立启动，不串行继承。
- 论文作者发布的官方 Vision-OPD-4B 只可作为可选参考，不得替代本项目 Base 或冒充本人训练结果。
- Teacher 使用裁剪图，Student 使用完整红框图；最终部署只需要 Student 和完整图。

## 2. 冻结实验矩阵

| 实验 | 实验 ID | 训练信号 | 计划完成时间 | 状态 |
|---|---|---|---:|---|
| Base / Vanilla 内部评测 | E-D4-001 | 不训练；internal eval-128 | Day 4 | PASS |
| 三项 Benchmark 协议与 Smoke | E-D5-001 | Base；每项固定 16 条端到端 Smoke | Day 5 | 未开始 |
| Base / Vanilla 外部基线 | E-D6-001 | ZoomBench、MMStar、V* Bench | Day 6 | 未开始 |
| Vision-OPD Smoke | E-D7-001 | 在线 Student Prefix + Crop Teacher Top-K JSD | Day 7 | 未开始 |
| Vision-OPD 64 | E-D8-001 | 同上 | Day 8 | 未开始 |
| Vision-OPD 1024 | E-D10-001 | 同上 | Day 10～12 | 未开始 |
| Vision-OPD 最终评测 | E-D12-001 | internal 128/64 + 三项外部 Benchmark | Day 12 | 未开始 |
| Cached Prefix 契约测试 | E-D14-001 | Base Cached Prefix + 同一 JSD | Day 14 | 未开始 |
| Cached Prefix 64 | E-D15-001 | 同上 | Day 15 | 未开始 |
| Cached Prefix 1024 | E-D16-001 | 同上 | Day 16～18 | 未开始 |
| Cached Prefix 最终评测 | E-D18-001 | internal 128/64 + 三项外部 Benchmark | Day 18 | 未开始 |
| GRPO Pilot/正式训练 | E-D23-001 等 | 规则 Reward + 组内相对优势 | Day 23～30 | 未开始 |
| GRPO 最终评测 | E-D28-001 | internal 128/64 + 三项外部 Benchmark | Day 28 | 未开始 |

Vision-OPD 与 Cached Prefix 只允许改变 prefix_source：前者为 online，后者为 cached。模型、数据、数据顺序、batch、步数、学习率、Teacher 图像、Top-K JSD、EMA、seed 和评测协议必须保持一致。

## 3. 代码状态

| 项目 | 冻结值 |
|---|---|
| 工作目录 | /root/autodl-tmp/Vision-OPD-main |
| Git commit | 04a87fe903cbab118fbe516836d18dfac261cd34 |
| 分支 | main |
| Remote | origin → https://github.com/wyh1992995539/Vision-OPD-main.git |
| origin/main | 与冻结 commit 一致 |
| 工作树 | 冻结前检查为 clean |

冻结前不存在未提交的个人修改，因此无需生成个人修改补丁。创建本文件及后续 Day 1 产物后，应使用独立提交保存正式冻结状态。

禁止通过 git reset 或覆盖文件处理后续个人修改。每次正式实验必须记录实际 commit、工作树状态、配置和完整命令。

## 4. 模型冻结

| 项目 | 冻结值 |
|---|---|
| Base 模型 | Qwen3.5-4B |
| 本地路径 | /root/autodl-tmp/models/Qwen3.5-4B |
| 当前大小 | 约 8.8 GiB |
| 权重分片 | 2 个 Safetensors 分片，均已存在 |
| 历史验证 | 单卡 vLLM 服务、普通图片问答和 Processor 检查已通过 |

所有训练分支必须从该 Base checkpoint 启动。Cached Prefix 必须由训练前 Base 模型生成，不得由 Vision-OPD、GRPO 或其他训练后模型生成。

正式训练前应在 artifacts/runs/preflight/model_sha256.txt 中保存模型配置、索引、Tokenizer 和两个权重分片的 SHA256。

## 5. 数据与评测协议

只允许进行一次固定划分：

| Split | 数量 | 用途 |
|---|---:|---|
| train | 1024 | Vision-OPD、Cached Prefix、GRPO |
| eval | 128 | 内部主评测；训练期间禁止用于优化 |
| retention | 64 | 内部格式与同分布能力保持检查，不冒充外部通用能力评测 |

数据契约：

- Day 2 元数据已冻结：`yuanqianhao/Vision-OPD-6K` revision 为 `eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4`；`train.jsonl` 共 6,241 行、4,566,587 bytes，SHA256 为 `8ad2fb81da0f6fba1766545dc5f84cc2250e48704738757461b2d75aa31821df`。本地准备副本位于 `D:\\VisionOPD-data\\raw_meta\\train.jsonl`，不作为服务器训练路径。
- 项目主随机种子固定为 42；数据划分、DataLoader、训练和 rollout 均使用该值，各实验配置和启动脚本必须显式传入。
- 按原始图像 ID 或问题组划分，禁止同图或同源问题跨 split 泄漏。
- 每条样本生成稳定且可重复的 sample_id，同时保留原始来源 ID。
- Vision-OPD 使用完整红框图、裁剪图和问题。
- Cached Prefix 与 train 1024 的 sample_id 必须一一对应。
- GRPO 只使用规则可以可靠判分的封闭式样本，数量不足时报告实际数量。

评测协议分为两层：

1. 训练开发层：Base、Vision-OPD、Cached Prefix、GRPO 使用同一 internal eval-128、retention-64、generation config 和 evaluator version，用于快速回归、配对分析与内部能力保持检查。
2. 最终定版层：Base 运行一次 ZoomBench、MMStar、V* Bench；每个最终 Vision-OPD、Cached Prefix、GRPO checkpoint 在内部结果冻结后运行同一三项外部 Benchmark。

共同约束：

- Day 5 冻结 Benchmark 数据 revision/hash、官方划分、Prompt、图像预处理、生成参数、答案解析、Judge 模型/版本/Prompt/参数和成本预算。
- 下载后对 train/eval/retention 与 Benchmark 执行文件 SHA256、问题文本和感知哈希重叠审计；不得静默删除官方测试样本或隐瞒重叠。
- 外部 Benchmark 不用于调学习率、epoch、Prompt 或 checkpoint 选择。
- 保存逐样本预测、解析结果、Judge 来源和错误，不只保存汇总准确率。
- 无法可靠判定的样本标记为 unsupported，不强制判错，也不得只按成功样本计算准确率。
- 结果比较至少包含总体与题型指标、corrected、regressed、输出长度、格式、训练/推理时间、显存和费用。

## 6. 软件环境

当前环境路径为 /root/miniconda3/envs/vision-opd。

| 组件 | 当前版本 | 状态 |
|---|---|---|
| Python | 3.12.13 | PASS |
| PyTorch | 2.10.0+cu128 | PASS |
| PyTorch CUDA | 12.8 | PASS（编译版本） |
| Transformers | 5.5.0 | PASS |
| vLLM | 0.18.0 | PASS；历史单卡部署通过 |
| Ray | 2.53.0 | PASS |
| FlashAttention | 2.8.3.post1 | PASS；已编译 sm_120 |
| causal-conv1d | 1.6.1 | PASS |
| verl | 0.7.0.dev0 | PASS；editable 指向当前仓库 |

已知风险：

- pip check 存在 vLLM/Transformers、xFormers/PyTorch、NumPy/OpenCV/CuPy 等依赖元数据冲突。
- 核心包导入和历史普通推理已通过，因此不在 Day 1 盲目重装或降级；真实 Smoke 将作为运行时兼容性的最终证据。
- Conda 环境约占系统盘 12 GiB。当前不迁移；模型、数据、缓存、rollout 和 checkpoint 必须写入数据盘。
- 运行 Python、vLLM 或训练前设置 OMP_NUM_THREADS=8。
- HF_HOME、TORCH_HOME、PIP_CACHE_DIR 和 TMPDIR 应统一指向 /root/autodl-tmp 下的对应目录。

## 7. 硬件冻结与验收

| 项目 | 冻结值 |
|---|---|
| 节点 | 单机单节点 |
| GPU | 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition |
| 单卡显存 | 96 GB |
| GPU / node | 2 / 1 |

2026-08-20 已在实际双卡 AutoDL 实例完成硬件验收：

- nvidia-smi 显示 2 张 NVIDIA RTX PRO 6000 Blackwell Server Edition，每张 97887 MiB，驱动为 580.95.05。
- PyTorch 2.10.0+cu128、CUDA 12.8 可用，torch.cuda.device_count() 等于 2。
- 两个 NCCL rank 的 all-reduce 均得到 3.0，双卡通信通过。
- 原始结果保存于 artifacts/runs/preflight/hardware.txt。

双卡硬件 Gate 状态为 PASS。完整 FSDP、Ray/vLLM 和真实模型训练稳定性仍由后续对应 Smoke 验证。

## 8. 预算冻结

| 项目 | 冻结值 |
|---|---:|
| 单卡费用 | 5.98 元/小时 |
| 双卡费用 | 11.96 元/小时 |
| 硬上限 | 2000 元 |
| 双卡时长硬上限 | 约 167 小时 |
| 当前估算状态 | 删除 SFT 后，等待 Day 5 Benchmark Smoke 和各训练分支实测吞吐后重新冻结 |

每次启动训练前计算本次预计费用和累计预计费用。预计累计费用超过 2000 元时停止扩大训练规模，优先完成评测、报告和证据归档。

## 9. 存储冻结与数据获取决策

2026-08-20 检查结果：

| 挂载点 | 总容量 | 已用 | 可用 | 状态 |
|---|---:|---:|---:|---|
| 系统盘 / | 30 GB | 约 18 GB | 约 13 GB | 不得存放新增大型文件 |
| 数据盘 /root/autodl-tmp | 50 GB | 约 8.9 GB | 约 42 GB | 不满足完整数据下载安全线 |

官方完整数据约 37.5 GB，而当前数据盘剩余空间低于 45 GB 安全线。当前禁止直接运行完整数据下载。

最终方案已确定为异机处理：在本地具有足够空间的机器上完成确定性抽样、图像抽取、校验和数据冻结，只向服务器上传 train 1024、eval 128、retention 64 子集及 manifest、统计、人工 QA 和 SHA256。服务器不得下载完整原始数据。

上传前必须记录子集总大小；上传后重新记录服务器磁盘，确保仍为最终 checkpoint、临时合并空间和评测产物保留足够容量。

训练期间只保留当前实验 checkpoint。完成合并、独立进程加载测试和 SHA256 后才能删除原始分片；不得删除尚未通过加载测试的唯一模型。

## 10. 证据与归档规范

每个正式训练实验目录至少包含 config.yaml、command.txt、git_commit.txt、env.txt、train.log、metrics.jsonl、cost.json、checkpoint_sha256.txt，以及 eval/predictions.jsonl 和 eval/summary.json。

每个正式评测实验目录至少包含 protocol.yaml、command.txt、git_commit.txt、env.txt、model_sha256.txt、dataset_revision_sha256.json、predictions.jsonl、judge_details.jsonl、summary.json 和 cost.json。

只有同时满足训练完成、模型可重新加载、固定评测完成、日志与配置齐全，实验才能标记为完成。

Day 1 preflight 目录至少包含：

- README.md
- git_state.txt
- working_tree.patch
- env.txt
- pip_check.txt
- hardware.txt
- disk.txt
- model_manifest.txt
- model_sha256.txt
- storage_decision.md

## 11. Day 1 验收状态

| 验收项 | 当前状态 | 说明 |
|---|---|---|
| 项目目标与实验矩阵冻结 | PASS WITH AMENDMENT | 2026-08-24 删除 SFT 并加入三项外部 Benchmark；Day 1～4 证据不重跑 |
| 模型与数据规模冻结 | PASS | Qwen3.5-4B；1024/128/64 |
| Git 状态审计 | PASS | 冻结前工作树 clean |
| 软件环境审计 | PASS WITH WARNINGS | 核心导入通过；保留 pip check 风险 |
| configs/project_1024.yaml | PASS | 已创建并固定主种子 42 |
| preflight 原始证据 | PASS | 初始证据文件已创建并完成模型 SHA256 |
| 双卡硬件快照 | PASS | 两卡可见、型号与显存匹配，NCCL all-reduce 通过 |
| 数据获取方案 | PASS | 已确定为本地抽取冻结子集后上传 |

Day 1 验收已完成，Day 2～4 后续 Gate 也已完成并保留原始证据。当前下一步是 Day 5：冻结 ZoomBench、MMStar、V* Bench 协议，完成数据准备、重叠审计和每项 16 条端到端 Smoke。服务器仍不得在未完成磁盘大小预检时下载大型 Benchmark 数据；任何正式训练前仍须先完成对应的真实模型 Smoke。
