# Vision-OPD Day 1 工作简报（2026-08-20）

## 今日目标

按照更新后的 21 天计划，完成正式 Day 1 的项目状态、实验边界、数据协议、预算和证据规范冻结，为后续确定性抽样、统一评测及训练实验建立可追溯基线。

## 已完成工作

1. 重读并采用新的项目计划，确认旧计划中的论文阅读、代码理解、环境安装、Qwen3.5-4B 下载和普通多模态推理均作为前置成果，不再重复占用计划天数。
2. 新建 docs/project_freeze.md，删除已过期的 day1_project_freeze.md，统一记录项目目标、实验矩阵、数据协议、软硬件、预算、存储约束和验收状态。
3. 新建 configs/project_1024.yaml，将项目主随机种子固定为 42，并要求数据划分、DataLoader、训练和 rollout 显式使用同一数值。
4. 创建 artifacts/runs/preflight，归档 Git 状态、工作树补丁、软件版本、pip check、磁盘快照、模型文件清单及关键文件 SHA256。
5. 精简历史环境记录，保留安装版本、CUDA 扩展编译方式、单卡推理结论和 Processor Shape，删除重复日志与过长模型回答。

## 冻结的实验设计

统一使用 Qwen3.5-4B 和固定数据协议：

| Split | 数量 | 用途 |
|---|---:|---|
| train | 1024 | SFT、Vision-OPD、Cached Prefix、GRPO |
| eval | 128 | 统一主评测，训练期间禁止使用 |
| retention | 64 | 通用能力与格式保持检查 |

实验矩阵包括 Vanilla、SFT、Vision-OPD、Cached Prefix 和后续 GRPO。SFT、Vision-OPD 与 GRPO 均从同一个 Base checkpoint 独立启动，不串行继承。Vision-OPD 与 Cached Prefix 只允许改变 prefix_source，其他训练和评测条件必须一致。

## 环境与模型结论

现有 vision-opd 环境的 Python、PyTorch、Transformers、vLLM、Ray、FlashAttention、causal-conv1d 和本地 verl 均可用。Qwen3.5-4B 位于 /root/autodl-tmp/models/Qwen3.5-4B，两个权重分片完整，模型文件哈希已归档。

pip check 仍报告若干依赖元数据冲突。由于核心导入和历史单卡推理通过，本阶段不盲目重装，将实际兼容性留给真实 SFT/Vision-OPD Smoke 验证。

## 数据与存储决策

服务器数据盘约 50GB，当前剩余空间不足以安全下载完整数据，因此不扩容服务器，也不在服务器保留原始数据包。最终方案为：

- 在本地具有足够空间的机器下载并处理原始数据；
- 使用 seed 42 按原始图像或问题组进行确定性划分；
- 本地完成自动校验、至少 30 条人工 QA、统计和 SHA256；
- 只向服务器上传冻结后的 1024/128/64 子集及 manifest、统计、人工 QA 和哈希；
- 上传后复算 SHA256，并重新检查服务器剩余空间。

## 延期项与风险门槛

当前没有可用的双卡实例，因此双卡 RTX PRO 6000 可见性、显存、NCCL 和 FSDP 尚未验证，状态记录为 DEFERRED。该项不阻塞本地 Day 2～3 数据准备，但必须在任何 GPU Smoke 或正式训练前完成。

Day 1 当前状态为 CONDITIONAL PASS。编写本简报时，冻结产物已提交，最新提交为 2674f85。

## 下一步

进入新计划 Day 2：先审计原始元数据字段和分组 ID，再实现 scripts/prepare_project_subset.py。脚本必须固定 seed 42、生成稳定 sample_id，并重复运行得到一致的 train/eval/retention 划分。完成候选 manifest 和泄漏检查前，不开始训练。
