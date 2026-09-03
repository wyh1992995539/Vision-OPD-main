# Day 10 Vision-OPD 6241 全量数据、存储与 Data Gate 工作简报

> 执行日期：2026-09-03（UTC）
> 对应实验：`E-D10-6K-DATA-001`
> Day 10 最终状态：**PASS_TO_DAY11**
> GPU 使用：**false**

## 技术摘要

Day 10 已将冻结 revision 下的 6,241 条源记录全部构建为现行 `train-6241`，不再划分新的 eval、test、retention 或 holdout。原始 `train.jsonl` 已通过行数、文件大小和 SHA256 三重身份校验；重新生成的 candidate/train manifest 均为 6,241 行，历史 1,216 个 sample ID 全部完成对账，无缺失。

全量 Student 红框图和 Teacher crop 各 6,241 张，自动 QA 共检查 12,482 张图片，未发现缺失、零字节、解码失败、重复 sample ID 或路径越界。项目 Schema 的 `train_6241.parquet` 已生成并完成回读，行数、字段、顺序、sample ID 和 SHA256 已冻结。

新增样本的分层人工语义复核由项目负责人完成并确认无错误；该结论按 `MANUAL_USER_ATTESTATION` 保存，本次证据更新没有独立复查样本。数据验证完成后，项目负责人明确要求永久删除 7 个可重新下载的原始图片压缩包，共释放约 29.13 GiB；冻结 revision、压缩包大小、SHA256 和下载元数据仍保留。最终数据盘可用约 238.49 GiB，高于 120 GiB 硬下限和 130 GiB 建议值。

最终配置状态为 `data_gate_completed`，机器汇总为 `overall_data_gate_status=PASS`，可以进入 Day 11。

## 一、Day 10 任务完成情况

| 任务 | 目标 | 主要结果 | 状态 |
|---|---|---|---|
| Task 1 | 下载前记录 Git、磁盘、缓存和数据占用 | 保存首次快照与最新调用快照，数据盘容量为 300 GiB | PASS |
| Task 2 | 将缓存和临时目录迁移到数据盘 | HF、Torch、pip、XDG 和 TMP 路径统一指向 `/root/autodl-tmp` | PASS |
| Task 3 | 冻结 6K scope amendment 和配置 | 新建 amendment 与 `configs/project_6241.yaml`，保留全部历史配置和证据 | PASS |
| Task 4 | 校验源元数据并重建 manifest | 6,241 行、4,566,587 bytes、SHA256 匹配；历史 1,216 ID 缺失 0 | PASS |
| Task 5 | 登记历史 split 边界 | eval-128、retention-64 标记为 `historical_only=true`；现行仅 train=6,241 | PASS |
| Task 6 | 下载、解压并自动检查全量图片 | Student/Teacher 各 6,241；12,482 张图片自动 QA 问题数为 0 | PASS |
| Task 7 | 对新增样本做分层人工语义复核 | 项目负责人确认约 10 条分层样本无语义错配 | PASS（用户确认） |
| Task 8 | 生成项目 Schema 的训练 Parquet | `train_6241.parquet` 共 6,241 行 | PASS |
| Task 9 | 冻结清单、Schema、哈希与回读证据 | Parquet 回读、相对路径、字段和 SHA256 检查全部通过 | PASS |
| Task 10 | 处理原始压缩包并记录最终磁盘 | 7 个压缩包经明确要求后删除，最终可用空间约 238.49 GiB | PASS |

## 二、冻结的数据合同

| 项目 | 冻结值 |
|---|---|
| 数据源 | `yuanqianhao/Vision-OPD-6K` |
| revision | `eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4` |
| source | 6,241 |
| train | 6,241 |
| active eval/test/retention/holdout | 0/0/0/0 |
| 历史样本对账 | 1,216/1,216，缺失 0 |
| 主键 | 稳定且唯一的 `sample_id` |
| Student 输入 | 完整红框图 |
| Teacher 输入 | 对应 crop 图 |

原始元数据身份：

```text
path:   /root/autodl-tmp/data/vision_opd_6241/raw/train.jsonl
rows:   6241
bytes:  4566587
sha256: 8ad2fb81da0f6fba1766545dc5f84cc2250e48704738757461b2d75aa31821df
```

历史 `eval-128` 和 `retention-64` 只保留为 Day 1～9 审计证据，其中样本已经并入 `train-6241`，不得用于新 checkpoint 的选优、早停或能力结论。

## 三、图片与数据质量 Gate

### 1. 自动 QA

| 检查项 | 结果 |
|---|---:|
| 训练记录 | 6,241 |
| 唯一 sample ID | 6,241 |
| Student 图片 | 6,241 |
| Teacher crop | 6,241 |
| 自动检查图片总数 | 12,482 |
| 缺失/零字节/解码失败 | 0 |
| 自动 QA issue | 0 |

自动 QA 覆盖路径安全、文件存在、零字节、Pillow 完整解码、尺寸、bbox、full/crop 配对和 sample ID 唯一性。

### 2. 人工语义复核

项目负责人从新增 5,025 条记录中完成约 10 条分层人工复核，并确认完整图、红框、Teacher crop、问题和答案之间没有语义错配。机器证据中的类型为：

```text
semantic_alignment_checked=true
semantic_alignment_check_type=MANUAL_USER_ATTESTATION
semantic_alignment_result=PASS
semantic_alignment_independently_rechecked_during_update=false
```

这一结论表示负责人已完成并担保人工复核结果，不表示自动程序对 6,241 条记录进行了语义判断。

## 四、Parquet 与哈希冻结

正式训练输入：

```text
path:   /root/autodl-tmp/data/vision_opd_6241/train_6241.parquet
rows:   6241
bytes:  1280746
sha256: 142c972f182cc0bf90b2ab44a2255896643c5983e0eaf8c7c58f5a488a89031e
```

构建报告确认以下检查全部通过：

- split 数量符合配置；
- manifest 必需字段齐全；
- manifest 中图片路径为可移植相对路径；
- 全部图片路径可读；
- 无 sample、group 或 image overlap；
- Parquet Schema 和 6,241 行完整回读通过。

## 五、存储、缓存与压缩包处置

### 1. 缓存合同

`scripts/activate_vision_opd_6241.sh` 将以下目录统一放在数据盘：

```text
HF_HOME=/root/autodl-tmp/hf_cache
HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
TORCH_HOME=/root/autodl-tmp/torch_cache
PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
XDG_CACHE_HOME=/root/autodl-tmp/xdg_cache
TMPDIR=/root/autodl-tmp/tmp
```

### 2. 压缩包处置

数据 Gate、图片 QA 和 Parquet 回读全部通过后，项目负责人于 2026-09-03 06:54 UTC 明确要求删除 6,241 数据的原始图片压缩包。随后使用限定到 7 个明确路径的 `rm -f` 执行永久删除，命令于 2026-09-03 06:55:10 UTC 成功退出。

| 项目 | 结果 |
|---|---|
| 压缩包数量 | 7 |
| 删除总量 | 31,283,220,139 bytes，约 29.13 GiB |
| 处置方式 | `RM_F_AFTER_EXPLICIT_USER_REQUEST` |
| 是否转移归档 | 否 |
| 冻结 revision | 保留 |
| 每包大小和 SHA256 | 保留 |
| Hugging Face 下载元数据 | 保留 |
| 可恢复性 | 可按冻结 revision 重新下载并核对哈希 |

### 3. 最终磁盘快照

| 项目 | 结果 |
|---|---:|
| 数据盘总量 | 300 GiB |
| 数据盘已用 | 约 61.51 GiB |
| 数据盘可用 | 约 238.49 GiB |
| 数据盘使用率 | 21% |
| 6K 数据集目录 | 约 29.18 GiB |
| 系统盘可用 | 约 12.01 GiB |
| 120 GiB 硬下限 | PASS |
| 130 GiB 建议值 | PASS |

## 六、主要证据

- `configs/project_6241.yaml`
- `docs/amendments/full_train_6241_scope_amendment.md`
- `artifacts/data/vision_opd_6241/candidate_manifest.jsonl`
- `artifacts/data/vision_opd_6241/train_6241.jsonl`
- `artifacts/data/vision_opd_6241/vision_opd_6241_manifest.json`
- `artifacts/data/vision_opd_6241/vision_opd_6241_data_qa.json`
- `artifacts/data/vision_opd_6241/vision_opd_6241_stats.json`
- `artifacts/data/vision_opd_6241/vision_opd_6241_sha256.txt`
- `artifacts/data/vision_opd_6241/parquet_build_report.json`
- `artifacts/data/vision_opd_6241/parquet_sha256.txt`
- `artifacts/runs/E-D10-6K-DATA-001/pre_download_snapshot.json`
- `artifacts/runs/E-D10-6K-DATA-001/data_gate_summary.json`
- `artifacts/runs/E-D10-6K-DATA-001/archive_disposition.json`
- `artifacts/runs/E-D10-6K-DATA-001/final_storage_snapshot.json`

## 七、结论边界与 Day 11 交接

### 可以确认

- 6,241 条冻结源记录全部进入现行训练集，未建立新的内部留出集。
- Student/Teacher 图片各 6,241 张，自动图片 QA 与 Parquet Gate 全部通过。
- 新增样本的分层人工语义复核由项目负责人确认无错误。
- 原始图片压缩包是在验证通过且负责人明确要求后删除，不是转移归档。
- 数据盘剩余空间同时满足 120 GiB 硬下限和 130 GiB 建议值。
- Day 10 最终状态为 `PASS_TO_DAY11`。

### 不能扩大表述

- 人工复核约 10 条新增样本，不代表逐条人工核验全部 6,241 条。
- 自动 QA 验证结构、图片和路径完整性，不具备全量语义判定能力。
- Day 10 没有运行训练，也没有产生模型能力提升结论。
- 已删除的压缩包需要重新下载才能恢复，但冻结 revision、大小和 SHA256 足以重新核验身份。

### Day 11 入口

Day 11 将继续完成：

1. 使用训练时 Processor 对 6,241 条执行 prompt 长度与错误审计；
2. 重跑 train-6241 与三项外部 Benchmark 的 overlap；
3. 实现并验证 `6241 + 7` 的零权重尾批覆盖合同；
4. 生成并验证 Base Cached Prefix 6,241/6,241；
5. 冻结 Vision-OPD 6K 配置、预算、中止策略和 guarded preflight；
6. 使用新增记录与长度长尾样本完成真实 Pilot。

只有 Day 11 的长度、overlap、sampler、Cached Prefix、存储、预算和 Pilot Gate 全部通过，才能进入 Day 12 的 Vision-OPD 6,241 全量正式训练。
