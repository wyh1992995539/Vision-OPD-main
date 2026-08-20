# Vision-OPD Day 2 工作简报：元数据冻结与确定性划分

> 日期：2026-08-20  
> 状态：**PASS（仅 Day 2 元数据与划分 Gate）**  
> 对应计划：[Day 2：获取数据并实现确定性抽样](vision_opd_21_day_plan.md)

## 1. 当日目标

在不下载或解压官方全量图像的前提下，冻结 Vision-OPD-6K 元数据版本，并从中生成一次性的、可重复且无同原图泄漏的：

| Split | 数量 | 用途 |
|---|---:|---|
| train | 1024 | SFT、Vision-OPD、Cached Prefix、后续可验证 GRPO 子集 |
| eval | 128 | 统一主评测，训练期间不可使用 |
| retention | 64 | 通用能力和格式保持检查 |

## 2. 冻结输入

| 项目 | 值 |
|---|---|
| 数据源 | `yuanqianhao/Vision-OPD-6K` |
| 数据 revision | `eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4` |
| 元数据文件 | `train.jsonl` |
| 本地输入位置 | `D:\VisionOPD-data\raw_meta\train.jsonl` |
| 文件大小 | 4,566,587 bytes |
| 行数 | 6,241 |
| SHA256 | `8ad2fb81da0f6fba1766545dc5f84cc2250e48704738757461b2d75aa31821df` |
| 数据划分 seed | `42` |

输入 revision、大小、行数与 SHA256 同时写入了 [`configs/project_1024.yaml`](../configs/project_1024.yaml)。脚本在每次运行前都校验这些值；任意一项不一致即停止，不重新抽样。

## 3. 实现与方法

实现脚本：[`scripts/prepare_project_subset.py`](../scripts/prepare_project_subset.py)。

处理流程：

```text
冻结 train.jsonl
→ 逐行校验路径、bbox、问题与答案
→ 使用 original_images[0] 生成 group_id
→ 以 revision + 路径 + 问题 + 答案生成稳定 sample_id
→ 对 group_id 按 SHA256(seed | group_id) 排序
→ 划分 train / eval / retention
→ 输出 manifest、统计与三个 split
```

划分单位是原图组，禁止同一 `original_images[0]` 出现在多个 split。若官方记录没有显式的源样本 ID，`source_id` 使用 `revision:row:<行号>` 作为透明 fallback，并通过 `source_id_kind=frozen_row_fallback` 标明；它不被表述为官方原始 ID。

所有图片字段只保留相对路径，例如 `images/*.png` 和 `teacher_images/*.png`，不写入 Windows 绝对路径。

## 4. 实际结果

| 检查项 | 结果 |
|---|---:|
| 总元数据记录 | 6,241 |
| 有效记录 | 6,241 |
| 无效记录 | 0 |
| 唯一原图组 | 6,241 |
| 多记录原图组 | 0 |
| 选中 train / eval / retention | 1024 / 128 / 64 |
| 选中样本题型 | 1,216 条均为多选题 |
| 跨 split 原图组重叠 | 0 |
| 元数据字段问题 | 0 |

生成文件：

| 文件 | SHA256 |
|---|---|
| `artifacts/data/candidate_manifest.jsonl` | `f31e40c9f98df8837a697c4c01870eadd5e4fcdf03a7ab0f896577904f2b1b72` |
| `artifacts/data/split_manifest.json` | `df25a12d0427e5b16d1ddcdb98a343c176434adc018587504c3270474da7bb05` |
| `artifacts/data/candidate_stats.json` | `21798988501b842c3c14baad74d47e5b11ed543b10ea320eabda42f6a6dda24a` |
| `artifacts/data/train_1024.jsonl` | `0721e9c57ce55c64b27617fb35c4e419d690a7e68aadca8280e01ceb1e10efcc` |
| `artifacts/data/eval_128.jsonl` | `c591dfb74dbc883a270e53ddfec2801ac62c21f3f3c9894911060f63e03efbdd` |
| `artifacts/data/retention_64.jsonl` | `fcf6969dc27c50bf3854a162fea04378a3aac6df63cef48bf75e4cad71196bb3` |

`split_manifest.json`、`candidate_stats.json` 与三个 split JSONL 已复制到 `E:\VisionOPD-subset\manifests\`，其 SHA256 与仓库生成副本逐文件一致；全量 `candidate_manifest.jsonl` 仅保留在仓库的 artifacts 目录。

## 5. 可重复性验收

同一输入、同一 revision、同一配置和同一 seed 下，脚本连续运行两次；候选清单、split 总清单、统计文件以及 train/eval/retention 三个 JSONL 的 SHA256 全部一致。

执行入口：

```powershell
conda activate train
cd "C:\Users\19929\Desktop\大模型\Vision-OPD\Vision-OPD-main"
python scripts\prepare_project_subset.py --config configs\project_1024.yaml
```

## 6. 已完成与未完成边界

本日已完成：

- 官方元数据版本冻结、校验与审计；
- 1024/128/64 确定性划分；
- 原图组隔离与稳定 sample ID；
- manifest 的本地副本同步及哈希一致性验证。

本日未完成，且不得因此宣称完成：

- 未下载或解压全部官方 Student 图与 Teacher crop；
- 未验证任何图片文件存在、可解码或与 bbox/crop 成对对应；
- 未生成为服务器可用的 Linux 路径 Parquet；
- 未进行人工图片 QA；
- 未启动 SFT、Vision-OPD、Cached Prefix、GRPO 或任意训练。

因此，本日成果应表述为：**“完成 Vision-OPD-6K 元数据审计、确定性无同图泄漏划分与冻结 manifest。”**

## 7. 下一步：Day 3

1. 将官方 `images` 和 `teacher_images` 压缩包下载到 `D:\VisionOPD-data\raw\`；不下载 `original_images`。
2. 根据 `E:\VisionOPD-subset\manifests\` 中的 1,216 条相对路径，选择性提取全图和 crop 到 `E:\VisionOPD-subset\images\`、`E:\VisionOPD-subset\teacher_images\`。
3. 实现 `scripts/validate_project_data.py`，检查图片解码、文件路径、bbox、图像/crop 配对、数量与 SHA256。
4. 人工检查至少 30 条，并保存 `manual_qa_30.jsonl`。
5. 图片和清单上传到服务器后，重建带 Linux 路径的 Parquet；不得上传 Windows 路径 Parquet。
