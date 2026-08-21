# Vision-OPD Day 3 工作简报：图片子集、自动校验与人工 QA

> 日期：2026-08-21  
> 状态：**PASS（图片、QA、服务器 Parquet 与加载 Gate）**
> 边界：Vanilla 基线、Cached Prefix、SFT、Vision-OPD、GRPO 均未开始

## 1. 当日目标

在 Day 2 已冻结的 1024 条训练、128 条主评测和 64 条能力保持清单上，完成以下数据闭环：

1. 下载固定 revision 的官方 Student 全图与 Teacher crop 压缩包；
2. 不解压全部 6,241 条图片，只提取三个 manifest 指定的 1,216 对图片；
3. 对 2,432 张图片执行路径、配对、完整解码、尺寸、格式、bbox 和 SHA256 校验；
4. 按 split 与 bbox 面积分层固定抽取 30 条，人工检查全图、crop、问题和答案的语义一致性；
5. 将图片子集与 manifest 上传到服务器，复核 2,432 张图片 SHA256；
6. 从冻结 JSONL 生成 Linux 路径的 train/eval/retention Parquet，并用 `datasets` 实际加载验收。

本日不下载 `original_images`，不生成带 Windows 路径的最终训练 Parquet，也不启动 Vanilla、SFT、Vision-OPD、Cached Prefix 或 GRPO。

## 2. 冻结输入与存储位置

| 项目 | 值 |
|---|---|
| 数据源 | `yuanqianhao/Vision-OPD-6K` |
| 数据 revision | `eb5c1c2e7b9a7b6a619efe4161c7369c71bf8af4` |
| 原始元数据 | `D:\VisionOPD-data\raw_meta\train.jsonl` |
| 元数据记录数 | 6,241 |
| 元数据大小 | 4,566,587 bytes |
| 元数据 SHA256 | `8AD2FB81DA0F6FBA1766545DC5F84CC2250E48704738757461B2D75AA31821DF` |
| 原始压缩包目录 | `D:\VisionOPD-data\raw` |
| 冻结子集目录 | `E:\VisionOPD-subset` |
| 仓库 | `C:\Users\19929\Desktop\大模型\Vision-OPD\Vision-OPD-main` |
| 服务器数据根目录 | `/root/autodl-tmp/data/vision_opd_1024` |

数据分层保持为：

```text
D:\VisionOPD-data\raw\               # 官方压缩包，不进入 Git
E:\VisionOPD-subset\                  # 本地冻结图片子集，不进入 Git
Vision-OPD-main\
├── scripts\                           # 提取、校验、人工 QA 工具
├── tests\                             # 针对性测试
├── artifacts\data\                   # manifest、报告、统计和哈希
└── docs\                              # 工作简报

/root/autodl-tmp/data/vision_opd_1024/   # 服务器训练数据，不进入 Git
├── images/                         # 1,216 张 Student 全图
├── teacher_images/                 # 1,216 张 Teacher crop
├── manifests/                      # JSONL、哈希和构建报告
├── train_1024.parquet
├── eval_128.parquet
└── retention_64.parquet
```

## 3. 官方压缩包下载与哈希

实际下载了 Student 全图的 6 个分卷和 Teacher crop 的 1 个压缩包。Student 分卷必须按 `00` 到 `05` 的顺序作为一个连续 gzip/tar 流读取。

| 文件 | 大小（bytes） | SHA256 |
|---|---:|---|
| `images.tar.gz00` | 5,368,709,120 | `E72239BB03D393886E84AA2758EABCAD387B03E86A7D7CE7238178F5832F52D0` |
| `images.tar.gz01` | 5,368,709,120 | `3E0872E7AE37CC4B94019EC0F23AE274EC5C1F7DBEBFA087E6219E0DA9301979` |
| `images.tar.gz02` | 5,368,709,120 | `B0BD79677A7439E87745F39716DA57B85796DC241974032831EF67772F96BE1A` |
| `images.tar.gz03` | 5,368,709,120 | `4B230E8F814B22117126064E8EB6D4D7BA6F639860C6982A387CE8BB814E99EE` |
| `images.tar.gz04` | 5,368,709,120 | `D7872E12C1AB49CA4FD0773F1E6AA621A7C3EF8355CA7E72FC57E8CE6292CFC6` |
| `images.tar.gz05` | 1,496,961,022 | `44844E97BC43EE0BF8546AE50CA5B709DECB22C144BFDA410F3DDFE8C546060A` |
| `teacher_images.tar.gz` | 2,942,713,517 | `F2FB6541E8E1EA4E33114AFF9D511C5E8CE764C0972798151C8D5B1B5B91883E` |

7 个压缩包合计 31,283,220,139 bytes，约 29.13 GiB。下载完成时 `images.tar.gz00` 至 `images.tar.gz05` 全部存在，Teacher 包存在，没有 `.incomplete` 文件，也没有残留下载进程。

## 4. 确定性选择性提取

实现脚本：[选择性提取脚本](../scripts/extract_project_images.py)。

脚本不重新随机抽样，而是读取 Day 2 已冻结的三份 manifest：

| Split | 数量 | 用途 |
|---|---:|---|
| train | 1,024 | SFT、Vision-OPD、Cached Prefix，以及后续可验证 GRPO 子集 |
| eval | 128 | 固定主评测，禁止参与训练 |
| retention | 64 | 能力和格式保持检查 |
| **合计** | **1,216** | 本地项目子集 |

执行入口：

```powershell
python scripts\extract_project_images.py `
  --raw-dir "D:\VisionOPD-data\raw" `
  --subset-root "E:\VisionOPD-subset"
```

实际提取结果：

| 检查项 | Student 全图 | Teacher crop |
|---|---:|---:|
| 请求数量 | 1,216 | 1,216 |
| 在归档中找到 | 1,216 | 1,216 |
| 实际写入 | 1,216 | 1,216 |
| 缺失 | 0 | 0 |
| 重复归档成员 | 0 | 0 |
| 扫描归档成员 | 6,242 | 6,242 |

输出位置：

```text
E:\VisionOPD-subset\images\           # 1,216 张 Student 全图
E:\VisionOPD-subset\teacher_images\   # 1,216 张 Teacher crop
```

Student 图片合计 5,557,722,429 bytes，Teacher 图片合计 549,351,769 bytes；冻结图片子集合计 6,107,074,198 bytes，约 5.69 GiB。提取过程使用临时文件后原子改名，最终没有零字节文件或残留 `.partial` 文件。

提取报告：[extraction_report.json](../artifacts/data/extraction_report.json)。

## 5. 自动数据校验

实现脚本：[自动校验脚本](../scripts/validate_project_data.py)。

执行入口：

```powershell
python scripts\validate_project_data.py `
  --manifest-dir "artifacts\data" `
  --subset-root "E:\VisionOPD-subset" `
  --workers 4
```

自动检查覆盖：

- split 数量、`sample_id` 唯一性与跨 split `group_id` 隔离；
- manifest 中全图/crop 相对路径与 E 盘实际文件集合精确一致；
- 全图与 crop 核心文件标识一致，无重复引用；
- 2,432 张图片全部执行 SHA256 和 Pillow 完整解码；
- 图片格式、色彩模式和宽高有效；
- bbox 为四个整数、面积为正且不超出 Student 全图尺寸；
- `problem`、`answer` 非空，且 Prompt 中恰好包含一个 `<image>` 占位符。

实际结果：

| 检查项 | 结果 |
|---|---:|
| manifest 记录 | 1,216 |
| 唯一 `sample_id` | 1,216 |
| Student 图片 | 1,216 |
| Teacher 图片 | 1,216 |
| 完整解码与哈希图片 | 2,432 |
| 缺失文件 | 0 |
| 解码失败 | 0 |
| bbox 错误或越界 | 0 |
| 全图/crop 路径配对错误 | 0 |
| 自动校验问题总数 | 0 |
| 最终状态 | **PASS** |

统计摘要：

| 项目 | 结果 |
|---|---|
| 答案分布 | A=355，B=287，C=285，D=289 |
| Prompt 字符长度 | min=212，median=254，max=455；这是字符数，不是 tokenizer Token 数 |
| bbox 面积占比 | min=0.000081，median=0.010863，max=0.099733 |
| Student 尺寸 | 宽 1500～5304，中位数 2250；高 1500～3333，中位数 1500 |
| Teacher 尺寸 | 宽 26～6206，中位数 428；高 34～4946，中位数 431 |
| 图片格式和模式 | 2,432 张均为 PNG、RGB |

自动校验报告：[data_validation.json](../artifacts/data/data_validation.json)。统计文件：[data_stats.json](../artifacts/data/data_stats.json)。

自动报告明确保留 `semantic_alignment_checked=false`：自动校验能证明文件、解码、bbox 和路径配对，但不能证明图像内容、问题与答案在语义上正确。

## 6. 30 条人工语义 QA

实现工具：

- [人工 QA 抽样脚本](../scripts/prepare_manual_qa.py)
- [人工 QA 页面生成脚本](../scripts/render_manual_qa.py)

抽样使用 `seed=42`，不从30条中挑选容易样本。固定配额与分层如下：

| Split | 总数 | 小 bbox | 中 bbox | 大 bbox |
|---|---:|---:|---:|---:|
| train | 20 | 7 | 6 | 7 |
| eval | 5 | 2 | 1 | 2 |
| retention | 5 | 2 | 1 | 2 |
| **合计** | **30** | **11** | **8** | **11** |

每条人工检查：

1. Student 全图是否有清晰且位置合理的红框；
2. Teacher crop 是否对应红框区域；
3. 问题是否与目标区域匹配；
4. 标准答案是否与图像内容一致。

人工审查结果：

| 状态 | 数量 |
|---|---:|
| `pass` | 30 |
| `pending` | 0 |
| `suspected_badcase` | 0 |

最终人工 QA 文件：[manual_qa_30.jsonl](../artifacts/data/manual_qa_30.jsonl)。导出文件与归档文件逐记录一致，SHA256 为：

```text
F8F3FA0D2F3930535D9100037237B54FCBD3E2C6DA25ACE695D73FCE1006484B
```

该结果证明固定抽查的 30 条均通过，不表示人工逐条检查了全部 1,216 条，也不证明整个官方 6,241 条数据没有标注问题。

## 7. 关键产物哈希

| 文件 | SHA256 |
|---|---|
| `train_1024.jsonl` | `0721E9C57CE55C64B27617FB35C4E419D690A7E68AADCA8280E01CEB1E10EFCC` |
| `eval_128.jsonl` | `C591DFB74DBC883A270E53DDFEC2801AC62C21F3F3C9894911060F63E03EFBDD` |
| `retention_64.jsonl` | `FCF6969DC27C50BF3854A162FEA04378A3AAC6DF63CEF48BF75E4CAD71196BB3` |
| `split_manifest.json` | `DF25A12D0427E5B16D1DDCDB98A343C176434ADC018587504C3270474DA7BB05` |
| `candidate_stats.json` | `21798988501B842C3C14BAAD74D47E5B11ED543B10EA320EABDA42F6A6DDA24A` |
| `extraction_report.json` | `587BD3EEF15D101640AA59009617C83F6AD3AE88F965E7AC0053B636E51883D5` |
| `data_validation.json` | `FF913286F824B44DA6C94AFD01DBB8958FE432B23E1BF45E68291170895BFEEA` |
| `data_stats.json` | `4844D9F5B75EF1977E231A8E4353D2A50774FD875C7299BA06579FA0D94600EC` |
| `data_sha256.txt` | `5C64EF28AE56B7D0AD015D0ECF6FC6FCF158A47360B522871D0A96EB4A285D6D` |
| `manual_qa_30.jsonl` | `F8F3FA0D2F3930535D9100037237B54FCBD3E2C6DA25ACE695D73FCE1006484B` |

`data_sha256.txt` 包含 2,432 行，分别记录每张冻结 Student/Teacher 图片的 SHA256。上传服务器后必须重新计算并逐文件比较，不能只比较文件数量。

## 8. 代码与测试证据

本日新增：

```text
scripts/extract_project_images.py
scripts/validate_project_data.py
scripts/prepare_manual_qa.py
scripts/render_manual_qa.py
tests/test_extract_project_images.py
tests/test_validate_project_data.py
tests/test_manual_qa_tools.py
```

验证结果：

- 9 个针对性测试全部通过，新增 [Parquet 构建测试](../tests/test_build_project_parquet.py)；
- 覆盖分卷 gzip/tar 连续读取、manifest 契约、合法/越界 bbox、确定性分层抽样、HTML 数据转义、Parquet schema 与读回行数；
- `git diff --check` 通过；
- 服务器生成器和数据均位于外部数据目录，不将大文件纳入 Git。

## 9. 服务器 Parquet 与加载验收

使用 [Parquet 构建脚本](../scripts/build_project_parquet.py) 在服务器数据根目录执行构建。脚本不下载、不解压、不重新抽样，只接受已冻结的三份 JSONL manifest。

| 产物 | 行数 | 验收结果 |
|---|---:|---|
| `train_1024.parquet` | 1,024 | PASS |
| `eval_128.parquet` | 128 | PASS |
| `retention_64.parquet` | 64 | PASS |

服务器执行 `datasets.load_dataset("parquet", ...)` 后实际验证了每份 Parquet 的行数、必需字段，并逐条确认 `images` 与 `bbox_images` 中的 Linux 绝对路径存在。终端证据为 `SERVER_PARQUET_LOAD_GATE=PASS`。

此前，服务器已对 `manifests/data_sha256.txt` 完成图片校验：2,432 条图片记录全部通过，没有 `FAILED` 或缺失文件。

## 10. 完成与未完成边界

数据 Gate 已完成：

- 固定 1024/128/64 manifest 未重新抽样；
- 按 manifest 从官方压缩包选择性提取 1,216 对图片；
- 2,432 张图片自动校验与 SHA256 完成，问题数为 0；
- 30 条确定性分层人工 QA 完成，30 条均为 `pass`；
- 原始压缩包、manifest、提取报告、自动报告、统计、图片哈希和人工 QA 均可追溯。
- 图片子集已上传到服务器，服务器侧 2,432 张图片 SHA256 复核通过；
- Linux 路径 Parquet 已生成，并通过服务器实际加载验收。

以下事项尚未完成，不能因本日 PASS 而宣称完成：

- 尚未运行 Vanilla 基线、Cached Prefix 生成或任何训练；
- 尚未获得 checkpoint、训练 loss、模型能力提升或论文结果复现证据。

因此本日成果应表述为：

> 完成 Vision-OPD 固定 1,216 条项目子集的全图/crop 选择性提取、2,432 张图片自动解码与双端 SHA256 校验，并完成 30 条按 split 和 bbox 面积分层的人工语义 QA；已在服务器生成并实际加载 1024/128/64 三份 Parquet，数据 Gate 为 PASS，训练尚未开始。

## 11. 下一步

1. 冻结评测器、题型规则与 generation 参数；
2. 使用同一个训练前 Base checkpoint，在 `eval_128.parquet` 上运行 Vanilla 基线并保存逐样本预测；
3. 使用同一个 Base 为 `train_1024.parquet` 生成 Cached Prefix；
4. 在启动任何训练前，将 `run_vision_opd.sh` 改为外部数据路径、本地模型路径和两卡设置。
