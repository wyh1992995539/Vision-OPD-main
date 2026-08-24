# Vision-OPD 外部 Benchmark 冻结协议

> 实验 ID：`E-D5-001`  
> 协议状态：**FROZEN**  
> 冻结日期：2026-08-24（UTC）  
> 机器配置：[`configs/benchmark_eval.yaml`](../configs/benchmark_eval.yaml)  
> 配置 SHA256：`e99aa0c9b1e2bb9703d7d50bbe0c34be9f587be00c518baaeebc229e0f5da461`

## 1. 目的与适用范围

本协议冻结 ZoomBench、MMStar 和 V* Bench 的数据来源、版本、输入、生成、评分、Smoke、重叠审计和报告规则。它适用于以下四个模型分支：

1. Base / Vanilla；
2. Vision-OPD；
3. Cached Prefix；
4. GRPO。

四个分支必须使用相同的官方样本、Prompt、图像处理、生成参数、答案解析、Judge 和指标。除模型 checkpoint 外，不得为某一分支单独修改评测条件。

外部 Benchmark 只用于一次 Base 基线和各训练分支最终定版后的评测，不用于选择 checkpoint、调整训练超参数或根据准确率改变数据集。本文冻结的是本项目“4B、1024 条训练数据的小规模比较”所使用的外部评测协议，不代表对各 Benchmark 所有扩展指标的完整复现。

## 2. 冻结与变更规则

以下任一内容发生变化，均视为协议变化：

- 数据仓库、revision、split、样本集合或样本顺序；
- Prompt、system prompt、Chat Template 或答案格式要求；
- 图像来源、full/crop 视图、解码、缩放或像素上下限；
- generation seed、采样参数、上下文长度或最大输出长度；
- 答案提取、规则评分、Judge 模型、Judge Prompt 或 Judge 参数；
- 失败样本处理、分母、指标或分类汇总方式。

如确需变更，必须在本文追加带日期的 amendment，说明原因、影响范围、旧值、新值以及哪些历史结果失去可比性。不得覆盖或伪造原始协议记录。发生影响结果的变更后，三个 Benchmark 的 16 条 Smoke 必须全部重跑。

## 3. 官方来源与版本

远端 revision 于 2026-08-24 通过各仓库 `HEAD` 查询获得，并以完整 40 位 commit SHA 冻结。步骤 4 下载时必须显式传入这些 revision，禁止使用漂移的 `main`。

| Benchmark | 官方项目与协议代码 | 协议代码 revision | 官方数据 | 数据 revision | 官方预期规模 |
|---|---|---|---|---|---:|
| ZoomBench | [inclusionAI/Zooming-without-Zooming](https://github.com/inclusionAI/Zooming-without-Zooming) | `fdc0ba1a3dee916d8c38304d543ad414879e0c99` | [inclusionAI/ZoomBench](https://huggingface.co/datasets/inclusionAI/ZoomBench) | `b788097e57d30510c6877824833234a73bf80d25` | 845 |
| MMStar | [MMStar-Benchmark/MMStar](https://github.com/MMStar-Benchmark/MMStar) | `88f243ab4a39cb339530085c33aecb22819881a1` | [Lin-Chen/MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar) | `bc98d668301da7b14f648724866e57302778ab27` | 1,500 |
| V* Bench | [penghao-wu/vstar](https://github.com/penghao-wu/vstar) | `4ede6647959cfb59eeabd09286adf6a5f9478da0` | [craigwu/vstar_bench](https://huggingface.co/datasets/craigwu/vstar_bench) | `d9ae62c903da0c98336e85c5ee89cd863b04b4da` | 191 |

### 3.1 许可证边界

| 项目 | 数据许可证 | 代码许可证 | 本项目处理 |
|---|---|---|---|
| ZoomBench | Apache-2.0 | Apache-2.0 | 可按许可证用于研究评测并保留来源与许可信息 |
| MMStar | 官方数据页未明确声明 | 官方代码仓库未明确声明 | 允许本地研究评测；重新分发前必须另行复核，不自行推断许可证 |
| V* Bench | 官方链接的数据页未明确声明 | MIT | 允许本地研究评测；数据重新分发前必须另行复核 |

许可证未声明不等于没有限制。本项目产物默认只保存数据 revision、哈希、样本 ID、模型输出和汇总，不把完整官方图片或标注复制进代码仓库。

### 3.2 下载后必须完成的来源验证

上表规模来自官方页面，当前并未下载数据。步骤 4 下载后必须核对：

- revision 是否与本表完全一致；
- 实际 split、样本数、字段和原始文件名；
- 是否存在缺图、坏图、空问题、空答案或重复 ID；
- 原始文件和转换后数据的 SHA256。

任何实测差异都必须停止准备流程并记录，不能为了满足预期规模静默删除或补造样本。

## 4. 统一模型与推理协议

### 4.1 模型身份

Base / Vanilla 必须使用：

```text
/root/autodl-tmp/models/Qwen3.5-4B
```

冻结的 Base 权重哈希：

```text
model.safetensors-00001-of-00002.safetensors
26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61

model.safetensors-00002-of-00002.safetensors
cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188
```

每次正式评测必须重新计算实际 checkpoint SHA256。Vision-OPD、Cached Prefix 和 GRPO 使用各自独立训练分支的最终 checkpoint，不得串行继承。论文作者发布的官方 Vision-OPD 权重不能替代本项目 Base。

### 4.2 服务参数

| 参数 | 冻结值 |
|---|---|
| API | OpenAI-compatible Chat Completions |
| tensor parallel | 2 |
| dtype | bfloat16 |
| GPU memory utilization | 0.80 |
| max model length | 32,768 |
| max concurrent sequences | 8 |
| trust remote code | true |

`max_model_len=32768` 是本项目服务上限，用于容纳图像、Prompt 和官方评测配置允许的最长 8,192 Token 输出。模型本地配置支持更长上下文，但本项目不据此无限扩大评测资源。

### 4.3 System Prompt

统一使用：

```text
You are a helpful assistant. The assistant first thinks about the reasoning
process in the mind and then provides the user with the answer. The answer
is enclosed within <answer> </answer> tags, i.e., reasoning process here
<answer> answer here </answer>.
```

输入问题中的字面量 `<image>` 在构造多模态消息前移除。每条请求只传入一个当前协议指定的图像视图，标准答案、crop 图路径和另一视图不得泄露给被评模型。

### 4.4 图像处理

| 参数 | 冻结值 |
|---|---:|
| 每条请求图像数 | 1 |
| `min_pixels` | 65,536 |
| `max_pixels` | 16,777,216 |
| 保持宽高比 | 是 |
| Processor | 当前被评 checkpoint 自带 Processor |

图像必须先验证存在且可解码。不得为了让某个模型通过而单独压缩、裁剪或更换格式。ZoomBench full 与 crop 是两个独立评测视图，不能在一次请求中同时输入。

### 4.5 生成参数

| 参数 | 冻结值 |
|---|---:|
| seed | 42 |
| do sample | true |
| temperature | 0.7 |
| top-p | 0.8 |
| top-k | 20 |
| presence penalty | 1.5 |
| repetition penalty | 1.0 |
| max new tokens | 8,192 |
| return sequences | 1 |

这些值来自冻结 revision 下 ZoomBench 官方 `infer_without_tool.py` 的无工具评测配置。本项目把同一组参数用于三项 Benchmark，以保证四个项目模型之间使用同一生成协议。MMStar 和 V* 官方页面没有把这组参数声明为唯一强制值，因此最终报告应称其为“本项目冻结外评协议”，不能表述为所有 Benchmark 的唯一官方生成配置。

实际请求必须显式传入 seed。达到最大输出长度仍无法解析的样本记为无效预测并计错，不能增加单个样本的 Token 上限后补跑。

## 5. ZoomBench 协议

### 5.1 数据范围

- 官方数据：`inclusionAI/ZoomBench`；
- split：`test`；
- 预期样本数：845；
- 问题格式：选择题与开放题混合；
- 官方感知维度：6；
- 每条样本包含完整图和关键区域 crop，支持双视图协议。

六个官方维度为：

1. Fine-Grained Counting；
2. OCR；
3. Color Attributes；
4. Structural Attributes；
5. Material Attributes；
6. Object Identification。

步骤 4 转换时必须保存官方样本 ID 和题型；只有官方快照提供逐样本类别映射时才保存官方类别。冻结的 ZoomBench Parquet 只有 `id`、`query`、`response`、`bbox`、`question_type`、`image` 和 `crop_image`，未提供逐样本维度字段，官方转换脚本也未发布映射。因此本项目使用 `unavailable_official` 占位，禁止通过关键词、LLM 或人工猜测六维类别，也不计算 ZoomBench 官方分类准确率。

六个维度仅作为官方数据集级覆盖范围记录。ZoomBench 可复算的分解指标为 `question_format_accuracy`，分别报告 MCQ 与开放题。

若原始数据未提供显式题型字段，则只有同时出现至少两个带标签选项且标准答案可映射到唯一选项时，才归类为 `multiple_choice`；其余归类为 `open_question`。转换规则必须由测试覆盖，不能人工逐题选择评分器。

### 5.2 双视图和主指标

| 视图 | 输入 | 定位 | 指标名称 |
|---|---|---|---|
| Full image | 官方完整图 | 项目主结果 | `full_image_accuracy` |
| Cropped region | 官方关键区域 crop | oracle/诊断 | `cropped_region_accuracy` |

同时计算：

```text
zooming_gap = cropped_region_accuracy - full_image_accuracy
```

项目模型的主比较只使用 `full_image_accuracy`。crop 结果和 zooming gap 必须单独标为诊断，不得与 full 结果平均，也不得把 crop 结果称为单次完整图推理能力。

### 5.3 评分

选择题：

- 从 `</think>` 之后或最后一个完整 `<answer>...</answer>` 中提取最终答案；
- 只接受可唯一映射到合法选项的最终结论；
- 与标准选项精确比较；
- 无答案、歧义或越界选项计错；
- 不使用 LLM Judge 改判明确的错误选项。

开放题：

1. 先使用冻结版本的 MathRuler；
2. MathRuler 未判为正确的样本交给冻结的 Qwen Judge；
3. 保存 `judge_source=mathruler|llm`；
4. Judge 调用失败、输出不是唯一 `Yes`/`No` 或无法解析时计错并记录错误。

### 5.4 Judge

| 参数 | 冻结值 |
|---|---|
| 模型 | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| revision | `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` |
| dtype | bfloat16 |
| tensor parallel | 1 |
| temperature | 0 |
| max new tokens | 2,048 |

Judge Prompt：

```text
Your task is to judge whether the response expresses the same meaning as the answer of a question.
The question is: {question}
The answer is: {reference_answer}
The response is: {model_answer}
Please check and compare them and then judge. If the response is correct, your output should be Yes. Otherwise, your output should be No. Directly give me your output.
```

Judge 只允许用于 ZoomBench 开放题。Judge 本身的模型、revision、Prompt 或生成参数变化均属于协议变化。

### 5.5 报告指标

- full-image 总体准确率；
- crop 总体准确率；
- zooming gap；
- 六个官方维度分别在 full/crop 下的准确率；
- 无效输出率、Judge 调用率、推理错误数；
- 平均输出长度、平均与 P95 延迟。

## 6. MMStar 协议

### 6.1 数据范围

- 官方数据：`Lin-Chen/MMStar`；
- config/split：`val`；
- 原始文件：`mmstar.parquet`；
- 预期样本数：1,500；
- 题型：选择题；
- 6 个核心类别、18 个二级类别，每个核心类别官方预期 250 条。

转换时保存 `index`、`question`、`answer`、`category`、`l2_category` 和图像。问题和选项保持官方文本，不自行重排选项。

### 6.2 评分

- 使用确定性 A/B/C/D 最终选项解析；
- 只接受唯一、可靠的最终选项；
- 与官方答案字母精确比较；
- 解析失败或歧义计错；
- 禁止调用 LLM Judge；
- 所有失败样本保留在 1,500 条总分母中。

解析器不能简单取回答中的第一个大写字母，因为解释文本和选项复述可能包含多个字母。应优先解析 `<answer>`、显式 `Answer:` 或回复末尾的唯一结论，并保存原始输出和解析依据。

### 6.3 项目指标边界

本项目报告：

- image-input overall accuracy；
- 6 个核心类别准确率；
- 18 个二级类别准确率；
- 无效输出率与推理失败数。

MMStar 的完整 Multi-modal Gain（MG）和 Multi-modal Leakage（ML）协议还需要额外的无图 LVLM 和原始 LLM 无图运行。本项目当前没有冻结这些额外运行，因此不得声称完整复现 MG/ML，也不得在结果表中填造 MG/ML 数值。

## 7. V* Bench 协议

### 7.1 官方数据身份

V* 官方项目 `penghao-wu/vstar` 直接链接的数据仓库是 `craigwu/vstar_bench`。本项目冻结该仓库，而不是当前本地准备脚本使用的 `lmms-lab/vstar-bench` 镜像。

- split：`test`；
- 预期样本数：191；
- 题型：选择题；
- 类别：`direct_attributes`、`relative_position`。

步骤 4 必须从冻结 revision 读取数据并验证 191 条 ID、选项、label 和类别。不得用镜像数据即使其内容看似相同；如需证明镜像等价，应另做全字段和图像哈希对比，但正式来源仍保持官方链接仓库。

### 7.2 Prompt 与评分

若官方问题中尚无答案格式后缀，追加：

```text
Answer with the option's letter from the given choices directly.
```

若已存在完全相同的后缀，不重复追加。评分规则与 MMStar 相同：确定性解析 A/B/C/D、精确匹配、无效计错、禁止 LLM Judge。

### 7.3 报告指标

- 191 条总体准确率；
- `direct_attributes` 准确率；
- `relative_position` 准确率；
- 无效输出率和推理失败数；
- 平均输出长度、平均与 P95 延迟。

## 8. 16 条 Smoke 协议

### 8.1 用途

Smoke 只验证数据、图像、Prompt、推理、断点恢复、解析、Judge、汇总和成本记录是否贯通。Smoke 准确率不作为模型能力结论，也不得用于更换 Benchmark、修改 Prompt 或重新抽样。

### 8.2 抽样

每个 Benchmark 固定 16 条：

- seed：42；
- ZoomBench 按题型确定性分层，固定 12 条 MCQ 和 4 条开放题；
- MMStar 与 V* 按官方类别确定性分层；
- 类别内按稳定 sample UID 排序后确定性抽取；
- 最终 UID 写入 `artifacts/runs/E-D5-001/smoke_selection.json`；
- UID 清单在首次推理前冻结；
- 后续 Base、Vision-OPD、Cached Prefix 和 GRPO 复用同一清单。

如果某个类别数量不足，按固定类别顺序将剩余额度分配给仍有样本的类别，并在 selection manifest 中记录。不得通过查看答案或模型输出影响抽样。

### 8.3 Smoke Gate

每个 Benchmark 必须同时满足：

- 16 条记录、16 个唯一 sample UID；
- 图片存在且可解码；
- Prompt 和标准答案非空；
- 请求中没有标准答案或未使用视图泄漏；
- 16 条都有原始响应或明确错误记录；
- 无未记录的空响应、图片错位或请求错误；
- 每条都有解析结果和 `judge_source`/`score_source`；
- 中断后恢复不重复、不漏样本；
- `summary.json` 可由逐样本结果重新计算；
- 错误和无效样本保留在 16 条分母中。

## 9. 数据重叠审计协议

### 9.1 比较范围

项目侧检查：

```text
/root/autodl-tmp/data/vision_opd_1024/train_1024.parquet
/root/autodl-tmp/data/vision_opd_1024/eval_128.parquet
/root/autodl-tmp/data/vision_opd_1024/retention_64.parquet
```

对三个 split 的完整图和 crop 图分别与三个 Benchmark 的所有评测图比较。文本比较覆盖问题文本；如官方数据保留来源 ID，也应一并比较来源 ID。

### 9.2 三层检查

1. **文件 SHA256**：完全相同则标记 `exact_image_match`。
2. **规范化问题文本**：使用 Unicode NFKC、casefold、首尾去空白和连续空白折叠；完全相同则标记 `normalized_question_match`。
3. **感知哈希**：图像解码并处理 EXIF 方向后计算 64-bit pHash；Hamming distance ≤ 5 标记 `suspected_perceptual_match`，必须人工确认。

感知哈希命中不能自动认定泄漏，也不能自动删除官方样本。报告必须区分：

- 已确认精确重复；
- 疑似感知重复；
- 文本重复但图像不同；
- 未确认候选。

### 9.3 发现重叠后的报告规则

- 保留并报告官方全量结果；
- 另行报告去重诊断，不替代官方结果；
- 不静默修改官方测试集；
- 有确认或未解决的疑似重叠时，不把结果称为“完全独立测试”；
- 保存每个候选的双方 sample ID、split、哈希、距离和人工结论。

## 10. 逐样本与汇总产物

每条预测至少保存：

```text
benchmark
sample_uid
official_category (when available; ZoomBench uses unavailable_official)
question_format
view
dataset_revision
image_sha256
model_checkpoint_sha256
prompt
raw_model_answer
parsed_answer
reference_answer
is_correct
score_source / judge_source
finish_reason
prompt_tokens
completion_tokens
latency_seconds
retry_count
error
```

每项 Benchmark 保存：

```text
predictions.jsonl
scores.jsonl
summary.json
```

`summary.json` 至少包含总体、题型分解和可用的官方分类指标、总分母、correct/incorrect/invalid/error、Judge 调用量、输出长度、延迟、GPU 时间和成本。ZoomBench 不得输出伪造的六维分类准确率。只有成功样本作为分母的结果无效。

统一产物根目录：

```text
artifacts/runs/E-D5-001/
```

## 11. 预算规则

双卡费用：

```text
双卡 GPU 成本 = 双卡 wall time（小时）× 11.96 元
```

Judge 成本按实际部署方式记录 GPU 时间，或按 API 输入/输出 Token 单价分别计算。Day 5 根据 48 条 Smoke 外推四次完整外评：Base、Vision-OPD、Cached Prefix、GRPO，并增加 15% 重试余量。

若预计训练与评测总成本超过项目 2,000 元硬上限，停止扩规模并重新审计非必要诊断；不得根据 Smoke 准确率删除主 Benchmark。ZoomBench crop 诊断可在预算不足时通过 amendment 延后，但 full-image 主结果、MMStar 和 V* 主结果不能被无记录地删除。

## 12. 当前本地实现审计与步骤 4 修复清单

本节记录协议冻结时的代码现状。以下差距意味着当前 `eval/run_eval.sh` **尚不能直接用于 Day 5 Smoke**。

| 文件 | 当前问题 | 步骤 4 要求 |
|---|---|---|
| `eval/prepare_data.py` | 三个 `snapshot_download()` 未传冻结 revision | 从 YAML 读取并显式传入 revision，保存远端 commit 与原始哈希 |
| `eval/prepare_data.py` | V* 使用 `lmms-lab/vstar-bench` 镜像 | 改为官方项目直接链接的 `craigwu/vstar_bench` |
| `eval/prepare_data.py` | 转换记录未统一保存 source ID、题型、revision 和图像哈希 | 补齐稳定 UID、题型、来源与哈希字段 |
| `eval/infer.py` | 接受 `--seed` 但请求未显式传 seed | 从 YAML 读取并把所有生成参数实际传给服务 |
| `eval/infer.py` | 当前请求固定 `temperature=0`，未实现冻结 system prompt 和图像像素范围 | 统一从 YAML 构造消息、Processor 和生成参数 |
| `eval/infer.py` | `normalize_model_answer()` 未用于最终记录，且缺少 token、finish reason、延迟等字段 | 同时保存原始回答、规范化回答和完整运行元数据 |
| `eval/judge_qwenlm.py` | MMStar/V* 规则失败后可能进入 LLM Judge | 两项选择题禁止 Judge；无效或错误答案直接计错 |
| `eval/judge_qwenlm.py` | Judge 输入输出路径硬编码 | 使用标准运行目录并保存 Judge revision、Prompt hash 和逐条来源 |
| `eval/cal_acc.py` | 主要输出到终端，ZoomBench/MMStar 分类汇总不足 | 生成可复算的 `summary.json` 和官方分类指标 |
| `eval/run_eval.sh` | 不读取 YAML、没有固定 16 条 manifest，默认可能运行全量 | 增加 config、sample manifest、run root 和 smoke/full 模式 |

修复顺序建议：

1. 先改数据准备与 revision；
2. 再实现固定 Smoke manifest；
3. 再让推理读取 YAML；
4. 分离选择题规则评分与 ZoomBench 开放题 Judge；
5. 最后统一结果目录和汇总。

在这些差距修复并通过测试前，不下载后立即运行全量评测，也不启动正式 Smoke。

## 13. 步骤 3 验收

步骤 3 在满足以下条件后标记完成：

- [x] 三个 Benchmark 的官方项目、官方数据和完整 revision 已记录；
- [x] split、官方预期样本数和许可证边界已记录；
- [x] Prompt、图像处理、生成参数和输出格式已冻结；
- [x] ZoomBench full/crop、混合题型和 Judge 已明确；
- [x] MMStar 的项目指标与完整 MG/ML 边界已明确；
- [x] V* 官方数据身份与镜像差异已明确；
- [x] Smoke、重叠审计、失败分母和预算规则已冻结；
- [x] 当前本地实现差距已形成步骤 4 修复清单；
- [ ] 下载后数据规模、字段、图片和原始文件 SHA256 已验证——此项属于步骤 4，不阻塞步骤 3。

结论：协议设计与审计已冻结；当前无需下载 Benchmark。下一阶段是步骤 4的数据下载、revision 强制、转换与哈希验证。

## 14. 参考来源

- ZoomBench 官方项目：<https://github.com/inclusionAI/Zooming-without-Zooming>
- ZoomBench 官方数据：<https://huggingface.co/datasets/inclusionAI/ZoomBench>
- MMStar 官方项目：<https://github.com/MMStar-Benchmark/MMStar>
- MMStar 官方数据：<https://huggingface.co/datasets/Lin-Chen/MMStar>
- V* 官方项目：<https://github.com/penghao-wu/vstar>
- V* 官方项目链接的数据：<https://huggingface.co/datasets/craigwu/vstar_bench>
- ZoomBench 官方 Judge 模型：<https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507>


## 15. Amendment A-2026-08-24-ZOOM-CATEGORY

- 日期：2026-08-24（UTC）；
- 原因：下载后字段审计确认，冻结 ZoomBench Parquet 与官方转换脚本均未提供逐样本六维类别映射；
- 旧值：`categories: 6`、必需 `category_accuracy`、Smoke 按类别分层；
- 新值：六维类别仅作为数据集级描述；逐样本类别标记为 `unavailable_official`；必需指标改为 `question_format_accuracy`；ZoomBench Smoke 固定为 12 条 MCQ 与 4 条开放题；
- 不变项：数据仓库、revision、845 条样本、样本顺序、full/crop 图像、Prompt、答案、生成参数和评分规则均不变；
- 可比性：此前若存在 ZoomBench 六维分类汇总或按推测类别抽取的 Smoke，均作废；总体、题型、full/crop 与 zooming gap 指标不因本 amendment 改变定义；
- 后续要求：重新生成 ZoomBench 转换 JSON、验证与 SHA256，并在首次推理前重新冻结三个 Benchmark 的 Smoke UID。禁止把派生或推测标签称为官方类别。

## 16. 任务 4：训练数据与 Benchmark 重叠审计结果

审计于 2026-08-24（UTC）完成，命令为：

```bash
python scripts/check_benchmark_overlap.py \
  --config configs/benchmark_eval.yaml \
  --benchmarks zoombench,mmstar,vstar
```

覆盖范围与数据质量：

- 项目 train/eval/retention 共 1,216 条样本、2,432 个 full/crop 图像引用；
- 三个 Benchmark 共 2,536 条样本、3,381 个图像引用；
- 共计算并缓存 5,813 个图像引用的 SHA256 与 64-bit DCT pHash；
- 缺图、空问题、重复 sample UID、图像指纹错误均为 0。

| Benchmark | 官方样本 | 候选 | 确认重叠 | 已排除 | 未确认 | 确认影响率 |
|---|---:|---:|---:|---:|---:|---:|
| ZoomBench | 845 | 1 | 0 | 1 | 0 | 0.000% |
| MMStar | 1,500 | 0 | 0 | 0 | 0 | 0.000% |
| V* Bench | 191 | 4 | 4 | 0 | 0 | 2.094% |

人工复核结论：

- V* 的 4 个候选均位于项目 `train` split。每对图像尺寸一致、pHash Hamming distance 为 0，RGB 像素差异仅稀疏集中在项目标注框附近，确认为相同底图；
- ZoomBench 的 1 个候选仅问题模板完全相同；项目答案为 B、Benchmark 答案为 A，四种 full/crop 交叉 pHash 距离为 32、34、36、32，确认为不同图像并排除；
- 最终未确认候选为 0。

报告规则：

- ZoomBench 与 MMStar 当前未检测到确认重叠；
- V* 必须保留并报告官方 191 条全量结果，同时另报排除 4 条受影响样本后的 187 条诊断结果；
- V* 结果不得称为“完全独立测试”；诊断结果不得替代官方全量结果；
- 训练数据或 Benchmark revision 变化时必须重跑审计。

主要产物位于 `artifacts/runs/E-D5-001/overlap/`：

- `overlap_report.json`：机器可读汇总；
- `overlap_candidates.jsonl`：逐候选证据；
- `manual_review_decisions.json`：可复现人工裁决；
- `manual_review.csv`：审阅表；
- `overlap_report.md` 与 `overlap_report.html`：人类可读报告；
- `image_fingerprint_cache.json`：可续跑指纹缓存。

任务 4 验收：

- [x] 文件 SHA256、规范化问题文本和 64-bit pHash 三层检查已实现；
- [x] 项目 full/crop 与三个 Benchmark 的全部评测图已覆盖；
- [x] 候选证据、人工结论和未确认状态可区分；
- [x] 已执行完整审计，数据/指纹错误为 0；
- [x] 已冻结发现重叠后的官方全量与去重诊断报告规则；
- [x] `tests/test_benchmark_overlap.py` 已覆盖规范化、哈希、候选分类、人工裁决与端到端产物。

结论：Day 5 任务 4 已完成。下一项是任务 5：生成固定的三项 Benchmark Smoke 样本清单。

## 17. Amendment B-2026-08-24-SMOKE-SELECTION

- 日期：2026-08-24（UTC）；
- 原因：步骤 5 需要让后续 Base、Vision-OPD、Cached Prefix 与 GRPO 使用完全一致、与文件顺序无关的 Smoke 样本；
- 旧值：仅规定按题型/类别分层和 seed 42，类别内“按 sample UID 排序”的 seed 使用方式不够精确，MMStar 与 V* 未写明固定配额；
- 新值：类别内使用 `SHA256("42:{benchmark}:{sample_uid}")` 的十六进制摘要升序排名，按 `configs/benchmark_eval.yaml` 的显式配额截取：
  - ZoomBench：12 条 `multiple_choice`、4 条 `open_question`；
  - MMStar：coarse perception、fine-grained perception、instance reasoning、logical reasoning 各 3 条，math、science & technology 各 2 条；
  - V* Bench：direct_attributes、relative_position 各 8 条；
- 已冻结产物：`artifacts/runs/E-D5-001/smoke_selection.json`，SHA256 为 `dc5856cf6563e5b4a341f5131fcb33785ea36efd3c4ac7f239aebb428e0a392b`；
- 配置 revision：3；配置 SHA256 为 `e99aa0c9b1e2bb9703d7d50bbe0c34be9f587be00c518baaeebc229e0f5da461`；
- 选择前检查：所有 48 条均验证了 source revision、非空 UID/问题/答案、图像存在及可解码、分层配额和 UID 唯一性；
- overlap：本次确定性选择没有命中已确认的 V* 重叠样本；该事实只是 manifest 元数据，不能成为替换或重抽样理由；
- 不变项：官方数据 revision、完整评测集合、Prompt、图像处理、生成、评分和 Judge 协议均不变；
- 后续要求：Base、Vision-OPD、Cached Prefix 和 GRPO 都必须复用此 manifest；如重新抽样、改 seed、改配额或改变选择排序，必须创建新的 amendment 且历史 Smoke 不再可比。

选择器实现为 `scripts/select_benchmark_smoke.py`，默认拒绝覆盖已冻结 manifest；仅在显式 `--force` 后才能重建。


## 18. Amendment C-2026-08-24-ZOOMBENCH-SCORING

- 日期：2026-08-24（UTC）；
- 原因：ZoomBench Smoke 暴露出 5 条未决开放题，其中 4 条的参考答案与抽取出的最终答案都是完整单数字且数值明显不等。让 LLM Judge 判断这类确定性错误既增加资源消耗，也引入不必要的随机性；原冻结 30B Judge 对本项目规模还需要不成比例的约 32 GiB 本地存储。
- 旧值：`MathRuler -> Qwen3-30B-A3B-Instruct-2507 Judge`；
- 新值：`确定性单数字比较 -> MathRuler -> 固定 Base Qwen3.5-4B 语义 Judge`。

确定性规则冻结如下：只有参考答案和从 `<answer>` 中抽取的最终回答在去除首尾空白后，都完整匹配带可选正负号、可选千位分隔符的整数或十进制数时才启用；使用十进制精确值比较，相等计对、不等计错。数字嵌在句子中、范围、分数、单位、多个候选数字或缺失最终答案均不由此规则裁决。单数字不等时不得再交给 MathRuler 或 LLM Judge 覆盖。非适用项先由 MathRuler 判定；仅 MathRuler 未确认的语义未决项进入固定 Base 4B。

固定 Judge 身份与生成约束：

- 模型：`/root/autodl-tmp/models/Qwen3.5-4B`，服务名 `vision-opd-base`；
- 权重 SHA256：分片 1 为 `26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61`，分片 2 为 `cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188`；
- 所有 Base、Vision-OPD、Cached Prefix 与 GRPO 分支复用同一固定 Base Judge，禁止随被测分支更换 Judge；
- `temperature=0`、`enable_thinking=false`、`max_new_tokens=64`，system prompt 要求只输出 `Yes` 或 `No`；其他输出按失败计错并记录原文；
- 该结果必须标记为 `project_frozen_base_4b_judge`，不得声称采用 ZoomBench 官方 30B Judge，也不与旧 Judge 协议结果直接横向比较。

本 amendment 只改变评分协议，不改变数据 revision、48 个冻结 UID、full/crop 图像、被测模型 Prompt、图像预处理或推理生成参数。因此不重抽样，原 selection manifest SHA256 `dc5856cf6563e5b4a341f5131fcb33785ea36efd3c4ac7f239aebb428e0a392b` 继续有效；运行器改为验证抽样 seed、排序规则、分层配额、样本数和数据 revision，而不是把整份配置 SHA256 当作样本身份。

配置升至 revision 4，`configs/benchmark_eval.yaml` SHA256 为 `d50d420d760fa59bd8a139fa4615aed8a4b41c79ca969d5f194e95c2ad6c25b6`。重评分后，7 条开放题由确定性数字规则裁决，当前 Smoke 没有 MathRuler 命中，1 条真正语义未决项由 Base 4B 输出严格的 `No`，Judge 失败和待定均为 0。最终结果：MMStar full `14/16 = 87.50%`，V* full `15/16 = 93.75%`，ZoomBench full `5/16 = 31.25%`，ZoomBench crop `10/16 = 62.50%`，zooming gap 为 `+31.25` 个百分点。

被中断的 `/root/autodl-tmp/models/Qwen3-30B-A3B-Instruct-2507` 下载目录仅含约 1.8 MiB 配置/分词器文件且无权重分片，已在确认与现有 Base 目录分离后删除；`/root/autodl-tmp/models/Qwen3.5-4B`（8.8 GiB）未受影响。

## 19. Day 5 Task 6: 完整评测预算决策与 Day 6 启动 Gate

- 日期：2026-08-24（UTC）；
- 冻结输入：`budget_inputs.json` SHA256 为 `9f03f8c83108319e248753442ac080f2385dd2e70015fee7acea9ab39bbefe61`；`cost.json` 与 `full_eval_budget.md` 的 SHA256 已写入 `artifacts/runs/E-D5-001/budget_artifacts.sha256`；
- 定价来源：用户确认的 AutoDL 双卡实例价格为 `11.96 CNY / hour`。该价格单位已经是双卡实例小时，费用计算不得再次乘以 GPU 数；
- 工作量：完整 Day 6 运行固定为 3,381 个请求，其中 ZoomBench full/crop 为 1,690、MMStar 为 1,500、V* Bench 为 191；预计语义 Judge 为 56 条，理论上限为 448 条；
- 三档预算：实测吞吐为 2.18 小时 / 26.11 元；**保守执行预算为 3.61 小时 / 43.18 元**；最坏护栏为 5.99 小时 / 71.70 元；
- 决策：Day 6 获准按保守预算启动，并冻结单次完整外部 Base 评测的运行费用上限为 **100.00 元**。该上限包含推理、重试和 Judge；它低于最坏护栏外加余量，也远低于项目 2,000 元硬上限。

### 19.1 Day 6 启动条件

启动 `E-D6-001` 前必须全部满足：

1. 对 `configs/benchmark_eval.yaml`、`smoke_selection.json`、`budget_inputs.json`、`cost.json` 和 `budget_artifacts.sha256` 完成 SHA256 校验；配置 revision 必须仍为 4，且不得变更数据 revision、48 条 Smoke UID、Prompt、图像处理、生成参数、评分/Judge 协议；
2. 核验实际加载的是两分片 SHA256 与冻结值一致的原始 `Qwen3.5-4B` Base；不得加载官方 Vision-OPD 权重、训练后权重或替代 Judge；
3. vLLM 必须以冻结的双卡参数启动：`vision-opd-base`、tensor parallel size 2、bfloat16、GPU memory utilization 0.80、max model length 32768、max num seqs 8、seed 42；服务健康检查通过后才可提交第一个请求；
4. 三个转换后 Benchmark 文件存在且其冻结 SHA256 与数据 manifest 一致；运行目录可写，且可续跑预测文件、评分文件与错误记录均启用；
5. AutoDL 双卡小时价格仍为 11.96 元；若价格、硬件数量、模型服务参数或预算输入任一变化，必须先创建新的 amendment 并重新生成 `cost.json`，不得沿用本决策。

### 19.2 运行中止与交付条件

- 从第一个 Day 6 请求开始记录墙钟时间与累计实例费用；当预计或实际费用达到 80 元时暂停检查，当达到 **100 元** 时立即停止新的请求、保留 checkpoint/逐条记录并调查，不能按成功子集报告分数；
- 任何未记录的 API 错误、图片错位、空响应、重复 request key、配置/模型哈希不一致，或 Judge 输出未按冻结规则记录时，暂停运行并修复后以断点恢复；所有失败样本仍在官方分母中；
- 完成标准：ZoomBench 845 条的 full/crop、MMStar 1,500 条、V* Bench 191 条全部有逐条输入标识、原始输出、解析/评分来源、延迟和错误字段；V* 同时报官方全量 191 条与排除 4 个已确认重叠样本后的 187 条诊断，不将后者替代官方结果；
- Day 6 结束后，写入 `artifacts/runs/E-D6-001/` 的预测、评分、汇总、成本、模型/数据哈希和人类可读报告；不得覆盖 Day 4 的内部 128 条基线或 Day 5 Smoke 结果。

