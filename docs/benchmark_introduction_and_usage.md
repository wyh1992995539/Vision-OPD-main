# ZoomBench、MMStar 与 V* Bench 介绍及 R3 使用指南

> 适用协议：E-PAPER-BASEJUDGE-001 / R3 单卡
> 唯一配置：configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml
> 配置 SHA256：e71255e817b11c120b4ac22d7ace81d12ffe01e25f7ea94de2e2ffb62e592903

## 1. 三个 Benchmark 分别测什么

三个 Benchmark 不是重复测试同一种能力，而是从不同角度检查视觉语言模型。

| Benchmark | 核心问题 | 本项目主样本数 | 题型 | 主指标 |
|---|---|---:|---|---|
| ZoomBench | 模型能否从完整大图中识别细小目标和细粒度属性 | 845 | 621 选择题 + 224 开放题 | full-image accuracy |
| MMStar | 模型是否保持覆盖感知、推理、数学和科学的通用多模态能力 | 1,500 | 全部选择题 | overall accuracy + 六大类/二级类别 accuracy |
| V* Bench | 模型能否在大图中搜索目标，并判断目标属性与相对位置 | 191 | 全部选择题 | overall accuracy + 两类 accuracy |

R3 正式主评测只输入完整图，共 845 + 1,500 + 191 = 2,536 个视觉请求。所有失败、空输出、无法解析和最终 Judge 失败都保留在固定分母并计错。

## 2. ZoomBench

### 2.1 内容与能力

ZoomBench 面向“无需显式调用裁剪工具的细粒度视觉理解”。图片通常包含尺寸较小、位置不显眼或需要局部观察的目标，问题要求模型从完整图中定位并识别细节。

其覆盖方向包括：

- Fine-Grained Counting：精确计算局部对象数量；
- OCR：读取较小文字、号码或标识；
- Color Attributes：判断对象颜色或局部颜色属性；
- Structural Attributes：判断形状、结构和组成；
- Material Attributes：判断材质或表面属性；
- Object Identification：识别小目标、局部部件或具体对象。

冻结数据共有 845 条：

- multiple_choice：621 条；
- open_question：224 条；
- 每条通常同时带 full image、目标 bbox 和 oracle crop。

当前官方快照没有可靠的逐样本六维类别字段，所以本项目不能臆造六维分类准确率。可复算的官方分解是选择题和开放题两种 question format。

### 2.2 本项目使用方式

R3 主评测只输入 full image。模型必须在没有 oracle bbox、没有 crop 图、没有定位提示的情况下回答。

ZoomBench 的 crop 图只适合另行进行 oracle 诊断：

- full accuracy 表示实际完整图能力；
- crop accuracy 表示目标区域已经提供后的识别上限；
- zooming gap = crop accuracy − full accuracy；
- crop 和 gap 不进入 R3 主结果，也不能与 full 平均。

本项目当前 R3 Base：

- full：428/845，50.65%；
- multiple choice：347/621，55.88%；
- open question：81/224，36.16%。

### 2.3 评分指标

总体准确率：

accuracy = 正确样本数 / 845

题型准确率：

- multiple-choice accuracy = 正确选择题 / 621；
- open-question accuracy = 正确开放题 / 224。

R3 实际评分顺序：

1. 对模型最终输出运行 MathRuler；
2. ZoomBench 不使用 MMStar/V* 的首字母直接匹配分支；
3. MathRuler 未确认正确的样本交给固定 Qwen3.5-4B Base Judge；
4. Judge 只接受规范化后唯一的 Yes 或 No；
5. API 失败、空输出或最终无效 Judge 输出计错。

因此 ZoomBench 的得分既覆盖精确数值/文本匹配，也覆盖语义等价回答。

## 3. MMStar

### 3.1 内容与能力

MMStar 是通用多模态选择题 Benchmark，强调题目需要真实视觉信息，而不是只靠语言先验猜答案。冻结 val split 共 1,500 条，六个一级类别各 250 条：

1. Coarse Perception：场景、对象和整体内容识别；
2. Fine-Grained Perception：局部属性、细节、OCR、数量等精细感知；
3. Instance Reasoning：围绕具体对象、属性和对象间关系推理；
4. Logical Reasoning：图表、序列、常识与多步逻辑推理；
5. Math：几何、数量、算术和视觉数学问题；
6. Science & Technology：物理、生物、化学、工程、地理等视觉科学题。

数据还带二级类别，本项目 summary.json 会同时输出一级 official_category 和二级 official_l2_category，便于定位模型具体在哪类能力上提升或退化。

### 3.2 本项目使用方式

每条请求输入一张官方完整图和带选项的问题。R3 不传 system prompt，关闭 thinking，要求模型按原问题作答。

所有 1,500 条都留在分母。不能因为模型没有输出 A–D、输出过长、服务失败或 Judge 失败而从分母删除。

本项目当前 R3 Base：

| 一级类别 | 正确/总数 | Accuracy |
|---|---:|---:|
| Coarse Perception | 204/250 | 81.60% |
| Fine-Grained Perception | 177/250 | 70.80% |
| Instance Reasoning | 192/250 | 76.80% |
| Logical Reasoning | 202/250 | 80.80% |
| Math | 204/250 | 81.60% |
| Science & Technology | 147/250 | 58.80% |
| Overall | 1126/1500 | 75.07% |

### 3.3 评分指标

总体准确率：

accuracy = 正确样本数 / 1,500

一级类别准确率：

category accuracy = 该一级类别正确数 / 250

二级类别准确率：

l2 accuracy = 该二级类别正确数 / 该类别固定样本数

R3 评分顺序：

1. MathRuler；
2. 若模型答案能唯一映射到正确选项字母，first-letter match 判对；
3. 尚未判对或未解决的输出进入固定 Base Judge；
4. 最终无效输出按错计。

分类汇总的分母来自官方字段，不能自行重分组。

## 4. V* Bench

### 4.1 内容与能力

V* Bench 重点评估视觉搜索能力：目标可能只占完整图很小区域，模型需要先在大图中找到目标，再读取属性或判断目标之间的关系。

冻结 test split 共 191 条，分为：

- Direct Attributes：115 条。寻找目标后判断颜色、类别、状态、外观等直接属性；
- Relative Position：76 条。寻找一个或多个目标后判断上下、左右、邻接等相对位置关系。

它与 ZoomBench 的区别是：ZoomBench 更广泛地覆盖细粒度识别、OCR、计数和开放题；V* 更集中于大图目标搜索及目标属性/空间关系，而且当前 191 条均为选择题。

### 4.2 本项目使用方式

R3 使用 lmms-lab/vstar-bench 的冻结 revision：

b44023b4dca749ed8a76b85eb576627d05a1c174

每张请求图始终：

1. 解码；
2. 转换为 RGB；
3. 编码为 PNG；
4. 只有编码后超过 20 MiB 才按冻结实现缩小。

问题末尾带固定提示，要求直接输出选项字母。全部 191 条必须保留，分母固定为 191。

Day 5 审计确认其中 4 张图与本项目 train split 使用同底图。按当前项目决定，这 4 条不删除，不另报 187 条指标，后续所有模型都按相同“不重复假设”使用官方 191 分母。历史 overlap 证据继续保留。

本项目当前 R3 Base：

- Overall：160/191，83.77%；
- Direct Attributes：97/115，84.35%；
- Relative Position：63/76，82.89%。

### 4.3 评分指标

总体准确率：

accuracy = 正确样本数 / 191

类别准确率：

- direct_attributes accuracy = 正确数 / 115；
- relative_position accuracy = 正确数 / 76。

评分链与 MMStar 相同：MathRuler → 正确选项首字母匹配 → 固定 Base Judge。失败样本仍保留在 191 分母。

## 5. 三项评测的统一 R3 推理协议

| 配置项 | 冻结值 |
|---|---|
| system prompt | 无 |
| message role | 单个 user |
| 图像数量 | 1 张 full image |
| enable_thinking | false |
| temperature | 0 |
| max_tokens | 1024 |
| 禁止请求参数 | seed、top-p、top-k、presence penalty、repetition penalty |
| inference workers | 16 |
| inference retries | 最多 3 次 |
| 服务 | OpenAI-compatible Chat Completions |
| tensor parallel | 1 |
| GPU memory utilization | 0.75 |
| GDN prefill | Triton |
| Judge | 冻结原始 Qwen3.5-4B Base |
| Judge max_tokens | 2048 |
| Judge workers/retries | 16 / 最多 3 次 |

Base、Vision-OPD、Cached Prefix 和 GRPO 只能改变被测 checkpoint、模型角色、权重哈希和独立输出目录。

## 6. 哪些阶段需要 GPU

| 阶段 | GPU | 说明 |
|---|---|---|
| 数据准备 | 不需要 | 已完成时不要重复使用 force |
| 配置/数据验证 | 不需要 | 校验样本数、图像哈希和配置 |
| prepare-only | 不需要 | 冻结 checkpoint、数据和输出清单 |
| 被测模型推理 | 需要 1 GPU | 运行 12 条 Smoke 或 2,536 条正式请求 |
| MathRuler/规则评分 | 不需要 | 本地 CPU 评分 |
| Base Judge | 需要 1 GPU | 重新加载原始 Base，不加载训练模型 |
| 汇总与最终 Gate | 不需要 | 生成 scores、summary、hash、validation |

单卡上不能同时常驻被测模型和 Base Judge。正确做法是：先完成被测模型推理并关闭服务，再加载原始 Base 运行 Judge。

## 7. 首次或数据重建时的准备命令

以下命令在项目根目录执行。

~~~bash
cd /root/autodl-tmp/Vision-OPD-main
source /root/miniconda3/bin/activate vision-opd

R3_CONFIG=configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml

sha256sum "$R3_CONFIG"

python eval/prepare_paper_aligned_primary_data.py \
  --config "$R3_CONFIG" \
  --benchmarks zoombench,mmstar

python eval/prepare_paper_aligned_vstar.py \
  --config "$R3_CONFIG"

python eval/validate_paper_aligned.py \
  --config "$R3_CONFIG" \
  --output artifacts/runs/E-PAPER-BASEJUDGE-001/preflight/r3_data_recheck.json
~~~

sha256sum 必须输出：

~~~text
e71255e817b11c120b4ac22d7ace81d12ffe01e25f7ea94de2e2ffb62e592903
~~~

当前数据已经准备并验证完成，正常评测不需要再次运行两个 prepare 脚本。除非明确重建冻结数据，否则不要传 --force。

## 8. 新 checkpoint 的推荐完整流程

下面以 Vision-OPD 为例。Cached Prefix 和 GRPO 只需替换角色、checkpoint、服务名和输出目录。正式目录创建前，脚本会自动校验 R3 配置 SHA，并把配置、amendment、三份数据、请求契约、分母和恢复键与冻结 Base `run_manifest.json` 比较；任一差异都会直接拒绝运行。

### 8.1 定义本次身份

~~~bash
cd /root/autodl-tmp/Vision-OPD-main
source /root/miniconda3/bin/activate vision-opd

R3_CONFIG=configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml
TARGET_ROLE=vision_opd
TARGET_CKPT=/绝对路径/到/最终合并后的Vision-OPD-checkpoint
TARGET_MODEL_ID=vision-opd-r3-eval
TARGET_SMOKE=artifacts/runs/E-PAPER-BASEJUDGE-001/smoke_r3/vision_opd
TARGET_FORMAL=artifacts/runs/E-PAPER-BASEJUDGE-001/vision_opd
BASE_CKPT=/root/autodl-tmp/models/Qwen3.5-4B
JUDGE_MODEL_ID=vision-opd-base-judge
~~~

这些 shell 变量不会跨终端自动共享。每个新终端都要重新执行所需变量定义，或者把命令中的变量替换成明确的绝对路径和模型名。

checkpoint 必须是 vLLM 可以独立加载的完整或已合并模型目录，并包含模型配置、Tokenizer 和权重。不要把未合并 LoRA/训练临时目录直接当正式 checkpoint。

### 8.2 先做 4×3 条 Smoke

Smoke 每个 Benchmark 取 4 条，共 12 个视觉请求。

先在不启动 GPU 服务时冻结清单：

~~~bash
python eval/run_paper_aligned_eval.py \
  --config "$R3_CONFIG" \
  --model-role "$TARGET_ROLE" \
  --model-path "$TARGET_CKPT" \
  --model-id "$TARGET_MODEL_ID" \
  --output-dir "$TARGET_SMOKE" \
  --limit-per-benchmark 4 \
  --prepare-only
~~~

### 8.3 启动被测模型服务

在终端 A：

~~~bash
cd /root/autodl-tmp/Vision-OPD-main
source /root/miniconda3/bin/activate vision-opd

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

vllm serve "$TARGET_CKPT" \
  --served-model-name "$TARGET_MODEL_ID" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.75 \
  --trust-remote-code \
  --additional-config '{"gdn_prefill_backend":"triton"}' \
  --host 127.0.0.1 \
  --port 8000
~~~

R3 故意不显式覆盖 dtype、max-model-len 和 max-num-seqs，不能自行添加这些参数。

服务就绪后，在终端 B 检查：

~~~bash
curl http://127.0.0.1:8000/v1/models
~~~

### 8.4 运行 Smoke 推理

终端 B：

~~~bash
python eval/run_paper_aligned_eval.py \
  --config "$R3_CONFIG" \
  --model-role "$TARGET_ROLE" \
  --model-path "$TARGET_CKPT" \
  --model-id "$TARGET_MODEL_ID" \
  --output-dir "$TARGET_SMOKE" \
  --api-base http://127.0.0.1:8000/v1 \
  --limit-per-benchmark 4
~~~

完成后在终端 A 按 Ctrl-C 关闭被测模型服务。

### 8.5 加载固定 Base Judge

终端 A：

~~~bash
vllm serve "$BASE_CKPT" \
  --served-model-name "$JUDGE_MODEL_ID" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.75 \
  --trust-remote-code \
  --additional-config '{"gdn_prefill_backend":"triton"}' \
  --host 127.0.0.1 \
  --port 8000
~~~

终端 B：

~~~bash
python eval/run_paper_aligned_judge.py \
  --config "$R3_CONFIG" \
  --input-dir "$TARGET_SMOKE" \
  --judge-model-id "$JUDGE_MODEL_ID" \
  --judge-model-path "$BASE_CKPT" \
  --api-base http://127.0.0.1:8000/v1
~~~

Judge 完成后关闭 Base 服务。

### 8.6 Smoke 评分和 Gate

~~~bash
python eval/score_paper_aligned.py \
  --config "$R3_CONFIG" \
  --input-dir "$TARGET_SMOKE"

python eval/validate_paper_aligned.py \
  --config "$R3_CONFIG" \
  --run-dir "$TARGET_SMOKE" \
  --skip-data \
  --output "$TARGET_SMOKE/validation.json"
~~~

validation.json 的 status 和 run.status 都必须为 pass。Smoke 失败时先修复工程问题，不能用 Smoke 准确率选择 checkpoint 或修改协议。

## 9. 正式 2,536 条评测

Smoke PASS 后，重复完全相同的三阶段流程，但：

- 输出目录改为 TARGET_FORMAL；
- 删除 --limit-per-benchmark 4；
- 不与 Smoke 目录混用；
- 不覆盖已有 Base 目录。

### 9.1 正式清单与被测模型推理

~~~bash
python eval/run_paper_aligned_eval.py \
  --config "$R3_CONFIG" \
  --model-role "$TARGET_ROLE" \
  --model-path "$TARGET_CKPT" \
  --model-id "$TARGET_MODEL_ID" \
  --output-dir "$TARGET_FORMAL" \
  --prepare-only
~~~

启动被测模型服务后：

~~~bash
python eval/run_paper_aligned_eval.py \
  --config "$R3_CONFIG" \
  --model-role "$TARGET_ROLE" \
  --model-path "$TARGET_CKPT" \
  --model-id "$TARGET_MODEL_ID" \
  --output-dir "$TARGET_FORMAL" \
  --api-base http://127.0.0.1:8000/v1
~~~

完成后关闭被测模型，加载固定 Base Judge。

### 9.2 正式 Judge

~~~bash
python eval/run_paper_aligned_judge.py \
  --config "$R3_CONFIG" \
  --input-dir "$TARGET_FORMAL" \
  --judge-model-id "$JUDGE_MODEL_ID" \
  --judge-model-path "$BASE_CKPT" \
  --api-base http://127.0.0.1:8000/v1
~~~

完成后关闭 Base 服务。

### 9.3 正式评分与最终 Gate

~~~bash
python eval/score_paper_aligned.py \
  --config "$R3_CONFIG" \
  --input-dir "$TARGET_FORMAL"

python eval/validate_paper_aligned.py \
  --config "$R3_CONFIG" \
  --run-dir "$TARGET_FORMAL" \
  --skip-data \
  --output "$TARGET_FORMAL/validation.json"
~~~

正式 Gate 必须确认：

- predictions.jsonl：2,536 个唯一预测；
- scores.jsonl：2,536 个最终评分；
- Judge required 与 Judge completed 完全相等；
- ZoomBench full：845；
- MMStar full：1,500；
- V* full：191；
- pending Judge：0；
- 重复键和损坏 JSONL：0；
- validation status：pass。

## 10. 断点恢复方法

推理中断时，不要删除输出目录。重新启动同一 checkpoint、同一 served model name，并原样重跑 run_paper_aligned_eval.py 命令。脚本会：

- 读取 predictions.jsonl；
- 按 benchmark、view、sample_uid 去重；
- 保留已成功样本；
- 重试错误或未完成样本；
- 结束时原子压缩文件。

Judge 中断时同理，重新运行同一 run_paper_aligned_judge.py 命令。它只补齐未完成的 Judge 记录。

如果模型路径、权重哈希、model role、model ID、配置哈希、数据哈希或输出规模发生变化，manifest 会拒绝继续，防止不同实验混写。此时必须使用新的独立输出目录。

不要在正式结果冻结后使用 --retry-finalized-failures 改写结果。该选项只可在结果尚未冻结、明确要重试最终 Judge 格式/API 失败时使用。

## 11. 如何读取结果

每个正式目录包含：

| 文件 | 含义 |
|---|---|
| run_manifest.json | 模型、配置、数据、请求参数和预期数量 |
| predictions.jsonl | 逐样本原始回答、token、延迟、错误和重试 |
| judge_results.jsonl | 逐样本 Judge prompt、输出、决定和错误 |
| scores.jsonl | 解析结果、规则来源、Judge 来源和最终正确性 |
| summary.json | 总体、题型、一级/二级类别结果 |
| resume_status.json | 推理完成和可恢复状态 |
| judge_resume_status.json | Judge 完成和失败状态 |
| metrics.json | token、延迟、触顶和推理错误 |
| cost.json | GPU 数、阶段墙钟和客户端阶段成本 |
| artifact_sha256.txt | 正式产物哈希 |
| validation.json | 最终自动 Gate |

主比较只读取 summary.json 中：

- groups/zoombench/full；
- groups/mmstar/full；
- groups/vstar/full。

能力诊断再读取 question_format_groups、official_category_groups 和 official_l2_category_groups。训练后模型与 Base 比较时应同时报告绝对准确率和百分点变化，并检查同一批样本的配对变化，而不是只看一个总体均值。

## 12. 使用边界

- 三项外部 Benchmark 不能用于训练中挑 checkpoint；
- 不得根据外部分数修改训练超参数后反复重跑；
- 不得删除失败样本或改用成功样本分母；
- 不得让训练后的模型为自己做 Judge；
- 不得把旧 E-D6-001 与 R3 结果直接作主比较；
- 不得把固定 Base Judge 结果描述为使用 GPT-OSS-120B；
- 不得宣称精确复现论文 Table 2；
- 后续所有模型必须继续使用相同 R3 配置哈希；自动可比性 Gate 必须为 PASS。

旧 E-D5/E-D6、R1/R2 文件的用途和哈希解释见 `docs/benchmark_history_index.md`。
