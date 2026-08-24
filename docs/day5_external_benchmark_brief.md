# Day 5 外部 Benchmark 协议、数据与 Smoke 工作简报

## 技术摘要

Day 5 的六项任务已经全部完成，最终状态是：三个外部 Benchmark 的来源、revision、许可边界、输入与评分协议已冻结；ZoomBench、MMStar、V* Bench 已下载、转换并通过规模和图像验证；训练数据重叠审计已完成；固定 48 个样本的端到端 Smoke 已完成 64 个实际请求；完整评测预算和 Day 6 启动 Gate 已冻结。

本日没有执行三个 Benchmark 的完整 Base 评测。Smoke 的职责是验证数据、图像、Prompt、推理、断点恢复、答案解析、Judge、汇总和预算链路，Smoke 准确率不能作为最终模型能力结论。完整 3,381 请求的 Base 基线属于 Day 6。

关键结论：

- 三个转换后数据集共 2,536 条样本，样本数、UID、问题、答案和图像验证均通过。
- 重叠审计发现 V* Bench 有 4/191 条与项目 train split 使用相同底图；后续必须同时报告官方 191 条全量结果和排除这 4 条后的 187 条诊断结果。
- Smoke 共 48 个唯一 UID；ZoomBench 同时运行 full/crop，因此实际为 64 个请求。64/64 均有响应或明确记录，API 错误、重试和重复 request key 都为 0。
- ZoomBench 评分最终冻结为“确定性单数字规则 → MathRuler → 固定 Base Qwen3.5-4B 语义 Judge”。7 条开放题由数字规则裁决，1 条由 Base 4B Judge 输出 No，未决和 Judge 失败均为 0。
- 完整评测按 11.96 元/双卡实例小时估算：保守预算 3.61 小时、43.18 元；最坏护栏 5.99 小时、71.70 元；单次 Day 6 运行止损线冻结为 100 元。
- 最终回归测试为 58 passed、16 subtests passed；协议、配置、数据、Smoke、重叠和预算产物均有 SHA256 记录。

## 六项任务均通过验收

| Day 5 任务 | 主要执行内容 | 状态 | 核心证据 |
|---|---|---|---|
| 1. 冻结外部评测协议 | 冻结官方来源、revision、许可、样本规模、Prompt、图像处理、生成参数、评分与 Judge | 完成 | [benchmark_protocol.md](benchmark_protocol.md)、[benchmark_eval.yaml](../configs/benchmark_eval.yaml) |
| 2. 审计旧评测实现 | 审计 prepare/infer/judge/cal_acc/run_eval 的协议偏差，形成修复清单并补充新入口 | 完成 | 协议第 12 节、[frozen_benchmark_data.py](../eval/frozen_benchmark_data.py)、[run_smoke.py](../eval/run_smoke.py) |
| 3. 下载、转换与验证 | 固定 revision 下载三个数据集，保留原始数据，生成统一 JSON/图像与哈希 | 完成 | [dataset_manifest.json](../artifacts/runs/E-D5-001/dataset_manifest.json)、[data_validation.json](../artifacts/runs/E-D5-001/data_validation.json) |
| 4. 重叠审计 | 对项目 1,216 条与 Benchmark 2,536 条做 SHA256、文本、pHash 和人工复核 | 完成 | [overlap_report.md](../artifacts/runs/E-D5-001/overlap/overlap_report.md) |
| 5. 固定 Smoke | 确定性抽取 48 UID，完成 64 请求、续跑、解析、Judge 与汇总 | 完成 | [smoke_selection.json](../artifacts/runs/E-D5-001/smoke_selection.json)、[summary.json](../artifacts/runs/E-D5-001/smoke/base/summary.json) |
| 6. 冻结完整预算 | 基于 Smoke 外推 3,381 请求的时间、token、Judge 和费用，建立 Day 6 Gate | 完成 | [cost.json](../artifacts/runs/E-D5-001/cost.json)、[full_eval_budget.md](../artifacts/runs/E-D5-001/full_eval_budget.md) |

## 范围、数据身份与指标定义

### 三个 Benchmark 的冻结身份

所有远端 revision 均使用完整 40 位 commit SHA，下载时禁止使用会漂移的 main。

| Benchmark | 官方项目 | 冻结数据 | 数据 revision | split | 样本数 | 许可边界 |
|---|---|---|---|---|---:|---|
| ZoomBench | inclusionAI/Zooming-without-Zooming | inclusionAI/ZoomBench | b788097e57d30510c6877824833234a73bf80d25 | test | 845 | 数据和代码 Apache-2.0 |
| MMStar | MMStar-Benchmark/MMStar | Lin-Chen/MMStar | bc98d668301da7b14f648724866e57302778ab27 | val | 1,500 | 官方数据页和代码仓库未明确许可；只做本地研究评测，重新分发前复核 |
| V* Bench | penghao-wu/vstar | craigwu/vstar_bench | d9ae62c903da0c98336e85c5ee89cd863b04b4da | test | 191 | 代码 MIT；数据页未明确许可，重新分发前复核 |

V* 使用官方项目直接链接的 craigwu/vstar_bench，而不是旧脚本默认的 lmms-lab/vstar-bench 镜像。这样可以避免来源身份和 revision 无法对应。

### 统一模型、图像与生成定义

Smoke 使用原始 Qwen3.5-4B Base，模型目录为 /root/autodl-tmp/models/Qwen3.5-4B。两份权重分片 SHA256 已写入配置；任何训练后权重或官方 Vision-OPD 权重都不能替代本 Base。

服务参数冻结为：

- OpenAI-compatible chat completions；
- served model name：vision-opd-base；
- tensor parallel size 2、bfloat16；
- GPU memory utilization 0.80；
- max model length 32,768、max num seqs 8；
- seed 42。

输入只包含一个评测视图的图像。ZoomBench 分别测试 full 与 crop；两者不能同时放进同一个请求。图像保持纵横比，像素范围冻结为 min_pixels 65,536、max_pixels 16,777,216。

被测模型生成参数为 temperature 0.7、top_p 0.8、top_k 20、presence penalty 1.5、repetition penalty 1.0、max new tokens 8,192。System Prompt 要求最终答案放入 answer 标签，解析器优先读取最后一个完整标签。

### 指标边界

- ZoomBench：报告 full accuracy、crop accuracy、crop−full zooming gap 和题型准确率。官方快照没有逐样本六维类别映射，因此不计算或宣称官方 category accuracy。
- MMStar：报告图像输入总体、一级类别和二级类别准确率。MG/ML 需要额外的无图 LVLM 与 base-LLM 实验，不纳入本项目 Day 5 主协议。
- V* Bench：报告总体和类别准确率；Day 6 必须同时提供 191 条官方全量结果与排除 4 条确认重叠后的 187 条诊断结果。
- 所有失败、无效或截断样本保留在官方分母中，禁止只按成功请求计算准确率。

## 任务 1：协议从草案推进到 revision 4

首先创建并冻结 benchmark_eval.yaml 与 benchmark_protocol.md，覆盖官方地址、完整 revision、许可、样本数、Prompt、图像处理、生成、解析、Judge、产物结构、重叠规则、Smoke 和预算规则。

协议在执行过程中通过三次正式 amendment 修订：

| Amendment | 触发证据 | 决策 |
|---|---|---|
| A：ZOOM-CATEGORY | 下载后的 ZoomBench Parquet 和官方转换代码均没有逐样本六维类别字段 | 六维只保留为数据集级描述；逐样本标为 unavailable_official；Smoke 改按题型分层 |
| B：SMOKE-SELECTION | 仅写 seed 42 不能完全定义跨文件顺序的抽样结果 | 使用 SHA256("42:{benchmark}:{sample_uid}") 升序排名，并显式冻结各层配额和 48 UID |
| C：ZOOMBENCH-SCORING | 5 条开放题未决中有 4 条是明确单数字不等；使用 30B Judge 成本过高且没有必要 | 改为确定性数字 → MathRuler → 固定 Base 4B，仅让真正语义未决项进入 LLM |

最终配置为 protocol revision 4，SHA256 为 d50d420d760fa59bd8a139fa4615aed8a4b41c79ca969d5f194e95c2ad6c25b6。

## 任务 2：旧评测代码审计暴露了协议漂移风险

旧实现可以运行通用 Benchmark，但不能直接满足本项目的冻结要求。审计发现的主要差距包括：

- prepare_data.py 下载时没有强制 revision，V* 还指向非冻结镜像；
- 转换记录缺少统一 sample_uid、source revision、题型和图像哈希；
- infer.py 虽接受 seed，但请求没有完整落实冻结生成参数、System Prompt 和图像像素约束；
- judge_qwenlm.py 可能让 MMStar/V* 的选择题进入 LLM Judge，与确定性判分要求冲突；
- cal_acc.py 主要打印终端结果，无法从逐样本记录稳定复算 summary；
- run_eval.sh 原来读取 eval 目录中的 JSON，可能绕过 /root/autodl-tmp/benchmark_data 下的冻结转换数据；
- 原入口只有 --benchmark，而计划命令使用 --config 和 --benchmarks，CLI 契约不一致。

最终处理：

- 保留旧 --benchmark/--data_dir 接口，新增互斥的 --config/--benchmarks 冻结接口；
- 新增 frozen_benchmark_data.py，统一负责 revision 下载、转换、验证、manifest 和 SHA256；
- 将 run_eval.sh 的三个冻结 Benchmark 路径改到 benchmark_data/converted/{benchmark}/{benchmark}.json；
- 新增独立的 run_smoke.py 与 score_smoke.py，Day 5 Smoke 不再依赖旧脚本隐式默认值；
- 为配置、转换、抽样、重叠、推理恢复、解析、Judge 和预算补充回归测试。

## 任务 3：完成三个数据集下载、转换和数据冻结

### 原始数据与转换数据的区别

原始数据保留官方文件结构和内容，作为来源审计与重建依据，不被静默修改。转换数据是面向统一评测接口的派生层，主要变化包括：

- 从 Parquet、JSONL 或 datasets 记录中提取图像为本地 PNG；
- 统一 images、crop_images、query、response、source_id、sample_uid、source_revision；
- 增加 question_format、category、image_sha256、crop_image_sha256；
- 给 V* 追加冻结的直接输出选项字母 Prompt；
- 保留官方样本数和顺序身份，不因缺字段或重叠删除样本。

实际推理使用转换数据；原始数据只用于验证、重建和哈希审计。

| Benchmark | 原始数据 | 转换数据 | 转换后验证 |
|---|---:|---:|---|
| ZoomBench | 约 3.9 GiB | 约 3.8 GiB | 845 条；621 MCQ、224 开放题；845 full + 845 crop |
| MMStar | 约 97 MiB | 约 278 MiB | 1,500 条 MCQ；6 个一级类别各 250 条 |
| V* Bench | 约 366 MiB | 约 1,005 MiB | 191 条 MCQ；direct_attributes 115、relative_position 76 |

data_validation.json 显示三项均为 pass，样本 UID 数与样本数一致，没有缺图、空问题或规模偏差。raw_data_sha256.txt 保存 485 个原始文件哈希，converted_data_sha256.txt 保存 3,384 个转换产物哈希。

### 下载和转换时遇到的问题

#### 1. huggingface_hub 模块缺失

首次在 base Conda 环境执行时出现 ModuleNotFoundError: huggingface_hub。问题不是 Benchmark 缺失，而是使用了错误的 Python 环境。

解决方式是切换到 vision-opd 环境；requirements.txt 也固定了 huggingface_hub、mathruler、openai 和 vllm 版本。下载相关 import 改为调用时再加载，使不涉及下载的帮助和本地验证路径不被顶层 import 阻断。

#### 2. CLI 报缺少 --benchmark

脚本最初只接受单 Benchmark 的 --benchmark，计划命令却传入 --config 和 --benchmarks，因此 argparse 仍要求旧参数。

解决方式是保留兼容旧接口，同时新增冻结接口。现在以下命令是有效入口：

```bash
python eval/prepare_data.py \
  --config configs/benchmark_eval.yaml \
  --benchmarks zoombench,mmstar,vstar
```

两个接口不能混用，避免输出目录和 revision 语义含混。

#### 3. ZoomBench 本地 Parquet 转换占用大、重复读取

ZoomBench 的 Parquet 内嵌 full/crop 图片。整表读取会造成较高内存占用，而且下载完成后重复执行可能再次读取大图列。

解决方式是优先复用冻结的本地 Parquet，按 row group 流式读取；已存在且可解码的图像不再读取对应大图列；每组结束释放 PyArrow 内存并执行垃圾回收。转换前强制检查 845 行，转换后验证 full/crop 图像和哈希。

#### 4. V* 数据来源和格式不一致

旧代码使用 lmms-lab 镜像，并假定 Parquet 字段；官方项目实际直接链接 craigwu/vstar_bench，其冻结快照包含 test_questions.jsonl 和图片文件。

解决方式是改用官方链接数据源和固定 revision，优先复用本地 JSONL，必要时才通过 datasets 加载；统一生成 191 条转换记录和稳定 sample_uid。

## 任务 4：训练数据重叠审计识别出 V* 的 4 条污染风险

审计范围覆盖项目 train/eval/retention 共 1,216 条样本、2,432 个 full/crop 图像引用，以及三个 Benchmark 共 2,536 条样本、3,381 个图像引用。

三层方法为：

1. 文件 SHA256：字节完全相同；
2. 问题文本：Unicode NFKC、casefold、首尾去空白和连续空白折叠；
3. 64-bit DCT pHash：EXIF 方向修正后计算，Hamming distance ≤ 5 的候选必须人工确认。

| Benchmark | 样本数 | 候选 | 确认重叠 | 排除候选 | 未确认 | 影响率 |
|---|---:|---:|---:|---:|---:|---:|
| ZoomBench | 845 | 1 | 0 | 1 | 0 | 0.000% |
| MMStar | 1,500 | 0 | 0 | 0 | 0 | 0.000% |
| V* Bench | 191 | 4 | 4 | 0 | 0 | 2.094% |

V* 的 4 对候选均位于项目 train split。它们尺寸一致、pHash distance 为 0，RGB 差异很小且集中在项目标注框附近，人工复核确认为同一底图。ZoomBench 的 1 个候选只是通用问题模板相同；答案不同，四组 full/crop 交叉 pHash 距离为 32、34、36、32，因此排除。

处理原则不是删除官方测试样本，而是：

- 始终报告 V* 官方 191 条全量结果；
- 另报排除 4 条后的 187 条诊断结果；
- 不把 V* 结果描述为完全独立测试；
- 训练数据或 Benchmark revision 改变时重跑审计。

## 任务 5：固定 48 UID 并跑通 64 请求 Smoke

### 确定性抽样

每个 Benchmark 选择 16 条，seed 固定为 42。类别内排名为 SHA256("42:{benchmark}:{sample_uid}") 的十六进制升序，不能根据准确率或 overlap 结果重抽。

| Benchmark | 分层和配额 |
|---|---|
| ZoomBench | multiple_choice 12、open_question 4 |
| MMStar | coarse perception、fine-grained perception、instance reasoning、logical reasoning 各 3；math、science & technology 各 2 |
| V* Bench | direct_attributes 8、relative_position 8 |

selection manifest SHA256 为 dc5856cf6563e5b4a341f5131fcb33785ea36efd3c4ac7f239aebb428e0a392b。选中的 48 条没有命中已确认的 V* 重叠样本；这一事实只是记录，不能成为重新选择样本的依据。

### 端到端执行

服务使用双卡 vLLM 加载 Qwen3.5-4B。实际命令链为：

```bash
vllm serve /root/autodl-tmp/models/Qwen3.5-4B \
  --served-model-name vision-opd-base \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --seed 42 \
  --host 0.0.0.0 \
  --port 8000

python eval/run_smoke.py \
  --config configs/benchmark_eval.yaml \
  --api-base http://127.0.0.1:8000/v1 \
  --model-id vision-opd-base

python eval/score_smoke.py \
  --config configs/benchmark_eval.yaml \
  --judge-api-base http://127.0.0.1:8000/v1 \
  --judge-model-id vision-opd-base
```

run_smoke.py 每完成一个请求就原子写回 predictions.jsonl；重启时按 benchmark/view/sample_uid 恢复，避免重复和漏样本。评分 amendment 改变配置 SHA256 后，运行器不再错误地把整份配置哈希当成样本身份，而是核验真正决定抽样的 seed、排名规则、配额、样本数与数据 revision。

执行结果：

- 48 个唯一 sample_uid，64 个实际请求；
- inference error 0、retry 0、duplicate request key 0；
- 61 条 finish_reason=stop，3 条 finish_reason=length；
- 平均延迟 14.582 秒，P95 55.960 秒；
- Prompt tokens 129,575，Completion tokens 110,860；
- 3 个达到 8,192 token 上限的请求仍保留在分母中：MMStar source_id 1066、V* source_id 162、ZoomBench crop source_id fc4e54b27927b252c53bb8c0d17658d2。

### ZoomBench Judge 问题与最终修复

最初规则评分后，ZoomBench full 为 5 对、8 错、3 待 Judge；crop 为 10 对、4 错、2 待 Judge。5 个待定中，4 个其实是参考答案与最终回答都是单数字但数值不等，交给 LLM 判断没有必要。

同时，原计划的 Qwen3-30B-A3B-Instruct-2507 Judge 预计需要约 32 GiB 额外存储。中断下载的目录实际只有约 1.8 MiB 配置/分词器文件，没有权重分片；确认它与现有 8.8 GiB Qwen3.5-4B 目录分离后已删除，现有 Base 未受影响。

最终评分顺序冻结为：

1. 若参考答案和抽取答案都是完整单数字，使用 Decimal 精确比较；相等计对，不等计错；
2. 非适用项进入 MathRuler；
3. 只有 MathRuler 不能确认的语义未决项进入固定 Base Qwen3.5-4B Judge；
4. MMStar 和 V* 选择题始终禁止 LLM Judge。

Base 4B 第一次 Judge 调用输出了长篇解释，没有严格返回 Yes/No，因此按失败策略记录，未伪装成有效裁决。修复为 temperature 0、enable_thinking=false、max_new_tokens 64，并增加只允许输出 Yes 或 No 的 System Prompt。重跑后唯一语义项输出 No。

最终 64 条的 score source 为：

| Score source | 数量 |
|---|---:|
| 选择题 exact/explicit/trailing option | 55 |
| 选择题 invalid_or_ambiguous | 1 |
| 确定性数字相等 | 3 |
| 确定性数字不等 | 4 |
| 固定 Base 4B Judge | 1 |
| MathRuler | 0 |
| Judge failure / pending | 0 |

Smoke 结果如下。它们只验证链路，不作为完整模型能力结论。

| Benchmark / view | 正确 | 错误 | 准确率 |
|---|---:|---:|---:|
| MMStar full | 14 | 2 | 87.50% |
| V* full | 15 | 1 | 93.75% |
| ZoomBench full | 5 | 11 | 31.25% |
| ZoomBench crop | 10 | 6 | 62.50% |

Zooming gap 为 +31.25 个百分点。

## 任务 6：完整评测预算和 Day 6 Gate 已冻结

完整 Day 6 工作量为 3,381 个请求：

- ZoomBench 845 条 × full/crop = 1,690；
- MMStar 1,500；
- V* Bench 191。

基于 Smoke 分 Benchmark 的平均时延与 token 外推，完整运行预计使用约 481 万 Prompt tokens 和 582 万 Completion tokens。Smoke 的 8 个开放题实例中只有 1 个进入语义 Judge，因此预期约 56 个 Judge 实例；理论最坏为全部 448 个 ZoomBench 开放题视图进入 Judge。

双卡实例价格由用户确认是 11.96 元/小时，单位已经是双卡实例小时，不能再次乘以 GPU 数。

| 预算场景 | 有效并发 | 缓冲 | Judge 数 | 墙钟时间 | 费用 |
|---|---:|---:|---:|---:|---:|
| 实测吞吐 | 7.414 | 15% | 56 | 2.18 h | 26.11 元 |
| 保守执行 | 5 | 30% | 56 | 3.61 h | 43.18 元 |
| 最坏护栏 | 4 | 50% | 448 | 5.99 h | 71.70 元 |

Day 6 获准按保守预算启动；80 元为暂停检查点，100 元为单次运行硬止损线。达到止损线后必须停止提交新请求、保留 checkpoint 和逐条记录，不能按成功子集报告结果。项目总硬上限仍为 2,000 元。

## 问题—原因—解决方法总表

| 问题 | 根因 | 解决方法 | 防复发措施 |
|---|---|---|---|
| ModuleNotFoundError: huggingface_hub | 在 base 环境运行，项目依赖位于 vision-opd | 激活正确环境；下载依赖延迟导入 | requirements 固定版本，命令统一使用 vision-opd Python |
| prepare_data 要求 --benchmark | 旧 CLI 与冻结计划 CLI 不一致 | 增加 --config/--benchmarks，同时保留旧接口 | 两类参数互斥并有 CLI 测试 |
| ZoomBench Parquet 转换内存压力 | full/crop 图片嵌在大 Parquet，整表加载 | row-group 流式转换、按需读图片列、释放 Arrow 内存 | 本地 checkpoint 和幂等复用测试 |
| V* 转换源不一致 | 旧脚本使用镜像和错误格式假设 | 改用官方链接仓库、固定 revision、读取本地 JSONL | 配置强制 repo_id，测试拒绝静默替换 |
| run_eval 数据路径错误 | 默认读取 eval 目录 JSON | 冻结三项改读 benchmark_data/converted | 缺文件时明确提示 prepare_data 命令 |
| ZoomBench category 缺失 | 官方快照没有逐样本六维映射 | 标记 unavailable_official，按题型抽样与报告 | Amendment A，禁止伪造 category_accuracy |
| sample_uid 如何选择 | seed 未定义稳定排序，文件顺序会影响结果 | SHA256 seed/benchmark/UID 排名与显式配额 | manifest 默认拒绝覆盖，SHA256 冻结 |
| 发现 V* 重叠 | 项目 train 与官方测试存在同底图 | 保留官方全量，同时生成 187 条去重诊断 | 三层审计和人工决策均保存哈希 |
| ZoomBench 数字题大量待 Judge | MathRuler 未覆盖所有明确数字不等 | 添加严格的单数字 Decimal 比较 | 数字不等不可被 LLM 覆盖，单元测试锁定 |
| 30B Judge 存储过大 | 官方 Judge 与项目规模不匹配 | 改用固定 Base 4B，仅处理语义未决项 | 标记 project_frozen_base_4b_judge，不冒充官方 30B |
| Base 4B Judge 不严格输出 Yes/No | thinking 模式导致长篇解释并耗尽输出 | 关闭 thinking、限制 64 token、System Prompt 强约束 | 非 Yes/No 按失败计错并保留原文 |
| 评分 amendment 导致 manifest config hash 不一致 | 样本 manifest 错把整份配置哈希当身份 | 改核验抽样相关字段而非所有评分字段 | seed、排名、配额、样本数、数据 revision 单独验证 |
| 预算最初缺少实例价格 | 服务器无法推断 AutoDL 控制定价 | 用户提供 11.96 元/双卡小时 | 价格变化必须重新生成 cost.json 和 amendment |
| GPU 空闲仍占显存 | vLLM 已加载模型但没有请求 | Smoke 完成后可停止服务；Task 6 全部使用 CPU | Day 6 Gate 明确何时恢复服务和健康检查 |

## 验证、哈希和可复现性

主要验证结果：

- 三个 Benchmark 转换数据 validation status 均为 pass；
- overlap fingerprint/data error 为 0，未确认候选为 0；
- Smoke inference error、retry、duplicate request 均为 0；
- Judge pending 和 failure 均为 0；
- 预算最坏情形 71.70 元，低于 100 元执行止损线和 2,000 元项目硬上限；
- 最终测试：58 passed、16 subtests passed。

核心哈希：

| 产物 | SHA256 |
|---|---|
| configs/benchmark_eval.yaml | d50d420d760fa59bd8a139fa4615aed8a4b41c79ca969d5f194e95c2ad6c25b6 |
| docs/benchmark_protocol.md | 24e02e9d8a1b1de0ba0b8edacd52379ad2eccadfe9973282c41ef74b821d3dbb |
| smoke_selection.json | dc5856cf6563e5b4a341f5131fcb33785ea36efd3c4ac7f239aebb428e0a392b |
| Smoke predictions.jsonl | 02d90a298dea80036014b139569585fe3404c913e5019e0eb454aaa6f2e8e395 |
| Smoke scores.jsonl | 056e4f4b2222c84d2cb20f619e32b22b4a3247b5c2b987cd886b4ed2a953864c |
| Smoke summary.json | b0f42696d2b4a19bce40c4b2d080146fc82e2c7cf5ea8b26b5ae077361a1cd53 |
| budget_inputs.json | 9f03f8c83108319e248753442ac080f2385dd2e70015fee7acea9ab39bbefe61 |
| cost.json | 13fb707e790850b6c75e549db642689dfdb837396336b503096f7dc05744456d |

## 限制与解释边界

- Smoke 每个 Benchmark 只有 16 个 UID，准确率方差很大，只能证明链路可运行，不能替代完整评测。
- 3 条响应达到 8,192 token 上限，说明完整评测仍可能出现长输出；预算已经通过保守和最坏缓冲覆盖，但 Day 6 必须保留 finish_reason。
- ZoomBench 六维类别只能作为数据集级描述；没有官方逐样本标签就不能生成官方 category accuracy。
- V* 的 4 条确认重叠使其不能被称为完全独立测试；187 条诊断不能替代 191 条官方分数。
- MMStar 和 V* 数据许可证没有在官方数据页明确声明，本地研究评测不等于允许重新分发图片和标注。
- 预算是基于 Smoke 的描述性外推，不是性能保证。实例价格、硬件数量或服务参数变化时必须重算。
- Day 5 没有产生完整三项 Base Benchmark 分数；这属于 Day 6，不能把本简报中的 Smoke 分数当作项目最终基线。

## Day 6 的可执行下一步

启动 E-D6-001 前应依次完成：

1. 校验 benchmark_eval.yaml、数据 manifest、Smoke selection、budget inputs、cost 和协议 SHA256；
2. 重新校验 Qwen3.5-4B 两个权重分片，确认未加载训练后或官方 Vision-OPD 权重；
3. 以冻结的双卡 vLLM 参数启动 vision-opd-base，并通过 /v1/models 健康检查；
4. 验证三个转换数据文件与图片存在、哈希匹配，创建独立 E-D6-001 目录；
5. 按 3,381 个请求完整运行并持续记录墙钟和费用；80 元检查，100 元止损；
6. 对每条记录保留输入标识、原始回答、解析/判分来源、错误、finish reason、token 和延迟；
7. 汇总 ZoomBench full/crop/gap、MMStar 官方类别、V* 191 全量与 187 去重诊断，并生成完整成本与报告。

## 主要产物索引

- 协议与配置：[benchmark_protocol.md](benchmark_protocol.md)、[benchmark_eval.yaml](../configs/benchmark_eval.yaml)
- 数据准备：[prepare_data.py](../eval/prepare_data.py)、[frozen_benchmark_data.py](../eval/frozen_benchmark_data.py)
- 数据 manifest 与验证：[dataset_manifest.json](../artifacts/runs/E-D5-001/dataset_manifest.json)、[data_validation.json](../artifacts/runs/E-D5-001/data_validation.json)
- 重叠审计：[overlap_report.md](../artifacts/runs/E-D5-001/overlap/overlap_report.md)、[manual_review_decisions.json](../artifacts/runs/E-D5-001/overlap/manual_review_decisions.json)
- Smoke 抽样与执行：[select_benchmark_smoke.py](../scripts/select_benchmark_smoke.py)、[run_smoke.py](../eval/run_smoke.py)、[score_smoke.py](../eval/score_smoke.py)
- Smoke 结果：[predictions.jsonl](../artifacts/runs/E-D5-001/smoke/base/predictions.jsonl)、[scores.jsonl](../artifacts/runs/E-D5-001/smoke/base/scores.jsonl)、[summary.json](../artifacts/runs/E-D5-001/smoke/base/summary.json)
- 预算：[budget_inputs.json](../artifacts/runs/E-D5-001/budget_inputs.json)、[cost.json](../artifacts/runs/E-D5-001/cost.json)、[full_eval_budget.md](../artifacts/runs/E-D5-001/full_eval_budget.md)
- 测试：tests/test_benchmark_protocol.py、tests/test_benchmark_overlap.py、tests/test_smoke_selection.py、tests/test_run_smoke.py、tests/test_score_smoke.py、tests/test_freeze_budget_inputs.py、tests/test_estimate_benchmark_budget.py

结论：Day 5 已完成并具备审计、复算和进入 Day 6 的条件；下一阶段只应在 Day 6 Gate 全部通过后启动完整 Base 外部评测。

