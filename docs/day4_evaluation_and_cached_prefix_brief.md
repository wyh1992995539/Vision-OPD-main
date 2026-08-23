# Vision-OPD Day 4 工作简报：冻结评测、Vanilla 基线与 Cached Prefix

> 执行日期：2026-08-22 至 2026-08-23  
> 实验 ID：`E-D4-001`  
> 状态：**PASS（评测器 Gate、Vanilla-128 Gate、Cached Prefix-1024 Gate）**  
> 边界：本日完成的是训练前基线和离线前缀准备；SFT、Vision-OPD、Cached/off-policy 对照训练及 GRPO 尚未开始，不能据此声称模型能力得到提升或论文方法已复现。
>
> **后续范围修订（2026-08-24）**：SFT 分支已从活跃计划中删除；当前 Day 5 改为 ZoomBench、MMStar、V* Bench 的协议冻结、数据准备、重叠审计与每项 16 条端到端 Smoke，Day 6 为原始 Qwen3.5-4B Base / Vanilla 的三项完整外部基线。本简报正文保留 Day 4 执行当时的事实与边界，不因后续计划变化重写原始证据；当前执行入口以 [`vision_opd_21_day_plan.md`](vision_opd_21_day_plan.md) 和 [`project_freeze.md`](project_freeze.md) 的 amendment 为准。

## 1. 当日目标与完成结论

Day 4 的目标不是训练模型，而是先固定后续所有实验共用的“尺子”和训练前输入：

1. 为实验 `E-D4-001` 单独记录代码、模型、软件、硬件和磁盘状态；
2. 冻结 Vanilla 评测协议和 Cached Prefix 生成协议；
3. 实现确定性的选择题评测器，并冻结评测器身份；
4. 使用训练前同一个 Qwen3.5-4B Base checkpoint，在固定 `eval_128.parquet` 上建立 Vanilla 基线；
5. 使用同一个 Base checkpoint，为固定 `train_1024.parquet` 生成 1,024 条离线 Cached Prefix；
6. 保留逐样本输出、失败审计、协议哈希、代码哈希和结果哈希，形成后续可复算、可配对的证据链。

最终三个 Gate 均通过：

| Gate | 最终结果 | 结论 |
|---|---|---|
| 评测器与协议冻结 | 25 tests + 16 subtests 通过；评测器版本与 SHA256 已保存 | **PASS** |
| Vanilla Base / eval-128 | 67/128，Accuracy 52.34375%，无推理错误、无空响应 | **PASS** |
| Cached Prefix / train-1024 | 1,024/1,024，唯一且完整；SHA256 已冻结 | **PASS** |

## 2. 为什么必须先做这些工作

### 2.1 先冻结评测器，防止训练后“换尺子”

Vanilla、SFT、Vision-OPD 和 Cached/off-policy 对照必须使用同一数据、Chat Template、生成参数和评分逻辑。如果训练后再修改解析规则或只删除难例，模型间差异就可能来自评测变化，而不是训练方法。

因此本日先固定：

- `eval_128.parquet` 的 128 个 `sample_id`；
- Qwen3.5-4B Base checkpoint；
- Chat Template；
- 解码参数与最大回复长度；
- 选择题解析逻辑；
- `invalid_prediction` 计入全体分母并按错误处理的规则；
- 每条预测的原始文本、解析状态、结束原因和 Token 数。

### 2.2 先建立 Vanilla，才能判断后续训练是否有效

Vanilla 是不训练的 Base 模型结果。后续 SFT 或 Vision-OPD 的准确率只有与该结果在同一协议下比较，才可以判断提升、退化和错误迁移。

### 2.3 先生成 Cached Prefix，才能进行单变量对照

Cached Prefix 由训练前 Base 模型对固定 1,024 条 train 样本一次性生成。后续 Cached/off-policy 分支将复用这些固定前缀，而标准 Vision-OPD 使用训练时 Student 在线生成前缀。保持模型、数据、Teacher crop、损失和训练预算不变时，两者的主要变量才是前缀来源。

因此 Cached Prefix 是后续消融实验的输入，不是本日的训练结果，也不是能力提升证据。

## 3. 按执行顺序梳理

### 步骤 0：为 `E-D4-001` 建立独立运行快照

Day 1 的 `artifacts/runs/preflight/` 记录项目启动时的环境；Day 4 另建 `artifacts/runs/E-D4-001/`，是为了记录本次实验真正使用的代码、模型和硬件状态，避免把历史预检当作当前运行证据。

记录内容包括：

- Git commit、分支和工作区差异；
- Base 模型文件清单及逐文件 SHA256；
- Python、PyTorch、CUDA、Transformers、vLLM 版本；
- Chat Template 与生成配置哈希；
- GPU 型号、显存、驱动版本和磁盘使用情况。

实际服务器环境：

| 项目 | 记录值 |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| CUDA runtime | 12.8 |
| Transformers | 5.5.0 |
| vLLM | 0.18.0 |
| GPU | 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition |
| 单卡显存 | 97,887 MiB（约 95.6 GiB） |
| 驱动 | 580.95.05 |
| `/root/autodl-tmp` | 50G，总占用 15G，可用 36G（记录时） |

最初在无 GPU 状态下记录的软件环境，因此 `env.txt` 中 `CUDA available=False`；GPU 开启后又在 `hardware.txt` 中补录了两卡状态。这两个文件对应不同时点，不构成矛盾。

Base 模型的配置、Tokenizer、索引和两片权重均通过 `sha256sum -c`。其中两片模型权重 SHA256 为：

```text
model.safetensors-00001-of-00002.safetensors  26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61
model.safetensors-00002-of-00002.safetensors  cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188
```

### 步骤 1：确认本项目当前评分范围

检查冻结的 1,216 条项目数据后确认：

- `question_type` 全部是 `multiple_choice`；
- 标准答案全部属于 `A/B/C/D`。

因此 Day 4 只实现并冻结确定性四选一评分，不提前实现当前数据并不需要的短字符串、数字、颜色或方向等开放式评分规则。这样减少了未经真实数据验证的分支，但在评测器版本中明确记录 `supported_question_types=["multiple_choice"]`；未来加入其他题型时必须升级评测器版本，不能静默沿用。

### 步骤 2：冻结两套生成协议

统一配置写入 [`configs/day4_generation.yaml`](../configs/day4_generation.yaml)。两套协议共享：

- Base 模型：`/root/autodl-tmp/models/Qwen3.5-4B`；
- `seed=42`；
- 只输入 Student 全图和 Prompt；
- 禁止把 `bbox_images` 或标准答案传给模型；
- Chat Template：`chat_templates/perception_chat_template_qwen35.jinja`；
- `enable_thinking=false`；
- `max_new_tokens=256`；
- 保留原始文本、结束原因与回复 Token IDs。

两者只在用途需要的生成方式上不同：

| 项目 | Vanilla 评测 | Cached Prefix |
|---|---|---|
| 数据 | eval 128 | train 1,024 |
| 采样 | 否，确定性 | 是 |
| temperature | 0.0 | 1.0 |
| top_p | 1.0 | 1.0 |
| top_k | -1 | -1 |
| 返回数 | 1 | 1 |

最终生成协议 SHA256：

```text
configs/day4_generation.yaml  6f57e2f7b713e57e34a4b2364695ede32e23f02f8ecf160459c5cf52ecfe7636
Chat Template                 d9604b52b4e1f4b9ec68e065238c757a3d7efdebe1c3692d13a97df6f84c54db
```

### 步骤 3：实现评测器并先在无 GPU 环境测试

新增或完善：

- [`eval/internal_eval.py`](../eval/internal_eval.py)：解析 A/B/C/D、评分、聚合和错误状态；
- [`eval/run_internal_eval.py`](../eval/run_internal_eval.py)：读取 Parquet、请求兼容 OpenAI API 的 vLLM 服务、保存逐样本结果；
- [`tests/test_internal_eval.py`](../tests/test_internal_eval.py)：覆盖正确、错误、无答案、歧义、显式答案和末行答案等情况。

无 GPU 阶段完成了代码、测试、配置和版本冻结，不等待 GPU 才开始开发。评测器采用保守解析：只有出现可靠的最终选择证据才返回 A/B/C/D，不能可靠解析时标记 `invalid_prediction`，不调用外部 LLM Judge，也不根据标准答案猜测。

最终评测器身份：

```text
evaluator_id = e40cc751a2b732b5cd8eaeb4f4ca61754c75a3286800ab44bf4a2e72dbe7c689
git_commit   = 3b5d32b7c3bfeb6c791a1341ae32f0fac5cfb574
working_tree_clean_before_generation = true
```

`artifacts/eval/evaluator_version.json` 的 SHA256 校验为 `OK`。

### 步骤 4：启动双卡 vLLM，并完成 8 条 Vanilla Smoke

GPU 开启后，使用两张 RTX PRO 6000、BF16 和 Tensor Parallel 2 启动 vLLM，服务名固定为 `vision-opd-base`。第一次 `curl 127.0.0.1:8000/v1/models` 失败，是因为 tmux 会话已经创建但服务尚未真正启动；进入会话启动并等待模型加载完成后，接口返回模型信息。

最初 8 条 Smoke 暴露出两个解析漏项：

- 模型经过解释后只在最后一行输出裸字母，例如末行 `C`；
- 模型输出 `Answer: **B**` 或 `**B. Latin cross**` 等 Markdown/带标签格式。

这不是推理请求失败，而是评测器没有识别模型已经给出的最终选择。随后扩展解析规则，并增加回归测试，同时防止把选项列表中的最后一项误当最终答案。

修复后 8 条 Smoke：

| 指标 | 结果 |
|---|---:|
| 总数/唯一 sample_id | 8/8 |
| 推理错误 | 0 |
| 空响应 | 0 |
| invalid | 0 |
| 正确 | 3 |
| Accuracy | 37.5% |

Smoke 的用途是验证数据、服务、解析和落盘链路，不用于报告模型基线能力。

### 步骤 5：第一次正式 eval-128 暴露 4096 上下文不足

第一次正式运行时，vLLM 的 `max_model_len=4096`。结果为：

| 指标 | 结果 |
|---|---:|
| 总数 | 128 |
| 正确/错误 | 62/50 |
| invalid | 16 |
| Accuracy | 48.4375% |
| 推理错误/空响应 | 6/6 |

审计发现 6 条请求的图像与文本输入长度为 4,212～6,511，超过 4,096，vLLM 返回 HTTP 400；其余 invalid 中同时包含 256 Token 截断和评测器尚未识别的带标签结论格式。因此该轮结果被归档为 `eval_failed_maxlen4096`，没有冒充最终基线。

处理方式：

1. 将服务端 `max_model_len` 提升到 8,192；
2. 保持评测端 `max_new_tokens=256` 不变，避免同时改变回复预算；
3. 修正剩余可靠结论格式的解析逻辑并补充测试；
4. 先针对 13 条失败/修复样本运行定向 Smoke，再重跑完整 128 条。

13 条修复 Smoke 的结果为：13/13 请求完成、无空响应、无推理错误、全部成功解析；8/13 正确。该结果只证明修复 Gate，不代表总体准确率。

### 步骤 6：完成 8192 上下文的最终 Vanilla-128

最终统一服务配置：

```text
served_model_name = vision-opd-base
tensor_parallel_size = 2
dtype = bfloat16
max_model_len = 8192
max_num_seqs = 8
gpu_memory_utilization = 0.80
```

最终结果：

| 指标 | 结果 |
|---|---:|
| total / unique sample_id | 128 / 128 |
| correct | 67 |
| incorrect | 57 |
| invalid_prediction | 4 |
| inference_errors | 0 |
| empty_responses | 0 |
| Accuracy | **52.34375%** |
| 回复 Token 数 | min=3，mean=67.9375，max=256 |

按标准答案分组：

| 标准答案 | 正确/总数 | Accuracy |
|---|---:|---:|
| A | 9/27 | 33.33% |
| B | 18/26 | 69.23% |
| C | 13/31 | 41.94% |
| D | 27/44 | 61.36% |

最终 4 条 invalid 全部是回复达到 256 Token 后仍未给出可靠最终 A/B/C/D：3 条 `invalid_ambiguous`、1 条 `invalid_no_choice`。它们没有请求错误，也不是空响应，因此按冻结协议保留并计入错误分母，没有人工补答案或从分母中删除。

最终逐样本结果与摘要哈希：

```text
predictions.jsonl  9dc30fc71ee43b1ffd3e3fcfe4c442160e09d0c3b164a796c8287a8a73e806f1
summary.json       3477ccef70e976c16d58aa7a3526030c295a60badb96ba794a08eae438672bf3
```

### 步骤 7：实现 Cached Prefix 生成器并完成 8 条 Smoke

实现：

- [`scripts/generate_cached_prefix.py`](../scripts/generate_cached_prefix.py)；
- [`tests/test_generate_cached_prefix.py`](../tests/test_generate_cached_prefix.py)。

生成器按 `sample_id` 保存断点 checkpoint，校验数量、唯一性、缺失/多余 ID、空响应、推理错误和空 Token IDs，并输出 Parquet、报告与 SHA256。

8 条真实数据 Smoke 结果：

| 指标 | 结果 |
|---|---:|
| total / unique | 8 / 8 |
| 空响应/推理错误/空 Token IDs | 0/0/0 |
| 截断 | 0 |
| 状态 | **PASS** |

Smoke Parquet SHA256：

```text
f6586a591da2ec1b6f4f0b4fed9d80a7e84cb54b1dd9a694f35a81bd126afbca
```

### 步骤 8：生成完整 1,024 条 Cached Prefix，并修正截断校验策略

完整生成实际完成了 1,024/1,024 个请求，但第一次最终校验报错：

```text
actual_records=1024
unique_sample_ids=1024
missing=[]
extra=[]
empty_responses=0
inference_errors=0
empty_token_ids=0
truncated_responses=54
status=FAIL
```

问题不在数据丢失或服务失败，而在旧校验器把所有 `finish_reason=length` 一律视为生成失败。由于生成协议已冻结 `max_new_tokens=256`，这 54 条是模型在规定预算内生成的有效固定前缀。删除、重采样或只保留短回复会改变前缀分布，使 Cached/off-policy 组引入额外筛选变量。

因此修复为：

- 54 条截断前缀继续保留；
- 报告显式记录 `truncated_responses=54`；
- 新增 `truncation_policy=preserve_at_frozen_max_new_tokens`；
- 截断本身不导致 Gate 失败，但空响应、推理错误、空 Token、缺失、重复或多余样本仍会失败；
- 增加截断可通过与空前缀必须失败的回归测试。

服务器更新代码后，测试结果为：

```text
25 passed, 16 subtests passed
```

已有 `.inprogress.jsonl` 中恰好保存 1,024 条，因此最终封装从 checkpoint 恢复，不使用 `--overwrite`，没有再次请求模型或浪费 GPU。

最终 Cached Prefix 结果：

| 指标 | 结果 |
|---|---:|
| expected / actual / unique | 1,024 / 1,024 / 1,024 |
| duplicate / missing / extra | 0 / 0 / 0 |
| empty responses | 0 |
| inference errors | 0 |
| empty token IDs | 0 |
| truncated responses | 54（5.2734375%） |
| 截断策略 | `preserve_at_frozen_max_new_tokens` |
| 状态 | **PASS** |

最终输出：

```text
/root/autodl-tmp/data/vision_opd_1024/cached_prefix_base_1024.parquet
```

Parquet SHA256：

```text
e5992f9a8e074b207a829ba0f6ddda7d3426d2a2d877eb6c6d9b4845e60f9d46
```

`response_token_ids` 的来源被明确记录为 `base_tokenizer_reencoded_openai_response_text`，即使用 Base Tokenizer 对 OpenAI 兼容接口返回文本重新编码；本日没有把它描述为 vLLM 原始采样 Token ID。

### 步骤 9：解决 Git 分叉并归档最终证据

本地主要提交代码，服务器主要提交运行日志，期间出现：

- `main...origin/main [ahead 1, behind 1/2]`；
- `git pull --ff-only` 无法快进；
- 服务器旧代码未包含 `truncation_policy`，导致相同 checkpoint 再次被旧校验器判为失败。

没有使用 `git reset --hard`。处理顺序是：先保留服务器提交和失败日志，`git fetch` 后确认左右独有提交，创建备份分支，再将服务器提交 rebase 到最新 `origin/main`。代码更新后通过 25 个测试，才从 checkpoint 重新执行最终封装。

最终报告、日志、协议/代码哈希和外部 Parquet 哈希均归档到 Git；Day 4 最终归档提交为：

```text
6cba1eb  day4: finalize cached prefix generation gate
```

## 4. 途中问题、原因与解决结果

| 问题 | 根因判断 | 解决方式 | 最终结果 |
|---|---|---|---|
| 无 GPU 时是否需要记录硬件 | 无 GPU 本身也是运行状态，且软件开发不依赖 GPU | 先记录无 GPU 环境；开启 GPU 后补录真实两卡信息 | 环境时间点可追溯 |
| tmux 创建后 API 连接被拒绝 | 只创建/退出了会话，vLLM 尚未启动或未完成加载 | 在 tmux 中启动服务，读取日志并等待 `/v1/models` 可用 | 服务正常返回 `vision-opd-base` |
| Smoke 中明明有答案却被判 invalid | 解析器未覆盖裸末行、Markdown 粗体、`Answer:` 和带标签结论 | 扩展保守解析规则并增加误判防护测试 | 修复后 8 条 Smoke invalid=0 |
| 4096 正式评测有 6 条 HTTP 400 | 多模态输入长度超过服务端 4096 上下文 | 只把服务上下文升至 8192，保留回复上限 256；定向 Smoke 后完整重跑 | 128 条请求错误=0 |
| 最终仍有 4 条 invalid | 模型在 256 Token 内没有给出唯一最终选择 | 按预先冻结规则保留并计错，不人工修答案 | Vanilla 结果可审计 |
| Cached 1024 生成完却 Gate FAIL | 旧校验器把 54 条达到固定生成上限的前缀当作失败 | 显式冻结“保留截断前缀”策略并补测试 | 1,024 条最终 PASS |
| 担心重新封装再次消耗 GPU | 完整结果已在 `.inprogress.jsonl` 中 | 不加 `--overwrite`，从 1,024 条 checkpoint 直接封装 | 无需重新推理 |
| `libgomp: Invalid value for OMP_NUM_THREADS` | 环境变量值格式无效，与模型结果无关 | `unset OMP_NUM_THREADS` 后设为整数 `1` | 最终测试无该阻塞 |
| Git 无法 fast-forward | 本地代码提交与服务器证据提交形成分叉，且服务器一度仍是旧校验代码 | 查看左右提交、备份、rebase、测试、再推送 | 代码与证据统一到远端 |

## 5. 最终产物与证据位置

### 5.1 协议和实现

```text
configs/day4_generation.yaml
chat_templates/perception_chat_template_qwen35.jinja
eval/internal_eval.py
eval/run_internal_eval.py
scripts/generate_cached_prefix.py
tests/test_internal_eval.py
tests/test_generate_cached_prefix.py
```

### 5.2 评测器版本

```text
artifacts/eval/evaluator_version.json
artifacts/eval/evaluator_version.sha256.txt
```

### 5.3 Vanilla 正式结果

```text
artifacts/runs/E-D4-001/eval/predictions.jsonl
artifacts/runs/E-D4-001/eval/summary.json
artifacts/runs/E-D4-001/eval/run_8192.log
artifacts/runs/E-D4-001/eval/final_invalid_audit_8192.txt
artifacts/runs/E-D4-001/eval/result_sha256.txt
```

### 5.4 失败轮次与修复证据

```text
artifacts/runs/E-D4-001/eval_failed_maxlen4096/
artifacts/runs/E-D4-001/eval_smoke_8/
artifacts/runs/E-D4-001/eval_smoke_repair_13_8192/
```

### 5.5 Cached Prefix 结果

```text
artifacts/runs/E-D4-001/cached_prefix_smoke_8_8192/
artifacts/runs/E-D4-001/cached_prefix_1024/report.json
artifacts/runs/E-D4-001/cached_prefix_1024/parquet_sha256.txt
artifacts/runs/E-D4-001/cached_prefix_1024/generator_code_sha256.txt
artifacts/runs/E-D4-001/cached_prefix_1024/generator_git_commit.txt
artifacts/runs/E-D4-001/cached_prefix_1024/finalize_from_checkpoint.log
```

大型 Parquet 保存在服务器数据盘，不进入 Git；Git 只保存报告、哈希和生成日志。

## 6. Day 4 验收判断

### 已证实

- 固定 128 条主评测可由同一 Base 模型、同一协议完整执行；
- 评测器可确定性重算并保留逐样本审计信息；
- Vanilla Base 基线为 67/128，即 52.34375%；
- 固定 1,024 条 train 样本均已有训练前 Base 生成的离线前缀；
- Cached Prefix 与 train `sample_id` 一一对应，无缺失、重复、空响应、推理错误或空 Token；
- 4 条评测 invalid 和 54 条截断前缀均按事先声明的规则保留，没有通过删除难例改善结果；
- 代码、协议、评测器、逐样本预测、汇总报告和外部 Parquet 均有哈希或 Git 版本证据。

### 尚未证实

- SFT 未执行，且已在 2026-08-24 范围修订中取消，不再属于当前项目验收项；
- Vision-OPD 的在线 Student Prefix、Crop Teacher、Top-K JSD 和 EMA 是否已在真实训练链路闭环；
- Cached/off-policy Prefix 是否优于或劣于在线前缀；
- 任一训练方法是否提升准确率、能力保持或 regional-to-global gap；
- 论文级结果或完整 Vision-OPD 方法复现。

## 7. 后续范围修订与当前下一步

本简报完成时原定“Day 5 进入 SFT 数据适配与真实 Smoke”；该安排已被 2026-08-24 范围修订取代，不再作为执行指令。Day 4 已完成的 `67/128` 内部 Base 结果和 Cached Prefix-1024 保持有效，不重新评测、不重新生成。

当前进入 Day 5：

1. 冻结 ZoomBench、MMStar、V* Bench 的官方数据版本、Prompt、图像预处理、生成参数、评分规则和 Judge 配置；
2. 对三个 Benchmark 与现有 train/eval/retention 执行文件 SHA256、问题文本和感知哈希重叠审计；
3. 使用与本日相同、哈希可核验的原始 Qwen3.5-4B Base checkpoint，为每个 Benchmark 固定运行 16 条端到端 Smoke；
4. Smoke 只验证数据、推理、答案解析、Judge、断点恢复、逐样本保存和成本链路，不将 16 条准确率写入正式结果；
5. Smoke Gate 通过并冻结完整评测预算后，Day 6 才运行 Base / Vanilla 的三项完整外部 Benchmark。
