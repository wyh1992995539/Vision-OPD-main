# Day 20 GRPO 模型定版与统一 R4 评测工作简报（待定草案）

> 文档状态：**DRAFT / NOT EXECUTED**  
> 目标阶段：GRPO checkpoint 合并、冷加载、固定 1,024-token R4 外部评测  
> 启动前提：Day19 正式训练及最终 checkpoint Gate PASS  
> 说明：结果表中的 `[待填]` 均为运行后才能取得的数值，不代表已有成绩，不得用于模型比较或对外报告。

## 技术摘要

若 Day20 正常完成，应将 Day19 唯一冻结的最终 actor checkpoint 合并为 Hugging Face 模型，保存模型、tokenizer、processor、chat template 和来源 checkpoint 的身份与 SHA256，在新进程完成 5 条功能冷加载，然后使用与 Base、Vision-OPD、Cached 完全相同的固定 1,024-token R4 协议完成 ZoomBench、MMStar 和 V* Bench 共 2,536 条评测。

外部 Benchmark 只能评测冻结后的最终模型，不能用于选择 Day19 中间 checkpoint、修改 Reward 或继续调参。

## 一、实际完成的工作（待执行成功后确认）

| 顺序 | 具体操作 | 主要结果与证据 | 状态 |
|---:|---|---|---|
| 1 | 冻结 Day19 `global_step_780` 的 marker、文件清单、大小和 SHA256；确认训练日志与 coverage receipt PASS | final checkpoint freeze | 预期 PASS |
| 2 | 在独立 `.inprogress` 目录执行 FSDP→Hugging Face 合并，校验 tensor 数量、dtype、配置差异白名单和必需文件 | merge attempt、merged manifest、SHA256 | 预期 PASS |
| 3 | 原子晋级 merged 目录，并在全新 vLLM 进程固定加载 5 条工程样本；测试前后复核模型哈希 | cold reload receipt | 预期 PASS |
| 4 | 在任何 GRPO 外评分数产生前冻结模型身份、R4 配置、请求清单、Judge、分母、错误规则和输出 Schema | pre-score freeze | 预期 PASS |
| 5 | 每个 Benchmark 固定抽取 4 条执行 12 条 Smoke，完成生成、解析、Judge、评分和 validation | smoke manifests/results | 预期 PASS |
| 6 | Smoke Gate 通过后运行 2,536 条正式推理，保存逐样本原始回答、finish reason、token 数和错误 | predictions 与 inference metrics | 预期 PASS |
| 7 | 对明确答案直接判分，对歧义回答调用冻结 Base Judge；Judge 失败按协议计错并保留 | judge results、pending=0 | 预期 PASS |
| 8 | 生成 scores、summary、validation 和 SHA256，核对 2,536 个唯一请求键及固定分母 | R4 completion receipt | 预期 PASS |

模型合并、冷加载、Smoke、正式推理和 Judge 使用不同的会话与证据目录。任何失败重试都保留旧 attempt；正式请求清单冻结后不得因 Smoke 或中途分数改变成员。

## 二、模型定版

| 项目 | 目标 | 实际结果 |
|---|---|---|
| 最终 actor step | 780 | 780 |
| FSDP checkpoint 校验 | PASS | PASS（待凭据） |
| HF merge | PASS | PASS（待凭据） |
| merged 权重 SHA256 | 已记录 | 待运行生成并绑定凭据 |
| tokenizer/processor/template 哈希 | 已记录 | 待运行生成并绑定凭据 |
| 新进程冷加载 | 5/5 非空 | 5/5（待凭据） |
| inference error | 0 | 0 |
| 测试前后模型哈希 | 不变 | 一致（待哈希） |

冷加载使用固定工程样本，只证明模型可加载和产生合法输出，不报告为准确率或泛化结果。

## 三、R4 评测冻结合同

- Benchmark：ZoomBench 845、MMStar 1,500、V* Bench 191。
- 总请求：2,536。
- 输入、图像、prompt、temperature、最大输出 1,024、解析器、Judge 和错误处理与 Day16 三模型 R4 一致。
- V* 主分母固定为官方 191。
- 明确选项由 R4 保守解析器处理；歧义回答进入冻结 Base Judge。
- inference/Judge error 保留在固定分母中并计错，不静默删除。
- 先执行每个 Benchmark 4 条 Smoke，共 12 条；Smoke Gate 通过后再执行正式评测。

## 四、GRPO R4 结果回填

| Benchmark | N | Base R4 | Vision-OPD R4 | Cached R4 | GRPO R4 |
|---|---:|---:|---:|---:|---:|
| ZoomBench | 845 | 50.65% | 52.54% | 51.36% | `[待填]` |
| MMStar | 1,500 | 68.00% | 59.20% | 56.33% | `[待填]` |
| V* Bench | 191 | 82.72% | 71.73% | 73.30% | `[待填]` |
| Micro | 2,536 | 63.33% | 57.93% | 55.95% | `[待填]` |

以上 Base/Vision-OPD/Cached 是固定 1,024-token R4 参考值。Base MMStar 的 72.73% 属于选择性自适应输出上限协议，不能放入本表替换 68.00%。

## 五、完整性和运行指标

| 指标 | 目标 | 实际结果 |
|---|---:|---:|
| Smoke predictions/scores | 12/12 | 12/12 |
| 正式 predictions | 2,536/2,536 | 2,536/2,536 |
| 正式 scores | 2,536/2,536 | 2,536/2,536 |
| 唯一请求键 | 2,536 | 2,536 |
| 重复键 | 0 | 0 |
| pending Judge | 0 | 0 |
| inference errors | 完整披露 | `[待填]` |
| Judge failures | 完整披露 | `[待填]` |
| length outputs | 完整披露 | `[待填]` |
| completion tokens | 完整披露 | `[待填]` |
| 推理/Judge 墙钟 | 完整披露 | `[待填]` |
| 估算费用 | 完整披露 | `[待填]` |

## 六、可能问题与处置

| 可能问题 | 识别信号 | 具体处置 |
|---|---|---|
| FSDP 合并 OOM | merger 被 SIGKILL、cgroup OOM 或磁盘急降 | 先确认训练进程已退出和 cgroup 足够；使用 `.inprogress` 原子合并，失败目录不晋级 |
| merged 配置与 Base 不完全相等 | dtype 等字段变化 | 逐字段比较，只允许由实际 BF16 权重导致的精确白名单差异；未知差异直接停止 |
| 冷加载失败或输出为空 | vLLM 启动错误、5 条中存在空输出 | 保留服务日志，检查 tokenizer/processor/template 和权重完整性；未通过不得进入外评 |
| Smoke 出现请求键或路由错误 | prediction/score 数量不等、pending Judge、分母异常 | 停止正式评测，修复工程问题后重跑独立 Smoke，不修改模型 |
| 推理 OOM/API 超时 | inference error、服务退出或请求缺失 | 按冻结重试规则记录 attempt；仍失败则保留在分母并完整披露 |
| R4 解析器再次抓取普通大写字母 | 明确回答与 prediction 不一致 | 使用 Day16 冻结 parser 回归集阻断；不得临时增加有利回退 |
| Judge 不严格返回 Yes/No | Judge failure 增加 | 保存原始 Judge 输出并按冻结规则计错；不人工改判、不静默重试到成功 |
| 大量 `finish_reason=length` | 输出集中触达 1,024 | 固定主表仍按 1,024 完成；自适应诊断另立协议，不能只为 GRPO 单独改上限 |
| 外评分数不理想 | 某 Benchmark 低于 Base/VOPD/Cached | 如实完成 2,536 条并进入 Day21 分析，不重选 checkpoint、不回到训练调参 |

## 七、阶段结论与边界（待真实结果确认）

> Day20 GRPO 模型定版与 R4 评测预期 **PASS**。最终 actor `global_step_780` 完成合并及 5/5 新进程冷加载。固定 1,024-token R4 完成 2,536/2,536 条预测和 2,536/2,536 条评分；ZoomBench、MMStar、V* 分别为 `[待填]`、`[待填]`、`[待填]`。该结果使用本地冻结 Qwen3.5-4B Base Judge，不等同于论文 GPT-OSS-120B Judge。

Day20 只报告预先冻结的 1,024-token R4 协议结果。Base 的自适应输出上限结果、Smoke 结果和人工诊断均不得并入主表；本地替代 Judge、固定分母和单次冻结评测共同限定结论范围。

## 八、关键证据索引

- Day19 最终 checkpoint manifest
- GRPO merged HF 模型和 SHA256 清单
- 5 条新进程冷加载凭据
- GRPO R4 pre-score freeze
- 12 条 Smoke predictions/scores/validation
- 2,536 条正式 predictions/scores/Judge results
- R4 summary、validation 和 artifact SHA256
