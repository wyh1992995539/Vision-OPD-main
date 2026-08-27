# Day 6 Base Benchmark 评测与论文对齐工作简报

> 执行日期：2026-08-26～2026-08-27（UTC）  
> 验收状态：**PASS**

## 技术摘要

Day 6 的强制任务已全部完成，最终状态为 PASS。本日先完成旧冻结协议 E-D6-001 的 Base 全量评测与问题诊断，随后针对论文复现目标建立 E-PAPER-BASEJUDGE-001，完成官方 V* 转换、推理、固定 Base Judge、评分、断点恢复、自动验证和单卡正式 Base 评测。最终将 R3 单卡配置冻结为 Base、Vision-OPD、Cached Prefix、GRPO 的唯一现行外部 Benchmark 标准。

最终 R3 Base 主结果为：ZoomBench full 428/845（50.65%）、MMStar 1126/1500（75.07%）、V* Bench 160/191（83.77%）。正式运行包含 2,536 条预测、1,786 条所需 Judge 记录和 2,536 条评分，自动 Gate 为 PASS，无推理错误、重复键、损坏 JSONL 或待定评分。

由于无法获得论文使用的 GPT-OSS-120B Judge，本项目固定使用未经训练的原始 Qwen3.5-4B Base Judge。这是相对论文唯一预先声明的核心替代，因此结果属于“尽量对齐公开论文配方 + 固定本地 Base Judge”，不能声称精确复现 Table 2。

## 一、任务完成情况

| 工作项 | 完成内容 | 状态 |
|---|---|---|
| Day 6 启动 Gate | 校验模型、数据、配置、V* 191 分母、Judge、预算和输出结构 | PASS |
| 旧协议 Base 全量评测 | 完成 ZoomBench full/crop、MMStar、V* 共 3,381 请求 | PASS，保留为工程诊断 |
| 输出问题诊断 | 定位 204 条解析失败和 301 条 token 触顶的关系与根因 | 完成 |
| 论文参数对齐 | 明确公开推理参数、Judge 差异和不可完全复现边界 | 完成 |
| 新评测工程 | 实现官方 V* 转换、推理、Judge、评分、验证和断点恢复 | 完成 |
| R1/R2/R3 Smoke | 依次验证公开配方、三项可控差异修正和单卡成本修订 | PASS |
| R3 正式 Base | 2,536 条预测、Judge、评分、汇总、成本和哈希完整归档 | PASS |
| 后续统一标准 | R3 成为四个模型角色唯一现行配置，删除 R1/R2 YAML | 完成 |

可选的“论文作者官方 Vision-OPD-4B 参考行”未执行；该项不属于 Day 6 强制验收，不影响 PASS。

## 二、今天完成的主要工作

### 1. 冻结 V* Bench 报告口径

Day 5 重叠审计确认项目 train split 与 V* 官方测试集有 4 张同底图。经人工复核后，本日决定不从 Benchmark 删除这 4 条，也不增加 187 条去重诊断或透明度指标，而是按负责人决定采用“不重复假设”：

- 保留官方全部 191 条预测和评分；
- 总体与类别分母始终为 191；
- 旧 overlap 报告继续保留，不能重写历史证据；
- 后续所有模型沿用相同口径，确保横向比较一致。

该决定已经并入本简报和统一评测标准。旧 V* Markdown 说明随后删除；机器可读 amendment、哈希及 overlap 证据继续保留。

### 2. 完成旧协议 E-D6-001 全量评测

旧协议运行 ZoomBench full/crop、MMStar、V* 共 3,381 个请求，全部获得模型响应并完成评分。

| 指标 | 结果 |
|---|---:|
| ZoomBench full | 433/845（51.24%） |
| ZoomBench crop | 574/845（67.93%） |
| crop − full | +16.69 pp |
| MMStar | 1057/1500（70.47%） |
| V* Bench | 165/191（86.39%） |

该实验使用 thinking system prompt、随机采样和 max_new_tokens=8192，与后续论文对齐协议不同，因此定位为旧协议工程诊断结果，不作为训练后模型的主比较基线。

### 3. 定位解析失败和长度触顶问题

旧协议出现 204/3,381（6.03%）选择题输出无法可靠解析，以及 301/3,381（8.90%）达到 8,192 token 上限。逐样本交叉分析确认：

- 204 条解析失败中有 201 条同时触及 token 上限；
- 只有 3 条是在正常停止后仍输出错误格式；
- 100 条虽然触顶，但截断前已有可解析答案，因此触顶不自动等于错误；
- 3,381 条请求的推理错误为 0，问题不是 GPU、vLLM 或数据缺失。

根因是旧 system prompt 强制先展开推理再给最终答案，叠加 temperature、top-p、top-k 和 presence penalty 后容易长时间发散。单纯提高 max_new_tokens 可能减少部分截断，但会显著增加成本和延迟，也不会解决格式遵循问题。因此没有回改旧基线，而是在独立论文对齐实验中使用公开的非思考、确定性生成配方。

### 4. 建立论文对齐协议和 Judge 替代边界

对论文、公开仓库和现有脚本核对后，确认旧 Day 5 协议并非论文 Table 2 的推理配方。新协议冻结为：

- 无 system prompt；
- enable_thinking=false；
- temperature=0；
- max_tokens=1024；
- 不传 seed、top-p、top-k、presence/repetition penalty；
- 16 个并发 worker，失败最多重试 3 次；
- 主评测只使用 ZoomBench full 845、MMStar 1500、V* 191，共 2,536 请求。

论文 Judge GPT-OSS-120B 无法获得，也没有后续部署条件。解决方法是固定原始 Qwen3.5-4B Base 作为所有 checkpoint 的统一 Judge，禁止训练后模型自评，并在配置、报告和论文复现声明中明确该限制。

### 5. 完成 R1、R2、R3 工程迭代

R1 建立了论文对齐推理、Judge、评分和恢复主链路。Smoke 后发现三项仍可控制的公开配方差异，R2 完成修正：

1. 服务改为 TP=1、GPU memory utilization=0.75、GDN Triton；
2. ZoomBench/MMStar 保留源图字节；
3. V* 请求图始终转换为 RGB PNG。

正式 Base 运行前实例从双卡无损切换为单卡，因此 R3 只修订计费和 GPU 数量，不改变 R2 的推理语义。R2 Smoke 可以继承，正式运行使用 R3 单卡配置。

#### R1、R2、R3 配置明细

三版都属于 `E-PAPER-BASEJUDGE-001`，共同冻结以下核心评测契约：

| 配置项 | R1/R2/R3 共同值 |
|---|---|
| 适用模型 | Base、Vision-OPD、Cached Prefix、GRPO；只允许被测 checkpoint 不同 |
| 主请求 | ZoomBench full 845 + MMStar 1500 + V* 191 = 2,536 |
| V* 口径 | 保留全部官方样本，分母 191，不排除 4 条重叠 |
| 消息 | 无 system prompt；1 张 full image + user text |
| 生成 | `enable_thinking=false`、`temperature=0`、`max_tokens=1024`、单返回序列 |
| 请求参数 | 不传 top-p、top-k、presence/repetition penalty；最终 R3 明确禁止 seed |
| 执行 | 16 workers、最多重试 3 次、请求超时 3,600 秒 |
| 评分 | MathRuler → MMStar/V* 首字母匹配 → 固定 Base Judge |
| Judge | 原始 Qwen3.5-4B Base；无 system prompt；非思考；temperature 0；max_tokens 2048 |
| 失败策略 | API/空输出/无效 Judge 重试后留在官方分母并计错 |
| 恢复键 | `benchmark + NUL + view + NUL + sample_uid`，逐样本 append/flush 和原子压缩 |
| 输出 | predictions、judge_results、scores、summary、resume_status、manifest、metrics、cost、hash |

三版之间的差异和最终处置如下：

| 项目 | R1 | R2 | R3 |
|---|---|---|---|
| 定位 | 首个论文对齐实现与 Smoke | 修正 3 项可控公开配方差异 | 单卡正式及后续唯一标准 |
| 配置 SHA256 | `ad2ffc149ca228eb4d18778f1785b1bed036ab4fd444cc7e4a733850580cc595` | `637b538c2e5250443f926d467bc3b237955b79a90c42c35d4485a6ba3aec8fd4` | `e71255e817b11c120b4ac22d7ace81d12ffe01e25f7ea94de2e2ffb62e592903` |
| Amendment SHA256 | `b74c28646a2396c0b5c790ed67c3e9163b1c56a8d5a66a9c7731d60f349862a0` | `4aa0427bb31579809fe988935eb735b7d74204e860017b4108d3f867cf66f207` | `6f85950a0f566cd61d9cb957842da8dbc92478acf68c66059a9ed6454db92b60` |
| 服务 | TP=2、显存利用率 0.80，显式 dtype/调度限制 | TP=1、显存利用率 0.75、GDN Triton | 与 R2 相同 |
| Zoom/MM 图像 | 冻结准备时解码并重新编码 PNG | 保留官方源图字节 | 与 R2 相同 |
| V* 请求图 | 小于等于 20 MiB 保留源字节，仅超限时转 PNG | 始终解码为 RGB PNG，超 20 MiB 再缩小 | 与 R2 相同 |
| 计费元数据 | 双卡实例口径 | 双卡实例口径，但实际推理 TP=1 | 单卡 1 GPU，5.98 元/墙钟小时 |
| Smoke 目录 | `smoke/base` | `smoke_r2/base` | 继承已 PASS 的 R2 Smoke，无需重复 |
| 正式 Base | 未运行 | 未运行 | `base/`，自动 Gate PASS |
| 当前状态 | 历史 Smoke；YAML 已删除 | 历史 Smoke；YAML 已删除 | 唯一现行 YAML，禁止原地修改 |

R1 的数据准备会重新编码 ZoomBench/MMStar 图片，因此相应转换 JSON 哈希与 R2/R3 不同。R2 已核验 2,536 张主图和 845 张 crop 图哈希；R3 不改变数据或推理语义，只改变正式实例的 GPU 数量与成本口径。R3 的正式 Base 结果可直接作为三个训练模型的统一比较基线。

历史身份仍由以下文件保留：

- `paper_aligned_evaluation_amendment.yaml`：R1 协议；
- `paper_alignment_r2_amendment.yaml`：R2 三项修正；
- `paper_alignment_r3_single_gpu_cost_amendment.yaml`：R3 单卡成本修订；
- 三份 `.sha256`：已删除 R1/R2 YAML 的档案哈希及现行 R3 校验。

### 6. 完成正式 R3 Base 评测

| Benchmark | 正确/总数 | 准确率 |
|---|---:|---:|
| ZoomBench full | 428/845 | 50.65% |
| MMStar | 1126/1500 | 75.07% |
| V* Bench official | 160/191 | 83.77% |

完整性与资源结果：

- 视觉推理成功 2,536/2,536，推理错误 0；
- 最终评分 2,536/2,536，pending Judge 0；
- MathRuler 直接判对 85 条，选择题首字母直接判对 665 条；
- 需要 Base Judge 1,786 条，其中 1 条最终格式失败，按协议计错；
- 172/2,536（6.78%）达到 1,024 completion token 上限，仍统一解析评分；
- prompt tokens 4,287,226，completion tokens 550,330；
- 客户端观测墙钟：推理 736.89 秒，Judge 37.63 秒；
- 单卡客户端阶段估算成本 1.29 元，不含模型加载、服务关闭和空闲等待。

### 7. 冻结唯一后续评测标准

R3 已写入 docs/benchmark_protocol.md，并同步到项目计划。三个 Benchmark 的内容、指标和完整命令见 docs/benchmark_introduction_and_usage.md。后续 Base、Vision-OPD、Cached Prefix、GRPO 只允许改变被测 checkpoint、模型角色、权重哈希和独立输出目录，其他条件必须完全一致。

旧 R1/R2 YAML 已按负责人要求删除；其 amendment、哈希、Smoke 输出和清单继续保留为历史审计证据。

### 8. 完成 Benchmark 治理同步（2026-08-27）

Day 6 结束后的全仓审计发现主项目配置、README 和部分旧入口仍可能把操作者带回 E-D6。现已完成以下同步：

- `configs/project_1024.yaml` 指向唯一 R3 配置、R3 Base 清单和自动可比性 Gate；
- README 增加项目 R3 入口，并把 `eval/run_eval.sh` 标为 upstream 通用评测；
- 正式推理、Judge、评分、验收及三项数据重建入口强制校验冻结 R3 配置 SHA；
- `model_role=base` 强制校验原始 Qwen3.5-4B Base 权重；
- Vision-OPD、Cached Prefix、GRPO 正式运行必须与 Base 清单逐项比较配置、amendment、数据、请求契约、分母和恢复键；
- 七个 E-D5/E-D6 Python 入口取消默认旧配置，必须显式传 `--config`；
- 新增 `docs/benchmark_history_index.md`，删除四个会污染搜索的 `.orig` 文件；
- 新增机器可读 `benchmark_governance_sync_amendment.yaml`；不修改 R3 配置，因此其 SHA 和正式结果仍有效；
- 新门禁对原正式 Base 的独立复核为 PASS，结果保存为 `preflight/post_sync_base_validation.json`，原 Base `validation.json` 与冻结目录未改写。

同步后项目 vision-opd 环境完整回归为 **76 passed，另含 16 个 subtests passed**。

## 三、遇到的问题与解决方法

| 问题 | 判断 | 解决方法与结果 |
|---|---|---|
| V* 有 4 条确认同底图 | 事实成立，但不能仅凭当前项目子集反推原论文训练集污染 | 不删除官方样本；按负责人决定只报 191，统一用于所有模型 |
| 双卡阶段出现 Killed，两卡显存接近占满 | 当时服务或任务资源压力很高，不能仅凭一行日志断言唯一根因 | 分阶段运行数据准备、推理和 Judge；正式协议使用 TP=1 单卡并保留恢复文件 |
| ZoomBench full 2/4 容易被理解为异常 | 实际是 Smoke 已完成 2/共 4 条的进度，不是 GPU 数或正式集只跑一半 | 核对输出计数和恢复状态后继续；正式集最终为 845/845 |
| 204 条解析失败、301 条触顶 | 主要是旧 thinking + 采样协议诱发长输出，不是 Benchmark 缺失或服务错误 | 保留旧结果；建立独立非思考 R3，不在看到结果后篡改旧配置 |
| 与论文 Qwen3.5-4B 结果不同 | 推理参数、图像处理和 Judge 曾存在差异，且论文 Judge 不可用 | 修正三项可控差异；对不可控 Judge 使用固定 Base 替代并明确限制 |
| 无法使用 GPT-OSS-120B Judge | 当前无接口、权重和部署机会 | 固定原始 Base 4B Judge，所有模型共享，禁止自评 |
| V* 官方来源和编码实现存在差异 | 镜像、转换和请求编码会造成协议漂移 | 冻结 lmms-lab/vstar-bench revision；请求始终 RGB PNG，分母 191 |
| 双卡计费切换为单卡 | 需要避免把费用修订误当作推理协议变化 | 新增 R3 amendment，只改 GPU 数量和每小时成本；推理继承 R2 |
| 正式清单 model_id 为 Vision-OPD-4B | 这是服务别名，容易误解，但不是实际 checkpoint 身份 | 清单同时冻结 Base 路径和两份权重 SHA256，确认加载的是原始 Base；后续使用清晰服务名 |

## 四、旧协议与 R3 为什么不能直接比较

| 项目 | 旧协议 E-D6-001 | R3 主协议 |
|---|---|---|
| thinking | 开启并要求显式推理 | 关闭 |
| generation | temperature 0.7 等采样 | temperature 0 |
| 最大输出 | 8,192 | 1,024 |
| ZoomBench | full + crop | full 主指标 |
| 正式请求数 | 3,381 | 2,536 |
| 服务 | 双卡 TP=2 | 单卡 TP=1 |
| 定位 | 工程诊断 | 后续模型唯一比较基线 |

两套结果回答的是不同协议下的模型表现。R3 不是对旧结果的覆盖或“修分”，而是为论文方向复现建立的新实验族；后续训练模型必须只与 R3 Base 比较。

## 五、最终冻结身份与证据

- 唯一配置：configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml
- 配置 SHA256：e71255e817b11c120b4ac22d7ace81d12ffe01e25f7ea94de2e2ffb62e592903
- R3 amendment SHA256：6f85950a0f566cd61d9cb957842da8dbc92478acf68c66059a9ed6454db92b60
- 原始 Base 路径：/root/autodl-tmp/models/Qwen3.5-4B
- 正式目录：artifacts/runs/E-PAPER-BASEJUDGE-001/base/
- 正式报告：artifacts/reports/base_external_benchmarks_r3_single_gpu.md
- 自动验证：artifacts/runs/E-PAPER-BASEJUDGE-001/base/validation.json，状态 PASS
- 治理同步：artifacts/runs/E-PAPER-BASEJUDGE-001/preflight/benchmark_governance_sync_amendment.yaml
- 同步后独立复核：artifacts/runs/E-PAPER-BASEJUDGE-001/preflight/post_sync_base_validation.json，状态 PASS

## 六、遗留限制与下一步

遗留限制有两项：GPT-OSS-120B Judge 不可获得，所以不能声称精确复现论文；正式清单中的服务别名不够清晰，后续运行应使用与模型角色一致的名称。两项均不影响当前 Base checkpoint 身份或 Day 6 PASS。

下一步进入 Day 7：实现并验证 Vision-OPD 双卡训练入口，以 8～16 条真实数据完成训练 Smoke。训练过程中只使用 internal eval/retention 做开发判断；最终 Vision-OPD checkpoint 冻结后，再用完全相同的 R3 Benchmark 标准与本日 Base 结果比较。
