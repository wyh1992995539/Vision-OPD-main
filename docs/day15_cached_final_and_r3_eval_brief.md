# Day 15 Cached 定版与三组统一 R3 外部评测工作简报

> 后续状态（2026-09-10 UTC）：Day16 的 R4 重评分、Base MMStar 输出上限诊断与根因分析已经完成，见 [Day 16 工作简报](day16_6k_delivery_and_r4_work_brief.md)。Day15 本简报继续保留冻结 R3 结果，不回写为 R4。

> 执行日期：2026-09-08（UTC）  
> 实验 ID：E-D15-6K-FINAL-EVAL-001  
> Cached 新进程冷加载：**PASS，5/5**  
> Pre-score 设计冻结：**PASS**  
> Vision-OPD R3：**PASS，2536/2536**  
> Cached Prefix R3：**PASS，2536/2536**  
> Base / Vision-OPD / Cached 统一比较：**PASS**  
> 确定性 Bad Case 抽样：**PASS，18 条；人工分析留待 Day 16**

## 技术摘要

Day 15 首先对 Day 14 合并后的 Cached Student 运行独立新进程冷加载，固定 5 条训练样本全部得到非空回答，0 inference error，服务受控关闭，FSDP checkpoint 和 merged 模型在测试前后均未变化。

在产生任何新的 Vision-OPD/Cached R3 预测或分数之前，项目冻结了两个最终 checkpoint 身份、唯一 R3 配置、输出 Schema、比较列、固定 denominator 和 Bad Case 抽样规则。随后对 Vision-OPD 与 Cached 各运行 4×3 Smoke；两组均通过 12/12 prediction、score、Judge 和 validation Gate。Smoke 通过后，两组正式评测各完成 2,536/2,536 条预测和评分，并使用同一个冻结原始 Qwen3.5-4B Base Judge。已有 Base R3 结果保持只读，三组比较使用完全一致的数据、输入、生成、解析、Judge 和分母。

主要结果是：Vision-OPD 在 ZoomBench 从 Base 的 50.65% 提高到 52.54%，但 MMStar 和 V* 分别降至 68.00% 和 75.92%。Cached 在 ZoomBench 为 51.36%，V* 为 77.49%，后者比 Vision-OPD 高 1.57 个百分点；其 MMStar 为 62.47%，低于 Base 12.60 个百分点。结果没有触发 checkpoint 重选、协议修改或重新训练。

## 一、实际完成的工作

| 顺序 | 实际工作 | 主要结果 | 状态 |
|---:|---|---|---|
| 1 | Cached 最终模型新进程冷加载 | 5/5 非空，0 inference error，源 checkpoint 与 merged 模型未变化 | PASS |
| 2 | 实现比较工具 | 固定读取 Base/Vision-OPD/Cached 的 manifest、validation、scores 和 summary | PASS |
| 3 | 实现 Bad Case 工具 | 六个互斥变化类型，固定 seed，SHA256 排序，最多 18 条 | PASS |
| 4 | 冻结 pre-score 设计 | 两个模型身份、R3 SHA、Schema、denominator 和规则在新分数前冻结 | PASS |
| 5 | 冻结四份运行清单 | 两组 12 条 Smoke、两组 2,536 条正式清单；正式可比性 Gate PASS | PASS |
| 6 | Vision-OPD/Cached Smoke 推理 | 各 12/12，均为 0 inference error | PASS |
| 7 | Smoke Judge、评分和 Gate | Judge 10/10 与 8/8；两组 validation 均为 pass | PASS |
| 8 | Vision-OPD 正式推理 | 2,536/2,536，0 inference error | PASS |
| 9 | Cached 正式推理 | 2,536/2,536，0 inference error | PASS |
| 10 | 固定 Base Judge | Vision-OPD 1,989/1,989；Cached 1,936/1,936；pending 0 | PASS |
| 11 | 正式评分与 validation | 两组 predictions/scores 均 2,536，V* 分母 191 | PASS |
| 12 | 三组统一比较 | Base、Vision-OPD、Cached 请求键与协议身份完全一致 | PASS |
| 13 | 确定性 Bad Case 抽样 | 六个类型各 3 条，共 18 条；未修改分数或 checkpoint | PASS |
| 14 | 回归验证 | Day15 与 R3 pipeline 共 18 tests passed | PASS |

## 二、Cached 最终模型冷加载

| 项目 | 实际结果 |
|---|---:|
| 来源 FSDP checkpoint | E-D14-6K-CACHED-001/global_step_780 |
| merged 模型 | E-D14-6K-CACHED-001/merged_hf |
| 固定样本数 | 5 |
| 非空回答 | 5/5 |
| inference error | 0 |
| finish reason | 5/5 stop |
| 服务关闭退出码 | 0 |
| checkpoint 测试前后 | 未变化 |
| merged 模型测试前后 | 未变化 |
| 能力评测 | 否；仅验证独立加载与功能推理 |

这一步使用全新 vLLM 进程，不复用训练或合并进程。冷加载使用训练样本只验证功能，不产生准确率结论。

## 三、评测前冻结

pre-score freeze receipt 生成于 2026-09-08 14:32:40 UTC。当时 Vision-OPD 和 Cached 的 Smoke/正式目录均没有 predictions、scores、summary 或 validation 等新评分产物。

冻结内容包括：

- 唯一 R3 配置 SHA256：e71255e817b11c120b4ac22d7ace81d12ffe01e25f7ea94de2e2ffb62e592903。
- Base R3 manifest 和 validation 身份。
- Vision-OPD 与 Cached 的 merged manifest、权重 SHA256 和 5 条冷加载凭据。
- 正式请求数：ZoomBench 845、MMStar 1,500、V* 191，总计 2,536。
- 主键：benchmark、view、sample_uid。
- 比较输出列与百分点差值。
- 六个互斥 Bad Case 类型、固定 seed、SHA256 确定性排序、每类最多 3 条、总计最多 18 条。
- 禁止根据外部分数修改评分、重选 checkpoint 或改变协议。

## 四、Smoke Gate

| 模型 | Predictions | Scores | Judge | Inference errors | Validation |
|---|---:|---:|---:|---:|---|
| Vision-OPD | 12/12 | 12/12 | 10/10 | 0 | pass |
| Cached Prefix | 12/12 | 12/12 | 8/8 | 0 | pass |

Smoke 每个 Benchmark 固定 4 条，只用于工程 Gate。正式清单和模型身份在 Smoke 前已经冻结，Smoke 分数没有用于模型选择或协议修改。

## 五、正式 R3 结果

### 1. 主结果

| Benchmark | N | Base | Vision-OPD | VOPD−Base | Cached | Cached−Base | Cached−VOPD |
|---|---:|---:|---:|---:|---:|---:|---:|
| ZoomBench | 845 | 50.65% | 52.54% | +1.89 pp | 51.36% | +0.71 pp | −1.18 pp |
| MMStar | 1,500 | 75.07% | 68.00% | −7.07 pp | 62.47% | −12.60 pp | −5.53 pp |
| V* Bench | 191 | 83.77% | 75.92% | −7.85 pp | 77.49% | −6.28 pp | +1.57 pp |
| Micro 汇总 | 2,536 | 67.59% | 63.45% | −4.14 pp | 59.90% | −7.69 pp | −3.55 pp |

Micro 汇总只反映 2,536 条按样本加权的整体值。三项 Benchmark 的主结果应分别报告，不能用 micro 值替换各自官方 denominator。

### 2. 配对变化

| 比较 | 错→对 | 对→错 | 净变化 |
|---|---:|---:|---:|
| Base → Vision-OPD | 306 | 411 | −105 |
| Base → Cached | 287 | 482 | −195 |
| Vision-OPD → Cached | 255 | 345 | −90 |

Vision-OPD 对 ZoomBench 有小幅收益，但在 MMStar 和 V* 上未保持 Base 能力。Cached 相对 Vision-OPD 在 V* 有小幅恢复，在 ZoomBench 和 MMStar 上更低。该结果说明当前训练配置没有在三项外部能力上形成一致提升。

## 六、完整性与运行诊断

| 指标 | Vision-OPD | Cached |
|---|---:|---:|
| Predictions / Scores | 2,536 / 2,536 | 2,536 / 2,536 |
| Inference errors | 0 | 0 |
| Judge required / completed | 1,989 / 1,989 | 1,936 / 1,936 |
| Judge failures | 0 | 1 |
| Pending Judge | 0 | 0 |
| 触达 1,024-token 上限 | 163 | 138 |
| Completion tokens | 448,933 | 345,021 |
| 推理客户端墙钟 | 618.08 s | 543.38 s |
| Judge 客户端墙钟 | 33.81 s | 24.47 s |
| 估算费用 | 1.08 元 | 0.94 元 |

费用按 R3 单卡 5.98 元/小时、客户端观测到的推理与 Judge 阶段计算，不包括模型服务启动、关闭和空闲时间。两组新增正式评测估算合计约 2.03 元。

Cached 的 1 条 Judge failure 按冻结协议保留并计错，记录为 finalized，因此 Judge required/completed 仍一致且 pending 为 0。评分过程中 MathRuler 对个别 LaTeX 宏给出警告；逐样本 rule error 被保留，后续仍按冻结的 first-letter/Base Judge 链路处理，没有中断评分或改变分母。

## 七、Bad Case 冻结抽样

Bad Case 工具按预先冻结的六个互斥正确性变化类型抽取，每类 3 条，共 18 条：

| 类型 | 候选数 | 抽样数 |
|---|---:|---:|
| Base 错，Vision 与 Cached 都对 | 190 | 3 |
| Base 错，Vision 对、Cached 错 | 116 | 3 |
| Base 与 Vision 错、Cached 对 | 97 | 3 |
| Base 对，Vision 与 Cached 都错 | 253 | 3 |
| Base 对、Vision 错、Cached 对 | 158 | 3 |
| Base 与 Vision 对、Cached 错 | 229 | 3 |

选择顺序由固定 seed 与 request key 的 SHA256 决定。当前只完成机器抽样，人工解释、overlap 标注和代表案例报告属于 Day 16；抽样不会反向改变任何 score。

## 八、阶段结论与边界

Day 15 计划中的 Cached 冷加载、评测设计冻结、两组 Smoke、两组 2,536 条正式 R3、固定 Base Judge、完整评分、validation、三组比较和 Bad Case 机器抽样均已完成。

当前最终结论基于唯一冻结 R3 协议。Base 使用既有冻结正式结果，没有重复运行；Vision-OPD 和 Cached 只更换被测 checkpoint 和独立输出目录。所有失败与无效结果均保留在固定分母内。

Day 15 没有重选 checkpoint、重新训练或修改 R3。Day 15 完成时将人工分析和协议复核列为下一步；后续实际执行的 R4、输出上限诊断及其边界登记在 [Day 16 工作简报](day16_6k_delivery_and_r4_work_brief.md)。

## 九、关键证据索引

- Day15 冻结配置：configs/day15_final_eval_freeze.yaml
- Pre-score freeze：artifacts/runs/E-D15-6K-FINAL-EVAL-001/preflight/eval_design_freeze.json
- Cached 冷加载：artifacts/runs/E-D15-6K-FINAL-EVAL-001/cached_prefix/cold_reload/reload_validation_summary.json
- Vision-OPD Smoke：artifacts/runs/E-D15-6K-FINAL-EVAL-001/smoke/vision_opd/
- Cached Smoke：artifacts/runs/E-D15-6K-FINAL-EVAL-001/smoke/cached_prefix/
- Vision-OPD 正式 R3：artifacts/runs/E-D15-6K-FINAL-EVAL-001/vision_opd/
- Cached 正式 R3：artifacts/runs/E-D15-6K-FINAL-EVAL-001/cached_prefix/
- 三组比较：artifacts/runs/E-D15-6K-FINAL-EVAL-001/comparison.json
- 对比报告：artifacts/reports/prefix_ablation_6241.md
- Bad Case 样本：artifacts/runs/E-D15-6K-FINAL-EVAL-001/badcases.jsonl
- Bad Case 汇总：artifacts/runs/E-D15-6K-FINAL-EVAL-001/badcases_summary.json
- 比较实现：eval/compare_experiments.py
- Bad Case 实现：eval/build_badcases.py
- R3 会话入口：scripts/run_day15_r3_stage.py
