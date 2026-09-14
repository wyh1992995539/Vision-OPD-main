# Day 16 6K 项目复核、R4 重评分与输出上限诊断工作简报

> 执行日期：2026-09-09～2026-09-10（UTC）  
> 三模型固定 1,024-token R4：**PASS，3 × 2,536/2,536**  
> R4 新增 Base Judge：**190/190，失败 0，pending 0**  
> MMStar Base 自适应输出上限修复：**COMPLETE，1,091/1,500 = 72.73%**  
> Vision-OPD/Cached 自适应重跑：**DEFERRED_BY_USER_NOT_STARTED**

## 技术摘要

Day 16 没有重新训练或选择 checkpoint。实际工作集中在三项评测可信度修复：首先修复 R3 选择题解析器会从任意英文大写字母回退取答案的问题，在不改写 R3 原始预测的前提下建立 R4；随后对 Base、Vision-OPD、Cached Prefix 的同一批 2,536 条冻结输出重新解析、路由和评分；最后针对 Base MMStar 中 163 条 `finish_reason=length` 样本执行逐级、自适应的生成上限诊断与收口。

固定 1,024-token R4 下，Vision-OPD 只在 ZoomBench 高于 Base；Vision-OPD 和 Cached 在 MMStar、V* 均低于 Base。Base MMStar 将截断样本逐级扩展到 131,072 tokens 后，从 68.00% 恢复至 72.73%，说明 1,024-token 上限是重要误差来源，但仍无法单独解释与论文 78.53% 的全部差距。

## 一、实际完成内容

| 工作 | 结果 | 状态 |
|---|---|---|
| 冻结 R3 输入和文件哈希 | R4 前后 R3 源文件未变化 | PASS |
| 修复选择题解析与 Judge 路由 | 明确错误不再因 `Answer` 等单词首字母被误判 | PASS |
| 三模型 R4 重评分 | Base、Vision-OPD、Cached 各 2,536/2,536 | PASS |
| R4 新增 Judge | 190/190，失败 0，pending 0 | PASS |
| 三模型 R4 对比 | 请求键、模型身份、分母和协议完整 | PASS |
| Base MMStar 2K/4K Gate 诊断 | 71.47%，未进入论文 ±2 pp 区间 | COMPLETE |
| Base MMStar 自适应上限扩展 | 1,091/1,500，72.73% | COMPLETE |
| 最后一条循环输出收口 | 标记 `terminal_generation_degeneracy` 并计错 | COMPLETE |
| 论文差距根因分析 | 解析、输出上限、Judge、checkpoint/runtime 边界已拆分 | COMPLETE |
| Vision-OPD/Cached 自适应上限评测 | 用户明确暂缓，未产生新输出 | NOT STARTED |

## 二、三模型固定 1,024-token R4 主结果

| 模型 | ZoomBench | MMStar | V* Bench | 三项 Micro |
|---|---:|---:|---:|---:|
| Base | 50.65% | 68.00% | 82.72% | 63.33% |
| Vision-OPD | 52.54% | 59.20% | 71.73% | 57.93% |
| Cached Prefix | 51.36% | 56.33% | 73.30% | 55.95% |

模型间差值：

| Benchmark | Vision-OPD − Base | Cached − Base | Cached − Vision-OPD |
|---|---:|---:|---:|
| ZoomBench | +1.89 pp | +0.71 pp | −1.18 pp |
| MMStar | −8.80 pp | −11.67 pp | −2.87 pp |
| V* Bench | −10.99 pp | −9.42 pp | +1.57 pp |

R4 使用修正后的保守选项解析和冻结 Qwen3.5-4B Base Judge。它适合项目内部统一比较，但本地 Judge 不等于论文使用的 GPT-OSS-120B Judge，因此不能写成论文评测的精确复现。

## 三、R3 到 R4 的影响

| 模型 | ZoomBench | MMStar | V* Bench | Micro |
|---|---:|---:|---:|---:|
| Base | +0.00 pp | −7.07 pp | −1.05 pp | −4.26 pp |
| Vision-OPD | +0.00 pp | −8.80 pp | −4.19 pp | −5.52 pp |
| Cached Prefix | +0.00 pp | −6.13 pp | −4.19 pp | −3.94 pp |

Base MMStar 的 R3 为 1,126/1,500（75.07%），R4 为 1,020/1,500（68.00%）。两者使用同一份模型输出；差异来自解析和评分路由。全量迁移中有 116 条 R3 对转为 R4 错、10 条 R3 错转为 R4 对，证明旧首字母回退会产生方向不对称的虚高。

## 四、Base MMStar 输出上限诊断与收口

| 最终尝试上限 | 本档请求数 | 本档后仍为 length | 混合总正确 | 混合准确率 |
|---:|---:|---:|---:|---:|
| 1,024 | 1,500 | 163 | 1,020 | 68.00% |
| 2,048 | 163 | 83 | 1,065 | 71.00% |
| 4,096 | 83 | 46 | 1,072 | 71.47% |
| 8,192 | 46 | 21 | 1,086 | 72.40% |
| 16,384 | 21 | 7 | 1,089 | 72.60% |
| 32,768 | 7 | 3 | 1,091 | 72.73% |
| 65,536 | 3 | 1 | 1,091 | 72.73% |
| 131,072 | 1 | 1 | 1,091 | 72.73% |

原始 163 条截断样本中，162 条最终自然结束。最后一条在 131,072 tokens 仍循环，规范化段落重复率为 98.58%，没有产生最终选项，R4 无法解析且冻结 Judge 返回 No。该样本保留为模型生成退化并计错，最终 `unresolved_output_cap_count=0`。

72.73% 是选择性自适应混合结果，不能替代上表三模型统一固定 1,024-token R4 的 Base 68.00%，也不能与尚未执行自适应重跑的 Vision-OPD/Cached 直接比较。

## 五、结论与边界

Day 16 证明了两个独立问题：旧首字母解析会虚增部分选择题分数；固定 1,024-token 上限会使一部分长回答被截断。修复两者后，Base MMStar 仍比论文 78.53% 低 5.80 pp，剩余差异可能来自 Judge、精确 checkpoint revision、运行时和论文未披露配置，现有证据不能继续细分其贡献。

本日没有改写 R3、没有根据结果重选模型、没有重训 Vision-OPD/Cached。原排期中的统一 `docs/final_report.md` 和 `docs/interview_qa.md` 在当前仓库中不存在；本简报只登记实际已有的 R4、输出上限修复和根因分析证据，不把未找到的交付物写成已完成。

## 六、证据索引

- R4 专题简报：`docs/day16_r4_basejudge_rescore_brief.md`
- 输出上限修复简报：`docs/day16_mmstar_base_output_cap_repair_brief.md`
- 2K/4K Gate 简报：`docs/day16_mmstar_base_max_tokens_diagnostic_brief.md`
- 根因分析：`docs/day16_mmstar_base_gap_root_cause_analysis.md`
- 三模型详细对比：`docs/day16_paper_vs_local_model_comparison.md`
- R4 根凭据：`artifacts/runs/E-PAPER-BASEJUDGE-R4-001/`
- 自适应上限根凭据：`artifacts/runs/E-MMSTAR-BASE-MAXTOK-EXT-001/`
- R4 validation：`artifacts/runs/E-PAPER-BASEJUDGE-R4-001/validation.json`
- R4 comparison：`artifacts/runs/E-PAPER-BASEJUDGE-R4-001/comparison.json`
- 自适应完成凭据：`artifacts/runs/E-MMSTAR-BASE-MAXTOK-EXT-001/completion_receipt.json`
- 终止收口凭据：`artifacts/runs/E-MMSTAR-BASE-MAXTOK-EXT-001/terminal_closure/completion_receipt.json`
