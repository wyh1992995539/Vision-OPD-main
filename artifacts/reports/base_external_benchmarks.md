# Day 6 Base 外部 Benchmark 正式报告

## 技术摘要

原始 Qwen3.5-4B Base 已完成 3,381 个冻结请求，逐样本预测与评分均为 3,381 条，推理错误、重复键和待定 Judge 均为 0。ZoomBench full 为 **51.24%**，crop 为 **67.93%**，crop−full gap 为 **16.69 个百分点**；MMStar 为 **70.47%**；V* Bench 官方 191 条为 **86.39%**。

## 总体结果

| 指标 | 正确/总数 | 准确率 |
|---|---:|---:|
| ZoomBench full | 433/845 | 51.24% |
| ZoomBench crop | 574/845 | 67.93% |
| ZoomBench crop−full gap | — | +16.69 pp |
| MMStar | 1057/1500 | 70.47% |
| V* Bench 官方全集 | 165/191 | 86.39% |

## 官方类别结果

| Benchmark / 类别 | 正确/总数 | 准确率 |
|---|---:|---:|
| mmstar/full/coarse perception | 173/250 | 69.20% |
| mmstar/full/fine-grained perception | 155/250 | 62.00% |
| mmstar/full/instance reasoning | 190/250 | 76.00% |
| mmstar/full/logical reasoning | 201/250 | 80.40% |
| mmstar/full/math | 202/250 | 80.80% |
| mmstar/full/science & technology | 136/250 | 54.40% |
| vstar/full/direct_attributes | 97/115 | 84.35% |
| vstar/full/relative_position | 68/76 | 89.47% |

## 输出质量、Token 与延迟

- 推理错误：**0/3,381**。
- 无法可靠解析的选择题输出：**204/3,381（6.03%）**，均按官方分母计错。
- 达到 8,192 token 上限：**301/3,381（8.90%）**；触顶不自动等于无效，仍按最终答案解析。
- Prompt token：**5,013,561**；Completion token：**6,697,748**；合计：**11,711,309**。
- 请求延迟：均值 **14.82s**，中位数 **7.25s**，P95 **56.40s**，最大 **77.77s**。

## GPU 时间和成本

- 双卡 vLLM 服务时间：**1.962 小时**（7063 秒）。
- 双卡价格：**11.96 元/小时**。
- GPU 实例费用：**23.46 元**。
- 外部 Judge API 费用：**0.00 元**；语义 Judge 使用同一台本地 Base 4B 服务。
- 总费用：**23.46 元**。

## 范围与方法

评测使用冻结的 Qwen3.5-4B Base、Benchmark revision、Prompt、图像处理、生成与评分协议。ZoomBench 对 845 条分别执行 full/crop；MMStar 使用 1,500 条官方样本；V* 使用官方 191 条和 191 分母，不生成 187 条二级指标。选择题由冻结解析器判分；ZoomBench 开放题依次使用确定性数字、MathRuler 和固定 Base 4B Judge。

## 验证与限制

完整性检查全部通过：请求/评分数量、唯一键、四组分母、零推理错误、零待定 Judge、V* 191 分母及零重复均已复算。结果是冻结快照上的描述性基线，不构成统计显著性或因果结论。长度触顶与解析失败应在后续模型比较中使用同一规则。该报告使用精确表格而非图表，因为只有一个评测快照，表格更适合审计固定分母和类别值。

## 下一步

冻结本 Base 结果与哈希，更新 Day 6 为 PASS；后续 Vision-OPD 与 Cached Prefix 仅在内部 checkpoint 冻结后使用完全相同的外评协议比较，不用外部分数选 checkpoint 或重训。

## 进一步问题

训练后模型是否提升 ZoomBench full、是否保持 MMStar/V*，以及输出触顶与无效解析率是否变化，应在统一外评阶段做逐样本配对比较。
