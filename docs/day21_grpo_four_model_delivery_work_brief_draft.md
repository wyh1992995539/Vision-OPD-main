# Day 21 四模型比较、失败分析与最终交付工作简报（待定草案）

> 文档状态：**DRAFT / NOT EXECUTED**  
> 比较对象：Base、Vision-OPD、Cached Prefix、GRPO  
> 启动前提：Day20 GRPO 固定 1,024-token R4 评测完整通过  
> 说明：本简报只提供最终交付结构；所有 `[待填]` 均为运行后才能取得的数值，并须来自冻结机器证据。

## 技术摘要

若 Day21 正常完成，应在相同 R4 数据、输入、生成、解析、Judge 和固定分母上比较四个模型，并分别报告 GRPO 相对 Base、Vision-OPD、Cached 的百分点变化、配对修正/退化、类别差异、输出格式与长度、训练成本和失败案例。结果无论是否优于前三组，都应完整保留，不能据此外推多 seed 稳定性或修改已冻结 checkpoint。

## 一、实际完成的工作（待执行成功后确认）

| 顺序 | 具体操作 | 主要结果与证据 | 状态 |
|---:|---|---|---|
| 1 | 读取四模型冻结 summary/validation，校验 Benchmark、view、sample UID 请求键集合完全一致 | four-model identity gate | 预期 PASS |
| 2 | 按 Benchmark 计算 accuracy、正确数、固定分母和 GRPO 对三组的百分点差值 | 四模型主表 | 预期 PASS |
| 3 | 在共同请求键上计算错→对、对→错和净变化，按样本执行配对 bootstrap | paired comparison | 预期 PASS |
| 4 | 复用训练 overlap 清单，分别报告 overlap/non-overlap、类别、invalid、length 和 Judge 路由结果 | 分层诊断表 | 预期 PASS |
| 5 | 在查看案例内容前冻结 Bad Case 类型、seed、SHA256 排序和每类上限，再生成候选和人工复核表 | badcase freeze/results | 预期 PASS |
| 6 | 汇总 Base/VOPD/Cached/GRPO 的训练轨迹来源、损失、Teacher/reference、可更新参数、GPU 小时和费用 | 方法与成本表 | 预期 PASS |
| 7 | 更新最终报告、README、面试问答和证据索引；每个数字链接到机器产物 | 可投递文档集 | 预期 PASS |
| 8 | 运行数据/评测/文档回归、链接和 SHA256 检查，冻结完成凭据与里程碑状态 | Day21 completion receipt | 预期 PASS |

汇总脚本只读取冻结产物，不重新生成回答或修改任何 score。所有表格先保存机器可读 JSON/CSV，再渲染 Markdown；人工 Bad Case 注释与原始评分分开保存，不能反向改判。

## 二、四模型 R4 主表

| Benchmark | N | Base | Vision-OPD | Cached | GRPO | GRPO−Base | GRPO−VOPD | GRPO−Cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ZoomBench | 845 | 50.65% | 52.54% | 51.36% | `[待填]` | `[待填]` | `[待填]` | `[待填]` |
| MMStar | 1,500 | 68.00% | 59.20% | 56.33% | `[待填]` | `[待填]` | `[待填]` | `[待填]` |
| V* Bench | 191 | 82.72% | 71.73% | 73.30% | `[待填]` | `[待填]` | `[待填]` | `[待填]` |
| Micro | 2,536 | 63.33% | 57.93% | 55.95% | `[待填]` | `[待填]` | `[待填]` | `[待填]` |

主结论必须按三个 Benchmark 分开报告。Micro 只表示 2,536 条按样本加权的汇总，不能替代各 Benchmark 官方分母。

## 三、配对变化与不确定性

| 比较 | 错→对 | 对→错 | 净变化 | 配对 bootstrap 区间 |
|---|---:|---:|---:|---:|
| Base → GRPO | `[待填]` | `[待填]` | `[待填]` | `[待填]` |
| Vision-OPD → GRPO | `[待填]` | `[待填]` | `[待填]` | `[待填]` |
| Cached → GRPO | `[待填]` | `[待填]` | `[待填]` | `[待填]` |

置信区间只描述固定样本上的配对不确定性。当前训练是单 seed，不能据此声称训练随机性稳定或统计显著优于其他训练方法。

## 四、训练侧与输出侧指标

| 指标 | GRPO 实际值 |
|---|---:|
| 有效训练 prompt | 6,240 |
| 有效 rollout | 24,960 |
| optimizer updates | 780 |
| mean training Reward | `[待填]` |
| mixed-group fraction | `[待填]` |
| parse-invalid rate | `[待填]` |
| length rate | `[待填]` |
| A/B/C/D 输出分布 | `[待填]` |
| 总 completion tokens | `[待填]` |
| GPU 小时/墙钟 | `[待填]` |
| 峰值显存/CPU/磁盘 | `[待填]` |
| 估算费用 | `[待填]` |

训练 Reward 与外部 R4 accuracy 必须分开报告。规则 Reward 提升不自动证明外部 Benchmark 提升。

## 五、Bad Case 冻结抽样

建议冻结 16～24 条，覆盖以下互斥或可审计类型：

- Base 错、GRPO 对；
- Base 对、GRPO 错；
- Vision-OPD 与 GRPO 分歧；
- Cached 与 GRPO 分歧；
- GRPO 高 Reward 但 R4 错；
- GRPO 低/零 Reward 但外部结果对；
- 解析失败、冲突答案和长度截断；
- train-6241 overlap 与非 overlap 分层。

| 项目 | 实际结果 |
|---|---:|
| 候选数 | `[待填]` |
| 固定抽样数 | `[待填]` |
| 人工复核数 | `[待填]` |
| Reward 投机案例 | `[待填]` |
| 解析/长度失败案例 | `[待填]` |

抽样规则、seed 和排序必须在阅读案例内容前冻结。某一类型不存在时应记为 0，不能更换为更有利的案例。

## 六、方法与成本边界

最终报告需要明确：

- Vision-OPD 使用在线 Student 轨迹、crop Teacher、JSD/EMA；
- Cached Prefix 使用预先缓存的 Base 轨迹；
- GRPO 使用每题 4 条在线 rollout、确定性规则 Reward 和组内相对优势；
- 三个训练分支不属于只改变单一 objective 的严格单变量实验；
- GRPO 是看到前三组结果后设计的探索性扩展；
- 本地 R4 Judge 是 Qwen3.5-4B Base，未复现论文 GPT-OSS-120B Judge；
- 单训练 seed、双卡资源缩放和 response length 均限制结论外推。

## 七、可能问题与处置

| 可能问题 | 识别信号 | 具体处置 |
|---|---|---|
| 四模型请求集合不一致 | 缺键、重复键或分母不同 | 阻断主表，回到各自 validation 定位；不能用 inner join 静默缩小分母 |
| 把自适应 Base 72.73% 混入固定 R4 | 主表协议列不一致 | 固定 1,024 主表只使用 Base 68.00%；自适应结果单独列为诊断 |
| Micro 掩盖单项退化 | 总体提升但某 Benchmark 明显下降 | 三项结果优先逐项报告，Micro 仅作附加汇总 |
| overlap 处理改变主分母 | 去重后准确率被当作主结果 | 官方完整集合继续作为主结果；overlap/non-overlap 只做分层诊断 |
| Bad Case 选择偏向有利案例 | 看过内容后改变类别或 seed | 使用预先冻结的类型、seed 和哈希排序；空类别如实记 0 |
| Reward 与 R4 结论混淆 | 用训练 Reward 宣称外部能力提升 | 两套指标分表呈现，解释其目标和解析协议不同 |
| 单 seed 被写成稳定提升 | 没有重复训练却使用稳定/显著表述 | 明确单 seed 探索性限制；bootstrap 只描述固定测试样本不确定性 |
| Judge/解析失败被排除 | denominator 小于请求数 | 所有失败保留并按冻结规则计错，单列数量和原始错误 |
| 结果不符合预期 | GRPO 没有全面超过前三组 | 保留结论和失败分析；不修改 checkpoint、Reward 或外评协议来追求目标结果 |
| 文档数字无法回溯 | 表格与 JSON/log 不一致 | 阻断交付，逐项绑定源路径和 SHA256，通过自动校验后再发布 |

## 八、阶段结论与边界（待真实结果确认）

> Day21 四模型交付预期 **PASS**。GRPO 固定 1,024-token R4 在 ZoomBench、MMStar、V* 上分别为 `[待填]`、`[待填]`、`[待填]`，相对 Base 分别变化 `[待填]`、`[待填]`、`[待填]` pp。四模型均使用 2,536 个相同请求键和冻结 R4 分母。GRPO 训练完成 24,960 条有效轨迹与 780 次更新，总 GPU 小时 `[待填]`，估算费用 `[待填]`。结论基于单 seed 和本地替代 Judge，不声称完整复现论文或跨 seed 稳定提升。

最终交付必须同时呈现提升、持平和退化项，并保留失败样本与协议限制。当前比较属于单 seed、不同训练路线之间的探索性结果，不能写成严格单变量因果结论，也不能改变 Day15–16 已冻结的三模型成绩。

## 九、关键证据索引

- 四模型逐 Benchmark 主表与配对变化表
- 类别/overlap/invalid/length 分层报告
- 固定 Bad Case 清单及人工分析
- GRPO 训练与评测成本表
- 最终模型、配置、命令、日志和 SHA256 索引
- 更新后的最终报告、README 和面试问答
- Day18–21 完成凭据与最终里程碑状态
