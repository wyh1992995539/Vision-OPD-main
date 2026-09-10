# Day16 MMStar Base 最大生成长度诊断简报

> 实验：E-MMSTAR-BASE-MAXTOK-DIAG-001  
> 状态：**PASS，Gate 未通过，按预设规则停止完整重跑**  
> 目标：验证将 Base 的 MMStar 评测生成上限从 1,024 提升到 2,048/4,096，能否使本地结果进入论文 Base 78.53% 的 ±2 pp 范围。

## 冻结设计

- 基线：本地 Base R4，1,020/1,500，68.00%。
- 冻结选择：R4 中 163 条 finish_reason=length 的 MMStar 样本。
- 其他条件不变：相同 Base 权重、Prompt、图像、temperature=0、non-thinking、vLLM 参数、R4 解析器和冻结 Qwen3.5-4B Base Judge。
- 2,048 阶段失败且仍截断时，仅对残余样本测试 4,096。
- 继续完整 1,500 条的门槛：反事实总分至少 1,148/1,500，即 76.53%。

## 结果

| 方案 | 诊断样本正确 | 相对旧163条净增 | 剩余截断 | 反事实总正确 | 反事实准确率 | Gate |
|---|---:|---:|---:|---:|---:|---|
| R4 / 1,024 基线 | 20/163 | — | 163 | 1,020 | 68.00% | — |
| 2,048 | 65/163 | +45 | 83 | 1,065 | 71.00% | FAIL |
| 2,048 + 残余4,096 | 72/163 | +52 | 46 | 1,072 | 71.47% | FAIL |

2,048 的答案翻转：

- 旧错转对：52
- 旧对转错：7
- 旧对保持：13
- 旧错保持：91

加入残余4,096后的混合翻转：

- 旧错转对：60
- 旧对转错：8
- 旧对保持：12
- 旧错保持：83

## Judge 与截断

- 2,048：79条进入 Judge，1条 Judge 未按要求只输出 Yes/No，按冻结规则计错。
- 残余4,096：52条进入 Judge，2条 Judge 未按要求只输出 Yes/No，按冻结规则计错。
- 这些失败即使全部改判正确，反事实分数也最多约71.67%，不改变 Gate 结论。
- 4,096 后仍有46条截断，其中4条已正确。即使另外42条在8,192下全部转对，总分也最多约74.27%，仍低于76.53%。

## 结论

提高生成上限确实改善 MMStar，说明1,024-token截断是一个真实问题；但它只能把本地 Base 从68.00%提升到约71.47%，无法使结果接近论文78.53%。继续完整1,500条或测试8,192不具备通过预设门槛的数学可能性，因此按冻结 Gate 停止。

下一步应转向排查论文与本地评测协议的其余差异：GPT-OSS-120B Judge替换、精确Base checkpoint revision、MMStar Prompt/Chat Template、数据快照与图像处理，以及长篇推理而非直接选项回答的生成行为。

## 凭据

- 冻结配置：configs/mmstar_base_max_tokens_diagnostic.yaml
- Gate：artifacts/runs/E-MMSTAR-BASE-MAXTOK-DIAG-001/gate_receipt.json
- SHA256：artifacts/runs/E-MMSTAR-BASE-MAXTOK-DIAG-001/artifact_sha256.txt
- 会话：artifacts/runs/E-MMSTAR-BASE-MAXTOK-DIAG-001/session.json
