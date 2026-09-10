# MMStar 官方公开解析器偏差：项目论证备忘录

> 论证编号：`EVAL-MMSTAR-PARSER-RISK-001`  
> 记录日期：2026-09-10  
> 状态：**强工作假设（strong working hypothesis）**  
> 用途：解释论文 MMStar 绝对分数与本地 R4 修正版之间的差距，并约束项目报告中的表述边界。

> **样本来源边界：**本文审计的 1500 条 predictions 和确认的 93 条误判均由本项目本地运行 Qwen3.5-4B Base 生成，不是论文作者提供的原始输出。官方仓库未发布 Table 2 的逐样本 model answers/scores；因此这些数据证明公开解析器缺陷真实且可造成大幅偏差，但不直接给出论文 78.53% 的修正值。

## 核心论点

作者公开的 Vision-OPD 评测实现包含可复现的选择题首字母解析缺陷。该缺陷在本地完全相同的一批 Base 模型输出上造成了 `+7.07 pp` 的净分数膨胀；同时，本地 1024-token 评测上限相对公开运行脚本的默认值更短，受控延长输出又恢复了 `3.47 pp`。这两项协议效应的量级已经足以解释本地 R4 `68.00%` 与论文报告 `78.53%` 之间的表面差距，因此不能把全部 `10.53 pp` 直接解释成模型能力损失。

基于公开代码的来源、提交历史和完整评测入口，把“论文实际运行使用了相同或高度相近的解析器”作为高概率工作假设是合理的；但论文没有发布逐样本预测、评分记录或实际运行 commit，当前证据不能把这一假设提升为已证实事实。

## 已证实事实

1. 官方公开 `eval/judge_qwenlm.py` 的 `extract_first_option` 在规范匹配失败后，会退回搜索回答中的任意第一个大写字母。
2. 该逻辑从公开文件的首次提交 `e4ba5ef432c88927791b984cfa616bd56caaf914`（2026-05-29）起就存在，不是后续偶然引入的改动。
3. 对文本 `Answer: D`，当更具体的正则没有捕获 D 时，回退逻辑可以读取单词 `Answer` 的 A；若标签是 A，就会把明确错误答案判为正确。
4. 本地 R3 与 R4 使用字节完全相同的 MMStar Base predictions，SHA256 均为 `74c0f4ebc869576fe7397ddccc37dee2027f2b678478e1cf10c9a964f34b7684`，因此两者差异来自评分和路由，不是模型重新生成。
5. 本地 Base 的 R3 官方式解析分数为 `1126/1500 = 75.07%`，R4 修正版为 `1020/1500 = 68.00%`，净差 `106` 条、`7.07 pp`。
6. R4 全量交叉审计确认：93 条明确回答错误选项的样本被 R3 首字母规则直接判对；另有 19 条被 R3 首字母规则判对、但 R4 判为歧义并经 Judge 判错。
7. 对原 163 条 1024-token 截断样本进行 2048/4096 受控重跑后，R4 反事实分数从 `68.00%` 提高到 `71.47%`，恢复 `3.47 pp`。
8. 论文报告的 Base MMStar 为 `78.53%`，但没有发布可用于复核的逐样本结果。

## 支持假设的量级一致性

以下计算只能作为量级校验，不能作为可加性证明：

```text
本地 R4 修正版                         68.00%
本地旧解析器相对 R4 的净增幅           +7.07 pp
延长截断样本输出观察到的净增幅          +3.47 pp
-----------------------------------------------
量级合计                               78.54%
论文报告 Base                          78.53%
```

两者相差 `0.01 pp`，说明已经确认的协议差异在数量级上足以覆盖全部表面差距。不过，提高 token 会改变回答文本、解析路线和 Judge 输入，两个效应不能严格线性相加；不得把 `78.54%` 写成论文分数的重构结果。

## 证据强度

| 命题 | 证据等级 | 当前判断 |
|---|---|---|
| 官方公开解析器存在首字母误判 | 已证实 | 源代码、提交历史和本地样本均可复现 |
| 缺陷能显著抬高 MMStar Base 分数 | 已证实于本地输出 | 同一 predictions 配对重评净增 `7.07 pp` |
| 论文实际使用相同或相近解析器 | 强工作假设 | 公开完整评测流程支持，但缺少论文运行 commit/产物 |
| 解析器与生成上限足以解释差距量级 | 强支持 | 本地观测效应合计约等于 `10.53 pp`，但非严格可加 |
| 论文 78.53% 的确切修正版就是 68.00% | 不成立/未证实 | 本地输出、Judge、checkpoint 与论文运行不完全相同 |
| 论文 Vision-OPD 相对 Base 的 +1.07 pp 一定无效 | 未证实 | 不同模型格式偏差不会自动抵消，但需要论文逐样本输出重算 |

## 可用于项目报告的严谨表述

推荐中文表述：

> 我们在作者公开的 Vision-OPD 评测实现中确认了一个答案格式相关的选择题解析偏差：当规范选项提取失败时，评测器回退到回答中的任意首个大写字母，使 `Answer: D` 等明确答案可能被解析为 A。在字节完全相同的本地 Base MMStar 输出上，将该公开逻辑替换为保守的显式选项解析后，准确率从 75.07% 降至 68.00%，净变化 7.07 个百分点；延长被截断回答又带来 3.47 个百分点的受控改善。这表明评测协议差异足以解释论文报告值与本地修正版之间的误差量级，因而该差距不能被直接归因于模型能力下降。由于论文未发布实际评分 commit 与逐样本产物，我们将“论文运行使用相同或相近解析器”标记为强工作假设，而非已证实事实。

推荐英文表述：

> We identified an answer-format-dependent bias in the authors' released Vision-OPD evaluator. When canonical option extraction fails, the evaluator falls back to the first uppercase letter in the response, allowing an explicit answer such as `Answer: D` to be interpreted as A. On byte-identical local Base MMStar predictions, replacing this released rule with conservative explicit-option parsing reduced accuracy from 75.07% to 68.00%, a net change of 7.07 percentage points. A controlled extension of truncated responses recovered another 3.47 points. These protocol effects are therefore large enough to explain the scale of the gap between the paper-reported score and our corrected local score; the full gap should not be attributed directly to model capability. Because the paper does not release its exact evaluation commit or per-sample artifacts, use of the same parser in the reported run remains a strong working hypothesis rather than a verified fact.

## 不应使用的表述

- “论文一定使用了这个有缺陷的解析器。”
- “论文 Base 的真实分数就是 68.00%。”
- “论文结果造假”或“论文所有 Benchmark 无效”。
- “7.07 pp 与 3.47 pp 可以严格相加并精确复原 78.53%。”
- “修复解析器后论文 Vision-OPD 的提升一定会消失或反转。”

这些说法超出了现有证据。当前能有力支持的是“公开评测器存在实质缺陷”和“协议差异足以解释误差量级”。

## 对项目结论的影响

1. 论文 MMStar 数值应标记为“作者报告值，实际评分实现未完全可审计”。
2. 本地 R4 应作为 Base、Vision-OPD、Cached Prefix 之间的内部主比较指标。
3. 本地未复现论文 MMStar 绝对分数，不应被直接解读为 Base 模型能力低了 `10.53 pp`。
4. 本地 Vision-OPD 在同一 R4 下仍低于本地 Base，这一相对回归独立成立，不能由论文解析器风险消除。
5. 若将来取得作者逐样本输出，应在完全相同的 predictions 上同时运行公开解析器与 R4，直接测量论文模型受到的偏差。

## 反证与升级条件

出现以下任一证据后，应更新本论证：

- 作者提供 Table 2 实际评测 commit，证明使用了不同且正确的解析器；
- 作者发布逐样本 predictions/scores，可直接验证没有首字母误判；
- 使用论文确切 checkpoint、生成合同和 GPT-OSS-120B Judge 的独立复现实验表明解析修复对论文输出影响很小；
- 作者确认公开脚本仅为发布后重写版本，并提供论文内部评分实现。

若作者确认 Table 2 使用了当前公开解析逻辑，则“论文使用相同解析器”可从强工作假设升级为已证实事实，MMStar 绝对分数和模型间增量都应基于原始输出重新计算。

## 证据入口

- 根因分析：`docs/day16_mmstar_base_gap_root_cause_analysis.md`
- 机器可读分解：`artifacts/reports/mmstar_base_gap_root_cause.json`
- R3 Base：`artifacts/runs/E-PAPER-BASEJUDGE-001/base/`
- R4 Base：`artifacts/runs/E-PAPER-BASEJUDGE-R4-001/base/`
- max-token 诊断：`artifacts/runs/E-MMSTAR-BASE-MAXTOK-DIAG-001/`
- R4 解析器：`eval/mcq_parser_r4.py`
- 可重复诊断脚本：`eval/analyze_mmstar_base_gap.py`

## 二次独立复核（2026-09-10）

为排除此前 R3/R4 汇总或本地移植代码造成误判，项目执行了第二套独立验证：脚本通过 `git show official/main:eval/judge_qwenlm.py` 读取官方 Git 对象，并用 Python AST 只加载官方 `extract_answer`、`extract_first_option`、`extract_mcq_option` 和 `first_letter_match` 四个函数，然后直接重放冻结的 1500 条 Base MMStar predictions。

复核结果：

- 官方函数对 `Answer: D` 返回选项 A；参考答案为 A 时直接匹配成功，参考答案为 D 时匹配失败。
- 官方函数对 `Final Answer: **D**` 先截取为 `Answer: **D**`，随后返回选项 A。
- 官方函数重放得到 512 条直接匹配，与本地 R3 的 512 条 `first_letter` 样本键逐条完全一致，差异数为 0。
- 512 条中有 93 条在 R4 中是明确选项不匹配；这 93 条全部只有一个 R4 候选选项，92 条来自显式答案标记，1 条来自独立末行选项。
- 93 条的 R4 期望选项与预测选项全部不同，不存在“相同选项被 R4 错标为 mismatch”的情况。

因此可以确定：**选择题首字母误判真实存在于当前官方公开源码，并真实影响本地 MMStar 分数；它不是本地 R3 实现偏差，也不是只靠构造字符串得到的理论风险。** 尚不能确认的仍只有论文 Table 2 实际运行是否使用同一源码版本。

独立复核入口：`eval/recheck_official_mcq_parser.py`  
复核结果：`artifacts/reports/mmstar_official_parser_recheck.json`  
结果 SHA256：`artifacts/reports/mmstar_official_parser_recheck.json.sha256`
