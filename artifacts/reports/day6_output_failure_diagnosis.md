# Day 6 输出解析失败与长度触顶诊断

## 结论

问题主要来自冻结生成协议与模型长推理行为的交互，而不是 GPU/vLLM 故障或 Benchmark 数据缺失。204 条 `invalid_or_ambiguous` 中有 201 条同时以 `finish_reason=length` 达到 8,192 token；只有 3 条在正常停止后仍不符合 A–D 解析格式。

## 分解

| Benchmark / 视图 | 总数 | 解析失败 | 长度触顶 | 两者重叠 | 正常停止但解析失败 |
|---|---:|---:|---:|---:|---:|
| ZoomBench full | 845 | 19 | 56 | 19 | 0 |
| ZoomBench crop | 845 | 18 | 44 | 18 | 0 |
| MMStar full | 1,500 | 160 | 194 | 157 | 3 |
| V* full | 191 | 7 | 7 | 7 | 0 |
| 合计 | 3,381 | 204 | 301 | 201 | 3 |

301 条触顶请求消耗 2,465,792 completion token，占全部 completion token 的约 36.82%。其中 100 条虽然触顶，但在截断前已经出现可解析答案，因此长度触顶不等同于解析失败。

## 根因判断

1. 系统 Prompt 明确要求模型先展示推理过程，再输出 `<answer>`；这会鼓励长链推理。
2. `max_new_tokens=8192` 是停止护栏，不是长输出的直接起因，但允许发散持续到很高的上限。
3. `temperature=0.7`、`top_p=0.8`、`top_k=20` 和 `presence_penalty=1.5` 可能增加推理分支与持续生成；本次数据能确认相关性，不能单独证明各参数的因果贡献。
4. MMStar 的数学、科学与逻辑题更容易触发反复推导：160/204 个解析失败来自 MMStar，其中 science & technology 59、math 34、logical reasoning 26。
5. 仅 3 条属于正常停止后的格式问题：输出 `A: Yes`、输出超出 A–D 的 `E`、或输出数值而非选项字母。它们反映模型指令遵循与严格解析器之间的轻微不匹配。

## 不属于根因的事项

- GPU/vLLM：3,381 条请求全部返回，推理错误为 0。
- 数据丢失或重复：3,381 个唯一请求键，无重复。
- V* 191 汇总修订：只改变汇总口径，不影响生成长度或解析。

## 解释与后续

Day 6 是冻结协议下的正式 Base 基线，不应在看到结果后修改参数并覆盖原结果。后续 Vision-OPD/Cached 必须沿用相同协议，才能与 Base 公平比较。

若要研究更实用的推理协议，应建立独立实验 ID，测试关闭 thinking、缩短 `max_new_tokens`、降低 presence penalty、对选择题强制短答案，以及解析 `<answer>A: ...</answer>` 的兼容性；新实验只能作为协议诊断，不能替换 E-D6-001。

## 来源

- `artifacts/runs/E-D6-001/base/predictions.jsonl`
- `artifacts/runs/E-D6-001/base/scores.jsonl`
- `configs/benchmark_eval.yaml`
