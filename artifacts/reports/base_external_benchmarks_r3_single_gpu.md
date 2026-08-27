# E-PAPER-BASEJUDGE-001 Base 正式评测（R3 单卡）

## 结论

正式 Base 评测已完成并通过自动 Gate。2536 条视觉请求、1786 条所需 Judge 记录和 2536 条最终评分均完整；无推理错误、无重复键、无损坏 JSONL。V* Bench 使用全部官方 191 条作为分母。

本结果采用尽量对齐公开 Vision-OPD 配方的非思考推理，但因项目无法获得 GPT-OSS-120B，统一使用冻结的原始 Qwen3.5-4B Base Judge。因此不得宣称为论文 Table 2 的精确复现。

## 主结果

| Benchmark | Correct | Total | Accuracy |
|---|---:|---:|---:|
| ZoomBench full | 428 | 845 | 50.65% |
| MMStar | 1126 | 1500 | 75.07% |
| V* Bench official | 160 | 191 | 83.77% |

## 主要类别结果

| Group | Correct | Total | Accuracy |
|---|---:|---:|---:|
| ZoomBench multiple choice | 347 | 621 | 55.88% |
| ZoomBench open question | 81 | 224 | 36.16% |
| MMStar coarse perception | 204 | 250 | 81.60% |
| MMStar fine-grained perception | 177 | 250 | 70.80% |
| MMStar instance reasoning | 192 | 250 | 76.80% |
| MMStar logical reasoning | 202 | 250 | 80.80% |
| MMStar math | 204 | 250 | 81.60% |
| MMStar science & technology | 147 | 250 | 58.80% |
| V* direct attributes | 97 | 115 | 84.35% |
| V* relative position | 63 | 76 | 82.89% |

## 完整性与诊断

- 视觉推理成功：2536/2536；失败：0。
- 最终评分：2536/2536；pending Judge：0。
- MathRuler 直接判对：85。
- 选择题首字母直接判对：665。
- 需要 Base Judge：1786；有效 Yes/No：1785。
- Judge 最终格式失败：1；按冻结协议计错并保留原始输出。
- 达到 1024 completion-token 上限：172/2536（6.78%）；仍按最终输出解析与评分。
- Prompt tokens：4,287,226；completion tokens：550,330。
- 并发请求的逐样本延迟总和：11,699.85 秒；平均：4.61 秒。
- 客户端观测阶段墙钟：推理 736.89 秒，Judge 37.63 秒，合计 774.52 秒。
- 单卡实例时价：5.98 元/小时；上述客户端观测阶段估算成本：1.29 元。
- 成本不含服务启动、模型加载、关闭和空闲等待时间。

## 冻结身份

- 配置：`configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml`
- 配置 SHA256：`e71255e817b11c120b4ac22d7ace81d12ffe01e25f7ea94de2e2ffb62e592903`
- Amendment SHA256：`6f85950a0f566cd61d9cb957842da8dbc92478acf68c66059a9ed6454db92b60`
- Base HF revision：`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Serving：TP=1、GPU memory utilization=0.75、GDN prefill backend=triton。
- Inference：无 system prompt、enable_thinking=false、temperature=0、max_tokens=1024。
- Judge：冻结原始 Qwen3.5-4B Base、enable_thinking=false、temperature=0、max_tokens=2048。

## 证据入口

- 正式目录：`artifacts/runs/E-PAPER-BASEJUDGE-001/base/`
- 汇总：`summary.json`
- 逐样本预测：`predictions.jsonl`
- Judge 记录：`judge_results.jsonl`
- 逐样本评分：`scores.jsonl`
- 正式 Gate：`validation.json`
- 产物哈希：`artifact_sha256.txt`
