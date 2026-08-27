# Vision-OPD 统一外部 Benchmark 评测标准（R3 单卡）

> 状态：**FROZEN / SOLE ACTIVE STANDARD**
> 生效日期：2026-08-27（UTC）
> 实验族：`E-PAPER-BASEJUDGE-001`
> 唯一现行机器配置：[`configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml`](../configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml)
> 配置 SHA256：`e71255e817b11c120b4ac22d7ace81d12ffe01e25f7ea94de2e2ffb62e592903`

## 1. 适用范围

本标准统一用于 Base / Vanilla、Vision-OPD、Cached Prefix、GRPO。跨模型评测只允许改变被测 checkpoint 的路径、角色、权重哈希和独立输出目录；数据与顺序、Prompt、图像处理、生成参数、解析、Judge、失败策略、分母和报告格式必须相同。训练后 checkpoint 禁止担任 Judge。

外部 Benchmark 仅评测冻结 Base 和各分支最终定版 checkpoint，不用于挑 checkpoint 或调训练参数。结果必须表述为“尽量对齐公开论文配方的非思考推理 + 固定本地 Qwen3.5-4B Base Judge”；因 GPT-OSS-120B 不可获得，不得声称精确复现论文 Table 2。

## 2. 冻结与变更

正式运行前必须确认 R3 YAML 的 SHA256 等于页首值。数据、Prompt、图像编码、服务或 generation 参数、解析、Judge、失败策略、分母、汇总口径中的任一变化均构成新协议，必须新增 amendment、递增 revision、重跑 Smoke，并重新建立全部模型的可比基线。不得原地修改 R3 后沿用旧哈希或历史结果。

## 3. 数据与主指标

| Benchmark | 数据 revision | split | 主视图 | 固定分母 |
|---|---|---|---|---:|
| ZoomBench | `b788097e57d30510c6877824833234a73bf80d25` | test | full | 845 |
| MMStar | `bc98d668301da7b14f648724866e57302778ab27` | val | full | 1,500 |
| V* Bench | `b44023b4dca749ed8a76b85eb576627d05a1c174` | test | full | 191 |

正式评测合计必须恰好为 2,536 条视觉请求。V* 的 4 条已人工确认重叠样本按已冻结的“不重复假设”保留，官方分母固定 191。ZoomBench crop 只能另做诊断，不进入主比较。缺图、推理失败、空输出、无法解析和 Judge 最终失败均留在分母并计错。

## 4. 模型、服务、输入与生成

Base 固定为 `/root/autodl-tmp/models/Qwen3.5-4B`，HF revision 为 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。两个权重分片 SHA256 为：

```text
26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61  model.safetensors-00001-of-00002.safetensors
cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188  model.safetensors-00002-of-00002.safetensors
```

每个训练模型正式评测前必须记录 checkpoint 路径与权重哈希。服务固定为 OpenAI-compatible Chat Completions、TP=1、GPU memory utilization=0.75、trust remote code=true、GDN prefill backend=triton；`dtype`、`max_model_len`、`max_num_seqs` 跟随公开启动命令默认值。

每条请求只有一个 user message，无 system prompt，输入一张 full image。ZoomBench/MMStar 保留源图字节；V* 始终转 RGB PNG，编码超过 20 MiB 时按冻结实现缩小。被测模型固定 `enable_thinking=false`、`temperature=0`、`max_tokens=1024`、16 workers、最多重试 3 次；不得传 `seed`、`top_p`、`top_k`、presence penalty、repetition penalty。触及 token 上限不自动判错，仍解析最终输出。

## 5. 评分与 Judge

评分顺序固定为 MathRuler、MMStar/V* 官方选择题首字母匹配、固定 Base Judge。Judge 始终使用上述原始 Qwen3.5-4B Base，无 system prompt，冻结官方 Judge Prompt，`enable_thinking=false`、`temperature=0`、`max_tokens=2048`、16 workers、最多重试 3 次。只有规范化后唯一的 Yes/No 有效；最终 API 或格式失败计错并保留原始记录。这是相对论文唯一预先声明的核心替代。

## 6. 输出与断点恢复

四个正式目录分别为：

```text
artifacts/runs/E-PAPER-BASEJUDGE-001/base/
artifacts/runs/E-PAPER-BASEJUDGE-001/vision_opd/
artifacts/runs/E-PAPER-BASEJUDGE-001/cached_prefix/
artifacts/runs/E-PAPER-BASEJUDGE-001/grpo/
```

每个目录必须包含 `predictions.jsonl`、`judge_results.jsonl`、`scores.jsonl`、`summary.json`、`resume_status.json`、`run_manifest.json`、`metrics.json`、`cost.json`、`artifact_sha256.txt` 和 `validation.json`。预测与评分各 2,536 条；汇总必须能完全由逐样本文件重建。

恢复键固定为 `benchmark + NUL + view + NUL + sample_uid`。逐条 append/flush；启动和结束时原子压缩；成功记录不重跑，错误记录按最多 3 次策略重试。

## 7. 成本、报告与 Base 参考

R3 为单卡实例：5.98 元/GPU 小时，实例 5.98 元/墙钟小时。报告必须包含主准确率及类别、无效/失败、Judge 调用/失败、token、平均/P95 延迟、推理/Judge 墙钟、GPU 小时和成本，并注明是否排除模型加载与空闲时间。

正式 Base 参考结果为 ZoomBench `428/845`、MMStar `1126/1500`、V* `160/191`。证据见 [`base_external_benchmarks_r3_single_gpu.md`](../artifacts/reports/base_external_benchmarks_r3_single_gpu.md) 和 [`base/`](../artifacts/runs/E-PAPER-BASEJUDGE-001/base/)。后续模型只与该 Base 作同协议比较。

## 8. 历史版本政策

`E-D5-001`、`E-D6-001` 和 R1/R2 Smoke 只作为历史工程诊断。按负责人 2026-08-27 决定，旧可执行配置 `configs/benchmark_eval_paper_basejudge.yaml` 与 `configs/benchmark_eval_paper_basejudge_r2.yaml` 已删除。R1/R2 amendment、哈希记录、Smoke 输出和清单继续保留以便审计；旧 `.sha256` 只是档案记录，不再是可执行校验清单。R3 配置和正式 Base 产物不得删除或覆盖。
