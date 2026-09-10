# Benchmark 自适应输出上限统一协议 V1

> 协议 ID：`BENCHMARK-ADAPTIVE-OUTPUT-CAP-V1`  
> 冻结实验：`E-BENCH-ADAPTIVE-CAP-V1-FREEZE-001`  
> 状态：**规则已冻结；Vision-OPD 与 Cached 暂不重跑**

## 适用范围

该协议用于 Base、Vision-OPD 和 Cached Prefix 三个冻结模型，并覆盖 `zoombench/full`、`mmstar/full`、`vstar/full`。每个模型、每个 benchmark 独立选择现有 R4 结果中 `finish_reason=length` 的样本；没有截断的样本不得重跑。

## 统一执行规则

1. 现有 1,024-token R4 结果作为不可变起点。
2. 仅对上一阶段仍为 `finish_reason=length` 的样本继续生成。
3. 上限与并发依次固定为：2,048/16、4,096/16、8,192/8、16,384/4、32,768/2、65,536/1、131,072/1。
4. 样本第一次以非 `length` 结束后立即停止，不得根据答案正确与否继续重试。
5. 禁止 best-of、人工挑选、参考答案驱动的重试和人工改分。
6. 保持各模型的 checkpoint、数据 revision、sample UID、Prompt、图像、Chat Template、non-thinking、temperature=0、服务参数、R4 解析器和冻结本地 Base Judge 不变。
7. 最终混合结果只替换原本截断的样本，并保留每次尝试、来源阶段和 SHA256。
8. Judge 或推理存在错误时保持未完成状态，不得静默计入最终修正版。

## 终止退化规则

样本达到 131,072 tokens 后仍为 `length`，且完整轨迹、无错误最终评分以及规范化段落重复率不低于 95% 均成立时，标注为 `terminal_generation_degeneracy`。保留原始 `finish_reason=length` 和当前评分，不使用参考答案修正，并将其视为已处置的模型生成错误。

若任何证据条件不成立，协议保持未完成；必须先冻结新修正案，不能临时更改阈值或评分方法。

## 当前冻结状态

| 模型 | ZoomBench 截断 | MMStar 截断 | V* 截断 | 当前状态 |
|---|---:|---:|---:|---|
| Base | 7 | 163 | 2 | MMStar 已完成：1091/1500；其余待处理 |
| Vision-OPD | 10 | 153 | 0 | 按用户要求暂缓，未启动 |
| Cached Prefix | 9 | 128 | 1 | 按用户要求暂缓，未启动 |

Vision-OPD 的 V* 当前截断数为0，因此未来执行时只需完成静态无截断审计，无需生成。其余项目只重跑表中的截断样本。

## 报告口径

每个模型必须同时报告原始 R4 分数和自适应最终分数，并列出原始截断数、自然停止数、终止退化数、未解决上限数、Judge error 和 inference error。只有 `unresolved_output_cap_count=0` 才能登记为该模型的最终本地内部 benchmark 结果。

该协议使用本地冻结 Qwen3.5-4B Base Judge，因此仍不等同于论文 GPT-OSS-120B Judge 协议。

## 冻结凭据

- 配置：`configs/benchmark_adaptive_output_cap_protocol_v1.yaml`
- 配置 SHA256：`55ec71409ac0d285ab4647d982828f9277c6cfe5cbf518208c856490b7294759`
- 静态冻结脚本：`eval/freeze_benchmark_adaptive_output_cap_protocol.py`
- 冻结凭据：`artifacts/runs/E-BENCH-ADAPTIVE-CAP-V1-FREEZE-001/protocol_freeze_receipt.json`
- 延后执行状态：`artifacts/runs/E-BENCH-ADAPTIVE-CAP-V1-FREEZE-001/deferred_execution.json`
- SHA256 清单：`artifacts/runs/E-BENCH-ADAPTIVE-CAP-V1-FREEZE-001/artifact_sha256.txt`
- 未来运行目录：`artifacts/runs/E-BENCH-ADAPTIVE-CAP-V1-001/`

本次冻结不创建未来运行目录、不启动 vLLM、不使用 GPU，也不更改 Vision-OPD 或 Cached 的现有预测和评分。
