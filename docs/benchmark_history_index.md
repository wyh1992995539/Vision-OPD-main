# Benchmark 协议版本与历史产物索引

> 当前唯一标准：R3 单卡 `E-PAPER-BASEJUDGE-001`
> 更新日期：2026-08-27

## 现行、可执行并可进入主比较的版本

- 配置：`configs/benchmark_eval_paper_basejudge_r3_single_gpu.yaml`
- 配置 SHA256：`e71255e817b11c120b4ac22d7ace81d12ffe01e25f7ea94de2e2ffb62e592903`
- 协议：`docs/benchmark_protocol.md`
- 管理同步：`artifacts/runs/E-PAPER-BASEJUDGE-001/preflight/benchmark_governance_sync_amendment.yaml`
- Base 参考：`artifacts/runs/E-PAPER-BASEJUDGE-001/base/`

Base、Vision-OPD、Cached Prefix、GRPO 只允许改变被测 checkpoint、`model_role`、明确的服务名和独立输出目录。正式运行必须通过配置 SHA、Base 身份及跨模型可比性自动门禁。

## 历史工程诊断，不得进入 R3 主表

| 版本 | 主要位置 | 当前用途 |
|---|---|---|
| E-D5-001 | `configs/benchmark_eval.yaml`、`artifacts/runs/E-D5-001/` | 数据准备、Smoke、重叠与早期协议历史 |
| E-D6-001 | `artifacts/runs/E-D6-001/` | 旧双视图/长输出评测工程诊断 |
| R1/R2 | `artifacts/runs/E-PAPER-BASEJUDGE-001/smoke*` 与 preflight 哈希/amendment | 论文参数对齐和单卡迁移的审计链 |

历史文件中的以下结论已被 R3 取代：

- V* 187 条去重诊断要求；当前只报告官方 191 条主结果。
- ZoomBench full/crop 双请求主评；当前主评只使用 845 条 full，crop 为可选诊断。
- `enable_thinking=true`、采样和 8,192 token；当前为非思考、temperature 0、1,024 token。
- 训练后模型继续沿用 E-D6；当前必须沿用 R3 并只与 R3 Base 比较。
- `craigwu/vstar_bench` 作为本地主来源；当前冻结来源为 `lmms-lab/vstar-bench` 指定 revision，并已完成内容等价核验。

## 哈希解释

旧 `hash_manifest.json` 和 `.sha256` 文件记录的是生成当时的时间点快照。若其中引用 `docs/benchmark_protocol.md`、`docs/vision_opd_21_day_plan.md` 或 `configs/project_1024.yaml` 等后续持续维护的路径，当前文件不再等于旧哈希属于预期现象，不能据此否定历史运行产物。

正式 R3 的配置文件和 Base 运行目录仍必须逐字节匹配各自冻结哈希，不适用上述“可变文档路径”解释。

## 旧入口政策

下列入口保留是为了重放历史实验，不是当前默认：

- `eval/run_eval.sh`
- `eval/run_smoke.py`、`eval/score_smoke.py`
- `eval/run_day6.py`、`eval/score_day6.py`
- `scripts/day6_preflight.py`、`scripts/day6_verify_frozen_inputs.py`
- 旧重叠、Smoke 选择和预算冻结脚本

执行这些入口必须显式接受历史协议警告。新的正式评测应使用 `run_paper_aligned_eval.py`、`run_paper_aligned_judge.py`、`score_paper_aligned.py` 和 `validate_paper_aligned.py`。

## ZoomBench 字段口径
