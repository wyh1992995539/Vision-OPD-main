# Vision-OPD 6241 Day11 最终汇总 Gate

- 状态：**FAIL_EVIDENCE_INTEGRITY**
- 报告生成使用 GPU：`false`
- 正式训练授权：`false`
- Pilot 运行期 CPU 容量：240.00 GiB
- 本报告进程 cgroup（不可作启动证据）：2.00 GiB
- 当前磁盘可用：95.52 GiB
- 正式磁盘门槛：120.00 GiB

## 最新诊断证据

- 证据状态：`PASS_DIAGNOSTIC_EVIDENCE`
- 固定输入对照：`PASS_MATCHED_ACTOR_INPUTS_PENDING_MEMORY_REVIEW`；不允许整段显存因果归因。
- 压力更新：16 步；有效回复 128 条；长度 1024～1024 tokens。
- NVML / 同步物理显存峰值：90.00% / 90.59%。
- CPU 峰值：183.21 GiB。
- 证据错误：`[]`
- 正式候选仍待独立绑定验证；诊断 PASS 不授权训练，也不证明完整断点续训。

## 检查

| 检查 | 结果 |
| --- | --- |
| `latest_diagnostic_evidence_integrity` | PASS |
| `static_gate_pass_and_inputs_current` | FAIL |
| `training_freeze_matches_formal_config_and_policy` | FAIL |
| `pilot_16_pass_and_frozen` | PASS |
| `pilot_64_pass_and_frozen` | PASS |
| `cold_reload_pass_bound_to_pilot_64` | PASS |
| `budget_frozen_and_sources_current` | FAIL |
| `formal_data_and_drop_last_contract` | PASS |
| `pilot_resource_summary_bound_to_training_run` | PASS |
| `pilot_runtime_capacity_met_reviewed_240_gib` | PASS |
| `disk_free_meets_formal_120_gib` | FAIL |
| `formal_cpu_floor_matches_reviewed_240_gib` | PASS |
| `diagnostic_gpu_peaks_below_formal_abort_line` | PASS |
| `diagnostic_cpu_peak_below_formal_abort_line` | PASS |
| `at_least_two_post_warmup_steps_observed` | PASS |
| `long_response_training_pressure_observed` | PASS |
| `diagnostic_length_and_warmup_match_formal` | PASS |
| `formal_uses_natural_eos` | PASS |
| `formal_candidate_validation_bound` | FAIL |
| `formal_config_released` | FAIL |

## 阻塞项

- `budget_frozen_and_sources_current`
- `disk_free_meets_formal_120_gib`
- `formal_candidate_validation_bound`
- `formal_config_released`
- `static_gate_pass_and_inputs_current`
- `training_freeze_matches_formal_config_and_policy`

## 下一步

1. Free enough disk to meet the 120 GiB formal floor.
2. Use the evidence-bound 240 GiB CPU floor; freshly recheck the launcher cgroup. Refresh final static/config/budget source bindings after resource and candidate changes.
3. Prepare and validate the final natural-generation candidate with deferred loading; preserve historical diagnostic runtimes and do not reuse forced-EOS/replay checkpoints as formal training starts.
4. Bind candidate source/configuration, checkpoint validation and refreshed budget in the promotion workflow; this diagnostic-only revision intentionally cannot authorize training.
5. Set configs/vopd_6241.yaml status to ready_for_formal_training only after every prior check passes, then rerun this script.

此文件是决策快照；资源变化、配置变化或证据文件变化后必须重新生成。
