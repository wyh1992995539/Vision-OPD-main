# Vision-OPD 6241 Day11 最终汇总 Gate

- 状态：**READY_TO_UNBLOCK_FORMAL_CONFIG**
- 报告生成使用 GPU：`false`
- 正式训练授权：`false`
- Pilot 运行期 CPU 容量：240.00 GiB
- 本报告进程 cgroup（不可作启动证据）：2.00 GiB
- 当前磁盘可用：148.59 GiB
- 正式磁盘门槛：120.00 GiB

## 最新诊断证据

- 证据状态：`PASS_DIAGNOSTIC_EVIDENCE`
- 固定输入对照：`PASS_MATCHED_ACTOR_INPUTS_PENDING_MEMORY_REVIEW`；不允许整段显存因果归因。
- 压力更新：16 步；有效回复 128 条；长度 1024～1024 tokens。
- NVML / 同步物理显存峰值：90.00% / 90.59%。
- CPU 峰值：183.21 GiB。
- 证据错误：`[]`
- 正式候选：`PASS_CANDIDATE_VALIDATION`；已绑定自然生成与最终 checkpoint。
- 新预算：计划 7.04 双卡小时 / 84.24 元；候选 PASS 仍不等于正式配置已放行。

## 检查

| 检查 | 结果 |
| --- | --- |
| `latest_diagnostic_evidence_integrity` | PASS |
| `historical_static_gate_preserved` | PASS |
| `pilot_16_pass_and_frozen` | PASS |
| `pilot_64_pass_and_frozen` | PASS |
| `cold_reload_pass_bound_to_pilot_64` | PASS |
| `candidate_gate_freeze_current` | PASS |
| `candidate_validation_pass_and_bound` | PASS |
| `validated_candidate_source_current` | PASS |
| `candidate_budget_refrozen_and_below_project_cap` | PASS |
| `formal_data_and_drop_last_contract` | PASS |
| `pilot_resource_summary_bound_to_training_run` | PASS |
| `pilot_runtime_capacity_met_reviewed_240_gib` | PASS |
| `disk_free_meets_refrozen_formal_floor` | PASS |
| `formal_cpu_floor_matches_reviewed_240_gib` | PASS |
| `diagnostic_gpu_peaks_below_formal_abort_line` | PASS |
| `diagnostic_cpu_peak_below_formal_abort_line` | PASS |
| `at_least_two_post_warmup_steps_observed` | PASS |
| `long_response_training_pressure_observed` | PASS |
| `candidate_length_and_warmup_match_formal` | PASS |
| `formal_candidate_uses_natural_eos` | PASS |
| `candidate_gpu_peak_below_formal_abort_line` | PASS |
| `candidate_cpu_peak_below_formal_abort_line` | PASS |
| `candidate_checkpoint_validation_bound` | PASS |
| `formal_candidate_validation_bound` | PASS |
| `formal_config_released` | FAIL |

## 阻塞项

- `formal_config_released`

## 下一步

1. Promote the source-bound candidate into configs/vopd_6241.yaml with a dedicated immutable receipt.
2. Bind the promoted config/policy hashes and this candidate Gate receipt in the formal launcher.
3. Refresh cumulative AutoDL billing and live resources immediately before formal --run.

此文件是决策快照；资源变化、配置变化或证据文件变化后必须重新生成。
