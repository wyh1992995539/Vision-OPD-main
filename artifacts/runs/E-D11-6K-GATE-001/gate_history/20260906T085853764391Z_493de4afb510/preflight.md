# Vision-OPD 6241 Day11 最终汇总 Gate

- 状态：**BLOCKED_ADDITIONAL_RESOURCE_VALIDATION**
- 报告生成使用 GPU：`false`
- 正式训练授权：`false`
- Pilot 运行期 CPU 容量：240.00 GiB
- 本报告进程 cgroup（不可作启动证据）：2.00 GiB
- 当前磁盘可用：422.54 GiB
- 正式磁盘门槛：120.00 GiB

## 检查

| 检查 | 结果 |
| --- | --- |
| `static_gate_pass_and_inputs_current` | PASS |
| `training_freeze_matches_formal_config_and_policy` | PASS |
| `pilot_16_pass_and_frozen` | PASS |
| `pilot_64_pass_and_frozen` | PASS |
| `cold_reload_pass_bound_to_pilot_64` | PASS |
| `budget_frozen_and_sources_current` | PASS |
| `formal_data_and_drop_last_contract` | PASS |
| `pilot_resource_summary_bound_to_training_run` | PASS |
| `pilot_runtime_capacity_met_reviewed_224_gib` | PASS |
| `disk_free_meets_formal_120_gib` | PASS |
| `formal_cpu_floor_refrozen_to_at_least_224_gib` | FAIL |
| `pilot_gpu_peak_below_98_percent_abort_line` | FAIL |
| `at_least_two_post_warmup_steps_observed` | FAIL |
| `long_response_training_pressure_observed` | FAIL |
| `formal_config_released` | FAIL |

## 阻塞项

- `at_least_two_post_warmup_steps_observed`
- `formal_config_released`
- `formal_cpu_floor_refrozen_to_at_least_224_gib`
- `long_response_training_pressure_observed`
- `pilot_gpu_peak_below_98_percent_abort_line`

## 下一步

1. Refreeze the formal CPU prelaunch floor to at least 224 GiB; Pilot runtime capacity was 240 GiB and launch capacity must be rechecked.
2. Complete targeted post-warmup and long-response GPU resource validation.
3. Set configs/vopd_6241.yaml status to ready_for_formal_training only after every prior check passes, then rerun this script.

此文件是决策快照；资源变化、配置变化或证据文件变化后必须重新生成。
