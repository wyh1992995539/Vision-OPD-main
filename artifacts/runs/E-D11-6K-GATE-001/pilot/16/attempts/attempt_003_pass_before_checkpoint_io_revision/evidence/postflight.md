# Vision-OPD 6241 Pilot 16 Postflight

- 状态：**PASS**
- 训练链路通过：`true`
- 阶段 Gate 通过：`true`
- 正式训练授权：`false`

| 检查 | 结果 |
| --- | --- |
| `guard_pass` | PASS |
| `training_preflight_pass` | PASS |
| `run_invocation_matches` | PASS |
| `no_duplicate_metric_steps` | PASS |
| `no_traceback_or_oom` | PASS |
| `log_prob_evidence_complete` | PASS |
| `checkpoint_complete` | PASS |
| `telemetry_complete_two_gpus` | PASS |
| `exact_contiguous_steps` | PASS |
| `required_metrics_present_and_finite` | PASS |
| `jsd_loss_finite` | PASS |
| `learning_rate_nonnegative` | PASS |
| `zero_lr_only_within_warmup` | PASS |
| `positive_learning_rate_observed` | PASS |
| `student_update_matches_learning_rate` | PASS |
| `teacher_optimizer_unchanged` | PASS |
| `teacher_direct_gradient_absent` | PASS |
| `teacher_ema_updated_each_step` | PASS |
| `crop_teacher_active_each_step` | PASS |
| `generation_errors_zero` | PASS |
| `prompt_truncation_zero` | PASS |
| `response_within_frozen_limit` | PASS |

此报告只证明 Pilot 工程与机制状态，不是模型能力结论，也不能作为正式训练 checkpoint。
