# 128×16 Formal Candidate Validation Postflight

- 状态：**PASS_CANDIDATE_VALIDATION**
- 训练 Gate：`true`
- 候选验证 Gate：`true`
- 正式训练授权：`false`
- 观测步骤：`16/16`
- 回复最大长度：`471.0 / 1024`

| 检查 | 结果 |
| --- | --- |
| `guard_pass` | PASS |
| `live_launch_gate_pass` | PASS |
| `source_bindings_current` | PASS |
| `training_preflight_pass` | PASS |
| `run_invocation_matches` | PASS |
| `checkpoint_io_revision_matches` | PASS |
| `no_unapproved_hydra_overrides` | PASS |
| `no_duplicate_metric_steps` | PASS |
| `no_traceback_or_oom` | PASS |
| `log_prob_evidence_complete` | PASS |
| `checkpoint_complete_at_step_16` | PASS |
| `telemetry_complete_two_gpus` | PASS |
| `gpu_peak_below_abort_line` | PASS |
| `cgroup_peak_below_abort_line` | PASS |
| `cgroup_oom_counters_stable` | PASS |
| `post_warmup_steps_complete` | PASS |
| `normal_eos_config_bound` | PASS |
| `natural_eos_observed` | PASS |
| `formal_training_not_authorized` | PASS |
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

此报告只决定 128×16 候选验证是否通过，不会自行放行 6,241 条正式训练。
