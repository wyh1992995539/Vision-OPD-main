# Cached Prefix 6,241 Pilot Postflight

- 状态：**PASS**
- 训练链路通过：`true`
- 冷加载与阶段 Gate 通过：`true`
- 严格 prefix-source 消融：`PASS`
- 正式训练授权：`false`


## 本次执行结论

- 8/8 optimizer steps 连续完成，64/64 条 cached prefix 被解析，所有 step 的 online generation calls 均为 0。
- Student 在正学习率 step 更新；Teacher 无 optimizer 更新、无直接梯度，EMA 在每个 step 生效。
- `global_step_8` checkpoint 文件完整；独立合并后的模型完成 5/5 固定样本冷加载推理，非空响应 5/5、inference error 0。
- 首次冷加载只因启动进程未从 `PATH` 找到虚拟环境中的 `vllm` 命令而失败。修复为按当前 Python 环境定位 `vllm` 后，复用已校验的同一 merged model 完成重试；源 checkpoint 前后 manifest SHA256 相同。
- 从 guarded Pilot 启动到修复后冷加载完成共 `1020.25` 秒（约 `17.00` 分钟），按双卡合计 `14 元/小时`估算为 `3.97 元`。
- 当前结论为“具备 Day 14 正式训练技术资格”；本次未启动 780-step 正式训练，因此 `formal_training_authorized=false`。

权威收尾凭据：`artifacts/runs/E-D13-6K-CACHED-PILOT-001/evidence/completion_receipt.json`。

| 检查 | 结果 |
| --- | --- |
| `checkpoint_io_revision_matches` | PASS |
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
| `cached_prefix_config_active` | PASS |
| `cached_runtime_metrics_all_steps` | PASS |
| `online_generation_calls_zero_each_step` | PASS |
| `cached_records_resolved_full_batch_each_step` | PASS |
| `cached_records_resolved_total_64` | PASS |
| `cached_parquet_sha256_current` | PASS |
| `static_contract_audit_pass_and_current` | PASS |

780-step 外推：`{"target_optimizer_steps": 780, "checkpoint_count": 2, "startup_seconds_estimate": 84.61981565319002, "pilot_final_checkpoint_seconds": 93.81768359988928, "scenarios": {"median": {"steady_step_seconds": 31.097145339474082, "projected_total_seconds": 24576.45344075933, "projected_dual_gpu_hours": 6.826792622433148, "projected_cost_cny": 95.57509671406407}, "mean": {"steady_step_seconds": 34.724914621295675, "projected_total_seconds": 27402.48571129835, "projected_dual_gpu_hours": 7.611801586471764, "projected_cost_cny": 106.5652222106047}, "conservative_max": {"steady_step_seconds": 42.75425139814615, "projected_total_seconds": 33657.339060464874, "projected_dual_gpu_hours": 9.34926085012913, "projected_cost_cny": 130.88965190180784}}, "status": "MEASURED_PROJECTION_NOT_YET_FROZEN"}`

该报告只在 8-step 训练、Cached 零在线生成、Teacher EMA、checkpoint 和 5 条冷加载全部通过后，才把消融状态标记为 PASS。
