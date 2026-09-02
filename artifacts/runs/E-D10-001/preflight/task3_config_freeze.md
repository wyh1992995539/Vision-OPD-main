# Day 9 Task 3：E-D10-001 正式配置冻结

> 产物状态：**COMPLETE**  
> Config Gate：**PASS**  
> 配置 SHA256：`5977d0b7adda448287d7410431c9461a6f6f53c04792390b9b13d9529a00b30c`

## 结论

`E-D10-001` 正式配置已冻结：1024 条、global batch 8、128 optimizer steps、完整 1 epoch，从冻结 Base 冷启动。任务 3 已完成，但磁盘与 Git Gate 仍独立阻止 Day 10。

## 正式训练合同

- 样本：1024
- global batch：8
- optimizer steps：128
- epoch：1
- 要求完整 epoch：True

## 冻结参数

| 字段 | 冻结值 |
|---|---|
| `experiment.id` | `E-D10-001` |
| `experiment.name` | `vision-opd-qwen35-4b-day10-formal-1024` |
| `experiment.group_name` | `E-D10-001` |
| `experiment.prefix_source` | `online` |
| `experiment.seed` | `42` |
| `paths.model` | `/root/autodl-tmp/models/Qwen3.5-4B` |
| `paths.train_file` | `/root/autodl-tmp/data/vision_opd_1024/train_1024.parquet` |
| `paths.output_dir` | `artifacts/runs/E-D10-001` |
| `data.expected_train_rows` | `1024` |
| `data.train_batch_size` | `8` |
| `data.max_prompt_length` | `8192` |
| `data.max_response_length` | `256` |
| `data.truncation` | `error` |
| `data.shuffle` | `True` |
| `data.dataloader_num_workers` | `0` |
| `actor.learning_rate` | `2e-06` |
| `actor.ppo_mini_batch_size` | `8` |
| `actor.use_dynamic_batch_size` | `True` |
| `actor.gradient_checkpointing` | `True` |
| `actor.max_token_length_per_gpu` | `8448` |
| `rollout.n` | `1` |
| `rollout.tensor_model_parallel_size` | `1` |
| `rollout.gpu_memory_utilization` | `0.45` |
| `rollout.log_prob_micro_batch_size_per_gpu` | `1` |
| `rollout.agent_num_workers` | `2` |
| `self_distillation.loss_mode` | `vopd` |
| `self_distillation.top_k` | `100` |
| `self_distillation.alpha` | `0.5` |
| `self_distillation.teacher_always_on` | `True` |
| `self_distillation.teacher_model_source` | `legacy` |
| `self_distillation.teacher_regularization` | `ema` |
| `self_distillation.teacher_update_rate` | `0.05` |
| `self_distillation.dont_reprompt_on_self_success` | `True` |
| `self_distillation.include_environment_feedback` | `False` |
| `self_distillation.importance_sampling_clip` | `2.0` |
| `resources.nodes` | `1` |
| `resources.gpus_per_node` | `2` |
| `training.expected_samples` | `1024` |
| `training.total_optimizer_steps` | `128` |
| `training.total_epochs` | `1` |
| `training.require_full_epoch` | `True` |
| `training.save_frequency` | `-1` |
| `training.test_frequency` | `-1` |
| `training.max_actor_ckpt_to_keep` | `1` |
| `training.resume_mode` | `disable` |

## 相对 Day 8 的变化

| 字段 | Day 8 | E-D10-001 |
|---|---|---|
| `experiment.id` | `E-D8-001` | `E-D10-001` |
| `experiment.name` | `vision-opd-qwen35-4b-day8-stability-64` | `vision-opd-qwen35-4b-day10-formal-1024` |
| `experiment.group_name` | `E-D8-001` | `E-D10-001` |
| `paths.train_file` | `/root/autodl-tmp/data/vision_opd_1024/train_day8_64.parquet` | `/root/autodl-tmp/data/vision_opd_1024/train_1024.parquet` |
| `paths.output_dir` | `artifacts/runs/E-D8-001` | `artifacts/runs/E-D10-001` |
| `data.expected_train_rows` | `64` | `1024` |
| `data.shuffle` | `False` | `True` |
| `data.dataloader_num_workers` | `4` | `0` |
| `training.expected_samples` | `64` | `1024` |
| `training.total_optimizer_steps` | `8` | `128` |

`dataloader_num_workers` 从 4 降到 0，用于移除 Day 8 checkpoint 保存后出现的 DataLoader 子进程 `Killed` 风险。该项不改变模型计算公式，但可能降低数据加载吞吐。正式全量数据恢复 seeded shuffle；Day 8 的 64 条审计子集因自身已有冻结顺序而关闭 shuffle。

## 不变量 Gate

- `formal_config_gate`：PASS
- `sample_budget_matches_steps`：PASS
- `legacy_smoke_removed`：PASS
- `fresh_base_start`：PASS
- `final_checkpoint_only`：PASS
- `dataloader_child_processes_disabled`：PASS

## 可执行命令

仅 CPU preflight：

```bash
bash scripts/run_vopd_2gpu.sh --config configs/vopd_1024.yaml --preflight-only
```

以下训练命令已经冻结，但只有 Day 9 全部 Gate 通过后才允许执行：

```bash
bash scripts/run_vopd_2gpu.sh --config configs/vopd_1024.yaml --run
```
