# Day 11 Drop-last 独立审计

> 状态：**PASS**  
> 审计 ID：`E-D11-6K-DROP-LAST-001`

## 结论

训练集 6241 条，global batch=8，1 epoch 使用 verl 原生
`drop_last=True`，因此形成 780 个完整 batch、
6240 条有效训练记录、0 条 padding，并丢弃打乱后尾部
1 条。

由于 `shuffle=true`，静态审计不能把 Parquet 物理末行认定为被丢弃样本；准确 sample ID
必须由 Pilot/正式运行回执记录。

## 检查

| 检查项 | 状态 |
|---|---|
| `parquet_rows_match_project_source` | PASS |
| `training_source_rows_match_project` | PASS |
| `batch_size_matches_training` | PASS |
| `one_epoch_frozen` | PASS |
| `shuffle_enabled` | PASS |
| `dataloader_seed_matches_training_seed` | PASS |
| `tail_is_nonempty` | PASS |
| `project_optimizer_steps_match_floor_division` | PASS |
| `project_effective_samples_match` | PASS |
| `project_padding_is_zero` | PASS |
| `project_dropped_rows_match_remainder` | PASS |
| `project_full_coverage_sampler_disabled` | PASS |
| `project_tail_policy_is_native_drop_last` | PASS |
| `training_optimizer_steps_match` | PASS |
| `training_effective_samples_match` | PASS |
| `training_padding_is_zero` | PASS |
| `training_dropped_rows_match` | PASS |
| `training_tail_policy_is_native_drop_last` | PASS |
| `training_full_coverage_padding_disabled` | PASS |
| `training_does_not_require_full_epoch_coverage` | PASS |
| `abort_policy_matches_contract` | PASS |
| `native_trainer_sets_drop_last_true` | PASS |

## 原生训练器证据

- 文件：`/root/autodl-tmp/Vision-OPD-main/verl/trainer/ppo/ray_trainer.py`
- SHA256：`a490a53f6651e493ca9c6f93ce81a6230743773b7a58cfeca6ddcaebe85a489c`
- `drop_last=True`：True
