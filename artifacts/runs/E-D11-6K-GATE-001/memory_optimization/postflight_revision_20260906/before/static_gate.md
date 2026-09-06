# Day 11 6K 静态 Gate

- 状态：**PASS_PENDING_GPU_PILOT**
- GPU 使用：`false`
- 可进入 GPU Pilot：`true`
- 正式训练授权：`false`
- 最终 `preflight.json`：未生成；只有运行时 Pilot 与预算复算通过后才能生成。

## 静态检查

| 检查 | 结果 |
| --- | --- |
| `prompt_length_pass` | PASS |
| `prompt_resource_only_amendment_valid` | PASS |
| `overlap_audit_complete_with_disclosure` | PASS |
| `drop_last_pass` | PASS |
| `cached_prefix_pass` | PASS |
| `training_config_static_freeze_current` | PASS |
| `formal_config_fail_closed_only_on_day11_status` | PASS |
| `pilot_16_static_preflight_pass` | PASS |
| `pilot_64_static_preflight_pass` | PASS |
| `pilot_16_has_two_steps` | PASS |
| `pilot_64_has_eight_steps` | PASS |
| `pilot_16_is_prefix_of_pilot_64` | PASS |
| `pilot_ids_unique` | PASS |
| `warmup_aware_student_update_contract` | PASS |
| `three_way_offload_resource_contract` | PASS |
| `pilot_guard_stage_contracts_valid` | PASS |
| `pilot_64_peak_review_memory_floor` | PASS |

## 尚未完成

- 16 条、2 steps、1024-token 双卡真实训练 Pilot。
- 16 条通过后执行 64 条、8 steps 稳定性 Pilot。
- 使用实测吞吐重算 780 steps 墙钟和费用。
- 冻结最终配置哈希并生成 `artifacts/runs/E-D11-6K-GATE-001/preflight.json`。

## 下一条安全命令

```bash
conda run --no-capture-output -n vision-opd python scripts/run_vopd_6241_pilot_guarded.py --stage 16 --preflight-only
```
