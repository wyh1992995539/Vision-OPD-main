# Day 11 Vision-OPD 6241 双卡 Pilot 运行手册

当前静态状态为 `PASS_PENDING_GPU_PILOT`。Pilot checkpoint 只用于机制、稳定性和预算证据；正式训练必须再次从原始 Base 冷启动。

资源修订：Pilot-16 两次运行均被显存 guard 中止；第二次只开启 Reference
offload 仍在 GPU 0 达到 95.57%。当前候选方案为三类 offload 全开，rollout
memory fraction=0.40，vLLM compilation_config.cudagraph_capture_sizes=[1,2,4,8]，
cgroup 启动下限 192 GiB。适配层保留显式 max_model_len=9216，避免被模型
默认上限 262144 覆盖。运行期 95% 中止阈值保持不变；新方案待 GPU 实测。

第二次失败原始日志、遥测、分片、回执和修改前配置已归档至
`artifacts/runs/E-D11-6K-GATE-001/pilot/16/attempts/attempt_002_reference_offload_gpu_pressure/`，
文件校验记录为 `archive_manifest.json`。残留分片不作为可恢复 checkpoint。

## CPU-only 预检

```bash
conda run --no-capture-output -n vision-opd \
  python scripts/run_vopd_6241_pilot_guarded.py \
  --stage 16 --preflight-only
```

预期：`status=PASS`、`gpu_used=false`。64 阶段在 16 postflight 通过前应被 `stage_prerequisite_pass` 顺序门阻止。

## 开卡后的启动要求

- 恰好两张 GPU 可见，且每张卡启动前显存占用不超过 10%。
- 本机 8000 端口没有遗留的独立 vLLM 服务。
- 可用磁盘不少于 120 GiB。
- cgroup 内存上限不少于 192 GiB；192 GiB 是候选资源下限，实际峰值仍须由 Pilot 验证。
- AutoDL 累计费用及观测 UTC 时间不超过 15 分钟。
- 不得与旧日志、rollout、checkpoint、telemetry 或 postflight 碰撞。

## 运行 16 条 / 2 steps

```bash
conda run --no-capture-output -n vision-opd \
  python scripts/run_vopd_6241_pilot_guarded.py \
  --stage 16 --run \
  --current-autodl-cost-cny <当前累计费用> \
  --billing-observed-at-utc <UTC时间>
```

守护器会自动运行静态/资源 Gate、训练期遥测、中止规则、`global_step_2` checkpoint 检查和 postflight。只有 `training_gate_pass=true` 与 `stage_gate_pass=true` 才会解锁 64 阶段。

### Warmup-aware Student 更新规则

- 每一步必须记录有限、非负的 `actor/lr`；缺失、非数值、NaN/Inf 或负值立即失败。
- 当 `actor/lr=0` 且 step 位于冻结的 10-step warmup 窗口内时，允许 Student parameter probe delta 为 `0`；这表示 optimizer 尚未施加有效学习率，不是“Student 没训练”。
- 当 `actor/lr>0` 时，Student parameter probe delta 必须 `>0`。运行期连续 2 个正学习率 step 未更新触发中止。
- Pilot postflight 必须至少观察到 1 个 `actor/lr>0` 的 step，因此不能用全部为零学习率的短 Pilot 冒充更新链路已验证。
- Teacher 规则不随 warmup 放宽：直接梯度必须为 0、optimizer delta 必须为 0、EMA 每步必须实际执行且 delta `>0`。

## 运行 64 条 / 8 steps

```bash
conda run --no-capture-output -n vision-opd \
  python scripts/run_vopd_6241_pilot_guarded.py \
  --stage 64 --run \
  --current-autodl-cost-cny <当前累计费用> \
  --billing-observed-at-utc <UTC时间>
```

64 postflight 会生成 780 steps 的三档墙钟/费用外推。训练检查通过后保持 `PASS_TRAINING_PENDING_RELOAD`，直到提供冷重载报告：

```bash
conda run --no-capture-output -n vision-opd \
  python scripts/audit_vopd_6241_pilot.py \
  --stage 64 --reload-report <冷重载报告.json>
```

## Fail-closed 退出码

- `41`：启动前 Gate 未通过，GPU 训练未启动。
- `40`：运行期中止规则或守护器异常。
- `42`：训练退出为 0，但 checkpoint 不完整。
- `43`：checkpoint 通过，但 postflight 机制检查失败。

任何 Pilot 结果都不直接授权正式训练。只有 16、64、冷重载和预算复算全部完成后，才能生成 Day11 最终 `preflight.json` 并解锁 Day12。
