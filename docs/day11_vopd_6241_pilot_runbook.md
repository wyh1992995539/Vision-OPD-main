# Day 11 Vision-OPD 6241 双卡 Pilot 运行手册

当前静态状态为 `PASS_PENDING_GPU_PILOT`。Pilot checkpoint 只用于机制、稳定性和预算证据；正式训练必须再次从原始 Base 冷启动。

资源修订：Pilot-16 两次运行均被显存 guard 中止；第二次只开启 Reference
offload 仍在 GPU 0 达到 95.57%。当前候选方案为三类 offload 全开，rollout
memory fraction=0.40，vLLM compilation_config.cudagraph_capture_sizes=[1,2,4,8]，
cgroup 启动下限 192 GiB。适配层保留显式 max_model_len=9216，避免被模型
默认上限 262144 覆盖。历史运行期中止线为 95%；当前 GPU 中止线经用户授权调整为 98%，CPU 仍为 95%。

第三次 Pilot-16 已通过：2 steps、完整 checkpoint、postflight PASS。
cgroup 实测峰值 210.6143 GiB（包括文件缓存），按 95% 守护线推算
配额应至少为 221.6993 GiB；Pilot-64 阶段启动下限上调至 **224 GiB**，
建议 **256 GiB**。这仅是基于已观测峰值的下限，不保证更长 Pilot 没有更高峰值。
Pilot-16 的历史配置及有效策略保留不变。正式训练的历史 192 GiB 下限
不再视为已验证安全，须在 Pilot-64/冷重载通过后重新冻结，当前仍禁止正式训练。
GPU 0/1 峰值为 91.82%/96.65%；GPU 1 连续 2 个采样、cgroup 1 个采样超过 95%，
未达到当时连续 3 次中止条件，仍有资源风险。这是旧 95% 策略下的历史证据。
复核记录：`pilot/64/preflight/resource_review.json`（相对本次 Gate 目录）。

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
- Pilot-64 cgroup 内存上限不少于 **224 GiB**，推荐 **256 GiB**。192 GiB 仅保留为 Pilot-16 历史候选基线，不是新的安全结论。
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

## Checkpoint 内存优化修订（CPU 已验证，待新 Pilot-16）

旧 Pilot-16 峰值 210.61 GiB 落在保存阶段。旧遥测未保存 memory.stat，不能精确拆分
匿名内存与文件缓存；进程树 RSS 合计还可能重复统计共享页。分析记录见
`artifacts/runs/E-D11-6K-GATE-001/checkpoint_memory_revision/analysis.md`。

当前保存实现缩短 model state_dict 临时引用生命周期，并由双卡启动器显式启用
`++actor_rollout_ref.actor.checkpoint.fsdp_flush_reclaim=true`：逐分片使用原生 torch.save，
文件完成后 fsync，再对本文件请求 DONTNEED。缓存建议可能不生效，也不能消除
单个分片写入期间的峰值。新增 memory.stat 分类遥测，但运行期守护仍按 memory.current 判断。

算法 YAML、checkpoint 内容和频率、224 GiB 阶段门槛保持不变；当前 GPU 中止线为 98%，CPU 为 95%。
新启动回执和 postflight 绑定保存源码哈希；旧 Pilot-16 PASS 保留为历史证据，
不会解锁新实现的 Pilot-64。需先归档旧运行到独立目录（不得覆盖），再进行新 Pilot-16
及 checkpoint 冷重载验证。本次没有执行归档、删文件或启动 GPU。

## GPU 中止线调整

用户明确授权将当前 6241 正式训练及 Pilot 的 GPU 显存中止线从 95% 提高到 98%。
规则为任一卡占用比例达到或超过 98%，连续 3 个采样触发中止；采样周期仍为 5 秒。
CPU cgroup 中止线仍为 95%，其他资源/阶段 Gate 不变。该调整减少显存安全余量，不能避免瞬时 CUDA OOM。
历史运行的策略副本和回执不改写；下次启动必须使用当前策略，不复用旧的费用例外策略副本。

## 2026-09-05 Pilot-64 冷重载完成

本节是最新状态；上文“待新 Pilot-16”等内容保留为历史实施记录。
当前新保存实现的 Pilot-16 已 PASS，Pilot-64 step 8 checkpoint 的 Student 冷重载也已 PASS，
`pilot/64/evidence/postflight.json` 已由 `PASS_TRAINING_PENDING_RELOAD` 更新为 `PASS`。
旧 postflight 和冻结记录已保存在 `pilot/64/cold_reload_attempt_002/` 中，不改写历史启动回执。

- 来源：`pilot/64/checkpoints/global_step_8`；所有源文件及 latest marker 前后 SHA256/大小/mtime 一致。
- 合并模型：`pilot/64/cold_reload/merged_hf`，保留原始 checkpoint，不删除优化器分片。
- 成功报告：`pilot/64/cold_reload_attempt_002/reload_validation_summary.json`。
- 确定性 SHA256 选取 5 条 Pilot-64 **训练样本**；只发送 Student 图像和 prompt，不读外部 Benchmark、不注入 bbox/答案。
- TP=2、eager、单请求、上下文上限 9216、输出上限 1024；这些是冷加载推理资源参数，不修改训练算法。
- 5/5 推理成功，实际输出 28/93/25/59/25 token，全部 `stop`；原文及 response token IDs 已留存。
- 第一次服务启动因无效 `OMP_NUM_THREADS` 失败（不是 OOM）；专用入口显式设为 4，
  MKL 线程数也设为 4。重试先核验源 checkpoint 与 merged 权重哈希，再复用导出文件启动全新服务。
- 验证结束已关闭服务，双卡显存回到 0 MiB。关联测试：20 passed、6 subtests passed；
  排除 1 项依赖旧 Day8 实物 checkpoint 的测试。

从仓库根目录发起一次新的冷重载验证（必须使用不存在的新输出目录）：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 conda run --no-capture-output -n vision-opd \
  python -m scripts.vopd_6241_pilot_reload --run \
  --output-dir artifacts/runs/E-D11-6K-GATE-001/pilot/64/cold_reload_attempt_003 \
  --reuse-merged-from artifacts/runs/E-D11-6K-GATE-001/pilot/64/cold_reload_attempt_002
```

仅检查输入时去掉 `--run`。入口拒绝覆盖任何已有非空结果目录。
不要删除 `cold_reload/`：成功报告复用了其中的合并模型，首次失败仅指服务初始化失败。

此 PASS 只证明保存后的 Student 可以重新加载并生成文本，不证明优化器/Teacher EMA/RNG
可以精确续训，也不证明模型效果或 1024-token 训练峰值安全。正式训练仍未授权：
仍需处理训练显存余量、长响应/完整 warmup 覆盖、资源与预算复核及 Day11 汇总 Gate；
当前磁盘约 69.45 GiB 可用，低于正式策略要求的 120 GiB，不自动清理任何 checkpoint。
