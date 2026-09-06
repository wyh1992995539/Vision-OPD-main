 # Day11：显存优化的 CPU 准备阶段

> 最新进展：修复后的 pressure v2 已完成 128 条、16 步、全部 1024-token 回复及 warmup 后验证。
> 最新诊断证据现已接入正式汇总 Gate；下面的早期“尚未完成”描述保留为历史记录。
> 详见 [最新证据与 Gate 决策分层](day11_latest_validation_gate.md)。诊断通过不代表正式候选已放行。

## 结论与范围

本轮只实施代码、CPU 测试和独立 A/B 静态配置，不启动 GPU，不解锁正式训练。
正式配置及历史 Pilot 结果保留；不能把历史成功结果视为新优化路径已通过 GPU 验证。

## 已实施的改动

1. **分阶段显存记录**：按 rank/PID 写入 JSONL，记录 Student forward、Teacher forward、backward、optimizer load/step/offload、Teacher EMA，以及 rollout 返回后的 actor update 边界。
   每个标记同步 CUDA，并记录区间峰值、当前 allocated/reserved 和设备 free/total；同时保留累计峰值，避免重置峰值计数后漏报原有 perf 指标。
   这些记录不覆盖 vLLM 进程内部阶段；整卡显存仍以原有运行时遥测为补充。
2. **优化器状态延迟加载**：已有 optimizer offload 开启时，允许状态在前向和反向阶段保留在 CPU，仅在 optimizer step 前加载，之后立即卸载并等待 D2H 完成。
   部分加载、更新异常和诊断写入异常均有清理路径。
3. **默认行为不变**：新增 `actor.defer_optimizer_state_load=false`、`actor.memory_profile_dir=null`；正式训练 YAML 不变。
   A/B 两组均开启相同诊断，仅 deferred 组开启延迟加载。数据、seed、batch、response 上限、LR/warmup、JSD、Top-K、梯度累积及 EMA 不变。
4. **实验隔离与源码绑定**：新目录与历史 Pilot 分离；guard 检查配置和相关源码哈希，并记录实际 Hydra overrides；不降低 GPU、CPU 或磁盘门槛。

## CPU 验证

执行命令：

```bash
cd /root/autodl-tmp/Vision-OPD-main
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n vision-opd \
  python -m pytest -q \
  tests/test_actor_memory.py tests/test_memory_experiment.py \
  tests/test_vopd_6241_pilot_guard.py tests/test_checkpoint_io_contract.py \
  tests/test_vopd_gpu_threshold_98.py
```

结果：`32 passed, 3 subtests passed`。
测试覆盖禁用诊断时不调用 CUDA、区间峰值记录、异常卸载、非有限梯度跳过更新、混合精度调用顺序及源码绑定。
CPU 小模型四次更新中，两条路径的模型参数、AdamW 状态、学习率调度和 Teacher EMA 状态完全一致；包含梯度累积和首步零学习率。
混合精度测试使用模拟 scaler；CPU 状态搬运不等于真实 CUDA/FSDP/GradScaler 验证。

## 预期收益与局限

- 若峰值来自 optimizer 状态与前向/反向临时张量同时驻留，延迟加载有机会降低该阶段的 allocated 峰值。
- 若峰值来自 optimizer step、vLLM/KV cache、模型本身或缓存分配器，整卡峰值可能不明显下降，甚至不下降。没有实测前不承诺节省多少 GiB。
- CUDA reserved 不一定随张量卸载立即减少；不能仅凭 allocated 降低宣称整卡安全余量提高。
- 状态搬运、同步及诊断有时间开销；诊断 A/B 的耗时不能直接替换正式训练预算。
- 没有缩短 response、减少 batch、改变损失归一化、关闭 Teacher 或修改 EMA；本轮也没有实现 logits 分块等更深层优化。

## 后续 GPU A/B 入口（本轮不执行）

准备脚本为 `scripts/vopd_memory_experiment.py`，仅生成配置并检查静态 Gate；若目录已存在会拒绝覆盖。
生成目录：`artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/{baseline,deferred}`。
两组各复用 Pilot-64 的 64 条数据、8 次更新，输出到各自的 `run/`。

可先做不启动 GPU 的检查：

```bash
conda run --no-capture-output -n vision-opd python scripts/run_vopd_6241_pilot_guarded.py \
  --stage 64 \
  --policy artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/baseline/policy.yaml \
  --preflight-only
```

GPU 开启后，核实空闲双卡、实时 cgroup 容量与磁盘，再将 `--preflight-only` 改为：

```text
--run --current-autodl-cost-cny <最新累计费用> --billing-observed-at-utc <对应UTC时间>
```

完成 baseline 并审查结果后，才考虑 deferred 的相同命令；只替换 policy 中的目录名。
原有预算和资源守卫继续生效。两组应在可比的冷启动状态下顺序运行，不并行争用显存。
结果查看：各自 `run/evidence/guard_summary.json`、`postflight.json`、`exit_receipt.json`、`memory_stages/*.jsonl` 和 `run/logs/train.log`。

2026-09-06 已接入 [A/B 专用结束审计和两组比较](day11_memory_ab_audit.md)。
单组诊断成功为 `PASS_MEMORY_AB_RUN`，不再等待 Pilot-64 冷重载。

2026-09-06 baseline 已完成 8/8 步及 checkpoint 保存。原始结束审计因分配器/物理显存
混用约束失败；修订后的独立离线重审为 `PASS_MEMORY_AB_RUN`，原失败记录保留。
详见 [计数口径修订与离线重审](day11_memory_ab_audit.md#2026-09-06-分配器计数修订与-baseline-离线重审)。
NVML 峰值 98.74% / 98.30%，CUDA 同步占用峰值 99.38%，显存余量问题仍在。
deferred 新源码绑定与跨版本比较 CPU 校验现已完成，当前优化组使用 `ab/deferred_v2/policy.yaml`。
静态 Gate 为 `PASS`，配对准备为 `PASS_COMPARISON_PREPARATION`；旧 deferred 保留但不再作为启动入口。
随后 deferred_v2 已完成 8/8 步、checkpoint 保存及 `PASS_MEMORY_AB_RUN` 结束审计，退出码 0。
正式保存的 [A/B 比较报告](../artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/deferred_v2/comparison_after_run.md)
为 `REVIEW_WORKLOAD_DIFFERENCE`，仍不授权正式训练。
NVML 峰值由 baseline 的 98.74% / 98.30% 降至 82.01% / 80.86%，但两组生成和微批次负载不同。
详见 [生成长度复核](../artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/deferred_v2/workload_review/workload_review.md)：
总回复 token 5939 → 5823（-1.95%），最长回复 629 → 389，Student 微批次 38 → 40。
CPU 复核已完成；同负载归因、近 1024-token 与 warmup 后压力验证尚未完成。
后续 [固定 actor 负载与压力验证的 CPU 准备](day11_fixed_workload_validation.md) 已完成：
四个隔离入口静态 PASS，133 项测试及 3 个 subtests 通过，未启动 GPU。
固定回放需先采集并封存完整原始批次；回放仅固定 actor 更新，不代表整条在线流程固定。
压力阶段为独立 128 条/16 步、强制长回复诊断，原正式配置和 Gate 保持不变。
两组完成后运行 `conda run --no-capture-output -n vision-opd python scripts/compare_vopd_memory_ab.py`；
它分别报告证据有效性、负载可比性和观测显存收益。原 Pilot-64 冷重载和正式放行要求继续保留。

**磁盘先决条件**：每次训练末尾仍会保存约 53 GiB checkpoint，不能靠 `save_freq=0/-1` 取消最终保存。
2026-09-06 扩容审计：训练盘已从 300 GiB 增至 **600 GiB**，实测可用约 **422.54 GiB**，状态为 `PASS_AB_STORAGE`。
两组 A/B 顺序运行的初始空闲需求约 173.12 GiB；加上 20 GiB 规划余量为 193.12 GiB，当前已满足，无须为这两组实验迁移文件。
详见[600 GiB 扩容审计](../artifacts/runs/E-D11-6K-GATE-001/memory_optimization/storage_expansion_20260906/report.md)及同目录的 `plan.json`、`sha256.txt`。
原 300 GiB／可用约 122.55 GiB 的盘点保留在 `memory_optimization/storage_preparation/`，作为扩容前历史证据。
启动下限仍为 120 GiB；启动前须复核实时可用空间。磁盘通过只解决存储条件，GPU 与 CPU 准入仍按各自 Gate 执行。
Pilot-64 级别启动最低 CPU 配额仍为 224 GiB；历史 240 GiB 不是当前运行进程配额的保证。

## 如何判断下一阶段通过

比较两组相同阶段的 allocated/reserved 区间峰值、整卡遥测、实际 prompt/response 长度、耗时、CPU 峰值、数值指标和 checkpoint 结果。
相同 seed 不能保证在线采样在 GPU 上逐 token 一致，必须结合实际长度解释差异。
优先判断是否降低最危险阶段的整卡峰值，而不是只看某个局部张量指标。

即使这组 A/B 通过，8 次更新仍未跨过 warmup=10，也不代表已覆盖 1024-token 长响应。
正式放行仍需后 warmup/长响应压力验证、正式 CPU 门槛重新冻结及 Day11 最终 Gate 通过。
