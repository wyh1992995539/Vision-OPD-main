# Day11：显存优化的 CPU 准备阶段

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

**磁盘先决条件**：每次训练末尾仍会保存约 53 GiB checkpoint，不能靠 `save_freq=0/-1` 取消最终保存。
本轮准备时可用约 122.55 GiB，启动下限 120 GiB；即使第一组能启动，保留其 checkpoint 后，第二组通常会被磁盘门槛阻止。
因此执行前需要安排额外磁盘或经授权的迁移；脚本不自动删除历史或 A/B checkpoint，也不降低门槛。
Pilot-64 级别启动最低 CPU 配额仍为 224 GiB；历史 240 GiB 不是当前运行进程配额的保证。

## 如何判断下一阶段通过

比较两组相同阶段的 allocated/reserved 区间峰值、整卡遥测、实际 prompt/response 长度、耗时、CPU 峰值、数值指标和 checkpoint 结果。
相同 seed 不能保证在线采样在 GPU 上逐 token 一致，必须结合实际长度解释差异。
优先判断是否降低最危险阶段的整卡峰值，而不是只看某个局部张量指标。

即使这组 A/B 通过，8 次更新仍未跨过 warmup=10，也不代表已覆盖 1024-token 长响应。
正式放行仍需后 warmup/长响应压力验证、正式 CPU 门槛重新冻结及 Day11 最终 Gate 通过。
