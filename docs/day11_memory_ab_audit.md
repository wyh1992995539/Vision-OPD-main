# Day11 显存 A/B：结束审计与结果比较

两组 A/B 的结束审计现在使用 `scripts/audit_vopd_memory_ab.py`，策略中只有独立 A/B 的
`require_cold_reload` 为 `false`。原 Pilot-64 的冷重载合同继续保留。
成功的单组结果为 `PASS_MEMORY_AB_RUN`，表示训练和显存诊断证据通过；优化是否有效由两组比较另行判断。

## 单组结束时会检查什么

guarded launcher 会自动调用专用脚本，沿用原训练审计的步数、warmup 数值规则、Teacher/EMA、
checkpoint、log-prob 证据、OOM/异常和双卡遥测检查。
再检查实际 Hydra overrides、启动时保存的源码 manifest、有效策略、每个 rank 的 8 次完整更新，
以及 Student/Teacher forward、backward、优化器加载/更新、EMA 等阶段记录。
deferred 组还必须记录更新后的优化器卸载，并满足延迟加载的阶段顺序。

输出仍在各组 `run/evidence/postflight.json`，配套 `.md` 与 `.sha256`；launcher 继续写 `exit_receipt.json`。
单组通过不意味着显存优化通过，也不会授权正式训练。

若需重做单组审计，使用：

```bash
cd /root/autodl-tmp/Vision-OPD-main
conda run --no-capture-output -n vision-opd python scripts/audit_vopd_memory_ab.py \
  --policy artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/baseline/policy.yaml
```

审计 deferred 时，将路径中的 `baseline` 替换为 `deferred`。
本命令用于训练结束后；训练前运行会得到 `NOT_RUN`，不会启动训练。

## 两组结束后比较

```bash
cd /root/autodl-tmp/Vision-OPD-main
conda run --no-capture-output -n vision-opd python scripts/compare_vopd_memory_ab.py
```

默认读取 `memory_optimization/ab/{baseline,deferred}/policy.yaml`，重新核查原始证据，
不依赖之前保存的 PASS 文本。结果写入 `memory_optimization/ab/comparison.json`、`.md`、`.sha256`。
这条命令可以在训练前执行，此时应返回 `WAITING_FOR_RUNS`，退出码为 2；观察到收益时返回 0，其余结论返回 1。

| 结果 | 含义 |
| --- | --- |
| `WAITING_FOR_RUNS` | 一组或两组尚未运行，不能比较 |
| `FAIL_AB_EVIDENCE` | 某组训练或证据审计失败 |
| `FAIL_AB_COMPARABILITY` | 配置/源码/数据顺序/硬件/CPU 配额不一致，或运行时段不满足先 baseline 后 deferred |
| `REVIEW_WORKLOAD_DIFFERENCE` | 实际长度或微批次形状不同；保留显存差值供复核，不自动认定收益 |
| `FAIL_MEMORY_HEADROOM` | deferred 的 GPU 或 CPU 观测峰值达到中止线 |
| `FAIL_GPU_PEAK_REGRESSION` | 至少一张卡的采样显存峰值升高 |
| `NO_CLEAR_MEMORY_BENEFIT` | 整卡最大峰值降幅不足预先约定的 0.5 GiB |
| `PASS_OBSERVED_MEMORY_REDUCTION` | 在可比较的诊断运行中观察到满足条件的显存收益 |

自动观察收益的标准：两组配置与源码等一致，逐步 prompt 最大长度、response 平均/最大长度及
各 rank 的 forward 形状一致；deferred 在两张卡均不回退，整卡最大峰值至少降低 **0.5 GiB**；
deferred 的采样 GPU 峰值和同步阶段设备占用均低于 **98%**，CPU 峰值低于 **95%**。
0.5 GiB 是此次比较预先声明的工程判据，不是论文参数或统计显著性阈值。
guard 仍沿用原连续采样中止规则；比较中的峰值准入更严格，因此训练成功也可能得到 `FAIL_MEMORY_HEADROOM`。

比较 JSON 同时保留按阶段 allocated/reserved 的差值、CPU 峰值增加量和运行耗时增加量。
整卡峰值来自 NVML 采样；阶段 allocated/reserved 来自 PyTorch 区间计数，不能相互替代。
rank 不直接当作物理 GPU 编号，同步阶段的设备占用独立参与安全余量检查。

## 结论范围

相同 seed 和长度统计不保证生成 token 完全相同；单次 A/B 只能提供本次观测收益，不能证明因果或统计显著性。
64 条、8 次更新仍未跨过 warmup=10，也没有自动覆盖 1024-token 长响应。
冷重载、后 warmup/长响应压力验证和正式 CPU 门槛冻结继续由后续 Gate 执行。
诊断同步会影响吞吐，比较耗时不能直接替代正式训练预算。

相关源码、比较判据和有效策略均在 A/B manifest 中绑定，并在启动时保存快照。
运行后改动这些输入会导致重新审计拒绝通过，应保留对应版本来复核；不能重写 manifest 给历史结果重新背书。
旧未启动 A/B 配置及此前 Gate 已保存在 `memory_optimization/postflight_revision_20260906/` 下。
