# Day11 最新诊断证据接入正式 Gate

本次仅更新 CPU 审计及测试，不启动 GPU，不修改正式训练配置、预算或资源门槛。

## 本次接入

- 固定 Actor 输入 A/B：`fixed_validation_v1/fixed_comparison.json` 及两组原始 postflight。
- 修复后的长回复压力验证：`fixed_validation_v1/pressure_v2/run/evidence/postflight.json`。
- 历史 Pilot-16、Pilot-64、冷重载、预算和静态 Gate 仍保留为独立证据，不覆盖、不追改。

校验代码见 `scripts/day11_validation_evidence.py`，只读取 JSON/YAML、源码与流式文件哈希。
不会导入 torch/vLLM/Ray，不加载 checkpoint 或反序列化回放张量，也不重新执行训练。
完整回放 payload 仍做流式 SHA-256 校验，因此需要读取约 8 GiB 文件，但无需把它们装入内存。

## 验证规则

1. 校验各隔离 runtime 的 manifest 和冻结文件，实际启动的配置、数据及 overrides，原始日志和输入收据。
2. 检查固定回放封存 bundle 及 payload 哈希；从两份报告重新比较硬件、CPU 容量、源码、微批次与完整 Actor 输入。
3. 检查比较报告内嵌运行与原始 postflight 一致，保留 `optimization_validated=false` 和 `whole_run_causal_claim_allowed=false`。
4. 压力证据需要完整 16 步、每步双 rank 收据、首批长度 Gate、全局与局部长度一致；从实际记录重算 warmup 后长回复覆盖。
5. 从 NVML、CPU 遥测和同步 CUDA 埋点重算峰值，检查硬件稳定、有效范围、双 rank/步数完整性及 OOM 计数。
6. 缺失文件、哈希变化、空检查、裸 PASS、不一致的长度/峰值或因果声明均不能放行；新证据失效时不退回旧短回复结果冒充覆盖。

## 决策分层

- `latest_diagnostic_evidence_integrity`：上述诊断证据是否完整一致。
- `diagnostic_gpu_peaks_below_formal_abort_line`：NVML 和同步物理显存峰值分别低于当前正式保护线。
- `diagnostic_cpu_peak_below_formal_abort_line`：诊断 CPU 峰值低于正式 CPU 保护线。
- warmup 后更新、长回复压力：使用最新有效压力记录，不再只读旧 Pilot-64 的预算覆盖字段。
- `formal_candidate_validation_bound`：本次保持未完成。历史固定回放和强制 EOS 诊断不能自动授权最终自然生成候选。

最后一项不是可以手动打开的开关。后续正式候选接入时，需要实现并测试独立的候选证据绑定验收；
本轮不提前设计/接受未经验证的自声明放行凭证。即使把 YAML 状态改成 ready 或扩大资源，也不能绕过此项。

固定对照与 pressure v2 的源码差异会列在报告中；二者分别验证，不能认为它们与正式 runtime 完全相同。
旧预算仍是旧 Pilot-64 的吞吐外推，压力结果不会自动改写费用；历史冷重载只证明 Student 功能推理，不证明训练恢复。

## 执行

```bash
cd /root/autodl-tmp/Vision-OPD-main
conda run --no-capture-output -n vision-opd python -m pytest -q \
  tests/test_day11_validation_evidence.py tests/test_vopd_6241_day11_finalize.py
conda run --no-capture-output -n vision-opd python scripts/finalize_day11_preflight.py
```

默认刷新 `artifacts/runs/E-D11-6K-GATE-001/preflight.json`、`.md`、`.sha256`。
刷新前会把旧三件套原样归档到同目录 `gate_history/<UTC时间>_<旧报告哈希>/`。
报告保留旧 Pilot 覆盖字段，新增最新诊断摘要、校验来源及审计脚本哈希。

命令退出 0 表示报告成功生成，不代表正式训练 PASS。应读取 `status`、`blocking_gates` 和
`formal_training_authorized`；本轮不会授予正式训练权限。

## 后续未完成事项

正式候选的 deferred/正常 EOS 启动链路及验证；CPU 门槛重新冻结；磁盘满足保存预算；
候选预算和 checkpoint 验证覆盖复核；最后才是正式配置放行。

## 本次执行结果（2026-09-06 UTC）

- 定向回归：`51 passed in 68.62s`，当前 2 GiB CPU cgroup 内完成，未使用 GPU。
- 最新诊断证据：`PASS_DIAGNOSTIC_EVIDENCE`，核验 1,774 个文件哈希。
- 压力覆盖：16 步、128 条有效回复，全部为 1024 tokens；每个 rank 在 step 11～16 覆盖长回复更新。
- NVML / 同步物理显存峰值：90.00% / 90.59%；CPU 峰值约 183.21 GiB。
- 刷新后正式 Gate：`BLOCKED_RUNTIME_RESOURCES`，`formal_training_authorized=false`。
- 剩余阻塞：磁盘可用 95.52 GiB（要求 120 GiB）、正式 CPU 门槛、正式候选验证绑定、配置最终放行。
- 旧报告归档：`artifacts/runs/E-D11-6K-GATE-001/gate_history/20260906T085853764391Z_493de4afb510/`。
- 新报告 JSON 与 Markdown 的 SHA-256 检查均通过。正式 YAML、abort policy 和历史实验文件未修改。
