# 固定负载验证 GPU 执行记录（2026-09-06）

结论：capture 和两组 fixed actor 对照完成；压力尝试因 `ignore_eos` 未传入实际采样器而主动停止。**尚不能进入正式训练。**

产物根目录：`artifacts/runs/E-D11-6K-GATE-001/memory_optimization/fixed_validation_v1/`。

| 阶段 | 最终结果 | 说明 |
| --- | --- | --- |
| capture | PASS_CAPTURE，guard 0 | 64 条 / 8 步，checkpoint 完整，8.104 GiB 完整输入已封存为 fixed_bundle.json |
| fixed_baseline | PASS_FIXED_ACTOR_RUN，guard 0 | 64 条 / 8 步，完整输入回放 |
| fixed_deferred | PASS_FIXED_ACTOR_RUN，guard 0 | 64 条 / 8 步，同一封存输入回放 |
| fixed comparison | PASS_MATCHED_ACTOR_INPUTS_PENDING_MEMORY_REVIEW | 硬件、CPU 配额、runtime 源码、全输入 SHA 和实际微批次全部匹配 |
| pressure | MANUALLY_STOPPED_INVALID_PRESSURE_WORKLOAD；FAIL_VALIDATION | 主动 SIGTERM，guard 40；未完成 16 步、未保存最终 checkpoint、无后 warmup 长回复覆盖 |

## 对照观测与解释边界

| 指标 | fixed_baseline | fixed_deferred |
| --- | --- | --- |
| GPU 0 整卡采样峰值 | 94.6895 GiB | 80.0156 GiB |
| GPU 1 整卡采样峰值 | 94.9531 GiB | 85.2637 GiB |
| 最坏整卡采样占比 | 99.3309% | 89.1947% |
| 最坏 CUDA 同步物理占用 | 99.9791% | 89.7767% |
| cgroup 内存峰值占 240 GiB 配额 | 79.0061% | 79.0778% |

baseline 同步峰值位于 rank 1 / step 7 / backward 后；deferred 位于 rank 1 / step 3 / backward 后。
两卡采样峰值分别下降约 14.67 和 9.69 GiB，支持继续做 deferred 压力验证。
但仅 actor 输入被固定，前面的 online rollout/ref/log-prob 计算及其显存池历史未固定；
不能把整个运行的峰值差额全部作因果归因。比较回执保留 `optimization_validated=false`、
`whole_run_causal_claim_allowed=false`、`formal_training_authorized=false`，没有改写成全面 PASS。
不以 PyTorch allocated/reserved 账面计数代替物理显存占用。

## 压力尝试为什么停止

压力配置及运行日志中的 `actor_rollout_ref.rollout.ignore_eos` 确实为 true。
但隔离 runtime 中 `verl/experimental/agent_loop/agent_loop.py` 的采样参数构造只传了
temperature、top_p、repetition_penalty 和 logprobs；
`verl/workers/rollout/vllm_rollout/vllm_async_server.py` 创建 SamplingParams 前也没有补充 ignore_eos。
所以开关没有进入实际请求，第一步真实回复仅 4–255 tokens；前三步六份 rank 记录均未达到 1000 tokens。

发现后只向已核实的本次 pressure 训练进程组 2577 发送 SIGTERM，保留 guard。
guard 记录 latest_step=2、return_code=-15、exit_code=40，2026-09-06 07:31:29 UTC 结束。
已有第 3 步 rank 输入记录不等于第 3 步已经完成。cgroup oom/oom_kill 均为 0，非 OOM 中止。
原 exit_receipt 的 postflight_status=null 原样保留；停止后单独运行结束审计，生成 FAIL_VALIDATION。
停止原因详见 `pressure/run/evidence/operator_stop.json`。没有修改运行中的源码、覆盖尝试或删除文件。

## 资源与后续

本轮双卡同为 RTX PRO 6000 Blackwell Server；GPU 0 UUID 与旧 A/B 不同，
本轮固定 baseline/deferred 使用同一新双卡身份，通过硬件一致性检查。
CPU 配额 240 GiB；压力启动前磁盘剩约 149 GiB。07:31:55 UTC 两卡显存均为 0 MiB。
最近用户账单观测：330 元，07:24:30 UTC；这不是训练结束后的最新账单。

下一步先做 CPU 修复：补齐配置到实际 SamplingParams 的透传测试，增加首批实际长度不达标的早停检查，
在新目录冻结压力专用源码绑定。不要重跑或修改已有 pressure 尝试。
之后才重新开卡运行压力验证，仍需真实 ≥1000-token、双 rank、warmup 后至少两步覆盖及原资源门槛。
正式训练还需自然生成验证、必要冷重载、正式资源门槛及预算冻结；本轮未放行。
