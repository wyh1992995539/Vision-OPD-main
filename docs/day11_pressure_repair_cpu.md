# Pressure v2：CPU 修复与验证

状态：`PASS_CPU_REPAIR_PENDING_GPU`。CPU 修复验证完成，未启动压力训练，也未放行正式训练。

新目录：`artifacts/runs/E-D11-6K-GATE-001/memory_optimization/fixed_validation_v1/pressure_v2/`。
最终回执：该目录下 `cpu_verification.json`；初始 `cpu_preparation.json` 和失败测试记录原样保留。

## 修复内容

- 隔离 agent loop 将 `config.ignore_eos` 传入生成请求。
- 隔离 vLLM server 在请求没有该字段时，采用配置中的 ignore_eos；显式的 false 不会被覆盖。
- 不改动原有最大生成长度及上下文上限，不新增强制 min_tokens，不修改损失、学习率、EMA 或 batch 参数。
- 首批生成完成后检查实际 response_mask：8 条回复必须全部为 1000–1024 个有效 tokens，张量宽度为 1024。
- 检查发生在 sample weighting 后、token balancing/ref/logprob/actor update 前；失败写入
  `run/evidence/first_batch_length_gate.json` 并抛出明确异常，不继续训练更新。
- 隔离 guard 在失败时也调用结束审计；缺少、失败或篡改阈值的首批回执不能通过。
- 首批通过不替代最终压力审计：仍须完成 16 步、checkpoint、双 rank 后 warmup 长回复覆盖，以及 GPU 98% / CPU 95% 门槛。

实现入口：`scripts/prepare_pressure_repair.py`、`scripts/pressure_runtime.py`、`scripts/audit_memory_validation.py`。
部署修改只进入新压力 runtime；正式训练源码、旧 capture/fixed 对照与原 pressure 失败尝试没有被改写。
新 manifest 绑定 549 个文件；测试使用的 helper 与新 runtime 副本已逐字节核对一致。

## CPU 验证结果

- 定向测试：25 passed。
- 关联回归：177 passed，8 subtests passed；包含上述 25 项，不应重复相加。
- 实际隔离 launcher 的 `--preflight-only`：PASS，gpu_used=false。
- 2 条测试警告来自 SWIG 类型的弃用提示，无测试失败。

参数链路测试执行实际 YAML 映射代码、agent worker、single-turn agent、server manager、vLLM server 方法，
并检查已安装 vLLM 的真实 SamplingParams 对象。Ray 传输、分词器和推理引擎使用替身，
所以测试证明的是参数传递及长度检查逻辑，不是真实 GPU 解码成功或显存压力通过。
负对照执行旧 runtime，确实复现配置 true 但 SamplingParams.ignore_eos=false。
覆盖 true/false、验证模式、服务端缺省值、显式请求覆盖和上下文截断上限。

可复跑 CPU 测试（仓库根目录；禁止自动加载无关插件并屏蔽 GPU）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  conda run --no-capture-output -n vision-opd python -m pytest tests/test_pressure_repair.py -q
```

检查新源码绑定，不调用训练：

```bash
conda run --no-capture-output -n vision-opd python scripts/memory_validation.py check \
  --directory artifacts/runs/E-D11-6K-GATE-001/memory_optimization/fixed_validation_v1/pressure_v2
```

不要重新执行 prepare 覆盖此目录；已有目录会被拒绝。要继续修改 runtime，需建立新尝试和绑定。
直接调用隔离 guard 时，policy 应使用绝对路径；推荐通过 memory_validation.py 的统一入口执行。

## 关于此前服务器/连接中断

不能依据一次连接断开就断言训练导致服务器崩溃。现有证据是：

- fixed_deferred 在 2026-09-06 07:13:23 UTC 正常结束，guard 0、完整审计 PASS。
- 其 cgroup 遥测共 116 条，最大相邻间隔约 9.17 秒，未见十分钟级监控缺口；oom/oom_kill 均为 0。
- 优化组日志未发现 CUDA out of memory、OutOfMemoryError、Traceback 或 NCCL Error。
- pressure 在 07:31:29 UTC 的 -15/guard 40 是发现无效压力负载后主动 SIGTERM，不是同一事件。
- 检查时内核启动时间为 2026-08-03，未见宿主机重启迹象。但 journal 无日志、dmesg 无权限，
  缺少平台/SSH/容器层证据，无法确定连接中断的根因，也不能完全排除这些层面的故障。

随后 CPU 测试期间，配额仅 2 GiB，常驻进程约占 1.4 GiB，重依赖测试两次退出 137（SIGKILL）。
当时 cgroup OOM 计数未增加，不能把 SIGKILL 直接认定为内核 OOM；内存余量不足是嫌疑而非确证。
扩大到 120 GiB 后，真实 vLLM 参数链路与完整回归均通过。此 CPU 测试事件需与此前训练连接中断分开。

## 下一步资源条件

本轮 CPU 验证时只检测到 1 张 RTX PRO 6000、CPU 配额 120 GiB、磁盘剩余约 149 GiB；GPU 显存 0 MiB。
这些资源足以完成本次 CPU 验证，**不满足压力训练的双卡及至少 224 GiB CPU 配额要求**。
后续需双卡空闲、CPU 配额至少 224 GiB（此前训练为 240 GiB）、磁盘实时检查和新鲜账单，
再单独授权/启动 pressure_v2。不要复用历史账单时间，也不要降低现有安全门槛。
