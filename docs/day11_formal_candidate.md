# 正式候选：deferred + 正常 EOS

> 后续更新：CPU 门槛现已冻结为 240 GiB，候选新收据位于 `formal_candidate_v2/`。
> 下文 192 GiB 和配置未修改的描述属于最初准备快照；当前结果以 [资源冻结记录](day11_resource_refreeze.md) 为准。

候选文件：`configs/vopd_6241_candidate.yaml`。
原 `configs/vopd_6241.yaml`、正式 abort policy 和历史隔离 runtime 均保持不变。
本轮只完成 CPU 准备，不启动训练、不放行正式配置。

## 已实施

- 开启 `actor.defer_optimizer_state_load=true`，保留三路 offload。
- 启动器新增 YAML → Shell → Hydra 参数映射；旧配置缺省仍为 false。
- 显式配置 `actor.memory_profile_dir`，记录至正式输出目录的独立显存文件；旧配置缺省为 null。
- Agent 透传 `ignore_eos`，Server 通过 setdefault 补默认值，显式 false 不会被默认 true 覆盖。
- 候选保持 `ignore_eos=false`；不复制固定回放、首批最小长度检查或压力测试豁免。
- 预检要求 deferred 配合 optimizer offload。候选保留 `formal_candidate_promoted=false`，仅修改 status 不能启动。

## 保持不变

模型、数据、模板、seed、batch=8、rollout n=1、LR=2e-6、warmup=10、JSD/Top-K/EMA、
回复上限 1024、shuffle=true、6241→6240 drop-last、780 步、390/780 保存合同全部保持。
CPU 下限暂时继承 192 GiB，明确标记为待重新冻结的占位值，不表示该容量足够。
GPU/CPU 中止线、磁盘要求和预算策略没有调整。

## 验证结果

`24 passed, 5 subtests passed in 8.10s`，当前 2 GiB CPU 环境完成，未使用 GPU。

覆盖真实配置映射、最终 Shell 参数展开、实际 AgentWorker→SingleTurn→Manager→Server 方法链，
以及全量数据预检。模拟 4-token 自然 EOS 回复可正常返回，不要求每条生成 1024 token。
全量预检按设计拒绝启动，失败项仅为候选状态和候选尚未验收；不是训练故障。

当前测试的 Ray、SamplingParams 和解码引擎使用替身，**不等于真实 vLLM 集成测试或 GPU 解码验证**。
源码审计确认 deferred 核心实现及 EOS 透传文件与成功 pressure v2 对应文件一致；
Actor 与诊断版的差异仅为移除了固定输入审计 hook。正式 Trainer 没有固定回放或强制长度 hook。

证据：`artifacts/runs/E-D11-6K-GATE-001/formal_candidate_v1/cpu_preparation.json` 及 `.sha256`。
状态为 `PASS_CPU_CANDIDATE_PREPARATION_PENDING_GPU`，`formal_training_authorized=false`。
收据绑定候选、原正式配置、策略、诊断来源和相关源码；源码或配置变化后需重新准备新收据。

```bash
cd /root/autodl-tmp/Vision-OPD-main
conda run --no-capture-output -n vision-opd python -m pytest -q \
  tests/test_formal_candidate.py tests/test_vopd_training_preflight.py
```

审计脚本为 `scripts/audit_formal_candidate.py`，默认拒绝覆盖已存在收据；重跑时指定新的 `--output`。

## 下一步

重新冻结 CPU/磁盘条件；准备独立的最终候选短程自然生成验证入口；进行真实双卡验证、
候选 checkpoint 和预算复核；最后完成候选证据绑定验收并正式放行。
本轮收据不满足最终 Gate 的 `formal_candidate_validation_bound`，也不是训练恢复验证。
