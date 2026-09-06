# 固定 actor 负载与长回复压力验证：CPU 准备

GPU 执行更新（2026-09-06）：capture 与 fixed baseline/deferred 已完成并封存比较；pressure 因采样开关未透传主动停止，未通过。详见 [GPU 执行记录](day11_fixed_validation_gpu_run_20260906.md)。下文保留原 CPU 准备时的状态和命令，已有尝试不能直接重跑。

后续 CPU 修复已完成：新 `pressure_v2` 的参数透传、首批长度 Gate 与测试通过；尚未运行 GPU 压力验证。详见 [Pressure v2 CPU 验证](day11_pressure_repair_cpu.md)。

状态：`PASS_CPU_PREPARATION_PENDING_GPU`。只完成 CPU 准备和测试，未运行新 GPU 任务，未授权正式训练。

## 入口与隔离

产物根目录：`artifacts/runs/E-D11-6K-GATE-001/memory_optimization/fixed_validation_v1/`。
每个阶段包含配置、policy、样本清单、独立 runtime 源码快照和 SHA256 manifest。
原训练源码、旧 A/B manifest、旧 FAIL/PASS 与 checkpoint 不修改。
首次准备因压力配置不符合原论文 EOS 合同而失败的产物保留在 `fixed_validation_preparation_attempt_001/`。

| 阶段 | 行数 / 更新数 | 目的 | 当前依赖 |
| --- | --- | --- | --- |
| capture | 64 / 8 | 保存 actor 更新前完整全局批次 | 实时 GPU/CPU/磁盘/账单检查 |
| fixed_baseline | 64 / 8 | eager optimizer，相同 actor 输入 | capture 成功后封存 bundle |
| fixed_deferred | 64 / 8 | deferred optimizer，相同 actor 输入 | 同一 sealed bundle |
| pressure | 128 / 16 | deferred 长回复、warmup 后与保存峰值 | 实时资源检查 |

四个隔离 launcher 的 `--preflight-only` 已执行并 PASS，报告在各 `run/preflight/pilot_guard_preflight.json`。
`run/` 中只有 CPU preflight，不代表训练已启动。总准备回执为根目录 `cpu_preparation.json`。

## 固定的内容，以及没有固定的内容

在 `RayPPOTrainer._update_actor` 设置全局 step、temperature 后，捕获完整 DataProto：
原始 Student/Teacher token IDs、response mask、attention/position、old log-probs、其他张量、
多模态输入、非张量字段和 meta_info。不是对解码文本重新分词。
使用仅含基础类型与 CPU tensor 的封装，`torch.load(weights_only=True, map_location='cpu')` 加哈希校验读取。
不支持的输入类型直接拒绝，不能静默丢弃字段；捕获额外 I/O 不用于性能比较。

两组 fixed 均从原 base 配置重新启动，不从 capture 的最终 checkpoint 续训。
每步在 actor 更新前回放同一个完整批次，并更新调用方持有的 DataProto，避免日志仍指向被替换前的批次。
记录每个 rank 的真实微批次顺序、token/mask/Teacher 张量哈希、多模态哈希和实际回复长度。
比较时要求两组完整 payload SHA、微批次计划、runtime 源码、GPU 身份与 CPU 配额匹配。

**限制：这是 actor-update-only 的离线诊断，不是原论文的正常在线训练。**
actor 之前的 online rollout/ref/log-prob 计算仍执行，且未固定；其显存池历史和耗时也可能不同。
因此即使输入匹配，也只返回 `PASS_MATCHED_ACTOR_INPUTS_PENDING_MEMORY_REVIEW`：
不能据此宣称全流程同负载、完整因果归因或正式训练放行，不能用该 checkpoint 替代正式在线训练。
GPU 数值确定性、CUDA/FSDP 状态搬运和真实分阶段内存收益尚未验证。

## 压力条件

128 条数据沿用原 tail-aware 选择算法，前 64 条与既有 Pilot-64 清单一致。
batch=8、rollout n=1、LR=2e-6、warmup=10、Top-K/JSD/EMA、模型和 offload 配置不变。
压力配置单独设置 `ignore_eos=true`，用于尽量生成至 1024-token 上限；这改变生成分布，
明确标记 `forced_length_not_paper_sampling`，不得作论文参数复现或模型质量结论。
只有隔离校验副本、且实验 ID 与诊断标记同时匹配时才允许此例外，原正式校验不变。

压力 PASS 必须同时满足：

- 16 个连续更新及原 loss、Student 更新、Teacher 无直接梯度、EMA、生成、checkpoint 审计通过。
- warmup=10 之后，每张卡各至少 **2 个不同步骤**包含实际有效回复长度 **≥1000 tokens** 的微批次。
- 所有实际回复长度在 1–1024，双 rank 每步样本覆盖完整。
- 整卡采样峰值和 CUDA 同步 marker 占用均严格低于 **98%**；cgroup 峰值严格低于 **95%**。

仅配置上限 1024、只有 warmup 内长回复、仅一张卡覆盖、或只完成短回复训练均不能通过。
强制长回复是资源压力验证，不代替未来的自然生成验证、必要冷重载、正式 CPU 门槛与预算冻结。

## CPU 验证

`tests/test_fixed_validation.py` 与旧 memory/Gate/checkpoint 回归合计 **133 passed, 3 subtests passed**。
运行记录：根目录 `cpu_tests.xml`。覆盖安全往返序列化、完整多模态字段、mask 检查、
真实动态微批次算法的分组一致性、已生成 trainer hook、启动后 bundle 变化拒绝和后 warmup 长回复覆盖。
现有 baseline 与 deferred_v2 重审仍为 `PASS_MEMORY_AB_RUN`。

当前 `check capture` / `check pressure` 通过源码依赖校验；两个 fixed 入口因为缺少
`fixed_bundle.json` 而拒绝。该 bundle 必须由新 capture 生成，不能用旧 rollout 文本冒充。

## 后续命令（本轮未执行 GPU）

在仓库根目录、vision-opd 环境执行。检查不调用 GPU：

```bash
conda run --no-capture-output -n vision-opd python scripts/memory_validation.py check \
  --directory artifacts/runs/E-D11-6K-GATE-001/memory_optimization/fixed_validation_v1/capture
```

开卡并获得最新账单后，按阶段逐个启动；占位符必须替换，不能复用历史账单时间：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n vision-opd \
  python scripts/memory_validation.py launch \
  --directory artifacts/runs/E-D11-6K-GATE-001/memory_optimization/fixed_validation_v1/capture \
  --current-autodl-cost-cny <当前累计费用> --billing-observed-at-utc <实际观测UTC时间>
```

capture 的 guard、checkpoint 和 `PASS_CAPTURE` 全部成功后，CPU 封存：

```bash
conda run --no-capture-output -n vision-opd python scripts/memory_validation.py seal \
  --directory artifacts/runs/E-D11-6K-GATE-001/memory_optimization/fixed_validation_v1/capture
```

之后用同样 launch 命令依次替换目录为 `fixed_baseline`、`fixed_deferred`，每次实时复核资源和账单。
不要并行。所有阶段只从当前配置的 base 启动，resume=disable；失败后不能直接覆盖原尝试。
两组完成后用 CPU 比较：

```bash
conda run --no-capture-output -n vision-opd python scripts/audit_memory_validation.py \
  --baseline-policy artifacts/runs/E-D11-6K-GATE-001/memory_optimization/fixed_validation_v1/fixed_baseline/policy.yaml \
  --deferred-policy artifacts/runs/E-D11-6K-GATE-001/memory_optimization/fixed_validation_v1/fixed_deferred/policy.yaml \
  --output artifacts/runs/E-D11-6K-GATE-001/memory_optimization/fixed_validation_v1/fixed_comparison.json
```

压力测试使用 launch 命令替换目录为 `pressure`；先审查已有对照结果再决定是否执行。
结果在每阶段 `run/evidence/{guard_summary,postflight,exit_receipt}.json`；日志为 `run/logs/train.log`。

原安全规则继续生效：启动 CPU 配额至少 224 GiB、空闲双卡、120 GiB 起始磁盘门槛、
GPU 98% / CPU 95% 连续样本中止、原 900 秒账单新鲜度与预算上限。
四次 checkpoint 总量预估约 **212.47 GiB**，还要加 capture bundle、日志和中间副本空间；
当前盘点约剩 317 GiB，但不以该快照代替每次启动的磁盘检查。没有删除或迁移历史文件。
