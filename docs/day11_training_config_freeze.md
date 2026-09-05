# Day 11：论文训练配置核对与 6K 双卡冻结说明

## 一、结论

本次以本地论文 `docs/Vision-OPD.pdf`（SHA256
`67f77a637fe691bba592477ef09e22773ded1a45b9eac6f83c75e3f32404e71b`）
和仓库作者入口 `scripts/run_vision_opd.sh` 为依据，重新核对 6K 正式训练配置。

核对确认旧 6K 配置的 `max_response_length=256` 与论文明确采用的
`1024` 不一致。现已将 Vision-OPD online 分支和 Cached Prefix 分支统一调整为
1024，并将总序列预算从 8448 调整为 9216。

当前冻结口径为 `paper_core_resource_scaled_2gpu`：论文核心算法参数对齐，
资源和运行参数按本机双卡 RTX PRO 6000 约束缩放。它不是官方八卡配置的完整复现。

配置继续保持 `blocked_until_day11_gate_passes`。本次只完成静态冻结，不授权
Day 12 正式训练。

## 二、论文和官方脚本披露值

论文第 4.1、4.3 节及表 3～6 明确披露：

- 训练数据约 6.2K，模型为 Qwen3.5-4B/9B；
- JSD beta 为 0.5；
- Top-K logits distillation 的 K 为 100；
- Teacher 使用 EMA，更新系数为 0.05；
- 最大 on-policy generation length 为 1024；
- 训练 1 epoch；
- 使用 non-thinking mode；
- 1024-token 结果优于论文消融中的 512-token 结果。

论文没有完整披露 batch、rollout n、学习率、warmup、offload 和 worker 数。
这些值以作者仓库的 `scripts/run_vision_opd.sh` 为补充依据。

## 三、逐项对照

| 参数 | 论文/作者入口 | 修改前 6K | 当前冻结值 | 判定 |
|---|---:|---:|---:|---|
| Base | Qwen3.5-4B | Qwen3.5-4B | Qwen3.5-4B | 对齐 |
| 数据 | 约 6.2K | 6241 | 6241 | 对齐 |
| epoch | 1 | 1 | 1 | 对齐 |
| JSD beta | 0.5 | 0.5 | 0.5 | 对齐 |
| distillation Top-K | 100 | 100 | 100 | 对齐 |
| EMA rate | 0.05 | 0.05 | 0.05 | 对齐 |
| max prompt | 作者入口 8192 | 8192 | 8192 | 对齐，待全量长度 Gate |
| max response | 论文/作者入口 1024 | 256 | 1024 | 已修正 |
| max sequence/token budget | 9216 | 8448 | 9216 | 已修正 |
| learning rate | 作者入口 2e-6 | 2e-6 | 2e-6 | 对齐 |
| LR warmup | 作者入口 10 steps | 未显式传入 | 10 steps | 已修正 |
| clip low/high | 作者入口 0.2/0.3 | 默认 0.2/0.2 | 0.2/0.3 | 已修正 |
| rollout sampling | temp 1、top-p 1、top-k -1 | 依赖默认值 | 显式冻结 | 已加固 |
| ignore EOS | false | 依赖默认值 | false | 已加固 |
| max reprompt | 作者入口 10240 | 依赖默认值 | 10240 | 已加固 |
| GPU | 作者入口 8 | 2 | 2 | 资源缩放 |
| global batch | 作者入口 96 | 8 | 8 | 资源缩放 |
| PPO mini batch | 作者入口 96 | 8 | 8 | 资源缩放 |
| rollout n | 作者入口 8 | 1 | 1 | 资源缩放 |
| DataLoader workers | 作者入口 8 | 0 | 0 | 稳定性缩放 |
| rollout agent workers | 作者入口 8 | 2 | 2 | 资源缩放 |
| actor/optimizer/reference offload | 作者入口 true/true/true | false/false/false | true/true/true | 第二次 Pilot 后修订，待重测 |
| rollout GPU memory utilization | 作者入口 0.7 | 0.45 | 0.40 | 降低推理缓存预算，待重测 |
| CUDA Graph capture sizes | 未显式限制 | 默认 | [1,2,4,8] | 限制图捕获规模 |
| vLLM fuse_allreduce_rms | 作者入口 false | 未显式传入 | false | 已对齐 |
| vLLM FlashInfer autotune | 作者入口 false | 未显式传入 | false | 已对齐 |
| tail batch | 作者入口原生 `drop_last=True` | 6241+7 全覆盖 | 6241 源数据、6240 有效、丢 1 | 已改回原生行为 |
| optimizer steps | 作者入口随 DataLoader 长度 | 781 | 780 | 与 batch 8 的 drop-last 算术一致 |
| checkpoint | 作者入口仅最终步（`save_freq=-1`） | 391/781 | 390/780 | 一次半程恢复点加最终点 |

因此后续报告可以写“论文核心算法参数对齐的双卡受控训练”，不能写“官方训练
配置完整复现”。batch 96、rollout n=8 和八卡资源对优化轨迹有实质影响。

### 当前选定方案与算法对齐双卡参考方案

当前唯一可执行候选仍是 `configs/vopd_6241.yaml`：2 GPU、global batch 8、
PPO mini batch 8、rollout n=1、780 steps。两个官方 vLLM 开关已经显式加入，
但没有借此改动已选择的缩规模 batch/rollout 参数。Pilot-16 首次 GPU 1 达到
95.42%；第二次开启 Reference offload 后 GPU 0 仍连续三次达到 95.57%。
当前候选开启 Actor/optimizer/Reference 三类 offload，并将 rollout fraction
降至 0.40、图捕获限制为 [1,2,4,8]；95% 显存中止阈值不变。修正本地 vLLM 适配层，使显式序列上限 9216 不再
被模型默认 262144 覆盖。确切峰值发生
阶段尚缺逐阶段显存证据，Actor/optimizer 与 rollout 重叠是待验证的解释。

另新增 `configs/vopd_6241_algorithm_aligned_2gpu.reference.yaml`，用于记录更接近
作者算法规模的双卡方案：global/PPO batch 96、rollout n=8、65 steps，三类
offload 开启，而 agent workers 保持 2、DataLoader workers 保持 0、rollout 显存
比例从 0.45 开始由 Pilot 上调。该文件为 `reference_only_not_selected`，不得被
guarded launcher 启动；只有独立 Pilot 通过且负责人明确切换配置后才能采用。

## 四、预算冻结

Day 8 的实测来自 response 上限 256。其旧 781-step 外推：

- 规划约 5.68 双卡小时、67.88 元；
- 保守约 10.09 双卡小时、120.64 元。

这些数值在 response 改为 1024 后只保留为历史下界，不能作为正式预算。
在 1024-token Pilot 完成前，预算采用 fail-closed 预留：

```text
墙钟硬上限 = 38 小时
双卡单价 = 11.96 元/小时
临时保守预留 = 38 × 11.96 = 454.48 元
```

Pilot 后必须使用实测启动、首步、稳态 step、response 长度分布和 checkpoint
保存时间重新生成规划值与保守值。如果新预计时间超过 38 小时且无明确原因，
正式训练保持 BLOCKED。

## 五、中止与存储策略

- 墙钟硬上限由 16 小时恢复为 38 小时；
- 启动前 cgroup 内存上限必须至少 192 GiB；三类 offload 会增加主机内存需求，仍待 Pilot 实测；
- GPU/cgroup 使用率达到 95% 且连续 3 次采样时中止；
- cgroup OOM、Teacher 直接梯度、Teacher optimizer 变化或非有限指标立即中止；
- Student 连续 2 个正学习率 step 不更新或 EMA 连续 2 step 不更新时中止；
- generation 连续 3 step 异常时中止；
- 日志心跳超时 900 秒，遥测连续失败 3 次时中止；
- 训练使用 verl 原生 `drop_last=True`；6241 条源数据在 batch 8、1 epoch
  下产生 780 steps、6240 条有效训练样本和 1 条丢弃样本。由于
  `shuffle=true`，丢弃的是 seed 42 打乱序列的末条，不保证是 Parquet 物理末行；
- checkpoint 只允许 step 390 和 780：`save_frequency=390` 在 780
  个总步数下触发半程点，verl 在最终 step 780 只执行一次保存；
- 启动磁盘门槛仍为
  `max(120 GiB, 2 × 57,034,960,981 bytes + 5 GiB)`，即 120 GiB。

response 长度不会改变 checkpoint 参数规模，因此现有 checkpoint 空间估计仍可使用；
运行时间、显存和吞吐必须由 1024-token Pilot 重测。

## 六、静态验证

执行结果：

- Shell 语法检查：PASS；
- 初次冻结的定向 pytest：25 passed，5 subtests passed；
- 本次三类 offload 修订的关联回归：47 passed，14 subtests passed；
- train Parquet：6241 行；
- 缺失图片路径：0；
- 论文核心参数检查：全部 PASS；
- 训练算术和 6241→6240 原生 drop-last 合同：PASS；
- 完整 preflight：预期 FAIL，唯一失败项为
  `config_not_explicitly_blocked=false`。

配置 SHA256：

- `configs/vopd_6241.yaml`：
  `1e8cae87c3ba65d4dbbc9f0e2fed0e12d7d78d90b657ac8dbb24524eeba7ff1f`
- `configs/cached_prefix_6241.yaml`：
  `495290e23af3e6a3232772aedab753152d7bd0bf42551a5d710bbe53e8af671f`
- `configs/vopd_6241_abort_policy.yaml`：
  `98caa92d425e3c3fe3e8ac61d4f8486c36fa90807dd64c78e89d6be1b80d55d3`

Warmup-aware 审计合同：每步必须记录有限、非负的 `actor/lr`；仅在
`actor/lr=0` 且 step 位于 10-step warmup 窗口内时允许 Student delta 为 0；
`actor/lr>0` 时 Student delta 必须大于 0，Pilot postflight 还必须至少观察到
一个正学习率更新 step。Teacher optimizer/gradient/EMA 检查不因 warmup 放宽。

## 七、剩余 Gate

静态冻结完成后仍需：

已有全量 prompt、Cached Prefix、overlap 和 drop-last 证据保持可追溯；本次
资源变更以原始配置 SHA256 和实际语义差异验证 prompt 审计复用条件。

1. 新资源方案下重新执行 Pilot-16；
2. 通过后执行 Pilot-64 和 checkpoint 冷重载；
3. 根据新增 offload 后的实测吞吐重算预算；
4. 生成总 `preflight.json` 并核对最终配置哈希；
5. 最后才可将状态切换为 `ready_after_day11_gate`。
