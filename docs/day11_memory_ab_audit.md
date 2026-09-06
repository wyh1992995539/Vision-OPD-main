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

首次结束审计由 launcher 自动执行。手工重审必须指定新的输出位置，不能覆盖原始结果。
对源码仍与启动 manifest 一致的运行，可使用：

```bash
cd /root/autodl-tmp/Vision-OPD-main
conda run --no-capture-output -n vision-opd python scripts/audit_vopd_memory_ab.py \
  --policy artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/baseline/policy.yaml \
  --output artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/baseline/reaudit_new.json
```

审计 deferred 时，将路径中的 `baseline` 替换为 `deferred`。
本命令用于训练结束后；训练前运行会得到 `NOT_RUN`，不会启动训练。

**2026-09-06 baseline 特例：** 原始自动审计因跨口径约束失败，现已修正审计源码。
此历史 baseline 应使用下文的离线快照重审，不能使用上面的严格当前源码命令。

## 两组结束后比较

以下无参数命令保留为原始同版本方案说明；当前 baseline / deferred_v2 配对必须使用文末显式命令。

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

## 2026-09-06 分配器计数修订与 baseline 离线重审

核查环境为 PyTorch 2.10.0 / vLLM 0.18.0。`StageMemoryRecorder` 调用的
`memory_allocated/reserved` 和区间峰值是当前进程当前设备的分配器记账；`mem_get_info`
是 CUDA 设备 free/total，NVML 遥测是整卡占用。不能要求前者必然小于后者的总容量。
[PyTorch 内存口径文档](https://docs.pytorch.org/docs/main/notes/cuda.html)区分了分配器统计与设备占用；
[vLLM cuMem 实现](https://docs.vllm.ai/en/stable/api/vllm/device_allocator/cumem/)展示了独立的休眠释放路径。
本地安装的 `vllm/device_allocator/cumem.py` 中，`sleep()` 调用 `unmap_and_release(handle)`，
保留 `pointer_to_data` 和内存池以供 `wake_up()` 重新映射。这支持“记账不等于物理驻留”的解释；
没有逐个 pool 的快照或专项 CUDA 复现实验，不能声称已量化本次异常全部来源。

本次 372 条阶段记录中，189 条 `reserved > device_total`，但没有 `allocated > reserved`
或 `peak_allocated > peak_reserved`。原始值不截断、不改写，报告增加计数和首个异常事件。
仍严格检查非负整数字节、allocated/reserved 次序、区间峰值下界与次序、free/total 有效性、
rank 内设备容量稳定、完整阶段链及双卡/CPU 遥测。物理余量继续使用 NVML 和 CUDA free/total。

已生成独立结果：
[baseline_reaudit.md](../artifacts/runs/E-D11-6K-GATE-001/memory_optimization/accounting_revision_20260906/baseline_reaudit.md)
及同名 JSON、SHA256 清单，状态 `PASS_MEMORY_AB_RUN`。原 `postflight.json` 的 FAIL、
`exit_receipt.json` 的 43、策略、manifest、原始遥测与训练源码未改写。
归档目录 `accounting_revision_20260906/before/` 保存旧审计源码和原失败结果。

再次离线重审时选择一个尚不存在的输出名：

```bash
conda run --no-capture-output -n vision-opd python scripts/audit_vopd_memory_ab.py \
  --policy artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/baseline/policy.yaml \
  --audit-source-snapshot artifacts/runs/E-D11-6K-GATE-001/memory_optimization/accounting_revision_20260906/before \
  --output artifacts/runs/E-D11-6K-GATE-001/memory_optimization/accounting_revision_20260906/baseline_reaudit_v2.json
```

离线模式只允许明确列出的审计文件发生版本变化，归档 SHA256 必须匹配启动 manifest；
报告记录原/新哈希。训练、采集、launcher、算法配置和策略绑定继续严格校验。
launcher 不传离线参数，仍拒绝旧哈希；不能用该模式启动旧配置。

本次真实物理峰值：NVML GPU0 **98.74%**、GPU1 **98.30%**，CUDA 同步 marker **99.38%**。
guard 的连续采样中止条件没有触发，不代表低于 98% 的峰值准入标准。
最大实际回复 629 tokens，warmup 后更新数为 0；正式训练仍不获准。

回归命令（CPU，无新训练）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n vision-opd pytest -q \
  tests/test_memory_ab_audit.py tests/test_memory_experiment.py tests/test_actor_memory.py \
  tests/test_vopd_6241_pilot_guard.py tests/test_checkpoint_io_contract.py tests/test_vopd_gpu_threshold_98.py
```

结果：**80 passed, 3 subtests passed**。包含真实异常字节数、坏数据拒绝、未变中止线、
离线归档篡改/训练源码变化拒绝和禁止覆盖原报告测试。

当时下一步是准备旧 deferred 的新绑定与跨版本验证；该 CPU 工作现已完成，见下节。

## deferred_v2 新绑定与跨审计版本校验（2026-09-06）

当前启动策略：
[deferred_v2/policy.yaml](../artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/deferred_v2/policy.yaml)。
旧 `ab/deferred/` 和历史 baseline 不覆盖，旧 deferred 仅保留审计用途，不再作为当前启动入口。
新目录的 `previous_binding/` 保存旧配置、有效策略来源、manifest 和样本清单；新 manifest
绑定当前全部 15 个源码/合同文件，并保留父绑定哈希。数据、算法、98% GPU / 95% CPU 中止线、
224 GiB 启动内存门槛及原账单规则均未改变。

已执行的 CPU 结果：

- `static_preflight.json`：`PASS`。
- [comparison_preparation.md](../artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/deferred_v2/comparison_preparation.md)：
  `PASS_COMPARISON_PREPARATION`，9 项检查全部通过。
- `comparison_before_run.json`：`WAITING_FOR_RUNS`，退出码 2 是预期等待状态，不是训练故障。
- baseline 用当前代码重新读取原始证据，通过；deferred 尚未创建 `run/`，未启动 GPU。

跨版本判定不再要求三个审计文件的启动哈希相同，但必须逐项验证：

1. 白名单只含 `audit_vopd_memory_ab.py`、`compare_vopd_memory_ab.py`、`vopd_memory_experiment.py`。
2. 两组源码字段集合完整；训练器、采集器、launcher、算法配置等其余哈希必须一致。
3. 两组由同一当前版本重新审计；旧审计文件归档哈希必须等于 baseline 启动 manifest，
   修订记录必须覆盖所有且仅覆盖发生变化的白名单文件，并核对当前文件哈希。
4. baseline 启动 manifest、有效策略和原始证据保持原样。candidate 仍须通过严格当前源码校验，
   不给 candidate 或 launcher 提供离线例外。
5. 数据字节哈希、64 条样本顺序、算法配置和有效守护策略在 CPU 阶段匹配；
   GPU UUID、容量、实际配额、顺序运行窗口与生成负载等在两组完成后再比较。

只排除两组实验 ID 和输出/清单路径等必要身份差异；不排除任何安全阈值。
跨版本可比不等于收益通过，0.5 GiB 最小降幅、不允许单卡回退、98% / 95% 峰值线及负载匹配继续生效。
首次准备检查因有效策略的顶层实验 ID 不同被拒绝，修正身份字段归一化后重新生成最终绑定；
该首次记录保留在 `accounting_revision_20260906/deferred_preparation_attempt_001/`，不是训练失败。

CPU 回归（运行记录为同目录 `cpu_tests.xml`）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n vision-opd pytest -q \
  tests/test_memory_ab_revision.py tests/test_memory_ab_audit.py tests/test_memory_experiment.py \
  tests/test_actor_memory.py tests/test_vopd_6241_pilot_guard.py \
  tests/test_checkpoint_io_contract.py tests/test_vopd_gpu_threshold_98.py
```

后续开双卡后，使用新 policy 执行原 guarded launcher；仍需实时资源检查和 900 秒内的新账单观测。
不能复用历史的 300 元观测时间。示意命令中的占位符必须换成实际数据：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n vision-opd \
  python scripts/run_vopd_6241_pilot_guarded.py --stage 64 \
  --policy artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/deferred_v2/policy.yaml \
  --run --current-autodl-cost-cny <当前累计费用> \
  --billing-observed-at-utc <该账单实际观测时间_UTC>
```

训练结束后的比较命令（输出名尚未使用；再次比较需新名字）：

```bash
conda run --no-capture-output -n vision-opd python scripts/compare_vopd_memory_ab.py \
  --baseline-policy artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/baseline/policy.yaml \
  --deferred-policy artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/deferred_v2/policy.yaml \
  --baseline-audit-source-snapshot artifacts/runs/E-D11-6K-GATE-001/memory_optimization/accounting_revision_20260906/before \
  --output artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/deferred_v2/comparison_after_run.json
```

如需重新做 CPU 配对检查，使用同一命令加 `--preflight-only`，并指定新的输出名。
结果比较始终重读原始证据，不信任手工编辑的 PASS；原比较报告不覆盖。
准备工作不消耗 GPU，不证明真实显存收益，不替代长回复/warmup 后验证。

## 已保存的正式比较与长度复核

deferred_v2 已完成 8 步及 checkpoint 保存，单组审计 `PASS_MEMORY_AB_RUN`。
主比较保存在 `deferred_v2/comparison_after_run.json`、`.md` 和 `.sha256`，
返回码 1 对应 `REVIEW_WORKLOAD_DIFFERENCE`，不是训练或报告生成失败；再次执行须选择新输出名。

复核目录为 `deferred_v2/workload_review/`：

- [workload_review.md](../artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/deferred_v2/workload_review/workload_review.md)：8 步长度表、数据质量风险和结论边界。
- `workload_review.json`：完整计算结果、输入路径和 SHA256。
- `step_lengths.csv`：逐步均值、最大值、token 总数、微批次数。
- `decoded_input_pairs.csv`：64 条输入配对及原行号，字符数不是 token 数。
- `workload_review.ipynb`：3 个代码单元已从头执行并通过 nbformat 验证。
- `sha256.txt`：以上产物完整性校验。

结果：总 token 5939 → 5823（-1.95%），均值 92.796875 → 90.984375，最长回复 629 → 389。
前 5 步中 4 步的 deferred 平均回复更长，不能简单称其整体负载更轻。
Student/Teacher 各 38 → 40 次前向微批次（第 4 步各 4 → 6 次），每个角色累计样本数仍均为 64。
26/64 条 rollout 行位置不同；按步骤＋唯一输入文本连接后，64/64 条输入匹配、标签全同，
9/64 条解码输出相同。没有按行号错配，也没有把字符长度当作真实 token 数。

额外使用 CPU、`torch.load(..., weights_only=True, map_location='cpu')` 核对 32 个 log-prob 文件：
16/16 个变体步骤的双 rank `num_valid_tokens` 合计与日志均值×8 相同，且 Student/Teacher 张量行数一致。
这些文件已展平有效 token；rollout 也未保存逐样本原始 token IDs/mask，仍无法恢复精确逐样本长度分布。

新增独立复核脚本 `scripts/review_memory_ab_workload.py`，未修改训练或原比较器源码。
系统 Python 的 nbformat/nbclient 已执行 notebook；无需 GPU。复跑选择新目录：

```bash
python scripts/review_memory_ab_workload.py \
  --comparison artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/deferred_v2/comparison_after_run.json \
  --output-dir artifacts/runs/E-D11-6K-GATE-001/memory_optimization/ab/deferred_v2/workload_review_new
```

定向测试 `test_memory_ab_workload_review.py`、`test_memory_ab_revision.py`、`test_memory_ab_audit.py`
合计 **93 passed**；主比较及复核产物哈希校验通过。
最终判断：可带限制分享观测结果，不能把约 16 GiB 降幅全部归因于延迟加载，也不能按 token 比例线性修正显存。
同 token/同计算负载验证及近 1024-token、warmup 后安全验证仍未执行。
