# Day 8 Vision-OPD 64 条稳定性报告

实验 ID：`E-D8-001`  
证据截止：`2026-08-31T09:26:52.817525+00:00`  
最终状态：**PASS_WITH_CAVEAT**

## 技术结论

Day 8 已完成，可以进入 Day 9 正式训练 Gate。固定 64 条数据连续完成 **8/8 optimizer steps**，全部记录数值有限；Student 每步发生参数更新，Teacher 每步均无 optimizer 直接更新、无梯度，并完成 EMA 更新。最终 `global_step_8` checkpoint 完整保存，关闭训练流程后合并并冷启动服务，冻结的 **5/5** 条样本均得到非空输出且推理错误为 0。

结论标记为 `PASS_WITH_CAVEAT`，不是无条件 `PASS`：训练在 checkpoint 已保存且进度达到 8/8 后出现一次 DataLoader worker `Killed`；没有同时采集的训练期 cgroup/RSS 快照，无法归因。日志中的显存峰值还高于冻结的单卡 96 GB 物理口径，因此只能作为 logger 诊断值，不能声称为可信的逐卡峰值。两项均不否定已保存模型的冷重载结果，但必须在 Day 9 修复观测性。

## Day 8 三项验收由六个证据 Gate 支持

| Gate | 状态 | 证据 |
|---|---|---|
| 固定 64 条输入与配置 | PASS | 64 rows；data SHA256 6e6502f352f2… |
| 连续训练与数值稳定 | PASS | 8/8 steps；全部数值有限；CUDA OOM=0 |
| Student/Teacher/EMA 契约 | PASS | Student 更新；Teacher optimizer delta=0、grad=0；EMA 8/8 |
| 最终 checkpoint 完整性 | PASS | global_step_8；13 个必需文件均非空并有 SHA256 |
| 关闭训练后的冷重载 | PASS | 5/5 非空输出；0 inference errors；源 checkpoint 未变化 |
| 1024 条耗时与费用外推 | PASS | Step 2–8 稳态三场景；包含启动、首步预热与最终保存 |

冷重载摘要还确认源 checkpoint 在合并前后未变化，合并模型清单 SHA256 为 `fd3990c1dd2516a89c086e17a56dc983eacc17d5c2032987c70e36536e2fdc50`，受控关闭后的服务退出码为 `0`。5 条重载 Smoke 的 3/5 正确率只证明推理链路可用，样本量太小，**不作为模型效果结论**。

## 8 步训练稳定，Step 6 是可解释的耗时高点

| Step | VOPD loss | Grad norm | Step 秒 | 生成秒 | 生成占比 | Prompt max | Response mean | Response 达上限 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01595 | 2.42 | 86.48 | 61.79 | 71.5% | 3503 | 128.5 | 25% |
| 2 | 0.03854 | 8.03 | 25.13 | 11.55 | 46.0% | 3741 | 84.0 | 25% |
| 3 | 0.08972 | 37.71 | 21.86 | 7.99 | 36.6% | 3976 | 4.1 | 0% |
| 4 | 0.05174 | 8.86 | 21.53 | 8.18 | 38.0% | 3977 | 5.0 | 0% |
| 5 | 0.08165 | 33.98 | 21.07 | 7.85 | 37.3% | 3554 | 3.9 | 0% |
| 6 | 0.06278 | 8.67 | 46.00 | 9.94 | 21.6% | 5664 | 38.8 | 0% |
| 7 | 0.04172 | 4.92 | 22.71 | 9.65 | 42.5% | 3371 | 43.2 | 0% |
| 8 | 0.05343 | 8.83 | 21.20 | 8.27 | 39.0% | 3600 | 3.4 | 0% |

8 步 loss 范围为 `0.01595`～`0.08972`，均值 `0.05444`；这是一段短稳定性运行，只能用于识别发散/非有限值，不能据此判断收敛。最大 grad norm 为 `37.71`，未出现 NaN/Inf。Step 6 用时 `46.00` 秒，主要来自 actor update `35.87` 秒，而不是生成异常。

全程 prompt 截断率为 0，最大 prompt 为 `5664` tokens；response abort 为 0。前两步各有 25% response 达到 256-token 上限，合计约 `4/64` 条，占 6.25%。这不是运行失败，但 Day 9 必须继续保留 response 截断监控。

展示说明：本报告没有为 8 个离散 step 另画趋势图。逐步表能保留全部精确值、异常点与单位，图表反而会增加尺度误读；这是有意的视觉省略。

## 1024 条训练预计约 1.02 双卡小时，建议按 1.75 小时保守预留

外推严格沿用 Day 8 配置：global batch 8，1024 条对应 128 steps。计算式为：

```text
总时长 = 启动/加载固定开销 + 首步预热 + 127 × 稳态单步统计 + 一次最终 checkpoint 保存
费用 = 总时长（小时）× 11.96 元/双卡小时
```

其中启动/加载固定开销用运行清单时间到日志结束时间反推，为 `231.45` 秒；首步 `86.48` 秒；最终保存 `114.09` 秒；稳态样本为 Step 2～8。

| 场景 | 稳态 step 秒 | 预计双卡小时 | 预计费用 |
|---|---:|---:|---:|
| 中位稳态 | 21.86 | 0.89 | ¥10.66 |
| 均值稳态（规划口径） | 25.64 | 1.02 | ¥12.25 |
| 稳态最大值（保守上界） | 46.00 | 1.74 | ¥20.84 |

Day 9 的预算基线采用“均值稳态”：约 **1.02 双卡小时 / ¥12.25**。启动前资源预留采用“稳态最大值”上界：约 **1.74 双卡小时 / ¥20.84**。它远低于计划中的 38 双卡小时停止线，但 64 条固定子集不保证覆盖 1024 条的长度尾部，因此这只是工程预算外推，不是 SLA。

Day 8 已记录的训练窗口约 `0.170` 双卡小时（`¥2.03`）；最终复用 merged 模型的冷重载窗口约 `0.041` 双卡小时（`¥0.49`）。这些是证据时间戳估算，不包含未被清单覆盖的空闲、失败尝试或云厂商计费舍入。

## 输入、指标与验证方法

- 训练总体：按 seed 42 的稳定哈希顺序从冻结 train-1024 选择 64 条；`shuffle=false`；8 条/global batch；1 epoch；8 optimizer steps。
- 模型与算法：Qwen3.5-4B Base 独立启动；Vision-OPD online prefix；Top-K 100；JSD alpha/beta 0.5；EMA rate 0.05。
- 稳态定义：排除含模型/内核预热的 Step 1，只用 Step 2～8 计算中位数、均值和最大值。
- 生成占比：`timing_s/gen ÷ timing_s/step`；全 8 步加权占比为 `47.1%`，Step 2～8 为 `35.3%`。
- checkpoint：13 个必需文件均有非零大小和 SHA256；目录大小 `53.12` GiB；冷重载确认源文件大小与 mtime 未变化。
- 训练期 CPU 指标 `perf/cpu_memory_used_gb` 来自 `psutil.virtual_memory().used`，是宿主机已用内存，不是训练进程 RSS。

## 限制、异常与鲁棒性检查

1. **结束阶段 worker 异常（中等影响）**：日志有 1 次 DataLoader worker `Killed` 和 1 个 traceback；发生在 8/8 与 checkpoint 路径输出之后，且 checkpoint 后续冷重载通过。没有训练期同步 cgroup/RSS 样本，不能宣称原因是或不是主机 OOM。
2. **显存峰值口径不可审计（中等影响）**：logger 报告 allocated `102.58` GB、reserved `120.13` GB；数值与 96 GB/卡的冻结硬件口径不一致，因此不用于容量结论。Day 9 应旁路采集每卡 `nvidia-smi` 峰值。
3. **工作树非 clean（低到中等影响）**：运行清单记录 commit `01f242469dd4ee2405b5eabaeddbc3f3bbf614e0`，但启动时有未提交 Day 8 文件。配置/数据哈希和运行时 Git 状态已保存，可追踪但复现体验弱于 clean commit。
4. **外推样本较小（中等影响）**：只有 7 个稳态 step，且 response 长度分布不均；报告同时给中位、均值与最大值场景，不把单一均值包装成确定预测。
5. **冷重载复用了 merged 模型（低影响）**：最终 PASS 窗口使用 `--reuse-merged`，因此 `147.70` 秒不含模型合并时间；合并产物自身有完整文件哈希，且源 checkpoint 未改变。

验证评估：**Share with caveats / 可带限制共享**。核心 Day 8 决策（是否进入 Day 9）证据充分；逐卡显存峰值与 worker 被杀原因仍未验证，不能从报告中推导这两项结论。

## Day 8 收尾与下一步

Day 8 到此关闭，不需要继续占用 GPU。下一任务是 Day 9：冻结 `configs/vopd_1024.yaml`，生成 `E-D10-001` preflight，加入训练期每卡显存采样、进程 RSS/cgroup 采样和 worker 异常中止/降级策略。只有 Day 9 全部 Gate 为 PASS 才启动 Day 10 的 1024 条正式训练。

仍需在 Day 9 回答两个问题：`dataloader_num_workers=4` 是否降为 0/1；正式训练的 GPU 预留采用 1.75 小时上界还是再加平台调度缓冲。外部 benchmark 不在 Day 9 运行。

## 可审计证据

- 训练配置：`configs/vopd_day8_64.yaml`
- 固定数据 preflight：`artifacts/runs/E-D8-001/preflight/preflight_summary.json`
- 运行清单：`artifacts/runs/E-D8-001/preflight/run_invocation.json`
- 原始训练日志：`artifacts/runs/E-D8-001/logs/train.log`
- checkpoint 清单：`artifacts/runs/E-D8-001/evidence/reload/checkpoint_manifest.json`
- 冷重载结论：`artifacts/runs/E-D8-001/evidence/reload/reload_validation_summary.json`
- 5 条推理摘要：`artifacts/runs/E-D8-001/reload_5/summary.json`
- 本报告机器摘要：`artifacts/runs/E-D8-001/evidence/stability_summary.json`
- 逐步结构化指标：`artifacts/runs/E-D8-001/metrics.jsonl`
- 费用口径：`artifacts/runs/E-D8-001/cost.json`
