# Day 9 Task 1：E-D10-001 正式训练时间与费用预算

> 生成时间：2026-09-01T08:35:24.233347+00:00  
> 产物状态：**COMPLETE**  
> Budget Gate：**PASS**

## 技术摘要

Day 8 的 1024 条训练外推已完成独立复算：正式训练为 128 个 optimizer steps，规划基线采用均值场景 **1.02 双卡小时 / ¥12.25**，启动预留采用保守场景 **1.74 双卡小时 / ¥20.84**。外推计算与源文件逐项一致，projection Gate 为 PASS。

仓库内非预测成本记录的可见小计为 **¥27.68**，但它遗漏 Day 4、Day 5 Smoke、Day 7，以及部分启动、空闲、失败尝试和计费舍入。因此该数字不是云平台累计账单。用户报告的 AutoDL 控制台累计费用为 **¥200.00**；加上保守预留后的预计累计费用为 **¥220.84**，剩余 **¥1779.16**，因此预算 Gate 为 `PASS`。

## 训练范围与费用定义

- 样本：1024 条；global batch：8；optimizer steps：128。
- 首步单独计入预热，后续使用 127 个稳态 step。
- 总时间：启动 + 首步 + `127 × 稳态 step` + 最终 checkpoint 保存。
- 费用：双卡小时 × `11.96 CNY/双卡小时`。
- 规划值用于预期；保守值用于启动前资源与费用预留，不是运行时 SLA。

## 三场景复算全部通过

| 场景 | 稳态 step 秒 | 双卡小时 | 预计费用 |
|---|---:|---:|---:|
| 中位稳态 | 21.86 | 0.8912 | ¥10.6589 |
| 均值稳态 | 25.64 | 1.0246 | ¥12.2542 |
| 稳态最大值 | 46.00 | 1.7427 | ¥20.8428 |

规划冻结值：`1.024596` 双卡小时、`¥12.254164`。资源预留值：`1.742705` 双卡小时、`¥20.842753`。

## 历史成本记录只能形成不完整小计

| 成本记录 | 类型 | 金额 | 计入可见小计 |
|---|---|---:|:---:|
| E-D5-001-projection | projection | ¥43.1787 | 否 |
| E-D6-001-base | observed_service_window | ¥23.4649 | 是 |
| E-PAPER-BASEJUDGE-001-smoke-r1 | observed_client_window | ¥0.0784 | 是 |
| E-PAPER-BASEJUDGE-001-smoke-r2 | observed_client_window | ¥0.3249 | 是 |
| E-PAPER-BASEJUDGE-001-base-r3 | observed_client_window | ¥1.2866 | 是 |
| E-D8-001-training-window | observed_evidence_window | ¥2.0316 | 是 |
| E-D8-001-reload-window | observed_evidence_window | ¥0.4907 | 是 |

Day 5 的 `cost.json` 是完整外评预算预测，不是已发生费用，因此不计入可见小计。其余行按文件中记录的观测窗口相加，但不能替代 AutoDL 账单。

## 数据质量发现与限制

- **HIGH / platform_total**：No AutoDL billing export or authoritative current cumulative charge is archived. The 2000 CNY project-cap gate cannot be marked PASS from repository evidence alone.
- **MEDIUM / E-D4-001_E-D5-001_E-D7-001**：Observed GPU cost files are absent for Day 4, Day 5 Smoke, and Day 7. The documented subtotal is incomplete and must not be presented as the cloud bill.
- **MEDIUM / measurement_coverage**：Several cost records exclude startup, shutdown, idle time, failed attempts, or billing rounding. Summed records represent documented workload windows, not invoice-level spend.
- **LOW / window_overlap**：Some paper-aligned records have duration and update time but no exact service interval. Exact overlap cannot be proven from the cost files, although runs are chronologically distinct.

这些缺口不会改变 E-D10-001 自身的 ¥20.84 预留，也不改变本次基于 AutoDL 平台控制值的预算判断；它们只意味着仓库小计仍不能冒充完整账单。

## 启动阈值与下一步

在为 E-D10-001 预留 ¥20.84 后，Day 10 启动前 AutoDL 平台显示的当前累计费用必须满足：

```text
current_autodl_cumulative_charge_cny <= 1979.157247
```

当前仓库可见小计与该阈值之间还有 ¥1951.48，但这不是可直接支配的余额，只表示需要由平台账单解释的最大未对账空间。

当前控制值已写入 Day 9 preflight。预算 Gate 只代表费用条件满足；Day 10 仍受配置、Git、磁盘等其他 readiness Gate 控制。

## 进一步问题

- 正式训练启动前，AutoDL 控制台累计费用是否仍接近本次记录的 ¥200.00？
- 是否存在仓库外的失败训练、空闲占用或已删除实验？
- 平台账单是否按整分钟、整小时或其他规则舍入？
