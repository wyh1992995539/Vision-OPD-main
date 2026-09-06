# Vision-OPD 6241 正式训练预算冻结

- 状态：**PASS_BUDGET_FROZEN_WITH_RESOURCE_CAVEATS**
- 计划值：8.98 双卡小时 / 107.38 元
- 资源预留：12.73 双卡小时 / 152.31 元
- 38 小时中止上限：454.48 元
- 正式训练授权：`false`

| 情景 | 双卡小时 | 增量费用（元） |
| --- | ---: | ---: |
| median | 7.99 | 95.60 |
| mean | 8.98 | 107.38 |
| conservative_max | 12.73 | 152.31 |

## 覆盖边界

- Pilot steps / warmup steps：8 / 10
- 实测最长响应 / 配置上限：392 / 1024 tokens
- 实测最高 GPU 比例 / 中止线：99.0326% / 98.00%

## 限制

- This freezes a measured extrapolation, not a provider billing guarantee.
- Pilot-64 ended before step 10 warmup completed.
- The longest observed response was shorter than the configured 1024-token limit.
- At least one GPU peak exceeded the 98% runtime abort ratio, although the configured three-sample streak did not trigger.
- The historical cumulative charge is expired launch evidence and is not reusable for formal launch authorization.
