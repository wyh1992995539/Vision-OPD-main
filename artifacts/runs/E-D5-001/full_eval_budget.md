# Day 5 Task 6: Full external-evaluation budget

- Experiment: E-D5-001, protocol revision 4
- Pricing: 11.96 CNY per dual-GPU instance hour (user_reported_autodl_dual_gpu_hourly_price_2026-08-24)
- Full workload: 3381 requests; expected semantic Judge 56, maximum 448
- Projected tokens: prompt 4813871, completion 5815151

| Scenario | Wall time | Cost (CNY) | Judge instances |
|---|---:|---:|---:|
| measured_throughput | 2.18 h | 26.11 | 56 |
| conservative_execution_budget | 3.61 h | 43.18 | 56 |
| worst_case_guardrail | 5.99 h | 71.70 | 448 |

Recommended execution cap: 100.00 CNY; project hard cap: 2000.00 CNY.

Method: benchmark-specific mean Smoke latency and token counts are multiplied by frozen full request counts. The measured scenario uses observed completion-span concurrency; the conservative and worst-case scenarios apply fixed lower concurrency and larger buffers. Judge calls are sequentially budgeted at 5 seconds each.
