# A/B 生成长度复核

**结论：需复核负载差异，不授权正式训练。**

两组各 64 条、8 步均完整。以步骤＋唯一解码输入配对，不按导出行号配对。

原比较生成时间（UTC）：2026-09-06T06:01:09.144590+00:00

总回复 token：5939 → 5823（-1.95%）；均值 92.7969 → 90.9844。

64/64 条输入配对且标签一致；26/64 条的行位置不同；仅 9/64 条解码输出文本相同。

| step | B mean | D mean | B max | D max | B tokens | D tokens | B/D Student 微批次 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 115.0 | 122.0 | 255 | 284 | 920 | 976 | 6/6 |
| 2 | 69.875 | 84.625 | 143 | 159 | 559 | 677 | 6/6 |
| 3 | 91.25 | 81.875 | 229 | 263 | 730 | 655 | 6/6 |
| 4 | 55.625 | 98.875 | 184 | 218 | 445 | 791 | 4/6 |
| 5 | 87.625 | 101.25 | 485 | 389 | 701 | 810 | 4/4 |
| 6 | 128.875 | 87.75 | 629 | 308 | 1031 | 702 | 4/4 |
| 7 | 97.625 | 84.125 | 189 | 148 | 781 | 673 | 4/4 |
| 8 | 96.5 | 67.375 | 225 | 125 | 772 | 539 | 4/4 |

## 质量判断与影响

- 高影响、高置信：8/8 步的平均/最大回复长度统计不同，前向微批次划分也不同；不满足既定同负载自动准入。
- 中影响、高置信：导出行顺序不同。已按唯一输入重新连接，避免把不同题目当成生成差异。
- 中影响、已知缺口：未导出逐样本原始 token IDs/response mask，不能重建精确逐样本 token 分布。
- 两组配置的 response 上限和记录的 response_width 均为 1024；实际有效 token 更少。固定张量宽度不等于相同计算量，remove-padding 和动态微批次仍会改变负载。
- 总 token 接近，且 deferred 并非每步更短；但较短的长尾与不同微批次仍是混杂因素，不能把显存降幅全部归因于延迟加载。
- temperature=1 的采样、执行数值差异或后续参数轨迹可能参与输出分歧；尚未通过受控重放定位原因。

## 下一步（尚未执行）

若需因果归因，准备同 token/同计算负载重放或明确的配对压力测试；若目标是资源安全，单独做近 1024-token 与 warmup 后更新验证。不要仅为得到 PASS 放宽现有负载判据。

## 口径与局限

- Totals reconstructed from logged response_mask mean * 8, independently reconciled to training logs; not retokenized decoded text.
- Rollouts omit sample IDs, raw generated token IDs and per-sample mask lengths. Join is step plus unique decoded input, not asserted raw multimodal/token identity.
- Decoded output character counts are not token counts. Per-sample exact token lengths and quantiles are unavailable from these exports.
- Shape counts describe microbatch calls, not a sample-by-sample paired compute trace. Dynamic batching differs.
- Same seed and temperature=1 do not establish identical generated tokens; the cause of divergence is not proven by these records.
- Allocator bookkeeping is not physical VRAM residency; no linear token-to-VRAM normalization or causal attribution is valid here.
- No post-warmup steps and no near-1024-token response safety validation. Formal training remains blocked.

## 证据

- 主比较：`../comparison_after_run.json` / `.md` / `.sha256`。
- 本目录 JSON 列出所有输入路径和 SHA256，CSV 保存逐步和逐输入配对记录。
- `workload_review.ipynb` 为可执行复核记录，仅使用 CPU。
- 未修改训练配置、原日志、checkpoint、历史 FAIL 或比较阈值。
