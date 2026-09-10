# Day16 R4 Base-Judge 重评分简报

> 状态：**PASS**。R3 原始预测和评分产物未改写；R4 已补评 190 条歧义回答，Judge 失败 0 条。

## 最终 R4 结果

| 模型 | ZoomBench | MMStar | V* Bench | 三项 Micro |
|---|---:|---:|---:|---:|
| Base | 50.65% | 68.00% | 82.72% | 63.33% |
| Vision-OPD | 52.54% | 59.20% | 71.73% | 57.93% |
| Cached Prefix | 51.36% | 56.33% | 73.30% | 55.95% |

## R3 到 R4 的变化

| 模型 | ZoomBench | MMStar | V* Bench | Micro |
|---|---:|---:|---:|---:|
| Base | +0.00 pp | -7.07 pp | -1.05 pp | -4.26 pp |
| Vision-OPD | +0.00 pp | -8.80 pp | -4.19 pp | -5.52 pp |
| Cached Prefix | +0.00 pp | -6.13 pp | -4.19 pp | -3.94 pp |

ZoomBench 不使用被修复的 first-letter 分支，因此结果不变。MMStar 变化最大，说明 R3 的任意大写字母回退与不同模型的长答案格式产生了显著交互。

## R4 模型间差值

| Benchmark | Vision-OPD vs Base | Cached vs Base | Cached vs Vision-OPD |
|---|---:|---:|---:|
| ZoomBench | +1.89 pp | +0.71 pp | -1.18 pp |
| MMStar | -8.80 pp | -11.67 pp | -2.87 pp |
| V* Bench | -10.99 pp | -9.42 pp | +1.57 pp |

R4 下 Vision-OPD 仅在 ZoomBench 高于 Base；MMStar 与 V* 均明显低于 Base。Cached Prefix 在 V* 高于 Vision-OPD，但仍低于 Base。

## 执行与凭据

- 新增 Judge：190 条，完成 190 条，失败 0 条。
- Judge：冻结 Qwen3.5-4B Base，单卡 TP=1，显存比例 0.75，GDN Triton。
- 三组均为 2,536/2,536，重复键 0，待评 0。
- R3 四个源文件的运行前后 SHA256 一致。
- 根凭据：artifacts/runs/E-PAPER-BASEJUDGE-R4-001/freeze_receipt.json、validation.json、artifact_sha256.txt。
- 详细对比：artifacts/runs/E-PAPER-BASEJUDGE-R4-001/comparison.json。

本地仍使用冻结 Qwen3.5-4B Base Judge 替代论文 GPT-OSS-120B Judge，因此 R4 修复了本地确定性解析与路由问题，但仍不能视为论文 Judge 的精确复现。
