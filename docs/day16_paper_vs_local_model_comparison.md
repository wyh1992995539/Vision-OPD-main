# Day 16 原论文模型、当前 Base 与训练后模型结果详细对比

> 对比日期：2026-09-10（UTC）  
> 论文来源：`docs/Vision-OPD.pdf`，主要使用 Table 1、Table 2、Table 3  
> 本地评测：`E-PAPER-BASEJUDGE-001`、`E-D15-6K-FINAL-EVAL-001`  
> 当前结论：**R4 已完成并修复 R3 选项解析缺陷。本地 Vision-OPD 只在 ZoomBench 获得小幅收益，MMStar 与 V* 均低于本地 Base；Cached Prefix 的 Micro 结果更弱，未复现论文整体提升。**

## 一、对比对象与模型身份

| 对象 | 骨干模型 | 训练方式 | 身份或权重 | 定位 |
|---|---|---|---|---|
| 论文 Vanilla 4B | Qwen3.5-4B | 未进行 Vision-OPD 训练 | 论文 Table 2；正文未给出可核对的 checkpoint SHA256 | 论文主对照 Base |
| 论文 Vision-OPD 4B | Qwen3.5-4B | 6.2K、online rollout、Crop Teacher、Top-K JSD、EMA | 论文 Table 2 | 与本地模型最接近的论文主结果 |
| 论文 Vanilla / Vision-OPD 9B | Qwen3.5-9B | 同一方法，参数规模更大 | 论文 Table 2 | 仅作论文上限参考，不能与本地 4B 直接比较 |
| 本地 Base | Qwen3.5-4B | 未训练 | Hugging Face revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 本地冻结基线 |
| 本地 Vision-OPD | Qwen3.5-4B | 6,240 有效样本、online prefix、780 trainer steps | `E-D13-6K-VOPD-FINAL-001/merged_hf`；权重 SHA256 `265a3c49fbee7a44d65cf8af732c4a9b71de5ad9ef799c935c4e38d07094862f` | 当前主训练模型 |
| 本地 Cached Prefix | Qwen3.5-4B | 6,240 有效样本、冻结 Base prefix、780 trainer steps | `E-D14-6K-CACHED-001/merged_hf`；权重 SHA256 `4120ba83e138444461a078b701d605ba72a07be6c61acd1dea440b86ab3d4aaf` | off-policy prefix 消融模型 |

论文与本项目使用同名 Qwen3.5-4B 架构，但论文正文没有提供精确基础权重 revision，因此无法证明论文 Base 与本地 Base 的每个权重字节完全一致。

## 二、三项共同 Benchmark 的主对比

以下是最接近的比较：论文 Qwen3.5-4B Vanilla/Vision-OPD，与本地 Base/Vision-OPD/Cached。所有数值均为准确率；“变化”表示相对各自 Base 的百分点变化。

| Benchmark | N | 论文 Base | 论文 Vision-OPD | 论文变化 | 本地 Base R3 | 本地 Vision-OPD R3 | 本地变化 | 本地 Cached R3 | Cached 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V* Bench | 191 | 84.29% | **92.15%** | **+7.86 pp** | 83.77% | 75.92% | **−7.85 pp** | 77.49% | −6.28 pp |
| ZoomBench | 845 | 47.69% | **59.76%** | **+12.07 pp** | 50.65% | 52.54% | **+1.89 pp** | 51.36% | +0.71 pp |
| MMStar | 1,500 | 78.53% | **79.60%** | **+1.07 pp** | 75.07% | 68.00% | **−7.07 pp** | 62.47% | −12.60 pp |

直接结论：

1. 本地 Base 在 V* 上仅比论文 Base 低 0.52 pp，在 ZoomBench 上反而高 2.96 pp，因此本地 Base 并非全面失效。
2. 本地 Vision-OPD 只保留了论文在 ZoomBench 上的提升方向，但增益从论文的 +12.07 pp 缩小到 +1.89 pp。
3. 论文 Vision-OPD 在 V* 上提升 7.86 pp，本地模型却下降 7.85 pp，提升方向完全相反。
4. MMStar 是论文用于检查能力保持的 holdout task。论文提升 1.07 pp，本地 Vision-OPD 下降 7.07 pp，说明本地训练出现明显能力遗忘或迁移。
5. Cached Prefix 在三项中均低于本地 Vision-OPD 或仅在 V* 小幅回升；它没有复现论文主方法的收益。

## 三、本地模型距离论文模型还有多远

这里直接计算“本地成绩 − 论文同角色成绩”。负数表示低于论文。

| Benchmark | 本地 Base − 论文 Base | 本地 Vision-OPD − 论文 Vision-OPD | Cached − 论文 Vision-OPD |
|---|---:|---:|---:|
| V* Bench | −0.52 pp | **−16.23 pp** | −14.66 pp |
| ZoomBench | +2.96 pp | **−7.22 pp** | −8.40 pp |
| MMStar | −3.46 pp | **−11.60 pp** | −17.13 pp |

本地 Vision-OPD 与论文模型的最大差距出现在 V*，达到 16.23 pp。由于本地 Base 在 V* 与论文 Base 很接近，这个差距主要在训练阶段产生。ZoomBench 的本地 Base 高于论文 Base，但训练后仍比论文 Vision-OPD 低 7.22 pp，说明当前训练只获得了论文增益的一小部分。

## 四、论文完整 Table 2：Qwen3.5-4B

本地只完成了 V*、ZoomBench、MMStar 三项，尚未评测 HR-Bench 4K/8K、MMVP、CV-Bench 和 POPE。因此当前不能声称完整复现或完整否定论文 Table 2。

| 类型 | Benchmark | 论文 Vanilla 4B | 论文 Vision-OPD 4B | 变化 | 本地结果 |
|---|---|---:|---:|---:|---|
| 细粒度视觉 | V* Bench | 84.29% | 92.15% | +7.86 pp | 已评测；本地变化 −7.85 pp |
| 细粒度视觉 | ZoomBench | 47.69% | 59.76% | +12.07 pp | 已评测；本地变化 +1.89 pp |
| 细粒度视觉 | HR-Bench 4K | 84.38% | 84.50% | +0.12 pp | 未评测 |
| 细粒度视觉 | HR-Bench 8K | 80.13% | 80.38% | +0.25 pp | 未评测 |
| Holdout | MMVP | 76.67% | 79.67% | +3.00 pp | 未评测 |
| Holdout | CV-Bench | 87.13% | 87.27% | +0.14 pp | 未评测 |
| Holdout | MMStar | 78.53% | 79.60% | +1.07 pp | 已评测；本地变化 −7.07 pp |
| Holdout | POPE | 88.28% | 89.14% | +0.86 pp | 未评测 |

论文 4B 的主要增益集中在 V*、ZoomBench 和 MMVP，而 HR-Bench、CV-Bench、MMStar、POPE 的变化较小，作用更接近“保持原能力”。本地结果的问题正是没有做到能力保持：ZoomBench 小幅上升的同时，MMStar 和 V* 明显下降。

## 五、论文 9B 结果参考

本地训练的是 4B，下面的 9B 结果只说明论文方法在更大模型上的表现，不能作为本地模型的直接验收线。

| Benchmark | 论文 Vanilla 9B | 论文 Vision-OPD 9B | 变化 |
|---|---:|---:|---:|
| V* Bench | 82.72% | 94.76% | +12.04 pp |
| ZoomBench | 52.07% | 65.80% | +13.73 pp |
| HR-Bench 4K | 85.75% | 88.13% | +2.38 pp |
| HR-Bench 8K | 80.63% | 85.50% | +4.87 pp |
| MMVP | 83.33% | 83.67% | +0.34 pp |
| CV-Bench | 88.29% | 88.40% | +0.11 pp |
| MMStar | 83.07% | 83.20% | +0.13 pp |
| POPE | 88.88% | 89.13% | +0.25 pp |

论文 9B 同样体现“细粒度任务大幅提升、holdout 基本保持”的模式。当前本地 4B 呈现“ZoomBench 小幅提升、其他能力明显回归”，与论文模式不同。

## 六、本地三模型的逐题迁移

| 比较 | Base/前一模型错→训练后对 | Base/前一模型对→训练后错 | 净正确数变化 |
|---|---:|---:|---:|
| Base → Vision-OPD | 306 | 411 | **−105** |
| Base → Cached Prefix | 287 | 482 | **−195** |
| Vision-OPD → Cached Prefix | 255 | 345 | **−90** |

这些数字证明本地训练并非完全没有学到新能力：Vision-OPD 修正了 306 道 Base 错题。但是同时损失了 411 道 Base 原本正确的题，回归比新增收益多 105 道。Cached 的回归更大。

## 七、主要能力分项

下表采用“剔除可明确确认的 first-letter 误报”敏感性口径，用于观察能力方向，不作为正式 R4 分数。

| Benchmark / 类别 | Base | Vision-OPD | Vision−Base | Cached | Cached−Base |
|---|---:|---:|---:|---:|---:|
| V* / relative_position | 82.89% | 63.16% | **−19.74 pp** | 75.00% | −7.89 pp |
| V* / direct_attributes | 84.35% | 82.61% | −1.74 pp | 79.13% | −5.22 pp |
| MMStar / logical reasoning | 76.80% | 64.00% | **−12.80 pp** | 56.40% | −20.40 pp |
| MMStar / fine-grained perception | 65.20% | 55.60% | **−9.60 pp** | 50.80% | −14.40 pp |
| MMStar / math | 80.80% | 72.00% | −8.80 pp | 72.40% | −8.40 pp |
| MMStar / instance reasoning | 72.40% | 64.80% | −7.60 pp | 63.20% | −9.20 pp |
| MMStar / coarse perception | 70.80% | 69.20% | −1.60 pp | 66.40% | −4.40 pp |
| MMStar / science & technology | 50.00% | 48.40% | −1.60 pp | 46.40% | −3.60 pp |
| ZoomBench / multiple choice | 55.88% | 57.33% | +1.45 pp | 55.72% | −0.16 pp |
| ZoomBench / open question | 36.16% | 39.29% | +3.13 pp | 39.29% | +3.13 pp |

Vision-OPD 的收益主要出现在 ZoomBench 的开放题和选择题；最大损失集中在空间相对位置、逻辑推理、细粒度感知与数学。这不是单一题型或一次推理服务故障能够解释的局部异常。

## 八、R3 原始分数与解析器敏感性结果

R3 已确认存在以下错误路径：模型输出 `Answer: D` 时，旧 first-letter 逻辑可能读取单词 `Answer` 的首字母 A；当参考答案恰好是 A 时，会把明确错误的 D 判成正确。

| 模型 | Benchmark | R3 原始正确数/总数 | R3 原始准确率 | 明确误报 | 剔除明确误报后 |
|---|---|---:|---:|---:|---:|
| Base | ZoomBench | 428/845 | 50.65% | 0 | 50.65% |
| Base | MMStar | 1,126/1,500 | 75.07% | 86 | 69.33% |
| Base | V* | 160/191 | 83.77% | 0 | 83.77% |
| Vision-OPD | ZoomBench | 444/845 | 52.54% | 0 | 52.54% |
| Vision-OPD | MMStar | 1,020/1,500 | 68.00% | 85 | 62.33% |
| Vision-OPD | V* | 145/191 | 75.92% | 2 | 74.87% |
| Cached Prefix | ZoomBench | 434/845 | 51.36% | 0 | 51.36% |
| Cached Prefix | MMStar | 937/1,500 | 62.47% | 48 | 59.27% |
| Cached Prefix | V* | 148/191 | 77.49% | 0 | 77.49% |

Micro 汇总：

| 模型 | R3 原始 | 剔除明确误报后 | 相对 Base 的调整后变化 |
|---|---:|---:|---:|
| Base | 67.59% | 64.20% | — |
| Vision-OPD | 63.45% | 60.02% | **−4.18 pp** |
| Cached Prefix | 59.90% | 58.00% | **−6.19 pp** |

解析缺陷明显改变绝对分数，并使 Cached 的原始差距看起来更大；但 Base 与 Vision-OPD 的明确误报数量分别为 86 和 87，剔除后相对差距仍为约 −4.18 pp。因此解析缺陷不能消除“本地 Vision-OPD 整体低于本地 Base”的结论。

该敏感性口径不是完整 R4。2026-09-10 已完成正式 R4：重新执行解析与路由，补跑新进入冻结 Judge 路径的 190 条样本；最终结果见本文件末尾和 `docs/day16_r4_basejudge_rescore_brief.md`。

## 九、推理行为与评分路由

| 指标 | Base | Vision-OPD | Cached Prefix |
|---|---:|---:|---:|
| 请求成功 | 2,536/2,536 | 2,536/2,536 | 2,536/2,536 |
| Inference errors | 0 | 0 | 0 |
| Completion tokens | 550,330 | 448,933 | 345,021 |
| 平均 completion tokens/题 | 217.01 | 177.02 | 136.05 |
| 达到 1,024-token 上限 | 172 | 163 | 138 |
| MathRuler 判对 | 85 | 57 | 145 |
| First-letter 判对 | 665 | 490 | 455 |
| Judge required | 1,786 | 1,989 | 1,936 |
| Judge failure | 1 | 0 | 1 |

三组推理均完整完成，0 inference error，因此当前差异不是请求缺失、服务崩溃或不同分母造成的。训练后模型回答明显变短，评分路径的组成也发生变化，这会和现有解析器/Judge 组合产生交互，进一步说明需要统一 R4 重评分。

## 十、训练设置对照

论文正文公开了 JSD、Top-K、EMA、response 1024 和 1 epoch；batch、rollout 与 GPU 等运行值来自仓库作者入口 `scripts/run_vision_opd.sh`。

| 参数 | 作者入口 | 本地 Vision-OPD / Cached | 影响 |
|---|---:|---:|---|
| GPU | 8 | 2 | GPU 数本身主要影响资源；它推动了下列训练参数缩放 |
| Global batch | 96 | 8 | 本地为 1/12，梯度估计噪声更大 |
| PPO mini-batch | 96 | 8 | 本地为 1/12 |
| Rollout n | 8 | 1 | 每个问题只采一条当前策略轨迹 |
| 每 epoch rollout 序列 | 49,920 | 6,240 | 本地为 1/8，on-policy 前缀覆盖明显减少 |
| Trainer steps / EMA events | 约 65 | 780 | 本地为 12 倍 |
| Student Adam updates，按实现估算 | 约 520 | 780 | 本地约为 1.5 倍，且单次 mini-batch 更小 |
| Learning rate | `2e-6` | `2e-6` | 未随 batch、更新次数重新标定 |
| Warmup | 10 trainer steps | 10 trainer steps | 作者对应 960 prompts/7,680 rollouts，本地对应 80/80 |
| EMA update rate | 0.05 | 0.05 | 系数相同，但相对样本量的 Teacher 更新速度不同 |
| Response length | 1,024 | 1,024 | 对齐 |
| Epoch | 1 | 1 | 对齐 |
| Top-K / JSD β | 100 / 0.5 | 100 / 0.5 | 对齐 |
| Prefix | 当前 Student online rollout | Vision=online；Cached=冻结 Base cache | Cached 不属于论文主方法 |

训练运行本身健康：Vision-OPD 和 Cached 均完成 780/780 steps，没有 NaN、OOM、abort 或 checkpoint 损坏。工程成功只证明权重按配置更新完成，不能替代外部能力验收。

## 十一、可比性限制

1. 论文与本地只在 V*、ZoomBench、MMStar 三项重合；本地缺少五项 Table 2 结果。
2. 论文使用的 Judge 记录为 GPT-OSS-120B，本地因不可用替换为冻结 Qwen3.5-4B Base Judge。
3. 本地 R3 存在已确认的 first-letter 解析缺陷；该缺陷已在独立 R4 协议中修复，R3 仅保留为历史结果。
4. V* 有 21/191 条测试图像与训练图像确认重合，因此不能称为完全独立测试；不过 Vision-OPD 在重合和非重合部分均未表现出论文式增益。
5. 论文的 Average 70.68%→77.07% 是六项细粒度 Benchmark 的算术范围；本地 micro 67.59%→63.45% 是三项共 2,536 条按样本加权的汇总，二者定义不同，禁止直接比较。
6. 本地 Cached Prefix 是额外消融模型，论文 Table 2 没有同名结果，不能把它当作论文 Vision-OPD 模型。

## 十二、最终判断

- **论文 Vision-OPD-4B：**在 V* 和 ZoomBench 获得大幅提升，并在 MMStar 等 holdout 上保持或略有提升。
- **本地 Vision-OPD：**确认获得小幅 ZoomBench 收益，但 V* 和 MMStar 明显低于本地 Base；当前不满足“复现论文整体收益”的验收条件。
- **本地 Cached Prefix：**整体弱于本地 Vision-OPD，尤其 MMStar 回归明显；它验证了 online prefix 对结果有价值，但当前两种训练都受相同资源缩放影响。
- **本地 Base：**R4 下仍是当前三项评测中更稳妥的通用模型；Vision-OPD 可作为研究 checkpoint 保留，目前不宜替代 Base。
- **下一步：**使用小规模受控 Pilot 调整 rollout、batch、EMA 时间尺度和学习率；只有 Pilot 显示 ZoomBench 提升且 V*/MMStar 不回归，才值得重新进行完整训练。

## 十三、本地 Base 与论文 Base 差异诊断

### 1. 差异大小与判断

| Benchmark | 论文 Base | 本地 Base R3 | 差值 | 粗略独立样本 z 值 | 判断 |
|---|---:|---:|---:|---:|---|
| V* Bench | 84.29% | 83.77% | −0.52 pp | −0.14 | 很小，合理 |
| ZoomBench | 47.69% | 50.65% | +2.96 pp | +1.22 | 可接受的复现偏差 |
| MMStar | 78.53% | 75.07% | −3.46 pp | −2.25 | 不能只用样本波动解释 |

这里的 z 值只是把固定 Benchmark 视为任务分布样本后的粗略不确定性参考，不代表 temperature=0 的重复运行会随机波动。本地推理基本是确定性的；真实差异来自权重版本、推理/图像实现或评分协议。

### 2. 三项 Benchmark 对评测链路的依赖不同

| Benchmark | N | 触及 1,024-token 上限 | MathRuler 判对 | First-letter 判对 | 进入本地 Base Judge | 明确 first-letter 误报 |
|---|---:|---:|---:|---:|---:|---:|
| ZoomBench | 845 | 7（0.83%） | 85 | 0 | 760（89.94%） | 0 |
| MMStar | 1,500 | 163（10.87%） | 0 | 512 | 988（65.87%） | 86 |
| V* Bench | 191 | 2（1.05%） | 0 | 153 | 38（19.90%） | 0 |

V* 大部分回答由选项字母直接判定，Judge 依赖最低，因此最接近论文 Base。MMStar 同时高度依赖 Judge、容易生成长回答，并受 first-letter 缺陷影响，所以它的绝对分数最不稳定。

### 3. MMStar 的生成长度影响很大

| Benchmark | 正常 stop 数量与准确率 | length 数量与准确率 |
|---|---:|---:|
| ZoomBench | 838，51.07% | 7，0.00% |
| MMStar | 1,337，81.53% | 163，22.09% |
| V* Bench | 189，84.13% | 2，50.00% |

MMStar 正常结束的 1,337 条本地 Base 回答准确率为 81.53%，并不低于论文的 78.53%；163 条触及长度上限的回答只有 22.09%，把总体拉低到 75.07%。这些 length 样本可能本身更难，不能假设延长输出就会全部恢复，但当前 `max_tokens=1024` 足以解释数个百分点的差异。论文正文没有明确公开评测 `max_tokens`，本项目采用的是后续公开命令口径，因此这里不能证明与论文运行完全相同。

### 4. Judge 替代会影响绝对可比性

论文评测记录使用 GPT-OSS-120B Judge；本项目因不可用，使用冻结 Qwen3.5-4B Base Judge。MMStar 有 988/1,500 条进入 Judge，这意味着约三分之二样本的最终判定由不同能力和偏好的 Judge 决定。Base 模型又负责判断自身生成内容，存在潜在自评偏差。ZoomBench 虽然 Judge 比例更高，但其最终总分恰好接近论文不能证明 Judge 等价。

### 5. 解析器缺陷使接近的原始分数可能具有偶然性

R3 first-letter 分支会在部分 `Answer: D` 输出中错误读取 `Answer` 的 A。Base MMStar 至少有 86 条明确错误选项被计为正确，相当于 5.73 个百分点。只剔除这类明确误报后，Base MMStar 为 69.33%，但这仍不是完整 R4，因为修复解析器后部分样本需要重新进入 Judge。

因此，本地 R3 的 75.07% 不能作为精确复现论文 78.53% 的可信测量；误报、截断和替代 Judge 可能相互抵消，使最终差值表面上只有 3.46 pp。

### 6. 其他合理来源

- 论文只给出 Qwen3.5-4B 名称，没有提供精确基础 checkpoint revision 和 SHA256；本地固定 revision 为 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- 本地使用 vLLM、TP=1、GDN Triton prefill 和默认 dtype；论文没有完整公开服务后端、硬件和数值内核。
- 本地固定了数据 revision 和文件 SHA256，但论文没有发布对应输入快照哈希，无法证明图像字节和数据 revision 完全相同。
- Prompt 基本对齐论文附录；V* 的本地 RGB PNG 转码、Chat Template 和 Processor 版本仍可能造成少量差异。

### 7. “是否合理”的分层结论

- **V* 的 −0.52 pp：合理。** 结果足够接近，可认为 Base 与论文在该项大致复现。
- **ZoomBench 的 +2.96 pp：合理。** 有限样本、权重 revision、服务栈和 Judge 差异可以产生这一量级变化。
- **MMStar 的 −3.46 pp：作为不同评测协议的结果可以解释，但不能称为精确复现。** 截断、替代 Judge 和解析器缺陷均是实质性差异。
- **作为项目内部基线：Base 仍然有用。** R4 已统一修复三模型的解析与路由；Base 在 MMStar 和 V* 仍领先，Vision-OPD 仅在 ZoomBench 领先。
- **作为论文绝对数值复现：当前 Base 仍不够。** R4 已验证通过，但本地 Judge 仍不是论文的 GPT-OSS-120B，且作者未公开精确基础权重与全部运行参数，因此不能声称精确复现论文 Table 2。

## 十四、证据索引

- 原论文：`docs/Vision-OPD.pdf`
- 作者训练入口：`scripts/run_vision_opd.sh`
- 本地训练配置：`configs/vopd_6241.yaml`、`configs/cached_prefix_6241.yaml`
- Base 正式 R3：`artifacts/runs/E-PAPER-BASEJUDGE-001/base/`
- Vision-OPD 正式 R3：`artifacts/runs/E-D15-6K-FINAL-EVAL-001/vision_opd/`
- Cached Prefix 正式 R3：`artifacts/runs/E-D15-6K-FINAL-EVAL-001/cached_prefix/`
- 三模型 R3 冻结比较：`artifacts/runs/E-D15-6K-FINAL-EVAL-001/comparison.json`
- 三模型 R4 冻结比较：`artifacts/runs/E-PAPER-BASEJUDGE-R4-001/comparison.json`
- R4 验证与冻结凭据：`artifacts/runs/E-PAPER-BASEJUDGE-R4-001/validation.json`、`freeze_receipt.json`
- 解析器敏感性与分类数据：`artifacts/reports/vision_opd_gap_diagnostic/diagnostic_data.json`
- 交互式诊断报告：`artifacts/reports/vision_opd_gap_diagnostic/report.html`
- Vision-OPD 最终模型：`artifacts/runs/E-D13-6K-VOPD-FINAL-001/merged_hf/`
- Cached 最终模型：`artifacts/runs/E-D14-6K-CACHED-001/merged_hf/`


## R4 评分修复的实现状态（2026-09-10）

R4 采用版本化修复，冻结 R3 配置与既有 `predictions.jsonl`、`judge_results.jsonl`、`scores.jsonl` 保持不变。代码入口为 `eval/mcq_parser_r4.py`，并由 `eval/score_paper_aligned.py` 中的 `judge.mcq_parser_revision: 4` 开关启用；未声明该字段时继续执行原 R3 逻辑。

三态规则如下：

1. 参考答案与模型答案都能解析出唯一显式选项，且相同：直接判对；
2. 两者都能解析出唯一显式选项，且不同：直接判错，不允许 Judge 推翻；
3. 模型答案无显式选项或包含相互冲突的显式选项：交给同一个冻结 Qwen3.5-4B Base Judge；
4. `Answer: **D**` 只会解析为 D，不再把单词 `Answer` 的 A 当作预测；
5. R4 识别 `<answer>B</answer>`、`Final Answer: B`、`Correct option: (B)`、`正确答案：B`、`\\boxed{B}` 和独立末行 B 等明确格式，不扫描普通句子中的任意大写字母。

针对性测试和原 R3 回归测试共 19 项，全部通过。正式 R4 已补跑以下冻结 Judge 请求，190/190 完成、0 失败：

| 模型 | MMStar 新增 Judge | V* 新增 Judge | 合计 |
|---|---:|---:|---:|
| Base | 26 | 108 | 134 |
| Vision-OPD | 27 | 0 | 27 |
| Cached Prefix | 28 | 1 | 29 |
| **合计** | **81** | **109** | **190** |

R4 已在新目录 `artifacts/runs/E-PAPER-BASEJUDGE-R4-001/` 完成，复用了提示文本和 Judge 身份完全一致的旧记录，并重新生成三组 `scores.jsonl`、`summary.json`、对比表、验证凭据和 SHA256。R3 原目录运行前后哈希一致，继续作为历史协议结果保留。

最终 R4：Base 为 ZoomBench 50.65%、MMStar 68.00%、V* 82.72%；Vision-OPD 为 52.54%、59.20%、71.73%；Cached Prefix 为 51.36%、56.33%、73.30%。

## MMStar 公开解析器风险论证（2026-09-10）

项目已将 `EVAL-MMSTAR-PARSER-RISK-001` 登记为强工作假设：公开评测器中已确认的答案格式偏差，在相同本地 Base predictions 上造成 7.07 pp 净分数膨胀；截断样本延长输出另观察到 3.47 pp 改善。两项协议效应足以解释论文 78.53% 与本地 R4 68.00% 的误差量级，因此该 10.53 pp 差距不能被直接归因于模型能力。

完整论证、限制和可引用文字见 `docs/mmstar_public_parser_bias_argument.md`。作者是否在 Table 2 实际运行中使用相同解析器仍缺少直接产物证明，故该部分保持为高概率推断。
