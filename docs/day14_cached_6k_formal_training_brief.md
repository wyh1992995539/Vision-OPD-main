# Day 14 Cached Prefix 6241 正式训练工作简报

> 执行日期：2026-09-08（UTC）  
> 实验 ID：`E-D14-6K-CACHED-001`  
> 正式训练：**PASS，780/780**  
> 最终 FSDP checkpoint：**PASS**  
> checkpoint SHA256：**PASS，13/13**  
> Cached Hugging Face 合并：**PASS，7/7 SHA256**

## 技术摘要

Day 14 从冻结原始 Qwen3.5-4B Base 冷启动 Cached Prefix 正式训练，使用与 Vision-OPD 主实验一致的 train-6241、drop-last、global batch 8、780 optimizer steps、学习率、上下文长度、Crop Teacher、Top-K JSD、EMA 和 checkpoint 合同。唯一目标变量仍是 Student prefix 来源：本实验从已冻结缓存读取，并在全部 780 steps 中保持 online generation calls 为 0。

训练从 2026-09-08 05:21:07 UTC 的实时启动 Gate 开始，到 10:08:04 UTC 正常退出，墙钟约 4 小时 46 分 57 秒。按双卡合计 14 元/小时估算费用约 66.95 元，该数值不是平台账单。最终有效样本为 6,240/6,241，padding 0，drop-last 1；`global_step_780` checkpoint 文件合同和独立 SHA256 复核均通过。

退出时监控器已确认训练和 checkpoint 为 PASS，但旧实现没有在子进程退出后读取日志缓冲区的最后一行，使原 guard 凭据中的 `latest_step` 停在 779。现已保留原凭据，依据完整日志重放补齐第 780 行 runtime metric，并修复监控器的 post-exit final drain。修正后的 guard 和 exit receipt 均记录 `latest_step=780`，完整日志 Gate 为 PASS，相关回归测试通过。

## 一、实际完成的工作

| 顺序 | 实际工作 | 主要结果 | 状态 |
|---:|---|---|---|
| 1 | 运行启动前静态与实时 Gate | 晋级、配置、缓存、资源、训练合同全部通过；实时启动凭据冻结 | PASS |
| 2 | 执行 Cached Prefix 正式训练 | 780/780 steps，6,240/6,241 有效样本，padding 0，dropped 1 | PASS |
| 3 | 审计完整训练日志 | step 1～780 连续；0 OOM、0 NaN、0 Traceback、0 NCCL fatal | PASS |
| 4 | 审计 Cached 与自蒸馏运行期 Gate | 每步 cache resolved=8、online generation=0、policy fallback=0、empty target=0、EMA=1 | PASS |
| 5 | 修正退出凭据尾行遗漏 | 保留原始 779 凭据，完整日志重放后修正为 780；监控器增加 final drain | PASS |
| 6 | 冻结最终 FSDP checkpoint | 13 个文件、57,034,962,958 bytes；文件合同和稳定性检查通过 | PASS |
| 7 | 生成并独立复核 checkpoint SHA256 | `sha256sum -c` 13/13 | PASS |
| 8 | 准备原子化 FSDP→Hugging Face 合并 | 独立 `merged_hf.inprogress`、输出合同、724 BF16 张量和 SHA256 校验已实现 | READY |
| 9 | 执行 Cached 合并 | 7 个文件、10,370,019,516 bytes；724 个 BF16 张量；独立 SHA256 7/7 | PASS |

## 二、正式训练结果

| 项目 | 实际结果 |
|---|---:|
| 实验 ID | `E-D14-6K-CACHED-001` |
| 启动 Gate | PASS |
| 完成 step | 780/780 |
| 有效训练样本 | 6,240/6,241 |
| global batch | 8 |
| padding / dropped | 0 / 1 |
| Cached records resolved | 每步 8，780/780 一致 |
| Online generation calls | 每步 0，780/780 一致 |
| Policy fallback | 每步 0，780/780 一致 |
| Empty target batch | 每步 0，780/780 一致 |
| EMA update applied | 每步 1，780/780 一致 |
| 致命日志匹配 | 0 |
| 开始时间 | 2026-09-08 05:21:07 UTC |
| 结束时间 | 2026-09-08 10:08:04 UTC |
| 墙钟 | 17,217 秒，约 4 小时 46 分 57 秒 |
| 估算费用 | 约 66.95 元，按双卡合计 14 元/小时 |

最终 step 780 的关键指标仍为有限值：`actor/vopd_loss=0.000303819`、`raw_jsd_token_mean=0.000721400`、`actor/grad_norm=0.091793`、`actor/lr=2e-6`。该步 Student optimizer delta 为正，Teacher optimizer delta 和直接梯度为 0，EMA 更新生效。训练 loss 只用于运行健康检查，不代替 Day 15 的统一外部能力评测。

## 三、凭据问题与修正

原训练子进程正常返回 0，最终 checkpoint marker 和目录均为 780/PASS。问题仅发生在监控器生成退出凭据时：旧循环在发现子进程退出后立即结束，没有再读取训练日志文件的尾部，因此最后一条 step 780 metric 没有进入 `runtime_metrics.jsonl`，原凭据记录为 779。

修正采取以下可审计流程：

1. 原始 guard 和 exit receipt 原样保存为 `*.pre_final_drain.json`，并在修正凭据中记录其 SHA256。
2. 从完整 `train.log` 重放 step 1～780，验证 step 连续、关键指标有限、所有 Cached/Teacher/EMA Gate 一致且无 fatal 日志。
3. 只补写缺少的 step 780 runtime metric；修正后集合严格等于 1～780。
4. 将当前 guard 和 exit receipt 修正为 `latest_step=780`，记录原因、方法、原凭据身份和修正脚本身份。
5. 修改 `monitor_vopd_training.py`，在训练进程退出后执行 final log drain，防止后续训练再次漏掉尾行。
6. 回归验证结果为 **23 passed，9 subtests passed**；Python 语法检查通过。

历史 promotion receipt 没有被重写。它冻结的是正式启动时的源码身份，`day14_live_launch_gate.json` 已证明当时 `promotion_receipt_valid=true`。监控器在训练完成后的修复会使“按当前源码重新执行启动前哈希校验”出现预期差异，测试现已明确区分历史启动凭据和未来新启动的当前源码校验。

## 四、最终 checkpoint 与 SHA256

最终 checkpoint：

```text
artifacts/runs/E-D14-6K-CACHED-001/checkpoints/global_step_780/
```

| 项目 | 实际结果 |
|---|---:|
| global step | 780 |
| world size | 2 |
| 文件数 | 13 |
| 总大小 | 57,034,962,958 bytes，约 53.12 GiB |
| marker | `latest_checkpointed_iteration.txt=780` |
| 文件合同 | PASS |
| 所有文件非空 | PASS |
| 哈希期间文件稳定 | PASS |
| 独立 SHA256 复核 | 13/13 PASS |
| checkpoint manifest SHA256 | `5bbe16155d54e00bf251f8bcf19ccf229b05a747309c9c4d30443c4ed2490eb2` |
| SHA256 清单自身 SHA256 | `dfe3472a3d77da4b4a0d0a58e00069b694867bf2f6e11bee530ddd4f605bab0b` |

## 五、Cached 合并结果

最终合并模型：

```text
artifacts/runs/E-D14-6K-CACHED-001/merged_hf/
```

| 项目 | 实际结果 |
|---|---:|
| merged 文件数 | 7 |
| merged 总大小 | 10,370,019,516 bytes，约 9.66 GiB |
| `model.safetensors` 大小 | 10,350,019,328 bytes |
| Tensor 数量 / dtype | 724 / BF16 |
| `model.safetensors` SHA256 | `4120ba83e138444461a078b701d605ba72a07be6c61acd1dea440b86ab3d4aaf` |
| merged manifest SHA256 | `1f6e67b7ac0bf333163c867152875e350d828045fdb87c3f46476ce9a50e7cc0` |
| SHA256 清单自身 SHA256 | `ee6beff4a9c5777d8dd0bfc84d8e64cf78edb7f7e90853baa88211c470df54fa` |
| 独立 SHA256 复核 | 7/7 PASS |
| 原子晋级 | PASS；`merged_hf.inprogress` 已不存在 |

合并前首先确认 cgroup 已从 2 GiB 恢复至 120 GiB，数据盘约有 182 GiB 可用。FSDP merger 正常返回 0，并在独立 `merged_hf.inprogress` 目录生成完整模型。

第一次后置校验发现来源模板与合并配置全对象不相等，因此按设计停止在临时目录，没有晋级最终模型。逐字段审计确认差异严格限定为 `text_config.dtype` 和 `vision_config.dtype` 从 `float32` 变为 `bfloat16`；其余字段完全一致，724 个权重张量也全部为 BF16。这是 Transformers `save_pretrained` 按实际合并权重 dtype 保存元数据的正常结果。

该次停止已冻结为 `merge_validation_attempt1.json`。随后使用独立校验器只允许上述两个精确转换，再次验证 7 个必需文件、非空文件、724 个 BF16 张量和配置差异白名单，通过后将临时目录原子晋级为 `merged_hf`。最终逐文件 SHA256 和独立 `sha256sum -c` 均为 7/7 PASS。

## 六、阶段结论与边界

Day 14 正式训练、完整运行期 Gate、最终 checkpoint、凭据尾行修正、checkpoint SHA256 和 Cached Hugging Face 合并均已完成并通过。当前既保留可恢复训练的 FSDP checkpoint，也已生成可供后续冷加载与评测使用的单目录 Cached Student。

外部 ZoomBench、MMStar 和 V* 结果尚未运行，也没有据此重选 checkpoint。按项目计划，合并后的 5 条新进程冷加载和统一 R3 外部评测属于 Day 15。

## 七、关键证据索引

- 实时启动 Gate：`artifacts/runs/E-D14-6K-CACHED-001/preflight/day14_live_launch_gate.json`
- 修正后 guard：`artifacts/runs/E-D14-6K-CACHED-001/evidence/guard_summary.json`
- 修正后退出凭据：`artifacts/runs/E-D14-6K-CACHED-001/evidence/exit_receipt.json`
- 原始 guard：`artifacts/runs/E-D14-6K-CACHED-001/evidence/guard_summary.pre_final_drain.json`
- 原始退出凭据：`artifacts/runs/E-D14-6K-CACHED-001/evidence/exit_receipt.pre_final_drain.json`
- 完整 runtime metrics：`artifacts/runs/E-D14-6K-CACHED-001/evidence/runtime_metrics.jsonl`
- checkpoint manifest：`artifacts/runs/E-D14-6K-CACHED-001/evidence/checkpoint_manifest.json`
- checkpoint 哈希凭据：`artifacts/runs/E-D14-6K-CACHED-001/evidence/checkpoint_hash_receipt.json`
- checkpoint SHA256 清单：`artifacts/runs/E-D14-6K-CACHED-001/checkpoint_sha256.txt`
- 合并凭据：`artifacts/runs/E-D14-6K-CACHED-001/evidence/merge_receipt.json`
- 合并 manifest：`artifacts/runs/E-D14-6K-CACHED-001/evidence/merged_manifest.json`
- merged SHA256 清单：`artifacts/runs/E-D14-6K-CACHED-001/merged_sha256.txt`
- 第一次合并校验停止记录：`artifacts/runs/E-D14-6K-CACHED-001/evidence/merge_validation_attempt1.json`
- 最终 Cached merged 模型：`artifacts/runs/E-D14-6K-CACHED-001/merged_hf/`
- 最终 checkpoint：`artifacts/runs/E-D14-6K-CACHED-001/checkpoints/global_step_780/`
- 训练日志：`artifacts/runs/E-D14-6K-CACHED-001/logs/train.log`
- 凭据/冻结/合并工具：`scripts/finalize_day14_cached.py`
- 合并临时目录严格晋级工具：`scripts/finalize_day14_cached_merge_staging.py`
- 修正后的监控器：`scripts/monitor_vopd_training.py`
