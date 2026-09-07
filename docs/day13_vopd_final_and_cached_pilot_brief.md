# Day 13 Vision-OPD 定版与 Cached Prefix Pilot 工作简报

> 执行日期：2026-09-07（UTC）  
> 实验 ID：`E-D13-6K-VOPD-FINAL-001`、`E-D13-6K-CACHED-PILOT-001`  
> 最终状态：**PASS**  
> Vision-OPD 模型定版：**PASS**  
> 严格 prefix-source 消融合同：**PASS**  
> Cached 正式训练技术资格：`eligible_for_formal_promotion=true`  
> Day 14 正式配置：**已晋级**，`formal_training_authorized=true`；训练仍为 `training_started=false`

## 技术摘要

Day 13 完成了两部分工作。第一部分将 Day 12 的 `global_step_780` 恢复 checkpoint 做逐文件 SHA256、FSDP→Hugging Face 合并和新进程冷加载，最终冻结出可独立加载的 Vision-OPD Student 模型。第二部分实现并审计 `prefix_source: online|cached`，证明 6,241 条 Base Cached Prefix 与训练样本、prompt、Student 原图、tokenizer 和生成协议逐条绑定；随后运行 64 条、8-step 双卡 Cached Pilot，验证 cached 分支确实绕过 online generation，同时保持 Student、Crop Teacher、Top-K JSD、EMA 和 checkpoint 合同。

Vision-OPD 最终 merged 模型包含 724 个 BF16 张量，固定 5 条冷加载推理全部非空且 0 inference error。Cached Pilot 完成 8/8 steps、解析 64/64 条缓存记录，所有 step 的 online generation calls 为 0；最终 checkpoint 和修复后的冷加载 5/5 通过。因此 `STRICT_PREFIX_SOURCE_ABLATION=PASS`，Cached-6241 已具备进入 Day 14 正式训练的技术资格。

Day 13 没有启动 780-step Cached 正式训练，也没有运行外部 Benchmark。Day 13 资格凭据生成时保持 `formal_training_authorized=false`；后续启动准备已通过独立 promotion receipt 将正式配置晋级为 `formal_training_authorized=true`，但 `training_started=false`，因此仍未发生 Day 14 GPU 训练。

## 一、实际完成的工作

| 顺序 | 实际工作 | 主要结果 | 状态 |
|---:|---|---|---|
| 1 | 审计 Day 12 最终 checkpoint | 13 个文件、57,034,962,830 bytes；逐文件 SHA256 与独立复核通过 | PASS |
| 2 | 执行 FSDP→Hugging Face 合并 | 生成 7 个文件、10,370,019,516 bytes 的 merged Student；724 个张量均为 BF16 | PASS |
| 3 | 新进程冷加载 Vision-OPD Student | 固定 5 条样本全部生成非空回答，0 inference error；源 checkpoint 与 merged 模型未变化 | PASS |
| 4 | 冻结 Vision-OPD 交付身份 | 保存 checkpoint manifest、merged manifest、SHA256 清单、merge receipt 和 final freeze | PASS |
| 5 | 实现 `online|cached` 分支 | online 保持默认；cached 按 sample ID 读取 Base response 并跳过在线 rollout 服务初始化 | PASS |
| 6 | 审计 6,241 条 Cached Prefix 合同 | 6,241/6,241、ID 唯一、prompt/原图绑定、response token 往返一致、0 inference error | STATIC_CONTRACT_PASS |
| 7 | 对比正式与 Pilot 配置 | Base、Processor、模板、图像、sampling、长度、batch、steps、loss、Teacher 和 EMA 等非前缀变量一致 | PASS |
| 8 | 运行 64 条、8-step 双卡 Cached Pilot | 8/8 steps；64/64 cache records resolved；online generation calls=0 | PASS |
| 9 | 保存并冷加载 Cached Pilot checkpoint | `global_step_8` 完整；修复启动路径后 5/5 非空、0 inference error | PASS |
| 10 | 生成 postflight、完成与晋级凭据 | `STRICT_PREFIX_SOURCE_ABLATION=PASS`；正式训练技术资格通过 | PASS_TO_DAY14 |
| 11 | 核对存储余量 | 最终数据盘约 235 GiB 可用，高于 ≥130 GiB 建议值；保留唯一恢复用 FSDP checkpoint | PASS，无需清理 |

## 二、Vision-OPD 最终模型定版

### 1. 恢复 checkpoint 冻结

| 项目 | 实际结果 |
|---|---:|
| 来源 | `E-D12-6K-VOPD-001/checkpoints/global_step_780` |
| global step | 780 |
| 文件数 | 13 |
| 总大小 | 57,034,962,830 bytes，约 53.12 GiB |
| marker | `latest_checkpointed_iteration.txt=780` |
| 非空文件检查 | PASS |
| SHA256 独立复核 | 13/13 PASS |

### 2. 模型合并与交付

最终模型目录：

```text
artifacts/runs/E-D13-6K-VOPD-FINAL-001/merged_hf/
```

| 项目 | 实际结果 |
|---|---:|
| merged 文件数 | 7 |
| merged 总大小 | 10,370,019,516 bytes，约 9.66 GiB |
| `model.safetensors` 大小 | 10,350,019,328 bytes |
| Tensor 数量 / dtype | 724 / BF16 |
| `model.safetensors` SHA256 | `265a3c49fbee7a44d65cf8af732c4a9b71de5ad9ef799c935c4e38d07094862f` |
| merged SHA256 独立复核 | 7/7 PASS |
| 冷加载推理 | 5/5 非空、0 error、全部正常 stop |

第一次合并时，运行环境的 cgroup 内存上限只有 2 GiB，merger 以退出码 137 结束，未生成或修改最终模型。将可用内存恢复至 120 GiB 后，在独立临时目录重新合并并完成校验，再原子晋级为 `merged_hf`。失败日志与成功凭据均保留。

本次冷加载只验证模型可以在独立新进程中加载并完成多模态生成，不验证训练恢复，也不等同于能力评测。

## 三、Cached Prefix 静态合同与实现

Cached 分支从同一个冻结原始 Base 启动，唯一目标变量是 Student prefix 的来源：

```text
Vision-OPD: prefix_source=online
Cached:     prefix_source=cached
```

实现保持 online 为默认路径；cached manager 根据 `sample_id` 读取离线 response，跳过在线 rollout server 初始化，并与 online 路径共享后续的 response mask、Student full-image forward、Crop Teacher、Top-K JSD、backward 和 EMA 更新。

### 1. 6,241 条缓存审计

| 检查项 | 实际结果 |
|---|---:|
| 缓存记录 / 唯一 sample ID | 6,241 / 6,241 |
| train-cache ID 集合 | 完全一致 |
| inference errors | 0 |
| response token 范围 | 2～1,024 |
| 正常 stop / 达长度上限 | 6,193 / 48 |
| tokenizer decode→encode | 6,241/6,241 精确往返 |
| prompt 与 Student 原图绑定 | 6,241/6,241 |
| Cache SHA256 | `3a68ff1f8c7e63082f188dcc1bc508d8c0f9d7eafbcee748f00fd3769e6193ef` |

正式 online/cached 配置与对应 Pilot 配置均完成数据、Actor、rollout、self-distillation、资源、训练、模型路径和模板合同对比。静态结论为 `STATIC_CONTRACT_PASS`；严格消融名称随后由真实双卡 Pilot 关闭。

## 四、Cached 双卡 Pilot 实测

Pilot 使用固定 64 条长尾样本、global batch 8，共 8 个 optimizer steps。

| Step | VOPD loss | LR | Student delta | Teacher optimizer delta | Teacher direct gradients | EMA applied | Cache resolved / online calls |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.012149 | 0 | 0 | 0 | 0 | 1 | 8 / 0 |
| 2 | 0.009591 | `2e-7` | `2.38e-7` | 0 | 0 | 1 | 8 / 0 |
| 3 | 0.006511 | `4e-7` | `4.77e-7` | 0 | 0 | 1 | 8 / 0 |
| 4 | 0.015029 | `6e-7` | `6.00e-7` | 0 | 0 | 1 | 8 / 0 |
| 5 | 0.017127 | `8e-7` | `8.02e-7` | 0 | 0 | 1 | 8 / 0 |
| 6 | 0.016591 | `1e-6` | `9.57e-7` | 0 | 0 | 1 | 8 / 0 |
| 7 | 0.015345 | `1.2e-6` | `1.16e-6` | 0 | 0 | 1 | 8 / 0 |
| 8 | 0.008619 | `1.4e-6` | `1.25e-6` | 0 | 0 | 1 | 8 / 0 |

Pilot 结果：

- 8/8 steps 连续且指标有限，64/64 缓存记录解析成功。
- 每个 step 的 online generation calls 都为 0。
- 正学习率 step 的 Student 参数 delta 均为正。
- Teacher optimizer delta 和直接梯度始终为 0；EMA 每步生效。
- prompt truncation、generation error 和 aborted ratio 均为 0。
- 最长 prompt 为 7,880 tokens，未超过 8,192；最长 response 为 998 tokens，未超过 1,024。
- `global_step_8` checkpoint marker、必需文件和非空检查通过。

### 资源遥测

| 指标 | 实测结果 | 说明 |
|---|---:|---|
| GPU 0 峰值显存 | 97.83% | 接近 98% 保护线 |
| GPU 1 峰值显存 | 99.25% | 瞬时峰值；未满足连续触发条件，无 OOM |
| cgroup 峰值内存 | 193,652,260,864 / 257,698,037,760 bytes，约 75.15% | 低于中止线 |
| Pilot 遥测采样 | GPU/cgroup/磁盘各 97 行 | 两张 GPU 均有完整记录 |
| 训练与修复后冷加载全流程 | 1,020.25 秒，约 17.00 分钟 | 包含冷加载路径修复间隔 |
| 估算费用 | 3.97 元 | 双卡 14 元/小时；不是平台账单 |

GPU 1 出现 99.25% 的单点显存峰值，但 guard、训练、checkpoint 和 postflight 均通过，且没有 CUDA OOM。该峰值应在 Day 14 前 3 steps 和完整正式运行中继续由同一守护策略监控，不应被解释为长期稳定占用只有 99.25%。

## 五、Cached checkpoint 冷加载问题与修复

Pilot 训练和 FSDP 合并首次均已成功。第一次冷加载失败发生在启动 vLLM 服务时：直接调用虚拟环境 Python 没有把对应 `bin` 目录加入 `PATH`，`shutil.which('vllm')` 返回空，进而触发 `FileNotFoundError`。这不是模型权重、checkpoint 或推理错误。

修复内容：

- 优先使用 `Path(sys.executable).with_name('vllm')` 定位当前 Python 环境中的 vLLM。
- 找不到时才回退到 `shutil.which('vllm')`。
- 在 reload receipt 中记录实现脚本路径、SHA256 和实际 vLLM 可执行文件。
- 复用第一次已验证的 merged model，不重跑训练、不覆盖第一次失败凭据。

重试结果：固定 5 条样本全部非空，inference error=0，受控关闭服务的退出码为 0；源 checkpoint 在冷加载前后的 manifest SHA256 完全相同。相关回归为 `11 passed, 3 subtests passed`。

## 六、正式训练外推与晋级结论

依据 8-step Pilot 的稳态 step、启动和 checkpoint 时间，780-step Cached 正式训练的当前外推为：

| 场景 | 预计墙钟 | 预计费用 |
|---|---:|---:|
| 中位稳态 | 6.83 小时 | 95.58 元 |
| 平均稳态 | 7.61 小时 | 106.57 元 |
| 保守最大值 | 9.35 小时 | 130.89 元 |

费用按双卡合计 14 元/小时估算，不要求每次人工填写累计费用或平台账单时间。该外推状态为 `MEASURED_PROJECTION_NOT_YET_FROZEN`，正式启动器仍应重新读取实时 GPU、cgroup、磁盘、端口与输出目录。

最终晋级结论：

```text
day13_eligibility_status=PASS
eligible_for_formal_promotion=true
STRICT_PREFIX_SOURCE_ABLATION=PASS
day14_config_status=ready_after_day13_gate
formal_training_authorized=true
training_started=false
```

`eligible_for_formal_promotion=true` 表示 Day 13 技术 Gate 已关闭。后续 promotion receipt 已授权 Day 14 正式配置；授权是启动资格，`training_started=false` 仍明确表示 780-step 正式训练尚未启动。运行方式见 `docs/day14_cached_formal_training_runbook.md`。

## 七、阶段结论与边界

Day 13 的计划任务全部完成：Vision-OPD checkpoint 哈希、模型合并、5 条冷加载和冻结交付完成；Cached 分支实现、6,241 条静态合同、严格前缀来源消融、64 条真实训练、checkpoint、冷加载和正式训练外推全部通过。

最终可用空间约 235 GiB，高于计划建议的 130 GiB，因此没有为了“完成清理任务”而删除唯一恢复用 FSDP checkpoint。后续若空间下降，应先确认 merged 模型、哈希和恢复需求，再按白名单清理。

本阶段没有运行 Cached 780-step 正式训练，也没有打开 ZoomBench、MMStar 或 V* 外部结果。Cached 模型能力与 Vision-OPD 的统一比较属于 Day 14～15，不能从 Pilot loss 或 5 条可加载 smoke 推导能力提升。

## 八、关键证据索引

Vision-OPD 定版：

- 最终冻结：`artifacts/runs/E-D13-6K-VOPD-FINAL-001/evidence/final_student_freeze.json`
- checkpoint manifest：`artifacts/runs/E-D13-6K-VOPD-FINAL-001/evidence/checkpoint_manifest.json`
- checkpoint SHA256：`artifacts/runs/E-D13-6K-VOPD-FINAL-001/checkpoint_sha256.txt`
- 合并凭据：`artifacts/runs/E-D13-6K-VOPD-FINAL-001/evidence/merge_receipt.json`
- merged SHA256：`artifacts/runs/E-D13-6K-VOPD-FINAL-001/merged_sha256.txt`
- 冷加载凭据：`artifacts/runs/E-D13-6K-VOPD-FINAL-001/cold_reload/reload_validation_summary.json`
- 最终 merged 模型：`artifacts/runs/E-D13-6K-VOPD-FINAL-001/merged_hf/`

Cached Prefix：

- 静态合同：`artifacts/runs/E-D13-6K-CACHED-PILOT-001/preflight/contract_audit.json`
- Pilot postflight：`artifacts/runs/E-D13-6K-CACHED-PILOT-001/evidence/postflight.json`
- Pilot 完成凭据：`artifacts/runs/E-D13-6K-CACHED-PILOT-001/evidence/completion_receipt.json`
- 晋级凭据：`artifacts/runs/E-D13-6K-CACHED-PILOT-001/evidence/formal_eligibility_receipt.json`
- 正式配置 promotion：`artifacts/runs/E-D13-6K-CACHED-PILOT-001/formal_promotion_v1/promotion_receipt.json`
- Day 14 运行手册：`docs/day14_cached_formal_training_runbook.md`
- Pilot checkpoint：`artifacts/runs/E-D13-6K-CACHED-PILOT-001/checkpoints/global_step_8/`
- 修复后冷加载：`artifacts/runs/E-D13-6K-CACHED-PILOT-001/cold_reload_attempt2/reload_validation_summary.json`
- 人读报告：`artifacts/reports/cached_prefix_contract.md`、`artifacts/reports/cached_6241_pilot.md`
