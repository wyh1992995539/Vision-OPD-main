# Day 8 Vision-OPD 64 条稳定性训练、冷重载与收尾工作简报

> 执行日期：2026-08-31（UTC）  
> 实验 ID：`E-D8-001`  
> 验收状态：**PASS_WITH_CAVEAT**

## 技术摘要

Day 8 已完成从“固定 64 条数据”到“checkpoint 冷重载”和“稳定性报告归档”的完整闭环。训练使用独立的 Day 8 配置，从原始 Qwen3.5-4B Base checkpoint 启动，按固定 seed 和稳定哈希规则选取 64 条样本，关闭 shuffle，以 global batch 8 连续完成 8/8 optimizer steps。

8 步的 VOPD loss、Student grad、Student 参数变化、Teacher optimizer 隔离和 EMA 更新均为有效有限值；prompt clip ratio 与 aborted ratio 全程为 0。最终 `global_step_8` checkpoint 完整保存，随后在训练进程结束后合并到独立目录，冷启动 vLLM，并对冻结的 5 条 eval 样本完成推理：5/5 输出非空、推理错误为 0、源 checkpoint 在合并前后未变化。

Day 8 最终判定为 `PASS_WITH_CAVEAT`。主要 caveat 是 checkpoint 保存后出现一次 DataLoader worker `Killed`，但缺少同时采集的训练期 cgroup/RSS 证据，无法确认根因；此外训练器记录的显存值与冻结的单卡 96 GB 物理口径不一致，不能作为可信的逐卡峰值。这两项不否定 checkpoint 的冷重载结果，但必须转入 Day 9 的正式训练观测性 Gate。

## 一、任务完成情况

| 工作项 | 完成证据 | 状态 |
|---|---|---|
| 独立 Day 8 配置 | `configs/vopd_day8_64.yaml` | PASS |
| 固定 64 条训练数据 | `train_day8_64.parquet`、固定选择清单与 SHA256 | PASS |
| 稳定性训练入口泛化 | `scripts/run_vopd_2gpu.sh` 支持显式 config、preflight 与运行清单 | PASS |
| 无 GPU Preflight | 64 rows、schema、图像、配置契约、样本顺序和哈希全部通过 | PASS |
| 连续训练 | 64 条、8/8 optimizer steps、1 epoch | PASS |
| Student/Teacher/EMA 契约 | Student 更新；Teacher optimizer delta=0、grad=0；EMA 8/8 | PASS |
| 数值与长度 Gate | 无 NaN/Inf；prompt clip=0；abort=0 | PASS |
| 最终 checkpoint | `global_step_8`，13 个必需文件均非空并有 SHA256 | PASS |
| 冷重载验证 | 源 checkpoint 不变；5/5 非空；0 inference errors | PASS |
| 1024 条时间与费用外推 | 均值约 1.02 双卡小时/¥12.25，保守约 1.74 小时/¥20.84 | PASS |
| Day 8 报告与机器摘要 | `vopd_64_stability.md`、`stability_summary.json` | PASS |
| 结束阶段进程清理 | checkpoint 后有一次 DataLoader worker `Killed`，根因未完全确认 | PASS_WITH_CAVEAT |

## 二、今天完成的主要工作

### 1. 新增独立 Day 8 配置和固定 64 条数据

新增 `configs/vopd_day8_64.yaml`，将 E-D8-001 与 Day 7 Smoke 解耦。Day 8 从同一个 Base checkpoint 独立启动，不继承 Day 7 checkpoint；冻结 global batch、rollout、Top-K JSD、EMA、长度上限、图像字段和保存策略。

新增 `scripts/prepare_vopd_stability_subset.py`，从冻结的 train-1024 中按以下规则选择 64 条：

```text
selection_key = sha256("42|<sample_id>")
按 selection_key 升序选择前 64 条，并保持该顺序
```

输出数据和选择清单默认拒绝覆盖，防止同一实验 ID 下的输入被静默重建。固定子集 SHA256 为：

```text
6e6502f352f2f6f0f17290aceba9aadef22e1ca3eb14487e4abb194a2d620c62
```

### 2. 将 Day 7 专用入口泛化为可审计训练入口

`scripts/run_vopd_2gpu.sh` 不再把 Day 7 的实验 ID、步数和保存参数写死在脚本中，而是从显式 YAML 配置读取，并支持：

- `--config <yaml>`：选择实验配置；
- `--preflight-only`：只校验、不占用 GPU；
- `--run`：preflight 通过后启动真实训练；
- 原样记录额外 Hydra overrides；
- 在运行前保存 config/data SHA256、样本顺序、Git commit、工作树状态、CUDA 设备和训练契约。

新增 `scripts/vopd_training_preflight.py`，在启动 GPU 任务前检查 Base 文件、Parquet 行数和 schema、完整图/裁剪图路径、64 个唯一 sample ID、固定选择清单、seed、batch、步数、长度、offload、Teacher 图像字段和保存策略。

### 3. 完成 64 条、8 步真实稳定性训练

E-D8-001 使用两张 RTX PRO 6000，完成 64 条样本、8 个 optimizer steps。训练日志、rollout、TensorBoard 和最终 checkpoint 全部写入独立实验目录。

训练结果证明 Day 7 的链路 Smoke 可以持续到完整的 64 条小规模 epoch：Student 每步被 optimizer 更新，Teacher 不参与 backward，也不被 optimizer 直接更新；Teacher 只在每步 optimizer 完成后的 EMA 阶段变化。

### 4. 补齐 checkpoint 冷重载验证

新增：

- `configs/vopd_day8_reload.yaml`；
- `scripts/vopd_day8_reload.py`；
- `scripts/run_vopd_day8_reload.sh`；
- `tests/test_vopd_day8_reload.py`。

冷重载流程不在原 checkpoint 目录内合并，避免调用会原地改变或清理源文件的旧入口。脚本先记录源 checkpoint 的文件大小、mtime 和 SHA256，再合并到 `merged_hf/`，启动自己的 vLLM 进程，等待服务就绪，完成冻结 5 条样本推理，最后只关闭自己启动的进程组，并再次核对源 checkpoint 未变化。

### 5. 生成稳定性报告并完成 Day 8 收尾

新增 `scripts/finalize_day8_stability.py`，从原始日志和冷重载证据自动复算：

- 8 步 loss、grad、长度、耗时和 Teacher/EMA 探针；
- 非有限值、CUDA OOM 和 worker 异常信号；
- checkpoint 必需文件、大小与哈希；
- 冷重载验证状态；
- Step 2～8 稳态吞吐；
- 1024 条的中位、均值和保守三场景费用。

收尾器生成 `metrics.jsonl`、`cost.json`、`checkpoint_sha256.txt`、机器摘要和正式稳定性报告，并将 Day 8 状态同步到项目计划、项目配置和运行手册。

## 三、冻结配置

| 配置项 | Day 8 冻结值 | 说明 |
|---|---:|---|
| Base model | Qwen3.5-4B | 从 Base 独立启动 |
| Seed | 42 | 数据、训练和 rollout 统一种子 |
| 样本数 | 64 | 固定稳定哈希选择 |
| Global batch | 8 | 8 个 optimizer steps 覆盖 64 条 |
| Shuffle | false | 样本顺序可审计 |
| Epoch | 1 | 完成固定 64 条一轮 |
| Rollout n | 1 | 每条 prompt 一条在线 Student 轨迹 |
| Prompt / response limit | 8192 / 256 | 长多模态输入和受控生成 |
| Student image | `images` | 完整图 |
| Teacher image | `bbox_images` | 裁剪图 |
| Learning rate | 2e-6 | 与后续正式训练起始配置一致 |
| Distillation Top-K | 100 | Top-K JSD |
| JSD alpha / beta | 0.5 / 0.5 | 对称混合分布 |
| EMA update rate | 0.05 | optimizer 后更新 Teacher |
| GPU | 2 | 单机双卡 FSDP |
| Checkpoint | 只保存最终 step | 当前仅保留 `global_step_8` |

## 四、真实训练与冷重载结果

### 1. 8 步训练指标

| Step | VOPD loss | Grad norm | Step 秒 | Generation 秒 | 生成占比 | Prompt max | Response mean | Response 达上限 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01595 | 2.42 | 86.48 | 61.79 | 71.5% | 3503 | 128.5 | 25% |
| 2 | 0.03854 | 8.03 | 25.13 | 11.55 | 46.0% | 3741 | 84.0 | 25% |
| 3 | 0.08972 | 37.71 | 21.86 | 7.99 | 36.6% | 3976 | 4.1 | 0% |
| 4 | 0.05174 | 8.86 | 21.53 | 8.18 | 38.0% | 3977 | 5.0 | 0% |
| 5 | 0.08165 | 33.98 | 21.07 | 7.85 | 37.3% | 3554 | 3.9 | 0% |
| 6 | 0.06278 | 8.67 | 46.00 | 9.94 | 21.6% | 5664 | 38.8 | 0% |
| 7 | 0.04172 | 4.92 | 22.71 | 9.65 | 42.5% | 3371 | 43.2 | 0% |
| 8 | 0.05343 | 8.83 | 21.20 | 8.27 | 39.0% | 3600 | 3.4 | 0% |

8 步 loss 范围为 0.01595～0.08972，均值 0.05444；全部指标有限。Step 1 包含首次生成和内核预热，不能代表稳态。Step 6 的 46.00 秒主要来自 actor update 增长到 35.87 秒，而 generation 只有 9.94 秒，因此不是生成服务卡死。

全程 prompt clip ratio 为 0，最大 prompt 5664 tokens；aborted ratio 为 0。前两步各有 25% response 达到 256-token 上限，估算合计 4/64 条，占 6.25%，需要在 Day 9 继续监控，但不构成 Day 8 失败。

### 2. Student、Teacher 与 EMA 边界

| 检查项 | 8 步结果 | 判定 |
|---|---|---|
| Student optimizer 后参数变化 | 每步均大于 0 | Student 正常更新 |
| Teacher optimizer 后参数变化 | 每步均为 0 | optimizer 未直接更新 Teacher |
| Teacher 非空梯度数 | 每步均为 0 | Teacher 不参与 backward |
| Teacher EMA 后参数变化 | 每步均大于 0 | EMA 正常改变 Teacher |
| `ema_update_applied` | 8/8 为 1 | 每步均完成 EMA |
| Teacher always-on / image swap | 8/8 为 1 | 裁剪图 Teacher 全程启用 |

### 3. Checkpoint 与冷重载

| 项目 | 结果 |
|---|---|
| 最终 step | `global_step_8` |
| Checkpoint 大小 | 53.12 GiB |
| 必需文件 | 13/13 非空并有 SHA256 |
| 合并目录 | `artifacts/runs/E-D8-001/merged_hf/` |
| 合并清单 SHA256 | `fd3990c1dd2516a89c086e17a56dc983eacc17d5c2032987c70e36536e2fdc50` |
| 源 checkpoint 是否变化 | 否 |
| 冻结推理样本 | 5 |
| 非空输出 | 5/5 |
| 推理错误 | 0 |
| 受控关闭后服务退出码 | 0 |

5 条样本中有 3 条答对，但这只是链路 Smoke，样本量不足，不能作为训练效果或精度提升结论。

### 4. 1024 条耗时与费用外推

1024 条在 global batch 8 下对应 128 steps。外推包含启动/加载固定开销、一次首步预热、127 个稳态 step 和一次最终 checkpoint 保存。

| 场景 | 稳态 step | 预计双卡小时 | 预计费用 |
|---|---:|---:|---:|
| 中位稳态 | 21.86 秒 | 0.89 | ¥10.66 |
| 均值稳态（规划口径） | 25.64 秒 | 1.02 | ¥12.25 |
| 稳态最大值（保守上界） | 46.00 秒 | 1.74 | ¥20.84 |

Day 9 使用均值场景作为预算基线，使用保守场景作为 GPU 时长预留。由于稳态样本只有 7 个 step，且固定 64 条不一定覆盖 train-1024 的全部长度尾部，该结果是工程预算外推，不是运行时 SLA。

## 五、遇到的问题与解决方法

| 问题 | 判断或根因 | 解决方法 | 验证结果 |
|---|---|---|---|
| Day 7 入口写死实验参数，不能直接审计 Day 8 | 实验 ID、步数和保存策略与启动脚本耦合 | 将入口改为显式读取独立 YAML；增加 `--config`、`--preflight-only`、`--run` 和运行清单 | Day 7 专用常量测试通过；E-D8-001 无 override 完成 8/8 |
| 固定 64 条若手工截取，难以证明可重复 | Parquet 行顺序或临时抽样可能变化 | 用 `sha256("42|sample_id")` 排序选择，保存 64 个 ID、源行、selection key 和子集 SHA256；默认拒绝覆盖 | Preflight 确认 64 个唯一 ID、顺序和 SHA256 全部一致 |
| 仅看到 8/8 不能证明 checkpoint 可恢复 | FSDP 分片可能不完整，训练进程内加载也不等于冷启动 | 新增独立冷重载配置和脚本；关闭训练后合并、启动新 vLLM、推理固定 5 条，并核对源 checkpoint 前后快照 | 13 个必需文件完整；5/5 非空；0 错误；源 checkpoint 未变化 |
| 旧合并入口可能原地改变或删除源 checkpoint | 旧脚本面向空间回收，不适合作为审计 Gate | 合并到独立 `merged_hf/`，源目录只读使用；保存 checkpoint 和 merged manifest | `source_checkpoint_unchanged=true`，两套哈希证据齐全 |
| 重载时提示 merged 目录非空 | 先前合并已经完成，默认重复写入会混淆模型身份 | 先核对 `merged_manifest.json`，再显式使用 `--reuse-merged`；历史预测只有在显式 `--overwrite-results` 时才替换 | 最终冷重载状态 PASS，复用模型哈希固定 |
| checkpoint 保存后 DataLoader worker 被 `Killed` | 出现在 8/8、checkpoint 路径和完整分片之后；缺少同时采集的训练期 cgroup/RSS，无法进一步归因 | 不把异常静默忽略，也不把已保存训练误判为失败；通过 checkpoint 清单和训练后冷重载确认产物有效，保留 caveat | checkpoint 可加载且 5/5 推理通过；根因仍未关闭，转入 Day 9 |
| 训练器显存峰值超过冻结的 96 GB/卡口径 | logger 值不能可靠解释为逐卡物理峰值，现有日志缺少旁路采样 | 报告保留原值但禁止用于容量结论；Day 9 增加每卡 `nvidia-smi` 时间序列、进程 RSS 和 cgroup 采样 | Day 8 不宣称逐卡峰值；观测性缺口已显式登记 |
| CPU 内存指标容易被误读 | `perf/cpu_memory_used_gb` 来自宿主机 `psutil.virtual_memory().used`，不是训练进程 RSS | 在报告中限定指标含义，不用它证明训练进程 OOM；Day 9 单独采集进程和 cgroup | 避免把宿主机已用内存错误归因到训练进程 |
| 手工汇总 8 步指标和费用容易抄错 | 原始指标集中在长日志行，费用还需要区分预热、稳态和 checkpoint | 新增收尾器自动解析日志、校验必需指标并输出三场景外推 | 机器摘要判定 `advance_to_day9`；相关 13 项测试通过 |
| 基础 Conda 环境缺少 `pyarrow`，且模块式 unittest 被同名 `tests` 包遮蔽 | QA 命令没有使用项目环境；环境中存在包名冲突 | 切换 `vision-opd` 环境，并以 `PYTHONPATH=.` 直接运行各测试文件 | 3+3+4+3，共 13 项 Day 8 测试全部通过 |

DataLoader worker 异常和逐卡显存峰值仍属于未完全解决的问题。当前采取的是“用独立证据确认 checkpoint 有效，同时将根因和观测性缺口带入 Day 9”，不是把 caveat 改写成已解决。

## 六、关键执行命令

### 1. 生成固定 64 条数据

```bash
conda run --no-capture-output -n vision-opd \
  python scripts/prepare_vopd_stability_subset.py
```

已有冻结输出时脚本默认拒绝覆盖；除非明确重建实验输入，否则不使用 `--overwrite`。

### 2. 无 GPU Preflight

```bash
scripts/run_vopd_2gpu.sh \
  --config configs/vopd_day8_64.yaml \
  --preflight-only
```

### 3. 双卡训练

```bash
CUDA_VISIBLE_DEVICES=0,1 scripts/run_vopd_2gpu.sh \
  --config configs/vopd_day8_64.yaml \
  --run
```

### 4. Checkpoint 冷重载

最终成功轮次复用了已审计的 merged 模型：

```bash
OMP_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=0,1 \
scripts/run_vopd_day8_reload.sh \
  --run \
  --reuse-merged
```

### 5. 重新生成 Day 8 收尾报告

该命令只读现有证据，不需要 GPU：

```bash
python scripts/finalize_day8_stability.py
```

## 七、结论边界与资源解释

### 可以确认

- 固定 64 条数据和样本顺序可重复核验；
- Vision-OPD 连续完成 8 个真实 optimizer steps；
- 8 步无 NaN/Inf，prompt 无截断，response 无 abort；
- Student 被 optimizer 更新；Teacher 无梯度且不被 optimizer 直接更新；
- EMA 每步执行并改变 Teacher；
- `global_step_8` 文件完整，可合并并在新服务中加载；
- 冻结 5 条样本的推理链路可用；
- 已形成可供 Day 9 使用的 1024 条时间和费用区间。

### 不能确认

- 64 条稳定性运行不能证明 loss 收敛或模型效果提升；
- 5 条冷重载 Smoke 的 3/5 正确率不能代表 internal eval 或外部 benchmark；
- 现有证据不能确定 DataLoader worker 被 `Killed` 的根因；
- 训练器显存日志不能证明可信的逐卡峰值；
- 1024 条外推不保证覆盖全部长度尾部和平台调度抖动；
- 运行从 dirty worktree 启动，虽然已保存 commit、Git 状态以及 config/data 哈希，但复现体验仍弱于 clean commit。

Day 8 记录的训练证据窗口约 0.170 双卡小时、¥2.03；最终冷重载窗口约 0.041 双卡小时、¥0.49。二者不包含未被清单覆盖的失败尝试、空闲时段或云平台计费舍入，不能当作完整账单。

## 八、最终证据入口

- Day 8 配置：`configs/vopd_day8_64.yaml`
- 固定选择清单：`artifacts/runs/E-D8-001/preflight/day8_64_selection.json`
- 训练 Preflight：`artifacts/runs/E-D8-001/preflight/preflight_summary.json`
- 运行清单：`artifacts/runs/E-D8-001/preflight/run_invocation.json`
- 原始训练日志：`artifacts/runs/E-D8-001/logs/train.log`
- 8 步 Rollout：`artifacts/runs/E-D8-001/rollouts/`
- 最终 Checkpoint：`artifacts/runs/E-D8-001/checkpoints/global_step_8/`
- Checkpoint 哈希：`artifacts/runs/E-D8-001/checkpoint_sha256.txt`
- 冷重载配置：`configs/vopd_day8_reload.yaml`
- 冷重载机器结论：`artifacts/runs/E-D8-001/evidence/reload/reload_validation_summary.json`
- 冷重载预测：`artifacts/runs/E-D8-001/reload_5/predictions.jsonl`
- 结构化逐步指标：`artifacts/runs/E-D8-001/metrics.jsonl`
- 费用与 1024 外推：`artifacts/runs/E-D8-001/cost.json`
- 稳定性机器摘要：`artifacts/runs/E-D8-001/evidence/stability_summary.json`
- 正式稳定性报告：`artifacts/reports/vopd_64_stability.md`
- Day 8 运行手册：`docs/day8_vopd_stability_runbook.md`

## 九、Day 9 交接建议

1. 冻结正式 `configs/vopd_1024.yaml` 和 E-D10-001 输出目录，不从 Day 8 checkpoint 续训，仍从同一 Base checkpoint 启动。
2. 将 `dataloader_num_workers` 从 4 降为 0 或 1，先做无 GPU 配置检查；若需要吞吐，再以新的短 Smoke 证明增加 worker 不会破坏结束阶段清理。
3. 增加每卡 `nvidia-smi` 采样、训练主进程及 worker RSS、cgroup `memory.current/memory.events` 采样，并把原始时间序列保存到 E-D10-001 evidence。
4. 设定中止条件：NaN/Inf、Teacher gradient 非 0、Teacher optimizer delta 非 0、连续生成错误、checkpoint 保存失败、可信逐卡显存逼近上限或磁盘不足。
5. 以 1.02 双卡小时/¥12.25 作为预算基线，以约 1.75 小时/¥20.84 作为启动前资源预留；若平台要求更稳妥，可在 Day 9 明确增加调度缓冲。
6. 当前 Day 8 目录同时保留约 53.12 GiB FSDP checkpoint 和 merged 模型。任何空间清理都必须在哈希、冷重载和后续使用边界确认后执行，不自动删除唯一训练状态。
7. Day 9 只做正式训练 Gate 和配置冻结，不运行 external benchmark；全部 Gate 通过后才进入 Day 10 的 1024 条训练。

Day 8 到此关闭。当前不需要继续占用 GPU；Day 9 的配置、preflight 和监控脚本开发可以在无 GPU 环境完成。
