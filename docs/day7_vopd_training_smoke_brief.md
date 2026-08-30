# Day 7 Vision-OPD 双卡训练入口与真实 Smoke 工作简报

> 执行日期：2026-08-30（UTC）  
> 实验 ID：`E-D7-001`  
> 验收状态：**PASS_WITH_CAVEAT**

## 技术摘要

Day 7 核心任务已完成。E-D7-001 在两张 RTX PRO 6000 上完成 2 个真实 optimizer steps 和 16 条在线 Student rollout；两步 VOPD loss 与 Student gradient 均为有限正值。运行时参数探针进一步证明：Student 在 optimizer 后发生变化，Teacher 在 optimizer 阶段保持不变且没有梯度，Teacher 仅在 optimizer 完成后的 EMA 阶段发生变化。

全量 1024 条实际 Processor Token 审计无处理错误、无超长样本；Smoke 两步的 prompt clip ratio 均为 0。由此可以确认在线 Student prefix、完整图 Student、裁剪图 Teacher、Top-K JSD、Student-only backward 和 EMA Teacher 的真实训练链路可执行。

本次仅有两个 step，因此结论限定为“链路 Smoke 通过”，不能据此判断长期稳定性、loss 收敛、训练后精度或 1024 条正式训练成本。训练结束时出现一个 DataLoader worker 清理警告，但它发生在训练达到 2/2 且 checkpoint 保存完成之后，不影响本次 Gate。

## 一、任务完成情况

| 工作项 | 完成证据 | 状态 |
|---|---|---|
| 双卡训练入口与冻结配置 | `configs/vopd_1024.yaml`、`scripts/run_vopd_2gpu.sh` | PASS |
| 真实在线训练链 | 2 steps、16 条 rollout；Student full image、Teacher bbox crop | PASS |
| Top-K JSD 与 Student gradient | 两步 `vopd_loss` 与 `grad_norm` 均有限且大于 0 | PASS |
| Student-only backward | Teacher 非空梯度数始终为 0 | PASS |
| Teacher optimizer 隔离 | Teacher optimizer delta 始终为 0 | PASS |
| optimizer 后 EMA | 两步 `ema_update_applied=1` 且 Teacher EMA delta 大于 0 | PASS |
| Smoke Prompt Gate | 两步 prompt clip ratio 均为 0 | PASS |
| Train-1024 Token 审计 | Student max 7880、Teacher max 2213；超长与错误均为 0 | PASS |
| Checkpoint | `global_step_2` 曾完整保存并核对分片；未重载；验证后清理 | PASS_WITH_CAVEAT |

## 二、冻结配置

| 配置项 | 冻结值 | 作用 |
|---|---|---|
| Base model | Qwen3.5-4B | Vision-OPD 实验统一起点 |
| GPU | 2 × RTX PRO 6000 | 双卡 FSDP Smoke |
| Global batch | 8 | 每个 optimizer step 使用 8 条样本 |
| Rollout n | 1 | 每条 prompt 生成一条在线轨迹 |
| Optimizer steps | 2 | Day 7 最低真实训练 Gate |
| Prompt / response limit | 8192 / 256 | 覆盖多模态长输入并控制 Smoke 输出 |
| Student image | `images` | 完整图视图 |
| Teacher image | `bbox_images` | 裁剪图视图 |
| Distillation top-k | 100 | Top-K JSD |
| Alpha | 0.5 | 蒸馏 loss 权重 |
| EMA update rate | 0.05 | optimizer 后更新 Teacher |
| Actor parameter offload | false | 参数留在 GPU |
| Optimizer offload | false | optimizer 状态留在 GPU |
| Reference parameter offload | false | Reference 参数留在 GPU |
| vLLM GPU utilization | 0.45 | 为训练与 rollout 留出显存余量 |

Day 7 是链路 Smoke，不是效果实验。Student 使用完整图并在线生成 response；Teacher 使用 bbox crop，但沿用同一 Student prefix。训练信号为 Top-K JSD，只对 Student 反向传播，optimizer 完成后再执行 EMA。

## 三、真实训练结果

### 1. Student、Teacher 与 EMA 边界

| 指标 | Step 1 | Step 2 | 验收解释 |
|---|---:|---:|---|
| `actor/vopd_loss` | 0.04209464 | 0.02820932 | 有限且大于 0 |
| `actor/grad_norm` | 5.213622 | 6.766190 | Student gradient 有效 |
| Probe elements | 512 | 512 | 轻量探针已启用 |
| Student optimizer 后最大参数变化 | 2.0266e-6 | 2.0018e-6 | Student 被 optimizer 更新 |
| Teacher optimizer 后最大参数变化 | 0 | 0 | optimizer 未直接更新 Teacher |
| Teacher 非空梯度数 | 0 | 0 | Teacher 不参与 backward |
| Teacher EMA 后最大参数变化 | 2.3842e-7 | 4.7684e-7 | EMA 改变 Teacher |
| `ema_update_applied` | 1 | 1 | 两步均执行 EMA |
| Prompt clip ratio | 0 | 0 | 无静默截断 |

运行时探针最多覆盖 32 个参数、每个参数 16 个元素，并兼容 FSDP/DTensor。它用于验证更新边界，不等同于完整模型权重差分。Student optimizer delta 为正、Teacher optimizer delta 为 0、Teacher gradient 为 0、Teacher EMA delta 为正，这四项组合证据直接验证了 Student-only backward 与 optimizer-after-EMA 顺序。

### 2. 在线生成与长度

| 指标 | Step 1 | Step 2 |
|---|---:|---:|
| Rollout 数 | 8 | 8 |
| Response mean | 95.75 | 54.75 |
| Response max | 256 | 125 |
| Prompt mean | 3375.625 | 3400.375 |
| Prompt max | 4169 | 3509 |
| Prompt clip ratio | 0 | 0 |
| Aborted ratio | 0 | 0 |

Step 1 有 1/8 response 达到 256 上限，对应 response clip ratio 0.125；该现象不影响 prompt 无截断 Gate。两步均无 aborted response。

### 3. 时间拆分

| 阶段 | Step 1 | Step 2 |
|---|---:|---:|
| Generation | 27.725 秒 | 9.518 秒 |
| Student forward | 41.403 秒 | 1.601 秒 |
| Teacher forward | 23.791 秒 | 0.843 秒 |
| Backward | 39.142 秒 | 3.867 秒 |
| Optimizer step | 0.165 秒 | 0.027 秒 |
| Teacher EMA | 6.231 秒 | 6.318 秒 |
| 主 step 墙钟 | 140.372 秒 | 24.158 秒 |
| 最终 checkpoint | 0 | 104.194 秒 |

Step 1 主要承担首次编译、CUDA Graph 和预热成本，不能作为稳态吞吐。Step 2 的主训练墙钟明显下降，但只有一个稳态候选点，仍不足以直接外推 1024 条训练；Day 8 至少 8 步的连续记录才是正式成本估算基础。

## 四、Train-1024 Prompt Token 审计

审计使用实际 Qwen3VLProcessor、冻结 Chat Template 和真实图像展开，不传 truncation 或 max-length 参数。分位数采用 nearest-rank 方法。

| 视图 | P50 | P95 | P99 | Max | 超长数 |
|---|---:|---:|---:|---:|---:|
| Student full image | 3366 | 3974 | 4638 | 7880 | 0 |
| Teacher bbox crop | 287 | 1527 | 1881 | 2213 | 0 |

全部 1024 行、两种视图均处理成功，错误数为 0。8192 Prompt 上限覆盖现有训练数据；Student 长尾主要来自完整图展开后的 image tokens，而非文本 tokens。

## 五、主要工程实现

### 1. 双卡入口与 Preflight

训练入口在启动 GPU 任务前检查：

- Base 模型目录与必要权重文件；
- train-1024 Parquet 行数和 schema；
- `images` 与 `bbox_images` 路径完整性；
- 两卡、batch、rollout、长度、offload、EMA 等冻结值；
- Chat Template 与配置哈希；
- Prompt 超长和静默截断 Gate。

入口通过 Hydra 覆盖官方 8 卡、大 batch 和长 response 默认值，并将成功日志、rollout、log-prob、checkpoint 和 evidence 分目录保存。

### 2. 运行时证据探针

新增低开销参数探针，在每个 optimizer step 的关键边界采样：

1. backward 后统计 Teacher 非空梯度；
2. optimizer 前保存 Student/Teacher 小规模参数样本；
3. optimizer 后比较 Student/Teacher 参数变化；
4. EMA 前保存 Teacher 样本；
5. EMA 后比较 Teacher 参数变化并记录是否执行。

证据进入 verl 现有 metrics 管道，因此同时保存在控制台日志与 TensorBoard。

## 六、遇到的问题与解决方法

| 问题 | 根因 | 修复与验证 |
|---|---|---|
| 初始 Token 审计进程被 `Killed` | 当时低 CPU cgroup 无法承载完整 Processor 导入与数据处理 | 先执行单条 Smoke，后在可用资源环境完成 1024 条审计；错误和超长均为 0 |
| Hydra 找不到 `data/legacy_data` | 官方配置组被仓库通用 `data/` ignore 规则漏掉 | 恢复官方语义配置并加入 `.gitignore` 例外 |
| `OMP_NUM_THREADS` 非法 | 容器环境变量为 0，libgomp 只接受正整数 | 启动脚本自动将非法值归一为 1 |
| 首次训练 CUDA `indexSelect` 越界 | 超大 FSDP flat parameter 上整数 `torch.linspace` 把最后索引舍入为 length | 改用 int64 整数除法；3 个 CPU 单测、40 亿长度 GPU 边界和真实双卡重跑通过 |
| 结束时 DataLoader worker 被 `Killed` | 出现在 2/2 与 checkpoint 保存后；现有证据不足以归因于 OOM | 保留为非阻断 caveat；Day 8 使用 cgroup/RSS 指标监控 |

失败轮次没有混入最终两步指标。相关修复均已进入代码或启动脚本，并通过针对性回归。

## 七、结论边界与资源解释

### 可以确认

- 在线 Student prefix、Student full image 和 Teacher crop image 链路正确；
- Top-K JSD 有效，Student gradient 有限；
- Teacher 没有梯度，也没有被 optimizer 直接更新；
- EMA 在 optimizer 后执行并改变 Teacher；
- 两步无 NaN/Inf，Prompt 无截断；
- train-1024 的 8192 Prompt 上限 Gate 通过。

### 不能确认

- 两步结果不能证明 loss 收敛或长期训练稳定；
- 尚未形成可信的 64/1024 条吞吐与费用外推；
- Day 7 checkpoint 保存后未执行重新加载与 5 条推理；
- 尚未衡量训练后的 internal eval/retention 或外部 Benchmark 表现。

日志中的 `perf/cpu_memory_used_gb≈193.9` 来自 `psutil.virtual_memory().used`，是宿主机整体已用内存的瞬时值，不是训练进程 RSS，也不是容器峰值。检查时 cgroup 记录为 `oom=0`、`oom_kill=0`，因此不能用 193.9 GiB 推断训练自身发生 CPU OOM。

## 八、证据保留与存储清理

Day 7 完整 FSDP checkpoint 约 54 GiB：FP32 Student model 分片约 20 GB，AdamW optimizer 分片约 34 GB，其余为 extra-state 与 HuggingFace metadata。

后续 E-D8-001 和 E-D10-001 均冻结为从 Base 启动，不依赖 Day 7 Smoke checkpoint。因此在记录分片清单后，永久删除 checkpoint、失败日志、Hydra 临时日志和失败 TensorBoard event。

| 产物 | 处置 | 理由 |
|---|---|---|
| 成功 `train.log` | 保留 | 两步最终指标原始来源 |
| 两步 rollout | 保留 | 16 条在线生成证据 |
| Token 审计 | 保留 | 8192 Prompt Gate 来源 |
| Runtime evidence JSON/Markdown | 保留 | Student/Teacher/EMA 机器可读证据 |
| 成功 TensorBoard event | 保留 | 指标可视化入口 |
| `global_step_2` checkpoint | 删除 | 后续从 Base 启动，不用于续训 |
| 失败日志与失败 TensorBoard event | 删除 | 修复已进入代码与测试，不参与最终指标 |
| `checkpoint_prune_manifest.json` | 保留 | 记录删除前分片、大小和不可恢复性 |

清理后 E-D7-001 从约 54 GB 降至约 1.5 MB，工作盘可用空间从约 25 GB 恢复到约 78 GB。删除后的 Day 7 Student 只能通过重跑 E-D7-001 恢复。

## 九、最终证据入口

- 冻结配置：`configs/vopd_1024.yaml`
- 双卡入口：`scripts/run_vopd_2gpu.sh`
- 成功日志：`artifacts/runs/E-D7-001/logs/train.log`
- Rollout：`artifacts/runs/E-D7-001/rollouts/1.jsonl`、`2.jsonl`
- Token 审计：`artifacts/runs/E-D7-001/preflight/prompt_length_summary.json`
- 运行时证据：`artifacts/runs/E-D7-001/evidence/runtime_evidence_summary.json`
- 运行时报告：`artifacts/runs/E-D7-001/evidence/runtime_evidence_report.md`
- Checkpoint 清理清单：`artifacts/runs/E-D7-001/evidence/checkpoint_prune_manifest.json`
- 成功 TensorBoard event：`tensorboard_log/Vision-OPD/E-D7-001/`

## 十、Day 8 交接建议

1. 从同一 Base checkpoint 启动固定 64 条数据，完成至少 8 个连续 optimizer steps。
2. 冻结 64 条样本 ID、数据顺序、seed 和配置哈希，确保后续 Cached Prefix 使用完全相同的对照条件。
3. 使用 cgroup memory 或进程 RSS 记录真实 CPU 内存，同时保留逐步显存、生成耗时占比和 loss。
4. 最多保留一个完整 checkpoint，避免 100 GB 工作盘被多个 54 GB 训练状态占满。
5. 训练结束后关闭进程，重新加载 checkpoint 并推理 5 条，补上 Day 7 未覆盖的可恢复性 Gate。
6. 使用 Day 8 的稳态 step 估算 1024 条训练时长与费用，不使用 Day 7 首步预热数据直接外推。

进入 Day 8 前还需要明确：固定 64 条的可重复抽取规则、checkpoint 只保存最终 step 还是滚动保留 1 份，以及未来是否需要把 EMA Teacher 状态纳入精确断点续训契约。
