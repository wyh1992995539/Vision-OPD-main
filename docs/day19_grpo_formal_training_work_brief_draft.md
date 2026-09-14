# Day 19 GRPO 6K 正式训练工作简报（待定草案）

> 文档状态：**DRAFT / NOT EXECUTED**  
> 目标实验：`E-D19-GRPO-TRAIN-001`  
> 启动前提：Day18 32/64 Pilot 全部 PASS，Reward、数据、参数和资源策略已经冻结  
> 说明：以下为训练正常完成的目标情景；所有 `[待填]` 均为运行后才能取得的数值，模型与产物哈希须由真实凭据生成。

## 技术摘要

若正式训练正常完成，本日应从冻结的原始 Qwen3.5-4B Base 独立启动，在 6,241 条源 prompt 上执行 1 epoch、原生 drop-last，完成 780 个外层迭代、6,240 个有效 prompt 和 24,960 条有效 rollout。最终应得到可恢复的 FSDP checkpoint、完整 coverage receipt、训练与资源指标，以及没有 Reward 路由、分组或数值错误的运行证据。

正式训练不能从 Day18 Pilot、Vision-OPD 或 Cached checkpoint 继续；Pilot 只用于冻结工程参数。

## 一、实际完成的工作（待执行成功后确认）

| 顺序 | 具体操作 | 主要结果与证据 | 状态 |
|---:|---|---|---|
| 1 | 读取 Day18 promotion receipt，逐项核对 Base、GRPO Parquet、Reward、配置、launcher 和 abort policy 哈希 | 正式配置冻结与来源绑定 | 预期 PASS |
| 2 | 在训练开始前重新检查双卡空闲、240 GiB 级 cgroup、磁盘安全余量、端口、Ray 残留进程和输出目录 | live launch gate | 预期 PASS |
| 3 | 从原始 Base 冷启动正式训练，确认数据 shuffle seed=42、原生 drop-last、每题 4 条 rollout 和自定义 Reward 动态加载 | run invocation 和最初 batch 的逐条映射 | 预期 PASS |
| 4 | 守护前 3 steps，核对 mixed group、PG/KL 分离、advantage、grad norm、clip fraction、actor delta 和数值有限性 | first-three-step gate | 预期 PASS |
| 5 | 按固定间隔采集 GPU、进程树 RSS、cgroup、磁盘、心跳和 step 指标；出现硬中止条件时受控终止进程组 | telemetry、runtime metrics、abort events | 预期 PASS |
| 6 | 在 step 390 原子保存中段 checkpoint，检查 marker、必需文件、稳定性和剩余磁盘后继续训练 | checkpoint 390 receipt | 预期 PASS |
| 7 | 完成 step 780 并执行 post-exit final drain，验证 step 1～780 连续、无重复和无尾行遗漏 | 完整训练日志和 guard/exit receipt | 预期 PASS |
| 8 | 冻结最终 checkpoint，生成逐文件 SHA256，并在独立进程执行 `sha256sum -c` | checkpoint manifest 与哈希凭据 | 预期 PASS |
| 9 | 从 dataloader/rollout 记录生成 coverage receipt，核对 6,240 个有效 prompt、1 个 dropped ID 和 24,960 条有效轨迹 | coverage receipt | 预期 PASS |
| 10 | 汇总训练指标、资源峰值、墙钟和费用，只据运行合同判定训练完成 | Day19 completion receipt | 预期 PASS |

正式运行目录必须在启动前为空；preflight 产物可位于独立子目录，但 logs、rollouts 和 checkpoints 发生碰撞时必须拒绝启动。可恢复的基础设施中断使用同一冻结配置和 checkpoint；任何 Reward、数据、batch、长度或算法修改都创建新实验并从 Base 重新验证。

## 二、冻结训练合同

| 项目 | 冻结目标 | 正常完成情景 |
|---|---:|---:|
| 源 prompt | 6,241 | 6,241 |
| epoch | 1 | 1 |
| prompt batch | 8 | 8 |
| rollout n | 4 | 4 |
| 外层迭代 | 780 | 780 |
| 有效 prompt | 6,240 | 6,240 |
| dropped prompt | 1 | 1 |
| padding | 0 | 0 |
| 有效轨迹 | 24,960 | 24,960 |
| optimizer updates | 780 | 780 |
| 初始化模型 | 原始 Base | 原始 Base |
| Reward SHA256 | Day18 冻结值 | 与 Day18 freeze 一致（待哈希） |
| 数据 SHA256 | `9550fac714fe030e6604e38b2e78db9970268d22464808a36f2a885ff1229206` | 一致 |
| response length | Day18 冻结值 | `[待填]` |

## 三、训练执行结果

| 指标 | 预期 | 实际结果 |
|---|---|---|
| preflight | PASS | PASS（待凭据） |
| 前 3 步人工/机器检查 | PASS | PASS（待凭据） |
| 完成 step | 780/780 | 780/780 |
| NaN/Inf | 0 | 0 |
| OOM | 0 | 0 |
| Reward 路由错误 | 0 | 0 |
| group 对齐错误 | 0 | 0 |
| checkpoint 保存 | 中段和最终 | PASS（待凭据） |
| 最终 checkpoint 校验 | PASS | PASS（待凭据） |
| coverage receipt | PASS | PASS（待凭据） |

训练中应逐步记录 Reward 均值/方差、mixed-group fraction、全对/全错组率、parse-invalid、A–D 输出分布、截断比例、entropy、KL、clip fraction、grad norm、response length、吞吐、显存、CPU 内存、磁盘和 optimizer update 计数。

## 四、训练指标回填表

| 指标 | 初始窗口 | 中段 | 最终窗口 |
|---|---:|---:|---:|
| mean Reward | `[待填]` | `[待填]` | `[待填]` |
| mixed-group fraction | `[待填]` | `[待填]` | `[待填]` |
| all-zero groups | `[待填]` | `[待填]` | `[待填]` |
| all-one groups | `[待填]` | `[待填]` | `[待填]` |
| parse-invalid rate | `[待填]` | `[待填]` | `[待填]` |
| length rate | `[待填]` | `[待填]` | `[待填]` |
| KL | `[待填]` | `[待填]` | `[待填]` |
| entropy | `[待填]` | `[待填]` | `[待填]` |
| clip fraction | `[待填]` | `[待填]` | `[待填]` |
| grad norm | `[待填]` | `[待填]` | `[待填]` |

Reward 上升本身不等于泛化能力上升。正式训练只按冻结运行健康与覆盖 Gate 判定是否完成，不依据外部 Benchmark 重选 checkpoint。

## 五、checkpoint、覆盖和恢复

建议保留中段 `global_step_390` 与最终 `global_step_780`，实际路径以 Day18 冻结配置为准。每个 checkpoint 应保存 actor、optimizer、scheduler、global step、数据顺序/RNG 所需状态和文件哈希。

coverage receipt 必须给出：

- 6,241 个源 sample ID；
- 6,240 个实际访问 sample ID；
- 唯一 dropped sample ID；
- 重复和遗漏计数；
- 24,960 条有效训练轨迹；
- 实际 optimizer update 数；
- 恢复前后的重复尝试与有效轨迹分开计数。

## 六、资源与成本

| 项目 | 实际值 |
|---|---:|
| GPU 型号/数量 | 2 卡（型号待运行凭据） |
| 训练墙钟 | `[待填]` |
| GPU 小时 | `[待填]` |
| 稳态 step 中位数/P95 | `[待填]` |
| 峰值显存/GPU | `[待填]` |
| 峰值 CPU RSS/cgroup | `[待填]` |
| 峰值磁盘 | `[待填]` |
| completion tokens | `[待填]` |
| 估算费用 | `[待填]` |

## 七、可能问题与处置

| 可能问题 | 识别信号 | 具体处置 |
|---|---|---|
| 启动身份漂移 | 配置/Data/Reward/Base 哈希与 Day18 freeze 不同 | 拒绝启动；生成新 freeze，不手工绕过哈希检查 |
| GPU、cgroup 或磁盘压力 | 显存/cgroup 连续越线，磁盘不足以原子保存 checkpoint | 受控停止并保留最近良好 checkpoint；只采用 Day18 已验证的资源修复 |
| Reward 或 mixed-group 崩溃 | invalid 率突升、连续窗口 mixed fraction 为 0、A–D 输出塌缩 | 暂停并抽查原始 rollout；区分模型全对/全错、采样失效和解析器问题，禁止在线改规则 |
| NaN/Inf 或梯度异常 | loss、KL、entropy、grad norm 非有限或持续超出 Pilot 范围 | 立即停止，保存故障前指标和 batch 身份；修复后新建 attempt，不覆盖原日志 |
| 组内重复或串组 | 同组四条完全重复、UID/图片/gold 不一致 | 停止训练并审计 rollout repeat、数据重排和 vLLM 请求映射；不能继续累计无效轨迹 |
| checkpoint 写入中断 | 临时目录、marker 与文件集不一致 | 保留 `.inprogress`，不更新 latest marker；从上一良好 checkpoint 恢复 |
| resume 后重复/漏训 | coverage receipt 出现重复 sample 或 sampler 位置跳变 | 将失败重试与有效轨迹分开计数；修复 sampler/RNG 恢复后重跑受影响区间 |
| 最后 step 指标未被 guard 读取 | checkpoint 为 780但 runtime metrics 停在 779 | 执行 final log drain 和完整日志重放，保存修正前凭据及其哈希 |
| Reward 上升但输出质量恶化 | 短格式投机、冲突答案、外观正确但语义错误 | 只作为训练异常分析；若确认为 Reward 漏洞则本次正式实验失效，升级 Reward 后从 Day18 重走 Gate |

## 八、阶段结论与边界（待真实结果确认）

> Day19 GRPO 正式训练预期 **PASS**。训练从冻结 Base 独立初始化，完成 780/780 个外层迭代、6,240/6,240 个有效 prompt、24,960/24,960 条有效轨迹和 780/780 个 optimizer updates。Reward/group/gold 路由错误为 0，NaN/Inf/OOM 为 0；最终 checkpoint 校验和 coverage receipt 均通过。本结论只说明训练合同完成，不代表外部能力已经改善。

Day19 的完成标准由冻结训练合同、覆盖率、数值健康和 checkpoint 完整性决定，不依据训练 Reward 或外部 Benchmark 重选 checkpoint。最终 actor 只有在 Day20 完成合并、冷加载和固定 R4 后，才能形成能力层面的结论。

## 九、关键证据索引

- `artifacts/runs/E-D19-GRPO-TRAIN-001/preflight/`
- `artifacts/runs/E-D19-GRPO-TRAIN-001/checkpoints/`
- `artifacts/runs/E-D19-GRPO-TRAIN-001/rollouts/`
- `artifacts/runs/E-D19-GRPO-TRAIN-001/evidence/`
- resolved config、启动命令、Git/文件哈希
- coverage receipt
- checkpoint manifest 与恢复状态
- 训练指标、资源遥测和成本报告
