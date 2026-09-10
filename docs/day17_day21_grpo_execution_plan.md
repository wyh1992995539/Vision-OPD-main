# Day17–Day21 GRPO 详细执行计划

编制日期：2026-09-10。起点：按负责人最新说明，Day16 已完成，进入 GRPO 扩展阶段。本文件是后续任务规划，未启动训练，未将候选参数标记为冻结或通过验收。

## 1. 本次核查结果与排期依据

已只读检查现行 6K 排期、Day16 R4 与输出上限修订、本地 verl、训练配置和全量训练 JSONL。

| 项目 | 已确认情况 | 对后续任务的影响 |
|---|---|---|
| 数据规模 | 6241 行、6241 个唯一 sample_id | 复用冻结数据，不重新划分、不筛成子集 |
| 题型 | 6241/6241 均为 multiple_choice | 首版只需四选一 Reward，不必实现无数据支持的数字/开放题分支 |
| 选项与 gold | 全部能抽出 4 个不同标签，gold 均在对应选项中；结构异常 0 | 全量可判分有结构基础，但还需奖励实现与真实输出审计 |
| gold 分布 | A=1740、B=1540、C=1461、D=1500 | 监测输出字母偏置；恒选 A 的训练集正确率约 27.88%，不能作为有效学习证据 |
| 项目实现 | 有通用 GRPO 配置与算法，没有本项目 grpo_6241 配置及正式入口 | Day17 包含真实实现工作，不只是改参数 |
| 旧 Reward | zoom_bench.py 的选择题分支使用任意大写字母回退，类型分支名称为 mcq | 必须新增独立训练 Reward，明确 multiple_choice→新路由映射，避免误入其他评分分支 |
| 本地批量语义 | fsdp_workers.py 会先将 ppo_mini_batch_size 乘 rollout.n，再按数据并行分片 | 顶层 mini-batch 不能想当然地填 32；需保存 resolved config 和实际更新计数 |
| 旧通用配置 | baseline_grpo.yaml 为 n=8、prompt batch=32、norm_adv_by_std_in_grpo=False | 不能直接视为本项目已冻结方案 |
| 评测现状 | 三组完整固定 1024 输出上限 R4 结果已存在；自适应协议只覆盖原三组且尚未全完成 | GRPO 先接入可比较的固定 1024 R4 评测；自适应结果另列 |
| 费用口径 | Day12 修订为双卡合计 14 元/小时，费用作估算记录 | 不恢复旧的逐次账单确认与累计费用自动门禁；资源异常监控仍保留 |

结构核查使用 artifacts/data/vision_opd_6241/train_6241.jsonl。它没有验证所有图片语义标签，也没有证明真实模型输出均可无歧义解析；不能据此提前宣称 Reward Gate PASS。

旧总计划里仍有 R3、零权重补齐、旧计费、Day16 未开始等历史表述。此次后续安排采用最新修订：源数据 6241、原生 drop_last、有效 6240、无补齐；历史产物保留。

## 2. 目标、边界与建议工时

完成一条从原始 Qwen3.5-4B 独立初始化的多模态 GRPO 分支，交付真实组采样、规则 Reward、组内优势、策略梯度更新、可重载 checkpoint 以及同协议外部比较的完整证据。

| 阶段 | 工作重点 | 主动工时建议 | 机器时间 | 完成条件 |
|---|---|---:|---|---|
| Day17A | 数据与 Reward 合同 | 3–4 h | CPU 审计 | 全量 6241 路由可用、测试通过 |
| Day17B | 配置、launcher、观测与静态预检 | 2–3 h | CPU 为主 | 实际配置正确、无 VOPD 分支残留 |
| Day18A | 32 prompt 真实更新验证 | 2–3 h | 实测 | 分组、优势、PG 梯度、参数更新成立 |
| Day18B | 64 prompt 稳定性与恢复验证 | 3–4 h | 实测 | 保存/恢复/冷加载、资源外推通过 |
| Day19 | 正式训练与监控 | 2–3 h | 780 个训练外层迭代，时长由 Pilot 外推 | 6240 有效 prompt、24960 条有效轨迹、最终 checkpoint |
| Day20 | 模型定版与外部评测 | 3–4 h | 合并、生成、Judge 分别估算 | 2536 条固定 1024 R4 评分完整 |
| Day21 | 四组比较、失败分析与归档 | 3–4 h | CPU 为主 | 结论与逐样本证据可追溯 |

总主动工时约 20–28 小时。Day17–21 是工作阶段，不保证五个连续自然日完成；若实现或训练超时，顺延日期而不压缩验收。32/64 Pilot 都从 Base 独立启动；正式训练也从 Base 重新启动，不接着 Pilot 权重训练。

## 3. Day17A：全量数据与选择题 Reward

### D17-1：复用并转换训练数据

实现 scripts/prepare_grpo_data.py：

- 输入现有冻结 6241 manifest/JSONL，核对 revision、SHA256、sample_id 集合和原始行号；输出新 GRPO 文件，保留原文件。
- 保留相同完整红框图、问题、四个选项与 prompt/template/processor 合同。GRPO policy 只看完整图，不传 bbox_images、Teacher crop、gold 或含答案的 extra_info。
- ground_truth 与 reward_route 只供 Reward 消费。建议新增独立 data_source=vision_opd_mcq_grpo_v1，禁止误路由到旧 zoom-bench 自动回退。
- 输出 images、prompt、reward_model.ground_truth、extra_info.sample_id、extra_info.options、extra_info.reward_route；实际 Schema 以本地 dataset/reward manager 接口为准。
- 原图只引用现有已验证路径，避免复制大图片集。
- 复用既有全量图片 QA；如果图像字节、processor 和模板不变，复用可绑定的长度证据。凡 prompt 编码发生变化，重算 6241 条实际多模态长度。
- 禁止静默 filter_overlong_prompts；超长必须报出 sample_id 后修正长度合同，不能少训练样本。

产物：GRPO JSONL/Parquet、数据哈希、逐条 reward_routes.jsonl、data_audit.json，以及 tests/test_grpo_parquet.py。

### D17-2：冻结奖励规则

新增独立纯函数 Reward 模块，建议 verl/utils/reward_score/vision_opd_mcq_grpo.py，避免更改历史评测 Reward 的行为。

首版采用纯结果奖励：唯一、合法、明确的最终选项与 gold 相同给 1；明确错误、空回答、冲突选项、无法解析或选项越界给 0，同时保留不同 reason。不给格式分、长度分、推理篇幅分，不在训练中调用 Base Judge 或其他 Reward Model。

解析约定：

1. 支持单个裸选项、明确的 Answer/Final answer 标记、合法 answer 标签及必要的括号/大小写变体。
2. prediction 必须先独立从回答中提取，再与 gold 比较；gold 不参与预测抽取。
3. 只允许该题真实选项 A–D。E/F、冲突的明确结论、单个标签内枚举多个候选都不能获得正分。
4. 明确处理解释区、最终答案区、多个答案区的优先级。普通推理中的选项讨论不自动算作最终答案；多个互相冲突的最终答案不能随意挑一个。
5. 不扫描任意大写字母，不从选项全文、单词片段或引用的题干猜选项。
6. 训练 Reward 可参考 R4 的保守抽取思想，但不能直接把需要 Judge 的 ambiguous 路由搬入训练；R4 评测实现及结果保持冻结。
7. 记录 finish_reason、截断状态与答案完整性。明确且完整的答案可按冻结规则评分；截断且无完整答案为 0。不能把所有 length 输出不加区分地写成语义错误。
8. 数据 gold 有歧义或不合法属于数据 Gate 阻塞；模型输出有歧义属于本条 rollout 的 parse_invalid。这两种状态不得混为一谈。

测试至少覆盖：A/B/C/D 正误、Answer: D 不被读成 A、普通英文首字母、含选项全文、否定句、引用别人的答案、多个冲突 answer 标签、单标签 A or B、越界、空串、缺失闭合标签、异常类型、长解释与末尾答案、改变 gold 不改变解析结果。

### D17-3：Reward 审计和验收

- 逐条输出 sample_id、选项、gold、normalized_gold、reward_route、scorable 状态和原因。
- 覆盖要求：source=6241、unique=6241、scorable=6241、unscorable=0、ambiguous_gold=0。
- 四个 gold 类别各人工抽至少 10 条，另查全部异常、复杂选项和已有可用长输出；检查图片/问题/选项/gold 关联。抽样不等于宣称全量语义标签已人工验证。
- 运行 Reward 与 Parquet 测试，验证 ground_truth 不进入模型输入、crop 不进入 GRPO。
- 如果任何条目无法可靠判分，修复可论证的规则后重审；确实无法覆盖则停止正式 6K GRPO，明确保留可判分 Pilot 作为机制验证，不静默删题。

## 4. Day17B：配置、入口与可观测性

### 候选参数

以下是待 Pilot 验证的起点，不是当前已有可执行配置。

| 项目 | 建议候选 | 冻结/验证要求 |
|---|---|---|
| Base | /root/autodl-tmp/models/Qwen3.5-4B | 与其他独立分支同一 Base 权重哈希 |
| 数据/epoch | 6241 行、1 epoch、shuffle seed=42、原生 drop_last | 有效 6240、dropped 1、padding 0 |
| prompt batch / n | 8 / 4 | 每个 batch 8 个组、32 条轨迹 |
| advantage | grpo，norm_adv_by_std_in_grpo=True | 显式配置，不继承通用 baseline 的 False |
| policy loss | vanilla clipped policy gradient | 禁用 VOPD/JSD/EMA 和 Teacher 构造，不只改 adv_estimator |
| loss 聚合 | token-mean | 记录与原始逐序列平均 GRPO 的差异，不称逐项复现原论文 |
| ppo_mini_batch_size | 顶层 8 | 本地 worker 扩为全局 32 responses；双卡 DP=2 时每卡 16，实测确认 |
| ppo_epochs | 1 | 每个 rollout batch 目标 1 次全局 optimizer update |
| LR | 1e-6；warmup 10 个实际 optimizer steps | 保守起点，不做宽泛搜索；因数值/更新尺度问题调整时保存新候选 |
| clip ratio | low=0.2，high=0.2 | 显式对称裁剪，记录 clip fraction |
| KL | actor use_kl_loss=True，coef=0.001，use_kl_in_reward=False | 固定原始 Base reference，不重复加 KL；类型使用本地支持的 low_var_kl 并冻结 |
| entropy coefficient | 0，仍记录 entropy | 监控策略变化，不额外引入奖励项 |
| rollout | temperature=1.0、top_p=1.0、top_k=-1、ignore_eos=False | 不使用外评的 greedy 来代替组采样 |
| prompt length | 8192 候选 | 由实际 Processor 编码证明不漏样本 |
| response length | 128 作首轮工程候选，必要时 256 | 如果训练数据上的正常答案持续截断，再审计 512/1024；冻结最终选择和成本 |
| micro-batch | 每卡 1 或等效动态 token batching | 验证有效全局 batch 不随显存修复变化 |
| 精度与资源 | 继承已验证的 BF16/FSDP、gradient checkpointing、必要 offload | reference 常驻/卸载和 rollout n=4 的峰值重新测量 |
| validation | val_before_train=False、test_freq=-1 | 不使用已并入训练的旧 eval-128/retention-64 |

response length 的升级只依据训练 Pilot 的 finish_reason、完整答案率和正常轨迹长度。Day16 已证明固定上限可能影响评价，因此不能把 128 当成已证明足够，也不能把外评的自适应上限逐题用于训练。参数选择记录为已看到先前三组结果之后的探索性扩展。

实现 configs/grpo_6241.yaml、scripts/run_grpo_2gpu.sh 及 guarded Python launcher。复用 Day12/14 的资源、心跳、checkpoint 验证能力，移除不适用于 GRPO 的 Teacher/JSD 指标门禁。launcher 在 --preflight-only 时不启动训练，输出所有 Hydra 合并后的参数、命令和哈希；不得复用其他分支的已通过 promotion receipt。

预检检查：Base 身份、全部 Reward 路由、crop/gold 输入隔离、真实 policy loss、固定 reference、无 Critic/Reward Model worker、batch 整除、长度预算、输出目录无冲突、GPU 空闲、CPU/cgroup 与磁盘、缓存目录、恢复策略。现有工作区有 Day16 修改；保存 commit 与 patch/文件哈希，不能 reset 或覆盖。

算法参考：[verl GRPO 文档](https://verl.readthedocs.io/en/latest/algo/grpo.html)与[DeepSeekMath](https://arxiv.org/abs/2402.03300)。实际参数与计数以本地代码和 Pilot 为准；上游示例不替代本地版本验证。

## 5. Day18A：32 prompt，证明真实 GRPO 更新

实验目录：artifacts/runs/E-D18-GRPO-PILOT-001/32/。

1. 从训练集预先按 gold、长度和视觉输入大小确定固定 32 条，保存 sample_id 与选择方法。不得按外部 benchmark 成败挑题。
2. 从 Base 启动，8 prompt/batch、n=4、1 epoch：预期 4 个外层迭代、128 条轨迹；前 3 步重点检查。
3. 保存每条 sample_id、group_uid、rollout_index、response、token/mask、finish_reason、解析选项、gold、reward/reason、advantage，以及 policy version。
4. 验证每组恰好 4 条；分布式重排、repeat/interleave、负载均衡之后 prompt、图片、gold、奖励和组标识仍对应。
5. 用实际 rollout 离线重算组内优势并比对训练 tensor。本地标准化使用 torch.std 的样本标准差；例如 reward=[1,0,0,0]，std=0.5，优势约 [1.5,-0.5,-0.5,-0.5]。全 0/全 1 组优势应为 0，且 response padding 不参与 loss。
6. 保存非零且有限的策略梯度证据、optimizer.step 计数、更新前后 actor 参数差异；单独记录 KL 与 PG。actor 参数有变化但仅由 KL/weight decay 驱动不算成功。
7. 不以 PG loss 标量必须非零作为门槛：组内中心化可能让标量接近 0，但梯度仍非零。
8. 人工查至少 20 条真实输出，涵盖正奖励、错误、解析失败、截断、重复及全部冲突答案；发现奖励漏洞则修复版本、重跑受影响 Pilot。

关键指标为 mixed_group_fraction：同题四条奖励有差异的组数/全部组数。全局平均 Reward 为 0.5，也可能所有组都是全对或全错，导致正确性梯度为零。

通过条件：至少有真实 mixed-reward 组和对应 PG 更新；无串组、路由/掩码错误或 NaN/Inf。所有组同奖励则先排查采样、解析、题目难度和截断，不能为了造出差异人工修改奖励或只保留易成功的组。

## 6. Day18B：64 prompt 稳定性、保存恢复与正式冻结

实验目录：artifacts/runs/E-D18-GRPO-PILOT-001/64/。

- 从 Base 独立启动固定 64 条，覆盖常规、最长 prompt、较大图像和四类 gold；预期 8 个外层迭代、256 条轨迹。
- 启动后检查前 3 步，完成全部 8 步。32 与 64 不同时占用同一双卡设备。
- 在中间保存 checkpoint，验证 optimizer、scheduler、global step、dataloader/sampler 位置与 RNG 状态的恢复。保存恢复路径与不中断路径的计数/顺序一致；不要求跨 GPU 推理逐 token 位级一致。
- 最终合并并在新进程冷加载 5 条训练工程样本，只验证可加载、非空、合法停止和无异常，不报告为泛化精度。
- 测量 rollout/actor/reference、checkpoint 保存及合并各阶段的 GPU、RSS/cgroup、磁盘峰值；不能套用 VOPD 的显存结论。
- 记录稳态 step 时长中位数、均值、最大值和长尾样本；8 步对 P95/长期稳定性的估计很弱。若混合组太少、编译占比大或趋势不明，延长到预先固定的 128/256 prompt 工程 Pilot，保留新 ID，不宣称正式训练已开始。

正式训练验收 Gate：

| Gate | 必须证据 |
|---|---|
| 数据/Reward | 6241 可判分，所有测试通过，真实高 Reward 无已知漏洞 |
| 算法 | 分组正确、relative advantage 正确、PG 梯度成立、无 VOPD/EMA 更新 |
| 更新计数 | resolved mini-batch 与 ppo_epochs 支持每外层迭代一次全局更新，实测吻合 |
| 稳定性 | loss/grad/KL 有限、无 OOM，mixed-group/parse-invalid/截断指标有明确解释 |
| checkpoint | 可保存、恢复、合并、冷加载，身份和哈希可追溯 |
| 资源 | GRPO 自身峰值和保存/合并峰值可承载，预留空间覆盖滚动 checkpoint 重叠 |
| 运行策略 | 墙钟、心跳、内存、磁盘及异常阈值在启动前写入配置 |

无收益不作为 Pilot 失败标准；无法证明有效优化信号则不得扩到正式训练。正式前冻结 Reward、数据、所有训练参数及版本、样本顺序和预计 dropped sample_id。

## 7. Day19：正式训练与异常处理

实验目录：artifacts/runs/E-D19-GRPO-TRAIN-001/。

冻结计数合同：

```text
source prompts                 6241
batch size                        8
rollouts per prompt               4
epochs                            1
outer training iterations       780
effective prompts              6240
dropped prompts                   1
padding rows                      0
effective rollout trajectories 24960
optimizer updates               780  # 仅在 Day18 已实测 mini-batch/ppo_epochs 合同后成立
```

恢复前后的重复尝试单独计数。24960 是有效训练轨迹，不把失败重试、Pilot、冷加载或外部评测生成混进去。

执行步骤：

1. 实时预检通过，从原始 Base 开始；检查前 3 个迭代的真实分组、Reward、PG 更新和资源。
2. 按冻结保存频率保留滚动恢复 checkpoint；建议中段 step 390 和最终 step 780，保留策略先在 Pilot 验证。写入/校验新 checkpoint 完成前，不删除唯一可恢复状态。
3. 每步保存奖励均值/方差、mixed-group fraction、全对/全错组率、invalid 率、A–D 输出分布、entropy、KL、clip fraction、grad norm、长度分位数、length 结束比例、吞吐、GPU/CPU/磁盘及 update 计数。
4. 训练中抽查高奖励长答案、重复答案和格式变化。指标反常时先检查模型输出与解析证据，不能只看 Reward 曲线。
5. 结束生成 coverage_receipt：已访问 sample_id、有效次数、dropped sample_id、重复/遗漏检查、实际 optimizer updates 与轨迹数。

异常策略建议（Day18 校准并冻结后启用）：

- NaN/Inf、错误 group/路由、gold 泄漏、OOM、磁盘无法安全保存：立即受控停止并保留证据，不覆盖良好 checkpoint。
- 连续窗口无 mixed group、invalid 比例明显上升、长度持续贴上限：记录告警并检查。全同奖励可能来自全对，也可能来自全错，不能不分原因自动记为模型崩溃。
- KL、entropy、梯度或 clip fraction 持续偏离 Pilot 范围：按冻结阈值暂停审查，不在同一实验中临时改 LR、KL 或 Reward。
- 可恢复的基础设施中断使用原配置与已验证恢复路径；算法/奖励变更创建新实验版本，从 Base 重新验证。

费用仅作估算与结算记录，不恢复逐次账单确认；墙钟和资源中止策略独立存在。

## 8. Day20：定版与评测协议衔接

### D20-1：模型定版

合并最终 actor，保存权重与 tokenizer/processor/template 的哈希，新进程冷加载 5 条。冻结最终训练步对应模型；不运行旧 eval-128/retention-64，不根据外部成绩挑中途 checkpoint。

### D20-2：固定 1024 R4 四组比较

建议优先完成可与原三组直接配对的评测：沿用冻结 R3 的生成/数据/图像合同，加 Day16 R4 的解析与固定 Base Judge 路由；GRPO checkpoint 与输出目录为新身份。先做 4×3 smoke，然后完成 ZoomBench 845、MMStar 1500、V* 191，共 2536 条。

需要在 GRPO 首次外评前新增可执行评测配置/适配并冻结：当前 R4 配置是对三组历史预测重评分的入口，不能直接冒充已支持新 GRPO 生成。复用 R4 parser/路由测试，保存新模型和协议的绑定凭据，确认现有 score_paper_aligned.py 未提交改动的实际版本/哈希。

固定 1024 R4 参考行（只作最终比较，不参与调参）：

| 模型 | ZoomBench | MMStar | V* |
|---|---:|---:|---:|
| Base | 50.65% | 68.00% | 82.72% |
| Vision-OPD | 52.54% | 59.20% | 71.73% |
| Cached Prefix | 51.36% | 56.33% | 73.30% |
| GRPO | 待评 | 待评 | 待评 |

验收：2536 个唯一结果键、无重复/待评分；推理/Judge 错误按冻结失败规则计数并完整披露，不能静默排除分母。训练 Reward 和 benchmark score 分开；训练模型不得自评。

### D20-3：自适应上限结果另列

目前 Base MMStar 的 72.73% 来自选择性自适应混合评测，不能与其他模型的固定 1024 分数直接构成统一主表。

现有 BENCHMARK-ADAPTIVE-OUTPUT-CAP-V1 只包含原三组，且 Vision-OPD/Cached 重跑明确暂缓。本次 GRPO 计划保留该暂停安排。若后续要四组自适应比较，先追加纳入 GRPO 的协议修订，保持逐档上限、停止条件、Judge、错误处理完全一致，并在相应模型执行完整后再汇总。

固定 1024 四组比较完成，不等于“自适应最终本地 benchmark 全部完成”；后者必须满足各组 unresolved_output_cap_count=0 等现行协议条件。

## 9. Day21：四组比较与交付

- 按 benchmark 分别报告 accuracy、正确数/分母、与 Base/VOPD/Cached 的差值、corrected/regressed 及共同 sample_id 上的配对结果；可给样本级配对 bootstrap 区间，但不能据此推断训练随机种子稳定性。
- 复用已有 overlap 审计，保留官方完整集合主分母；有重叠则单列诊断，不能静默删除。
- 单独报告训练侧 reward、有效组比例、解析成功率、截断比例、更新规模和 A–D 偏置。
- 成本表列出训练 prompt 数、有效覆盖、轨迹数、实际 response token 总量、长度上限、optimizer updates、GPU 小时、墙钟、峰值显存、费用。
- 四组方法表写清轨迹来源、监督源、损失、Teacher/reference、可更新参数；GRPO 与 VOPD 不属于只改变 objective 的单变量实验。
- 预先规定 bad-case 抽样规则，建议 16–24 条，覆盖 GRPO 修正/退化、与 VOPD 分歧、解析/长度失败和已观察到的奖励投机；某类不存在则如实注明。
- 已经看到原三组 benchmark 结果后设计 GRPO，报告为探索性后续实验；冻结后不借外部分数继续选 checkpoint 或反复调参。
- 汇总单训练 seed、Base Judge 替代论文 Judge、资源缩放、轨迹预算及长度差异、潜在数据重叠的限制。
- 更新 README、最终报告和证据索引，运行与本次改动相关测试；准备可审查 diff。提交/tag 按实际项目工作流完成，不把准备状态写为已发布。

## 10. 机器时间、费用与资源估算

正式训练先采用：

```text
T_train_hours = (780 × stable_mean_step_seconds + startup_seconds
                 + checkpoint_seconds + reserved_recovery_seconds) / 3600
training_cost_yuan = 14 × T_train_hours
GPU_hours = 2 × T_train_hours
```

启动/编译从稳态 step 统计中剔除后再单独计入，避免重复计费。合并与 benchmark/Judge 分别估算，单卡使用单卡实际计费，不把双卡单价直接套用。

仅作换算示例、不代表预测：30/60/120 秒一个迭代分别为 6.5/13/26 小时纯迭代时长，对应 91/182/364 元，尚未加启动、保存、恢复、合并和评测。

空间预检沿用现行至少 120 GiB 的安全基线，并用 GRPO 实测 checkpoint、滚动保存重叠、HF 合并与临时文件峰值重新计算，取更严格要求。CPU/cgroup 也根据 GRPO 的实测重新冻结，不直接宣称已通过 VOPD 的资源 Gate。未验证的唯一模型/恢复分片不得为腾空间而删除。

## 11. 交付清单与下一步顺序

以下均是计划新增路径，不表示文件已经实现：

```text
configs/grpo_6241.yaml
configs/grpo_6241_pilot_32.yaml
configs/grpo_6241_pilot_64.yaml
configs/grpo_6241_abort_policy.yaml
scripts/prepare_grpo_data.py
scripts/run_grpo_2gpu.sh
scripts/run_grpo_6241_guarded.py
scripts/audit_grpo_pilot.py
verl/utils/reward_score/vision_opd_mcq_grpo.py
tests/test_reward_rules.py
tests/test_grpo_parquet.py
tests/test_grpo_training_contract.py
artifacts/runs/E-D17-GRPO-DATA-001/
artifacts/runs/E-D18-GRPO-PILOT-001/32/
artifacts/runs/E-D18-GRPO-PILOT-001/64/
artifacts/runs/E-D19-GRPO-TRAIN-001/
artifacts/runs/E-D20-GRPO-EVAL-001/
artifacts/reports/grpo_64_stability.md
artifacts/reports/grpo_eval.md
```

每个运行目录保存 config/resolved_config、command、commit+patch 或文件哈希、环境、data/reward/model hash、metrics、rollout 证据、资源记录、费用、checkpoint 哈希及 gate/coverage receipt。评测额外保存逐样本 prediction、score、Judge 请求与协议哈希。

下一次实际执行顺序：D17-1 数据转换 → D17-2 独立选择题 Reward 与测试 → D17-3 全量路由验收 → D17B 配置/launcher 静态预检 → D18A 双卡真实 Pilot。优先解决奖励正确性，再投入长时间训练。
