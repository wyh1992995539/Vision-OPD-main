# Day 17 GRPO 数据、选择题 Reward 与训练静态合同工作简报

> 执行日期：2026-09-10（UTC）  
> 数据转换：**PASS，6,241/6,241**  
> `vision_opd_mcq_grpo_v1` Reward：**PASS，12,482/12,482 全量正反例**  
> GRPO 配置与 Hydra 静态预检：**PASS**  
> GPU/Pilot/正式训练：**未启动**

## 技术摘要

Day 17 完成了 GRPO 正式训练前的数据、Reward 和静态配置工作。冻结的 Vision-OPD 6,241 行数据被非破坏性转换为 GRPO Parquet，保留完整红框图、问题、A–D 选项和来源身份，移除 `bbox_images`、Teacher crop 及重复答案字段。全部样本路由到独立 `data_source=vision_opd_mcq_grpo_v1` 和 `reward_route=vision_opd_mcq_v1`。

训练 Reward 使用确定性二元规则，不部署神经 Reward Model，也不调用外部 Base Judge。唯一、合法、明确且等于 gold 的最终选项得 1；错误、空回答、冲突、无法解析、标签异常或越界选项得 0。模型答案先独立解析，再与 gold 比较，避免 gold 参与预测提取。

GRPO 候选配置已通过本地 Hydra 真实合并，明确使用原始 Qwen3.5-4B Base、rollout `n=4`、prompt batch 8、vanilla policy loss、组内标准化、单一 actor KL 路径，并关闭 Critic、神经 Reward Model、VOPD Teacher/JSD/EMA 和 Cached Prefix。当前只完成静态 preflight；受限执行环境只有 2 GiB cgroup 且不能执行 `nvidia-smi`，因此训练 loader、双卡运行和真实策略更新仍属于 Day 18 Gate。

## 一、数据盘整理与保留边界

在转换前检查了数据盘并按用户授权删除两批可清理内容。最终 Base、Vision-OPD merged 和 Cached merged 模型继续保留；此前用于精确恢复的旧 FSDP 中间 checkpoint 已不再作为可恢复来源。清理后工作盘约有 510 GiB 可用，为 GRPO Pilot、checkpoint 和 rollout 证据预留空间。

该清理不改变 GRPO 输入数据和已冻结评测产物。后续 Vision-OPD/Cached 可以从 Base 重新调参训练，但不能从已删除的旧 FSDP 状态精确续训。

## 二、GRPO 数据转换

| 项目 | 结果 |
|---|---:|
| 源数据 | 6,241 行 |
| 输出数据 | 6,241 行 |
| 唯一 sample_id | 6,241 |
| 可判分 | 6,241 |
| 不可判分 | 0 |
| `bbox_images` | 已移除 |
| 策略图像 | 每条 1 张完整红框图 |
| data source | `vision_opd_mcq_grpo_v1` |
| reward route | `vision_opd_mcq_v1` |
| Gold 分布 | A=1,740；B=1,540；C=1,461；D=1,500 |

输出字段固定为：`data_source`、`prompt`、`images`、`ability`、`reward_model`、`extra_info`。`reward_model.ground_truth` 只是 verl 的标准答案载体，不代表存在训练好的 Reward Model。

哈希：

- 源 Parquet SHA256：`142c972f182cc0bf90b2ab44a2255896643c5983e0eaf8c7c58f5a488a89031e`
- GRPO Parquet SHA256：`9550fac714fe030e6604e38b2e78db9970268d22464808a36f2a885ff1229206`

## 三、选择题 Reward 实现

Reward 支持裸选项、Answer/Final answer 标记、中文标记、合法 `<answer>` 标签、括号、大小写及 `\boxed{}` 等有限格式。它不会扫描任意大写字母，也不会从选项全文、普通推理讨论或英文单词片段猜答案。

主要诊断字段包括：`prediction`、`parse_valid`、`parse_status`、`parse_evidence`、`candidate_options`、`truncated`、`reward_route` 和 `reward_source`。数据合同错误直接抛出异常；模型输出解析失败保留原因并给 0 分。

验证结果：

- Reward 单元测试：12 passed，125 个格式子用例通过。
- 转换测试：3 passed。
- 全量正确模板：6,241/6,241 得 1。
- 全量确定性错误模板：6,241/6,241 得 0。
- 总试验数：12,482，失败 0。
- 动态文件路径加载：PASS。
- Reward SHA256：`27508c95922f7717c3b5ecad6d71a8c6d7c38a616518accb033d52d633676c20`。

## 四、GRPO 候选训练合同

| 参数 | 候选值 |
|---|---:|
| Base | `/root/autodl-tmp/models/Qwen3.5-4B` |
| 源 prompt | 6,241 |
| prompt batch | 8 |
| rollout n | 4 |
| 每外层迭代轨迹 | 32 |
| epoch | 1 |
| 外层迭代 | 780 |
| 有效 prompt | 6,240 |
| dropped prompt | 1 |
| 有效轨迹 | 24,960 |
| 顶层 PPO mini-batch | 8 |
| worker 展开后全局 mini-batch | 32 |
| 双卡 DP 每卡 mini-batch | 16 |
| policy loss | vanilla clipped policy gradient |
| loss aggregation | token-mean |
| advantage | GRPO，组内标准化开启 |
| LR | 1e-6，warmup 10 steps |
| clip | low=0.2，high=0.2 |
| KL | actor loss 0.001，`low_var_kl` |
| KL in reward | false |
| response length | 128，待 Pilot 审计截断后决定是否升至 256 |

本地 worker 会先将顶层 `ppo_mini_batch_size=8` 乘以 `rollout.n=4`，再按双卡数据并行拆分。因此记录的展开后全局 32、每卡 16 与本地实现一致。

## 五、静态预检和 fail-closed 入口

完成并测试了：

- `configs/grpo_6241.yaml`
- `scripts/grpo_training_preflight.py`
- `scripts/run_grpo_guarded.py`
- `scripts/run_grpo_2gpu.sh`
- `tests/test_grpo_training_preflight.py`

静态预检检查了 Base 文件、6,241 行和唯一 ID、所有图像路径、Reward 路由、Reward/Data 哈希绑定、gold 输入隔离、无 bbox crop、batch 算术、单一 KL 路径、无禁止组件，以及本地 Hydra 参数可合并性。最终回归为 18 passed、125 个 Reward 子用例通过。

`--preflight-only` 已实际执行并生成 `preflight.json`、`hydra_overrides.json`、`resolved_hydra_config.yaml` 和可复现 `command.txt`，没有启动 GPU。当前配置的 Pilot 和正式训练授权均为 false；对 `--run` 的测试按预期返回 `RUN BLOCKED`。

## 六、资源状态与下一 Gate

预检时资源快照：

- 数据与配置静态 Gate：PASS。
- 磁盘：约 509 GiB 可用，PASS。
- 当前 Codex 工具 cgroup：2 GiB，低于训练要求。
- `nvidia-smi`：当前受限进程无执行权限，GPU runtime Gate 无法验证。
- `training_started=false`、`gpu_used=false`。

下一步是 Day 18A 的固定 32-prompt 真实 GRPO Pilot。它需要真实双卡环境，执行 4 个外层迭代、128 条 rollout，并验证 mixed-reward group、组内 advantage、非零且有限的策略梯度、optimizer step、actor 参数变化、截断和 Reward 投机。32-prompt Gate 通过后才能进入 64-prompt 稳定性与恢复验证；正式 6,241 行训练仍未授权。

## 七、证据索引

- 数据转换实现：`scripts/prepare_grpo_data.py`
- GRPO Parquet：`/root/autodl-tmp/data/vision_opd_6241/grpo_train_6241.parquet`
- 数据与 Reward 根凭据：`artifacts/runs/E-D17-GRPO-DATA-001/`
- Reward 实现：`verl/utils/reward_score/vision_opd_mcq_grpo.py`
- Reward 验证：`artifacts/runs/E-D17-GRPO-DATA-001/reward_validation.json`
- 候选配置：`configs/grpo_6241.yaml`
- 静态预检根凭据：`artifacts/runs/E-D17-GRPO-CONFIG-001/preflight/`
- 静态预检报告：`artifacts/runs/E-D17-GRPO-CONFIG-001/preflight/preflight.json`
- resolved Hydra：`artifacts/runs/E-D17-GRPO-CONFIG-001/preflight/resolved_hydra_config.yaml`
- Day17–21 执行计划：`docs/day17_day21_grpo_execution_plan.md`
