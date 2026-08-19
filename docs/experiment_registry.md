# Vision-OPD 实验登记表

## 登记规则

- 实验 ID 使用 `E-D<Day>-<三位序号>`，例如 `E-D1-001`。
- 每次实验开始前填写目标、唯一改动与预期；结束后补齐实际结果、指标、产物和下一步。
- “唯一改动”应只描述相对直接基线的单一变量；多变量变更应拆成多个实验。
- Git commit 填实验实际运行时的完整 commit。无法追溯时必须写“未记录”，不可用登记时 HEAD 冒充。
- 状态统一使用 `PLANNED`、`RUNNING`、`PASS`、`FAIL`、`BLOCKED` 或 `INVALID`。
- 命令、日志、配置和 checkpoint 应保存到可追溯路径；没有产物时明确写“未记录”或“不适用”。

## 总览

| 实验 ID | 日期 | 目标 | 状态 | 关键结果 | 下一步 |
| --- | --- | --- | --- | --- | --- |
| E-D1-001 | 原始运行日期未记录；2026-08-13 可复现复跑 | 验证 Student–Teacher 最小单步梯度与冻结链路 | PASS | 复跑结果：loss 0.415912；student grad norm 0.561588；Student 更新；Teacher 无梯度且未更新 | Day 3 实现玩具 JSD 与 EMA；后续再验证完整 OPD 训练链路 |

## 实验详情

### E-D1-001 — Student–Teacher 最小示例

| 字段 | 记录 |
| --- | --- |
| 实验 ID | `E-D1-001` |
| 日期 | 原始运行日期未记录；2026-08-12 登记；2026-08-13 完成可复现复跑 |
| Git commit | 原始运行 commit 未记录；复跑时 HEAD 为 `79aa58e49f3c1adbcc9192f1563e0e970c2d415e`；脚本最近一次提交为 `e75be845c24bf2a0a23e9eb2b4290c31ff01aab5`，复跑时脚本未修改 |
| 目标 | 用最小 Student–Teacher 单步示例验证 Student 反向传播与参数更新，以及 Teacher 冻结链路 |
| 模型 | Student 与 Teacher 均为独立的 `torch.nn.Linear(4, 6)`；随机种子 `torch.manual_seed(42)`；Teacher 参数冻结 |
| 数据版本与样本数 | 固定随机生成的 Tensor；batch size 8，input dim 4，vocab size 6；不使用外部数据集 |
| GPU | 2026-08-13 复跑使用 CPU；该最小梯度验收不依赖 GPU |
| 配置或启动命令 | `export OMP_NUM_THREADS=8 && /root/miniconda3/envs/vision-opd/bin/python scripts/day1_student_teacher_minimal.py` |
| 唯一改动 | 基线链路验证，无对照实验变量；仅执行一次 Student–Teacher 单步更新 |
| 预期 | loss 可反向传播；Student 获得非零梯度并更新；Teacher 无梯度且参数保持不变 |
| 实际结果 | 2026-08-13 复跑：loss 0.415912；student grad norm 0.561588；`teacher has grad: False`；Student 参数已更新；Teacher 参数未更新；脚本全部断言通过并输出 `PASS` |
| 关键指标 | `loss=0.415912`；`student_grad_norm=0.561588`；`student_params_updated=true`；`teacher_has_grad=false`；`teacher_params_updated=false` |
| 产物路径 | 可复现脚本：`scripts/day1_student_teacher_minimal.py`；脚本 SHA-256：`dc088b557d6ab7cc64a1a409ddee7721f732662b74db5b8df74dd76fa1d6fd84`；结果摘要记录于本条目 |
| 状态 | `PASS` |
| 下一步 | Day 3 在小 Tensor 上实现 JSD、backward 和 EMA 更新；完整 OPD 与 GPU 训练链路在后续 smoke 阶段单独验证 |
| 结论边界 | 只证明单步梯度与冻结链路，不证明完整 OPD 或效果提升 |

## 新实验模板

复制下表并用新的实验 ID 建立详情；不要覆盖历史记录。

| 字段 | 记录 |
| --- | --- |
| 实验 ID | `E-Dx-xxx` |
| 日期 | YYYY-MM-DD |
| Git commit | 完整 commit SHA |
| 目标 |  |
| 模型 |  |
| 数据版本与样本数 |  |
| GPU | 数量、型号、显存 |
| 配置或启动命令 |  |
| 唯一改动 |  |
| 预期 |  |
| 实际结果 |  |
| 关键指标 |  |
| 产物路径 |  |
| 状态 | `PLANNED` |
| 下一步 |  |
| 结论边界 |  |
