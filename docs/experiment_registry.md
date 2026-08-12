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
| E-D1-001 | 实际运行日期未记录；2026-08-12 登记 | 验证 Student–Teacher 最小单步梯度与冻结链路 | PASS | loss 0.415912；student grad norm 0.561588；Student 更新；Teacher 无梯度且未更新 | 在可用 GPU 和完整依赖环境中验证完整 OPD 训练链路与效果指标 |

## 实验详情

### E-D1-001 — Student–Teacher 最小示例

| 字段 | 记录 |
| --- | --- |
| 实验 ID | `E-D1-001` |
| 日期 | 实际运行日期未记录；登记日期为 2026-08-12 |
| Git commit | 实验运行 commit 未记录；登记时基线 HEAD 为 `c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471` |
| 目标 | 用最小 Student–Teacher 单步示例验证 Student 反向传播与参数更新，以及 Teacher 冻结链路 |
| 模型 | 最小 Student–Teacher 示例；具体模型结构与初始化未记录 |
| 数据版本与样本数 | 最小示例输入；数据版本与样本数未记录 |
| GPU | 运行设备未记录；不能用登记时环境（CUDA 不可用、可见 GPU 0 张）反推实验设备 |
| 配置或启动命令 | 原始配置与启动命令未记录 |
| 唯一改动 | 基线链路验证，无对照实验变量；仅执行一次 Student–Teacher 单步更新 |
| 预期 | loss 可反向传播；Student 获得非零梯度并更新；Teacher 无梯度且参数保持不变 |
| 实际结果 | loss 0.415912；student grad norm 0.561588；Student 参数已更新；Teacher 无梯度且参数未更新 |
| 关键指标 | `loss=0.415912`；`student_grad_norm=0.561588`；`student_params_updated=true`；`teacher_has_grad=false`；`teacher_params_updated=false` |
| 产物路径 | 独立脚本、配置和日志路径未记录；本登记记录位于 `docs/experiment_registry.md` |
| 状态 | `PASS` |
| 下一步 | 固化可复现脚本、随机种子、输入和日志；随后在完整依赖与可用 GPU 环境中验证多步/完整 OPD 流程及效果指标 |
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
