# 正式 CPU 门槛与磁盘处理（2026-09-06）

CPU 门槛已冻结为 **240 GiB = 257698037760 bytes**。
正式 YAML、独立候选 YAML、正式 abort policy 三处一致；不修改 GPU/CPU 中止比例、训练算法或 checkpoint 保存合同。
这是配置门槛，不是修改 AutoDL 实例配额：本次 CPU 工作进程仍为 2 GiB，启动训练时必须重新检查 launcher 的实际 cgroup。

## CPU 证据

| 有效运行 | CPU 峰值 GiB | 实际容量 GiB |
| --- | ---: | ---: |
| Pilot-16 | 177.39 | 240 |
| Pilot-64 | 179.61 | 240 |
| fixed capture | 188.61 | 240 |
| fixed baseline | 189.61 | 240 |
| fixed deferred | 189.79 | 240 |
| pressure v2 | 183.21 | 240 |

从六次原始 cgroup 遥测重算，OOM/oom_kill 增量均为 0。最高峰值不是最新 pressure v2，不能只用后者定门槛。
保留 95% CPU 中止线，对应 228 GiB；最高已观察峰值到中止线约 38.21 GiB，到容量约 50.21 GiB。
选择实际成功使用的 240 GiB，不声称 224/220 GiB 已验证，也不保证 780 步永不出现更高峰值。

冻结证据：`artifacts/runs/E-D11-6K-GATE-001/resource_refreeze_v1/cpu_freeze.json` 及 `.sha256`。
原三份配置的精确快照位于同目录 `before/`。脚本 `scripts/freeze_formal_cpu.py` 检查仅 CPU 合同发生语义变化，
并绑定原始运行证据和新旧配置；修改来源、容量、算法或中止比例都会使验证失败。

候选新收据：`artifacts/runs/E-D11-6K-GATE-001/formal_candidate_v2/cpu_preparation.json`。
旧 candidate v1、静态 Gate、training_config_freeze 和 budget_freeze 保持历史原样，不追改哈希。
因此最终 Gate 会明确报告旧静态/配置/预算绑定已过期，后续应在正式候选验收流程中刷新，不可把它当成新的训练失败。
CPU 冻结本身不授权训练。

## 磁盘计划：仍需明确删除批准

盘点时训练盘总量 600 GiB，可用约 95.52 GiB。项目实验产物约 453 GiB，下载缓存基本已清空。
未确认可用的私有独立备份盘；系统盘仅约 9.6 GiB 可用，公共盘不能作为私有备份。

拟退休两组旧在线 A/B（不是 fixed A/B）的八个大分片：

- `memory_optimization/ab/baseline/run/checkpoints/global_step_8/actor/`
- `memory_optimization/ab/deferred_v2/run/checkpoints/global_step_8/actor/`

每组只包含两个 `model_world_size_2_rank_*.pt` 和两个 `optim_world_size_2_rank_*.pt`，脚本内部使用明确的八个文件白名单，不用通配符删除。
保留旧 A/B 日志、遥测、配置、审计与小文件，以及全部 Pilot、fixed A/B、pressure v2 checkpoint 和回放 payload。
旧 baseline 原始外层审计 FAIL 不会被重写为 PASS。

预计释放 **106.20 GiB**，可用空间增至约 **201.72 GiB**。
这覆盖现有 120 GiB 启动门槛，也覆盖先保存一次候选 checkpoint 再进入正式训练的约 193.12 GiB 规划（含 20 GiB 余量）。
实际仍须按清理后的 df 和启动时磁盘检查为准。

计划：`artifacts/runs/E-D11-6K-GATE-001/resource_refreeze_v1/disk_plan.json`。
**尚未删除真实文件；删除后这八个旧分片没有备份，不能恢复。**
得到明确同意后才可执行 `scripts/retire_old_online_ab.py --execute --confirm-delete-old-online-ab-shards`。
该脚本会先验证目标 inode/大小/mtime/链接类型、打开文件和保护清单，完成全部八个文件的流式哈希后才逐个 unlink，
最后写入清理收据与不可恢复标记；失败不会自动扩大清理范围。

## 验证命令

```bash
conda run --no-capture-output -n vision-opd python -m pytest -q \
  tests/test_formal_resource_refreeze.py tests/test_formal_candidate.py \
  tests/test_vopd_training_preflight.py tests/test_vopd_6241_day11_finalize.py \
  tests/test_day11_validation_evidence.py
```

清理执行测试仅删除临时目录里的合成小文件，不触及真实 checkpoint。

本次结果：`94 passed, 5 subtests passed in 72.76s`。CPU 冻结检查通过；刷新后的正式 Gate 仍未授权训练。
其 `FAIL_EVIDENCE_INTEGRITY` 表示旧静态/配置/预算哈希尚未随候选与资源变更重新绑定，原历史证据被保留，非最新训练出错。

## 磁盘清理执行结果

用户明确同意后，八个旧在线 A/B 模型/优化器分片已按白名单删除，状态 `PASS`。
删除前全部完成 SHA-256 和元数据复核；目标剩余数为 0，删除不可恢复。
释放约 106.20 GiB，训练盘由约 95.52 GiB 可用增至约 201.72 GiB（`df` 显示约 202 GiB）。

清理收据：`artifacts/runs/E-D11-6K-GATE-001/resource_refreeze_v1/disk_cleanup_receipt.json`。
两组旧在线 A/B 的日志、配置、遥测、postflight、小型 checkpoint 元数据均保留；
Pilot-16、Pilot-64、fixed capture/baseline/deferred 和 pressure v2 的完整 checkpoint 各 13 个必需文件均存在。

清理后正式磁盘 Gate 已通过；正式训练仍因旧静态/配置/预算绑定待刷新、候选验证待完成和配置未放行而保持禁止状态。
