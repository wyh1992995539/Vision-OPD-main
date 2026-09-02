# E-D10-001 基础 Gate 已通过，Day 10 仍等待中止条件冻结

> 生成时间：2026-09-02T06:07:07.519370+00:00
> Task 4：**COMPLETE**
> 决策：**PASS_TO_TASK5**
> GPU 使用：**false**

## 结论

E-D10-001 的数据、Base、正式配置、预算、Git、输出目录、日志路径、磁盘和 CPU-only launcher preflight 均已通过。任务 4 已完成，可以进入任务 5；当前仍不得执行正式训练，因为训练中止条件和观测性控制尚未冻结。

磁盘满足项目定义的 `2 × 最终 checkpoint 估算 + 5 GiB`。公式之外另有 46542057302 bytes（约 43.35 GiB）余量，容量 Gate 为 `PASS`。checkpoint 写入量仍然较大，因此任务 5 继续保留磁盘监控。

## 正式训练合同

| 项目 | 冻结值 |
|---|---:|
| 实验 | E-D10-001 |
| 样本数 | 1024 |
| global batch | 8 |
| optimizer steps | 128 |
| epochs | 1 |
| 完整 epoch | True |
| 配置 SHA256 | `5977d0b7adda448287d7410431c9461a6f6f53c04792390b9b13d9529a00b30c` |

配置从冻结 Base 冷启动，`resume_mode=disable`；`dataloader_num_workers=0`；只保留最终 checkpoint。

## Gate 证据

| Gate | 状态 | 证据 |
|---|---|---|
| data | PASS | 1024 rows; frozen SHA256 match=True |
| base | PASS | frozen shard hashes match=True |
| config_identity | PASS | E-D10 identity and 1024/128 formal settings |
| git_state | PASS | commit=f47e9947e0d6; clean=True; diff-check=PASS |
| output_directory | PASS | unexpected files=0 |
| log_path | PASS | /root/autodl-tmp/Vision-OPD-main/artifacts/runs/E-D10-001/logs/train.log |
| storage | PASS | required=119438631082; available=165980688384; shortage=0 |
| budget | PASS | current=200.0 CNY; projected=220.8427527465047 CNY |
| launcher_preflight | PASS | 1024 rows; missing images=0; gpu_used=false |
| config_hash_identity | PASS | 5977d0b7adda448287d7410431c9461a6f6f53c04792390b9b13d9529a00b30c |
| day8_cold_reload | PASS | 5 predictions; 0 inference errors |

审计源提交为 `f47e9947e0d62400a6659a9255a1cda33c2ccdbf`，审计开始时工作树 clean=`True`。本报告生成后产生的新文件需要单独提交，最终 clean 状态在提交后复核。

## 时间与预算

| 口径 | 双卡小时 | 费用 |
|---|---:|---:|
| 均值规划 | 1.0246 | ¥12.25 |
| 保守预留 | 1.7427 | ¥20.84 |

- 用户报告的 AutoDL 累计费用：¥200.00。
- 加入保守预留后的预计累计费用：¥220.84。
- 项目预算预计剩余：¥1779.16。
- 该累计费用是点时值；Day 10 启动前必须刷新 AutoDL 控制台。

## 磁盘口径

- Day 8 checkpoint：57034960981 bytes。
- 冻结公式要求：119438631082 bytes。
- 当前可用：165980688384 bytes。
- 高于最低要求：46542057302 bytes。

## 风险与任务 5 交接

| 严重度 | 状态 | 风险 | 交接动作 |
|---|---|---|---|
| INFO | PASS | Storage exceeds the frozen checkpoint-retention formula with additional headroom. | Retain filesystem monitoring in Task 5 because checkpoint writes are still large. |
| MEDIUM | MITIGATED_REQUIRES_MONITORING | Day 8 recorded one DataLoader worker Killed event after checkpoint save. | Task 5 must capture trainer RSS and cgroup memory events. |
| MEDIUM | REFRESH_BEFORE_LAUNCH | The controlling AutoDL cumulative charge is a user-reported point-in-time value. | Refresh the console value immediately before Day 10 launch. |
| HIGH | OPEN_TASK5 | Training abort conditions and monitoring are not yet frozen as executable controls. | Complete Task 5 before authorizing the --run command. |

Day 8 的 checkpoint 已完成 5/5 冷重载推理且没有 inference error。DataLoader worker `Killed` 和不可审计的逐卡显存峰值仍作为观测性 caveat 保留，不改写为已解决。

## 可复制命令

允许重复执行的 CPU-only preflight：

```bash
bash scripts/run_vopd_2gpu.sh --config configs/vopd_1024.yaml --preflight-only
```

正式训练命令已经冻结，但只有 Task 5 完成、AutoDL 费用刷新且所有 Gate 仍为 PASS 后才允许执行：

```bash
bash scripts/run_vopd_2gpu.sh --config configs/vopd_1024.yaml --run
```

## 证据来源

- `budget`：`artifacts/runs/E-D10-001/preflight/budget_projection.json`，SHA256 `56da1e5269f0d89329e6efba50ce891e8c0f253374adfc4e083f62a6180b093e`
- `data`：`artifacts/runs/E-D10-001/preflight/data_manifest.json`，SHA256 `edfd183340e8cc635512d8bcd137a618247cf27738e736b143f80ee48437cc63`
- `base`：`artifacts/runs/E-D10-001/preflight/base_model_manifest.json`，SHA256 `f8e7f3d922acb63a9dd9d6866d5bf70a4c5c3af1c93bc4546f8b99d179d8a11c`
- `launcher`：`artifacts/runs/E-D10-001/preflight/preflight_summary.json`，SHA256 `fcd9dadebce0fcfff102e2d6394b4895db0b771fd1e9b227798f947666fa7398`
- `readiness`：`artifacts/runs/E-D10-001/preflight/task2_readiness.json`，SHA256 `4d6b26c651a08aaa6f12771aabacb30366473edda5640f946b4a5f14b582821a`
- `config_freeze`：`artifacts/runs/E-D10-001/preflight/task3_config_freeze.json`，SHA256 `c5374f6bee49fc6d97e2e32b3ef0c62f83ac4cc4a412e748448ac5a33d11281c`
- `day8_stability`：`artifacts/runs/E-D8-001/evidence/stability_summary.json`，SHA256 `2f4fad9a3187f835df2295327d51aa4518c80afb07d7c634b86ff475884cde51`

未绘制趋势图：这些证据是离散的单次 readiness Gate，不是连续时间序列；表格能更准确地保留状态、单位和来源。
