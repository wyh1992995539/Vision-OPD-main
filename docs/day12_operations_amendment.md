# Day12 计费与启动流程修订（2026-09-07）

负责人更新：双卡合计 **14 元/小时**，不用每次监控费用。本修订将费用改为估算记录，
不要求每次提供 AutoDL 累计费用或 15 分钟内账单时间，也不以累计费用检查阻塞启动。
无自动账单查询、周期费用提醒或费用中止。38 小时墙钟上限与 GPU/CPU/磁盘、数值、
心跳、checkpoint 检查继续沿用冻结策略。项目历史 2000 元规划不作为本入口自动门禁。

## 当前入口

在仓库根目录执行静态预检（不会启动 GPU 训练）：

```bash
conda run --no-capture-output -n vision-opd python scripts/run_day12_vopd.py --preflight-only
```

实际正式启动命令为：

```bash
conda run --no-capture-output -n vision-opd python scripts/run_day12_vopd.py --run
```

计费来源是 `configs/day12_operations.yaml`，14 元是两张卡合计，不再乘 2。
启动前仍要求 Git 工作树干净、Day11 全部证据绑定有效、GPU 空闲、CPU 至少 240 GiB、
磁盘至少 120 GiB、输出无冲突。静态预检通过不表示实时资源检查已通过。

## 当前估算

复用自然 EOS 候选冻结的未舍入时长，只更换单价：

| 场景 | 双卡墙钟时间 | 预计增量费用 |
|---|---:|---:|
| 平均规划 | 7.043446635950425 小时 | 98.61 元 |
| 保守预留 | 8.074135051401228 小时 | 113.04 元 |
| 墙钟硬上限 | 38 小时 | 532.00 元 |

费用只在启动预检与结束回执中记录。结束估算为本次 launcher 实测秒数 ÷ 3600 × 14，
不是平台账单，不包含实例空闲时间或其他任务。训练失败退出也保留这份估算。

## 与 Day11 证据的关系

Day11 的历史单价、候选预算、promotion receipt、正式 YAML、abort policy 和原 launcher
已相互绑定 SHA256，因此保持原件。新入口完整调用原静态预检，并使用原训练 Shell 与
原资源监控器；历史预算 PASS 只表示当时证据通过，不表示已检查当前累计账单。

本修订是独立的运行流程变更，不重新声明 Day11 GPU 实验已经验证了新 launcher。
新入口、当前计费配置、Day11 Gate 和运行期监控器 SHA256 写入
`preflight/day12_operations_preflight.json`、`preflight/day12_live_launch_gate.json`
及 `evidence/exit_receipt.json`，使实际入口可追溯。

`scripts/run_vopd_6241_guarded.py` 保留为历史冻结入口；其账单参数要求不适用于上述新入口。
本次不运行训练；提交当前修改后，正式启动仍需重新执行实时资源检查。
