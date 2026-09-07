# Day 14 Cached Prefix 6241 正式训练运行手册

> 实验 ID：`E-D14-6K-CACHED-001`  
> 当前配置状态：`ready_after_day13_gate`  
> 正式配置授权：`formal_training_authorized=true`  
> 当前训练状态：`training_started=false`  
> 唯一正式入口：`scripts/run_day14_cached.py`

## 1. 已关闭的启动缺口

1. `configs/cached_prefix_6241.yaml` 已由 Day 13 技术资格凭据晋级，晋级后 SHA256 为 `247dedba3eca82ff11df516dd27fdff45c7d1cb5a4666ac89af3c2d9b54fe1bb`。
2. `configs/cached_prefix_6241_abort_policy.yaml` 已冻结 780-step Cached 正式训练的资源、指标、磁盘、checkpoint 和中止规则。
3. `scripts/run_day14_cached.py` 已实现静态 preflight、实时资源 Gate、启动授权、自动遥测、中止控制和最终 checkpoint 检查。

晋级凭据位于：

```text
artifacts/runs/E-D13-6K-CACHED-PILOT-001/formal_promotion_v1/promotion_receipt.json
```

该凭据绑定候选配置、Day 13 资格与完成凭据、正式策略、launcher、训练 preflight、运行监控器和底层训练脚本。任一绑定文件改变后，正式 preflight 会失败。

## 2. 强制启动流程

当前低资源实例只有 2 GiB cgroup 时，不执行全量静态 preflight。开卡并获得两张 GPU、至少 240 GiB cgroup 后，先运行：

```bash
cd /root/autodl-tmp/Vision-OPD-main
/root/miniconda3/envs/vision-opd/bin/python scripts/run_day14_cached.py --preflight-only
```

必须看到：

```text
"status": "PASS"
"training_started": false
No GPU training started.
```

随后使用 tmux 启动：

```bash
cd /root/autodl-tmp/Vision-OPD-main
mkdir -p artifacts/runs/E-D14-6K-CACHED-001/preflight
tmux new-session -d -s day14-cached \
  "bash -lc 'set -o pipefail; /root/miniconda3/envs/vision-opd/bin/python scripts/run_day14_cached.py --run 2>&1 | tee artifacts/runs/E-D14-6K-CACHED-001/preflight/day14_launcher_console.log; code=\${PIPESTATUS[0]}; echo \$code > artifacts/runs/E-D14-6K-CACHED-001/preflight/day14_launcher_exit_code.txt; exit \$code'"
```

禁止使用：

```bash
bash scripts/run_cached_prefix_2gpu.sh --run
```

底层入口没有 `VOPD_GUARD_ACTIVE=1` 时会以退出码 2 拒绝启动。

## 3. 静态与实时 Gate

静态 preflight 必须确认：

- promotion receipt 当前有效，配置与策略 SHA256 匹配。
- `prefix_source=cached`，从冻结原始 Base 冷启动，`resume_mode=disable`。
- 6,241 条源数据、6,240 条有效训练、padding 0、dropped 1、780 steps。
- Cache 6,241/6,241、Base/模板/采样/Token 合同有效。
- step 390 和 step 780 checkpoint 计划与策略一致。
- 费用按双卡 14 元/小时只作估算，不要求账单时间或累计费用 Gate。

实时 Gate 必须确认：

- 正好两张 GPU，启动前每张显存占用不超过 10%。
- cgroup 上限至少 240 GiB。
- 数据盘可用空间至少 130 GiB。
- 输出目录没有既有 logs、rollouts、checkpoints 或 evidence 冲突。
- 本地 8000 端口没有残留 vLLM 服务。

任一检查失败时返回退出码 41，`training_started=false`。

## 4. 自动中止条件

| 条件 | 动作 |
|---|---|
| Cached 单步解析记录不是 8 | 立即中止 |
| online generation calls 大于 0 或缺失 | 立即中止 |
| NaN/Inf | 立即中止 |
| Teacher 出现直接梯度或 optimizer delta 非 0 | 立即中止 |
| Student 或 EMA 连续 2 个有效 step 不更新 | 中止 |
| generation error 连续 2 steps | 中止 |
| 任一 GPU 显存连续 3 次达到 98% | 中止 |
| cgroup 内存连续 3 次达到 95% | 中止 |
| cgroup OOM/oom_kill 增加 | 立即中止 |
| 遥测连续失败 3 次 | 中止 |
| 日志在宽限后静默 15 分钟 | 中止 |
| 磁盘跌破 checkpoint reserve | 中止 |
| 墙钟达到 12 小时 | 中止 |

Pilot 曾出现 GPU 1 单点 99.25% 显存峰值，因此正式训练仍采用“98% 连续 3 次”规则：单点尖峰保留证据，持续压力自动中止。

## 5. 启动后的检查与监控

查看 tmux：

```bash
tmux attach -t day14-cached
```

退出查看但保持训练：按 `Ctrl-b`，再按 `d`。

查看训练日志：

```bash
tail -f artifacts/runs/E-D14-6K-CACHED-001/logs/train.log
```

查看守护器提取的最新 step：

```bash
tail -n 1 artifacts/runs/E-D14-6K-CACHED-001/evidence/runtime_metrics.jsonl
```

查看 GPU：

```bash
watch -n 5 nvidia-smi
```

前 3 steps 必须看到：

- `cached_prefix/resolved_records=8`
- `cached_prefix/online_generation_calls=0`
- loss、grad norm、LR 和参数 delta 为有限值
- 正学习率 step 的 Student delta 大于 0
- Teacher optimizer delta=0、direct gradients=0
- Teacher EMA delta 大于 0且 applied=1
- 没有 prompt truncation、response abort、OOM 或 guard trigger

自动遥测位于：

```text
artifacts/runs/E-D14-6K-CACHED-001/evidence/telemetry/
```

## 6. 正常完成标准

- 训练进度达到 780/780。
- `latest_checkpointed_iteration.txt=780`。
- `checkpoints/global_step_780/` 存在且 13 个必需文件全部非空。
- guard status=PASS、return code=0、trigger=null。
- runtime metrics 中每个 step 都为 cache resolved 8、online calls 0。
- `exit_receipt.json` 保存墙钟和估算费用。

Day 14 完成后再进入 Day 15：生成最终 checkpoint 清单与 SHA256、FSDP merge、5 条冷加载及统一 R3 外部评测。

## 7. 当前资源状态

本运行手册生成时，实例 cgroup 上限为 2 GiB，GPU 接口未开放，因此只完成晋级、轻量规则验证和 raw-entry 拦截验证，没有执行全量 6,241 条静态 preflight，也没有启动训练。开卡后必须重新运行本手册第 2 节的正式 preflight。
