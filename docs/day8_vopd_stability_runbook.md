# Day 8 Vision-OPD 64 条稳定性训练运行手册

实验 ID：`E-D8-001`。本入口用于 64 条、8 个连续 optimizer steps 的工程稳定性 Gate，训练从原始 Base checkpoint 独立启动，不继承 Day 7 checkpoint。

> 收尾状态（2026-08-31 UTC）：**PASS_WITH_CAVEAT**。8/8 steps、最终 checkpoint、5/5 冷重载和 1024 条时间/费用外推均完成。正式报告见 `artifacts/reports/vopd_64_stability.md`，机器结论见 `artifacts/runs/E-D8-001/evidence/stability_summary.json`。Day 8 已关闭，下一步是 Day 9 正式训练 Gate。

## 冻结输入

- 配置：`configs/vopd_day8_64.yaml`
- 来源数据：`/root/autodl-tmp/data/vision_opd_1024/train_1024.parquet`
- 固定子集：`/root/autodl-tmp/data/vision_opd_1024/train_day8_64.parquet`
- 抽样清单：`artifacts/runs/E-D8-001/preflight/day8_64_selection.json`
- 抽样算法：按 `sha256("42|<sample_id>")` 升序选择前 64 条，并保持该顺序
- global batch：8；shuffle：false；epoch：1；optimizer steps：8

固定子集不使用 `retention_64.parquet`。后者仍只用于内部 retention 评测。

## 生成与校验

首次生成固定子集：

```bash
conda run --no-capture-output -n vision-opd \
  python scripts/prepare_vopd_stability_subset.py
```

子集和清单已存在时脚本默认拒绝覆盖。只有在明确重建冻结输入时才使用 `--overwrite`；重建后必须重新审核清单和 SHA256。

无 GPU preflight：

```bash
scripts/run_vopd_2gpu.sh \
  --config configs/vopd_day8_64.yaml \
  --preflight-only
```

preflight 会检查 Base 文件、Parquet 行数/schema、两种图像路径、64 个唯一 sample ID、配置契约，以及清单中的 seed、样本顺序和子集 SHA256。结果写入 `artifacts/runs/E-D8-001/preflight/preflight_summary.json`。

## 启动训练

确认两张 GPU 空闲、磁盘能够保留一个约 54 GiB 的完整 checkpoint 后运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 scripts/run_vopd_2gpu.sh \
  --config configs/vopd_day8_64.yaml \
  --run
```

入口会在 GPU 训练前再次执行 preflight，并写入 `run_invocation.json`，记录 config/data 哈希、64 个 sample ID、Git commit、工作树状态、CUDA 设备和额外 Hydra overrides。日志、rollout、log-prob 和 checkpoint 分别进入 E-D8-001 对应子目录。

额外 Hydra override 会被原样记录，但正式 E-D8-001 不应覆盖 seed、batch、步数、数据顺序、Top-K、JSD、EMA、图像键和保存策略。需要改变这些冻结值时，应创建新实验 ID 和新配置。

## 训练后 Gate

训练完成后仍需执行以下工作，不能仅凭 8/8 steps 判定 Day 8 PASS：

1. 从日志汇总逐步 loss、grad、EMA、响应长度、阶段耗时和峰值显存，并确认无 NaN/Inf。
2. 保存 cgroup/RSS 证据，解释任何 worker `Killed`、Ray/vLLM 重启或 FSDP 警告。
3. 完全关闭训练进程后重新加载最终 checkpoint，对冻结的 5 条样本推理。
4. 只保留一个完整 checkpoint；合并、重载和 SHA256 验证前不得删除唯一分片。
5. 使用 Step 2～8 的稳态统计估算 1024 条（128 steps）时间和费用，生成 `artifacts/reports/vopd_64_stability.md`。

## Checkpoint 冷重载验证

冷重载使用独立配置 `configs/vopd_day8_reload.yaml` 和入口
`scripts/run_vopd_day8_reload.sh`。固定 5 条样本来自 `eval_128.parquet`，按
`sha256("42|<sample_id>")` 升序选择；样本 ID 已显式写入配置，不能临时替换。

无需 GPU 的静态 preflight：

```bash
scripts/run_vopd_day8_reload.sh --preflight-only
```

静态 preflight 检查 step 8、world size 2、两个 rank 的 model/optimizer/extra-state、
Hugging Face metadata、eval SHA256、5 个样本及图像、独立合并目录和磁盘空间。
当前无 GPU 或 CPU cgroup 内存不足只会形成预检警告；`--run` 会执行严格资源 Gate。

GPU 和至少 48 GiB CPU/cgroup 内存可用后，一键执行合并、冷启动服务、5 条推理和结果验证：

```bash
CUDA_VISIBLE_DEVICES=0,1 scripts/run_vopd_day8_reload.sh --run
```

脚本将模型合并到 `artifacts/runs/E-D8-001/merged_hf/`，不会调用会原地删除文件的
`scripts/merge_checkpoint.sh`，也不会修改 `global_step_8`。随后脚本启动自己的 vLLM
进程，等待 `/v1/models` 就绪，完成 5 条推理后只终止该进程组。

若合并已经成功、但服务或推理阶段失败，可在核对 `merged_manifest.json` 后复用合并模型：

```bash
CUDA_VISIBLE_DEVICES=0,1 scripts/run_vopd_day8_reload.sh --run --reuse-merged
```

若 `reload_5` 已有失败或历史预测，需要明确允许替换结果：

```bash
CUDA_VISIBLE_DEVICES=0,1 scripts/run_vopd_day8_reload.sh \
  --run --reuse-merged --overwrite-results
```

最终 PASS 要求：原始 checkpoint 的文件大小与 mtime 未变化、合并模型有完整哈希、
5 个 sample ID 和顺序完全一致、5 条输出非空、response token 为正、推理错误为 0。
最终机器可读结论写入
`artifacts/runs/E-D8-001/evidence/reload/reload_validation_summary.json`。

## 收尾报告复算

以下命令只读取已归档日志和冷重载证据，不使用 GPU：

```bash
python scripts/finalize_day8_stability.py
```

它会复算逐步稳定性、Teacher/EMA 契约、checkpoint 完整性和 1024 条三场景外推，并生成 `metrics.jsonl`、`cost.json`、`checkpoint_sha256.txt`、机器摘要与正式报告。若缺失 Step 1～8、必需 checkpoint 文件或冷重载 PASS，收尾器会失败而不会把 Day 8 标记为完成。
