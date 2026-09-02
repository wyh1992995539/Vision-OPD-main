# Day 10：6241 条新数据环境交接

本文件记录当前可复现状态。正式约束以
`docs/amendments/full_train_6241_scope_amendment.md`、
`configs/project_6241.yaml` 和 `configs/vopd_6241.yaml` 为准。

## 已完成

- 已锁定上游 revision、`train.jsonl` 的 SHA256、字节数和 6241 行记录数。
- 已生成 6241 条候选清单、训练清单和统计；旧 1216 个 sample ID 全部可追溯，缺失数为 0。
- 缓存、临时目录和数据根目录均固定到 `/root/autodl-tmp`。
- 已实现 6241 个真实样本 + 7 个零权重 padding 的 781 步完整覆盖协议及回执。
- 新训练入口默认阻断，只有 Day 11 的长度、重叠、cached-prefix 和 pilot 门禁全部 PASS 后才能启动。
- 旧 `E-D10-001` 启动入口已设为 historical-only，不能误启动旧 1024 实验。

## 当前未完成

全量图片下载被人工中断。当前没有后台下载进程，只有完整的
`raw/train.jsonl`；图片压缩包、选择性解压、图像 QA、
`train_6241.parquet`、prompt 长度审计、cached-prefix 和 pilot 尚未完成。
因此当前数据总门禁是 `INCOMPLETE`，不能开始正式训练。

## 恢复命令

```bash
cd /root/autodl-tmp/Vision-OPD-main
source scripts/activate_vision_opd_6241.sh
conda run --no-capture-output -n vision-opd \
  python scripts/prepare_vision_opd_6241.py --stage all
```

下载支持 Hugging Face 缓存续传。执行前脚本会按“剩余压缩包 + 40 GiB
保守解压预算 + 120 GiB 解压后安全空间”做动态门禁；下载完成后还会逐个核对
7 个压缩文件的字节数与 SHA256。脚本不会自动删除压缩包。

全量数据完成后，继续执行 Day 11 审计并生成
`artifacts/runs/E-D11-6K-GATE-001/preflight.json`。在该门禁和配置状态尚未切换为
`ready_after_day11_gate` 前，下面命令应返回失败且不得使用 GPU：

```bash
conda run --no-capture-output -n vision-opd \
  python scripts/run_vopd_6241_guarded.py --preflight-only
```

## 容量依据

旧 1216 对 Student/Teacher 图像实际占用约 6.11 GB；按样本均值投影，
6241 对约 31.34 GB。完整压缩包固定为 31,283,220,139 字节（约 29.13 GiB）。
脚本使用更保守的 40 GiB 解压预算，并要求结束后至少保留 120 GiB。
