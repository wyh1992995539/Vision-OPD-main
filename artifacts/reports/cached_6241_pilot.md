# Cached Prefix 6,241 Pilot

- 当前状态：**BLOCKED_BY_LIVE_RESOURCES**
- 静态契约：**STATIC_CONTRACT_PASS**
- 训练是否启动：`false`
- `STRICT_PREFIX_SOURCE_ABLATION`：**PENDING_TWO_GPU_PILOT**
- Cached 正式训练授权：`false`

已完成 6,241/6,241 条 sample ID、prompt SHA256、Student 原图、Base tokenizer response decode→encode、EOS、1,024-token 上限和 cache SHA256 检查。online 与 cached 的数据、样本顺序、batch、steps、学习率、rollout sampling、VOPD loss、Crop Teacher、Top-K JSD、EMA 和资源配置静态一致；定向回归验证 online 仍为默认路径，cached dispatch 的在线生成调用为 0。

实时 Gate 于 2026-09-07 UTC 执行。磁盘、输出目录、GPU 空闲、端口和静态凭据通过；实例只暴露 1 张 RTX PRO 6000，cgroup 内存上限为 120 GiB，未达到冻结的 2 GPU 与 224 GiB 保守启动线，因此没有创建训练进程。磁盘可用约 297.6 GiB，高于 120 GiB 启动线。

资源满足后运行：

```bash
cd /root/autodl-tmp/Vision-OPD-main
/root/miniconda3/envs/vision-opd/bin/python scripts/run_cached_prefix_pilot.py --run
```

该入口按双卡合计 14 元/小时记录估算，不要求累计账单或账单时间；它会自动执行 8-step 训练、运行熔断、`global_step_8` checkpoint、固定 5 条冷加载和 Cached 专用 postflight。只有全部通过才会记录 `STRICT_PREFIX_SOURCE_ABLATION=PASS`。
