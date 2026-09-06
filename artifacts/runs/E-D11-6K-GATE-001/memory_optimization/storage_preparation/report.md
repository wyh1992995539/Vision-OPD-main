# A/B 磁盘盘点与迁移准备

状态：盘点完成；尚无已确认的私有迁移目标。未复制、移动、删除任何训练文件，未开 GPU。

## 当前空间

训练盘总容量 300.00 GiB，可用 122.54 GiB。
系统盘与 /tmp 属于同一个 overlay，可用 9.70 GiB；不能把两者相加。
/autodl-pub 是只读公共盘，不能作为备份目标；tmpfs 不是持久化磁盘。

| 保留原位的对象 | 已分配 GiB |
| --- | ---: |
| /root/autodl-tmp/Vision-OPD-main/artifacts/runs/E-D11-6K-GATE-001/pilot/16/checkpoints | 53.12 |
| /root/autodl-tmp/Vision-OPD-main/artifacts/runs/E-D11-6K-GATE-001/pilot/64/checkpoints | 53.12 |
| /root/autodl-tmp/Vision-OPD-main/artifacts/runs/E-D11-6K-GATE-001/pilot/64/cold_reload/merged_hf | 9.66 |

完整目录占用、文件大小、inode、mtime 和挂载信息见 plan.json。
大张量未做全量内容哈希，这不是备份完成或 checkpoint 完整性认证。

## 两组顺序运行的容量计算

- 每组 checkpoint 按 53.12 GiB 估算；启动下限 120.00 GiB 不变。
- 第二组启动前已保留第一组 checkpoint，因此初始至少需 173.12 GiB 空闲。
- 当前最少还差 50.57 GiB；另留 20 GiB 规划余量后，建议增加至少 70.57 GiB。
- 这是空间规划，不是放行；日志、临时文件、checkpoint 大小变化仍需现场检查。

## 优先方案：扩容原训练盘

不改输出路径，不移动历史证据。按当前占用，300 GiB 扩到 400 GiB 可覆盖上述建议值；
这是容量目标，并非平台套餐或配额已获批准。调整后用 `df -B1 /root/autodl-tmp` 核实实际可用空间。

## 备选方案：归档未来 baseline 的 checkpoint

先由用户提供私有、持久化目标路径，并核实其空间和独立配额。
目标至少需约 73.12 GiB 空闲（含 20 GiB 余量）。
baseline 完成后才会有这个源目录；当前不迁移现有 Pilot-16/Pilot-64 或 merged HF。

迁移顺序：停止写入 → 源文件清单及全量 SHA256 → 复制到全新 staging 目录 →
核对目标文件集合/大小/SHA256、源未变更 → 写归档及恢复凭据 → 经另行授权才移除源副本 → 重新检查启动门槛。
仅复制不释放源盘空间；同盘改目录也不释放空间。不得通过删除唯一副本或降低门槛来放行。
现有审计可能引用原始绝对路径；不能擅自改历史 JSON 或假设软链接一定兼容，迁移前须验证恢复/审计流程。

## 下一步需要的选择

扩容当前训练盘，或提供已挂载的私有存储绝对路径。未确定前不执行迁移。
