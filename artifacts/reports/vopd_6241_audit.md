# Vision-OPD 6K 最终 Student 审计报告

> 状态（2026-09-07 UTC）：**PASS / Vision-OPD 定版完成**。Day 13 任务 1～3 已关闭；Cached 契约与 Pilot（任务 4～8）尚未执行。

## 结论

正式训练完成 780/780 steps，最终 `global_step_780` checkpoint、FSDP 合并模型和单卡冷启动均通过。合并模型包含 724 个 BF16 张量；固定 5 条训练样本推理全部非空，推理错误为 0。该 smoke 只证明模型可独立加载和推理，不代表准确率或外部 benchmark 结果。

## 冻结结果

| 项目 | 结果 |
|---|---:|
| 正式训练 guard | PASS，return code 0，latest step 780 |
| drop-last 合同 | 6241 source → 6240 effective，drop 1，padding 0 |
| checkpoint | 13 files，57,034,962,830 bytes，SHA256 独立复核通过 |
| merged Hugging Face 模型 | 7 files，10,370,019,516 bytes，SHA256 独立复核通过 |
| `model.safetensors` | `265a3c49fbee7a44d65cf8af732c4a9b71de5ad9ef799c935c4e38d07094862f` |
| 冷启动推理 | 5/5 非空，0 inference error，均正常 `stop` |
| 源 checkpoint / merged model | 冷加载前后 stat 不变 |

## 关键证据

- 最终冻结凭据：`artifacts/runs/E-D13-6K-VOPD-FINAL-001/evidence/final_student_freeze.json`
- checkpoint 哈希凭据：`artifacts/runs/E-D13-6K-VOPD-FINAL-001/evidence/checkpoint_hash_receipt.json`
- 合并凭据：`artifacts/runs/E-D13-6K-VOPD-FINAL-001/evidence/merge_receipt.json`
- merged SHA256：`artifacts/runs/E-D13-6K-VOPD-FINAL-001/merged_sha256.txt`
- 冷加载凭据：`artifacts/runs/E-D13-6K-VOPD-FINAL-001/cold_reload/reload_validation_summary.json`
- 逐样本输出：`artifacts/runs/E-D13-6K-VOPD-FINAL-001/cold_reload/predictions.jsonl`

## 执行说明

第一次合并尝试因当时 cgroup 内存上限只有 2 GiB，以退出码 137 结束，没有产生或修改模型文件。内存上限提升至 120 GiB 后，官方 FSDP merger 在独立临时目录成功完成，校验通过后原子晋级为 `merged_hf`。失败和成功日志均保留用于审计。

历史 eval-128/retention-64 和外部结果未用于本次定版，也没有据此重选 checkpoint。原始 FSDP checkpoint 仍保留；清理大型 actor/optimizer 分片应在确认后单独执行。
