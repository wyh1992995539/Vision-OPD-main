# Day16 MMStar Base 原始输出上限修复简报

> 主实验：`E-MMSTAR-BASE-MAXTOK-EXT-001`  
> 收口实验：`E-MMSTAR-BASE-MAXTOK-CLOSURE-001`  
> 最终状态：**COMPLETE，输出上限导致的评测歧义已归零**  
> 最终 R4 + 冻结本地 Base Judge：**1,091/1,500 = 72.73%**

## 任务与约束

本次只重跑原始 R4 结果中 `finish_reason=length` 的 MMStar 样本，并在每一档仅继续处理上一档仍截断的样本。模型权重、MMStar 数据快照、Prompt、图像、`temperature=0`、non-thinking、R4 解析器和冻结 Qwen3.5-4B Base Judge保持不变。原始 R3、R4 和 2,048/4,096 诊断产物均通过 SHA256 锁定且未被覆盖。

这是一条**选择性自适应混合评测线**，不是用同一上限重跑全部 1,500 条；其用途是消除既有 1,024-token 输出中的截断歧义。

## 自适应结果

| 最终尝试上限 | 本档请求数 | 本档自然结束 | 本档后仍为 `length` | 混合总正确 | 混合准确率 |
|---:|---:|---:|---:|---:|---:|
| 1,024（原始 R4） | 1,500 | — | 163 | 1,020 | 68.00% |
| 2,048 | 163 | 80 | 83 | 1,065 | 71.00% |
| 4,096 | 83 | 37 | 46 | 1,072 | 71.47% |
| 8,192 | 46 | 25 | 21 | 1,086 | 72.40% |
| 16,384 | 21 | 14 | 7 | 1,089 | 72.60% |
| 32,768 | 7 | 4 | 3 | 1,091 | 72.73% |
| 65,536 | 3 | 2 | 1 | 1,091 | 72.73% |
| 131,072 | 1 | 0 | 1 | 1,091 | 72.73% |

原始 163 条截断样本中，162 条最终自然停止。提高上限带来净增 71 个正确样本，使 R4 从 68.00% 提升到 72.73%，但仍低于论文 Base 的 78.53%，差 5.80 pp。因此，1,024-token 上限是已证实的重要误差来源，但不能单独解释论文与本地结果的全部差距。

## 最后一条为什么不再继续抬高上限

`mmstar:source_id:998` 从 1,024、2,048、4,096、8,192、16,384、32,768、65,536 一直到 131,072，每次都精确以 `finish_reason=length` 结束。在 131,072 档：

- 原始回答约 353,172 个字符；
- 8,023 个规范化段落中只有 114 个唯一段落；
- 重复段落实例占 98.58%；
- 同一句最多精确重复 365 次；
- R4 没有解析到最终选项；
- 冻结 Judge 严格返回 `No`；
- 参考答案为 B，该样本保持计错。

这已经不是“合理答案尚差少量 tokens 才完成”，而是模型在候选词和推理路径之间循环。继续把有限上限提高到 262K 只会延长同一退化输出，无法保证自然结束，也会把模型错误误写成评测配置错误。

收口视图因此保留原始 `finish_reason=length` 作为证据，将该样本标注为 `terminal_generation_degeneracy`，保持错误评分，并把 `unresolved_output_cap_count` 置为 0。这里的“完成”表示所有样本均已有可审计的最终处置，不表示 1,500 条都由模型主动输出了停止符。

## Judge 与稳定性检查

8,192 阶段出现过 1 次 Judge 未严格输出 Yes/No；该样本随后在 16,384 阶段自然停止并重新评分，最终混合结果中：

- Judge error：0；
- inference error：0；
- 待评分：0；
- 样本数及唯一 `sample_uid`：1,500；
- 正确数：1,091；
- 收口前后分数变化：0。

另有一个可复现性限制：`temperature=0` 不能保证不同并发批次的字节级输出相同。一个样本在并发为 2 的 32,768 档触顶，随后在并发为 1 的 65,536 档于 19,968 tokens 自然停止。这与 GPU 数值路径和调度变化一致，因此本实验保留了每一档的完整选择轨迹，不能只保存最终答案。

## 结论

原始输出上限问题已经按可审计方式完成修复：162 条通过加长输出自然完成，1 条被证明是生成循环并作为模型错误终止处置。最终可信的本地内部结果是 **72.73%**，比原始 R4 高 **4.73 pp**，仍不能复现论文 78.53%。后续最高价值工作仍是对齐论文 Judge、checkpoint revision、服务运行时和论文实际使用的评测 commit，而不是继续为单个循环样本无限提高 token 上限。

本轮 GPU 推理已结束，vLLM 服务和 tmux 会话均已退出，可以关闭 GPU。

## 凭据

- 自适应配置：`configs/mmstar_base_adaptive_max_tokens.yaml`
- 自适应运行器：`eval/run_mmstar_base_adaptive_max_tokens.py`
- 自适应完成凭据：`artifacts/runs/E-MMSTAR-BASE-MAXTOK-EXT-001/completion_receipt.json`
- 最终混合结果：`artifacts/runs/E-MMSTAR-BASE-MAXTOK-EXT-001/final_hybrid/`
- 收口配置：`configs/mmstar_base_output_cap_terminal_closure.yaml`
- 收口运行器：`eval/finalize_mmstar_base_output_cap_repair.py`
- 收口完成凭据：`artifacts/runs/E-MMSTAR-BASE-MAXTOK-EXT-001/terminal_closure/completion_receipt.json`
- 退化证据：`artifacts/runs/E-MMSTAR-BASE-MAXTOK-EXT-001/terminal_closure/terminal_evidence.json`
- SHA256 清单：`artifacts/runs/E-MMSTAR-BASE-MAXTOK-EXT-001/terminal_closure/artifact_sha256.txt`

## 后续统一协议冻结

三模型后续统一使用 `BENCHMARK-ADAPTIVE-OUTPUT-CAP-V1`。规则、模型身份、当前 R4 源文件哈希、逐级上限、并发、退化阈值和报告字段已经冻结在 `configs/benchmark_adaptive_output_cap_protocol_v1.yaml`。Vision-OPD 与 Cached 本轮状态均为 `DEFERRED_BY_USER_NOT_STARTED`，没有启动新评测。协议说明见 `docs/benchmark_adaptive_output_cap_protocol_v1.md`。
