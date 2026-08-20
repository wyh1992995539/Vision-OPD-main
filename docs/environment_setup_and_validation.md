# Vision-OPD 环境安装与推理验收记录

> 原执行日期：2026-08-12～2026-08-13
> 文档定位：更新后项目计划的已完成前置工作，不对应新计划中的 Day 2 数据抽样任务。

## 1. 验收结论

已在单张 NVIDIA RTX PRO 6000 Blackwell Server Edition 上完成独立环境安装、CUDA 扩展编译、Qwen3.5-4B 下载、vLLM 部署、普通图片问答和 Processor 输入检查。该记录证明软件环境与基础多模态推理链路可用，不证明双卡训练、Vision-OPD 完整链路或模型效果提升。

| 项目 | 实际值 | 结果 |
|---|---|---|
| Conda 环境 | /root/miniconda3/envs/vision-opd | PASS；位于系统盘 |
| Python | 3.12.13 | PASS |
| PyTorch | 2.10.0+cu128 | PASS |
| PyTorch CUDA | 12.8 | PASS |
| 历史验证 GPU | 1 × RTX PRO 6000 Blackwell，96GB | 单卡 PASS |
| Transformers | 5.5.0 | PASS |
| vLLM | 0.18.0 | PASS；单卡服务启动成功 |
| Ray | 2.53.0 | PASS |
| FlashAttention | 2.8.3.post1 | PASS；按 sm_120 编译 |
| causal-conv1d | 1.6.1 | PASS |
| verl | 0.7.0.dev0 | PASS；editable 安装 |
| Qwen3.5-4B | /root/autodl-tmp/models/Qwen3.5-4B | PASS；约 8.8 GiB |

双卡 GPU、NCCL 和 FSDP 尚未由本记录验证，必须在任何 GPU Smoke 或正式训练前单独完成。

## 2. 安装摘要

基础环境与项目包：

```bash
conda create -n vision-opd python=3.12 -y
conda activate vision-opd
pip install --upgrade pip
pip install --no-deps -r requirements.txt
pip install -e . --no-deps
```

Blackwell 对应的 CUDA 扩展：

```bash
FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=8 NVCC_THREADS=1 \
pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3.post1

pip install --no-cache-dir --no-build-isolation causal-conv1d==1.6.1
```

关键经验：

- FlashAttention 2.8.3.post1 使用 FLASH_ATTN_CUDA_ARCHS 控制编译架构；本机只需 sm_120。
- TORCH_CUDA_ARCH_LIST 对该版本 FlashAttention 的构建路径无效。
- causal-conv1d 1.6.1 会编译多个预设架构，不能仅靠 TORCH_CUDA_ARCH_LIST 限制。
- 编译过程长时间无输出时，可检查 ninja、nvcc 和 cicc 进程确认是否仍在运行。

## 3. 模型与服务验证

模型目录包含两个 Safetensors 权重分片，索引引用完整，无 .incomplete 或 .part 残留文件。

历史实测启动命令：

```bash
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0 vllm serve /root/autodl-tmp/models/Qwen3.5-4B \
  --served-model-name Qwen3.5-4B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80
```

普通图片 Smoke 可使用更保守的建议参数：

```bash
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0 vllm serve /root/autodl-tmp/models/Qwen3.5-4B \
  --served-model-name Qwen3.5-4B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --enforce-eager
```

历史推理设置：

| 参数 | 值 |
|---|---|
| enable_thinking | false |
| temperature | 0 |
| max_tokens | 512 |
| API | http://127.0.0.1:8000/v1 |

两次图片问答均返回非空中文结果：

- 图表样本正确识别最高分模型 Vision-OPD-9B 和分数 79.7，但出现无依据的指标猜测。
- 糖豆样本给出总数、颜色和逐项计数，但部分图案描述前后不一致。

因此仅将两次结果记为“推理链路 PASS”，不作为准确率或模型能力结论。

## 4. Processor 验证

实测组件：

| 组件 | 实际值 |
|---|---|
| Processor | Qwen3VLProcessor |
| Image Processor | Qwen2VLImageProcessor |
| Tokenizer | TokenizersBackend |
| 原图 | 2807 × 1426，RGB |

单图输入的关键字段：

| 字段 | Shape | Dtype | 作用 |
|---|---|---|---|
| input_ids | 1 × 3987 | int64 | 文本、特殊 Token 和视觉占位 |
| attention_mask | 1 × 3987 | int64 | 有效位置掩码 |
| mm_token_type_ids | 1 × 3987 | int64 | 多模态 Token 类型 |
| pixel_values | 15840 × 1536 | float32 | 视觉 Patch |
| image_grid_thw | 1 × 3 | int64 | 图像 Patch 网格 |

该结果确认图像经过 Processor 后形成视觉 Patch，并通过视觉特殊 Token 进入模型输入链路。

## 5. 已知问题与处理

| 问题 | 结论 |
|---|---|
| Conda 镜像 TLS 失败 | 属于网络链路问题；改用可用 channel，不关闭 SSL 校验 |
| 系统盘空间有限 | 环境暂不迁移；模型、数据、缓存和 checkpoint 放入 /root/autodl-tmp |
| OpenMP 提示 | 运行 Python、vLLM 或训练前设置 OMP_NUM_THREADS=8 |
| 依赖元数据冲突 | 核心导入和历史推理通过；保留到真实 Smoke 检验，不盲目重装 |
| 双卡未验证 | 状态为 DEFERRED；首次 GPU Smoke 前必须完成 |

缓存与临时目录应指向数据盘：

```bash
export HF_HOME=/root/autodl-tmp/hf_cache
export TORCH_HOME=/root/autodl-tmp/torch_cache
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export TMPDIR=/root/autodl-tmp/tmp
export OMP_NUM_THREADS=8
```

## 6. 当前有效边界

- 环境安装、模型完整性、单卡服务、普通图像推理和 Processor 字段检查已完成。
- 当前项目使用 Qwen3.5-4B，不下载或训练 Vision-OPD-9B。
- 新计划的数据方案为本地抽取冻结子集后上传，服务器禁止下载完整原始数据。
- 环境详细快照、依赖检查、磁盘信息和模型 SHA256 以 artifacts/runs/preflight 为准。
- 下一阶段是新计划 Day 2 的元数据审计与确定性抽样，不再进行与项目无关的玩具 Tensor 实验。
