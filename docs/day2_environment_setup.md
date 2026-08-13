# Vision-OPD Day 2：环境安装与 CUDA 扩展编译记录

## 1. 目标与结果

### 目标

在 AutoDL 的 RTX PRO 6000 Blackwell Server Edition 实例上，建立可运行 Vision-OPD 的独立 Conda 环境，并完成仓库依赖、`verl`、FlashAttention 和 `causal-conv1d` 的安装验证。

### 结果

环境与普通 VLM 推理均已验证通过。Qwen3.5-4B 已完整下载到数据盘，单卡 vLLM 部署、图片问答以及 Processor 字段和 shape 检查均已完成；当前可以进入 Day 3 的玩具蒸馏与代码映射阶段。

已验证的软件与硬件组合：

| 项目 | 实际值 | 状态 |
| --- | --- | --- |
| Conda 环境 | `vision-opd`，路径 `/root/miniconda3/envs/vision-opd` | 已创建并激活；当前位于系统盘 |
| Python | 3.12.13 | 已通过 |
| PyTorch | `2.10.0+cu128` | 已导入 |
| PyTorch CUDA | 12.8 | 已识别 |
| GPU | 1 × NVIDIA RTX PRO 6000 Blackwell Server Edition | 单卡 PyTorch 与 vLLM 已验证；双卡 NCCL/FSDP 延期至 Day 9 |
| vLLM | 0.18.0 | 已导入并完成单卡服务部署 |
| Transformers | 5.5.0 | 已导入并完成本地 Processor 验证 |
| Ray | 2.53.0 | 已导入 |
| FlashAttention | 2.8.3.post1 | 已导入，按 `sm_120` 编译 |
| causal-conv1d | 1.6.1 | 已导入 |
| 本地项目包 | `verl-0.7.0.dev0` | 已以 editable 模式安装 |
| Qwen3.5-4B | `/root/autodl-tmp/models/Qwen3.5-4B`，约 8.8 GiB | 2 个权重分片完整，单图推理通过 |
| OpenAI-compatible API | `http://127.0.0.1:8000/v1` | 模型查询和图片问答通过 |

实际验证命令等价于：

```python
import torch
import flash_attn
import causal_conv1d
import vllm

print(torch.__version__)          # 2.10.0+cu128
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)         # 12.8
print(flash_attn.__version__)     # 2.8.3.post1
print(vllm.__version__)           # 0.18.0
```

## 2. 执行过程

### 2.1 Conda 环境创建

项目 README 指定 Python 3.12，因此创建独立环境：

```bash
conda create -n vision-opd python=3.12 -y
conda activate vision-opd
```

创建阶段曾访问清华 Conda 镜像失败，错误为：

```text
CondaHTTPError: HTTP 000 CONNECTION FAILED
SSLEOFError: EOF occurred in violation of protocol
```

该问题是到镜像站的 TLS/网络链路中断，可能受代理、VPN 或镜像临时状态影响；不是 Python 或项目依赖错误。环境最终创建成功。

### 2.2 仓库依赖与本地包安装

按 README 顺序执行：

```bash
pip install --upgrade pip
pip install --no-deps -r requirements.txt
pip install -e . --no-deps
```

其中：

- `requirements.txt` 包含完整的 PyTorch 2.10、CUDA 12.8、vLLM、Ray、Transformers 等固定版本，因此下载与安装体积较大；
- `--no-deps` 只是不递归解析依赖，不能避免安装依赖文件中已经逐条列出的 CUDA 运行库；
- `pip install -e . --no-deps` 成功安装了 `verl-0.7.0.dev0`，使仓库源码以可编辑方式被当前环境引用。

安装过程中系统盘从约 20 GB 剩余空间降至约 11 GB。pip 缓存曾占用约 3.5 GB，后续应在确认不需重装后清理缓存。

### 2.3 FlashAttention 编译

README 要求额外安装：

```bash
pip install flash-attn --no-build-isolation
```

首次源码编译耗时很长。监控发现其默认同时生成多个 GPU 架构的 CUDA 内核：`sm_80`、`sm_90`、`sm_100` 和 `sm_120`，且开始时 `ninja -j 1`，导致单架构任务也需要较长时间。

确认本机 GPU 为 Blackwell 后，停止首次构建并改为只编译实际需要的 `sm_120`：

```bash
FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=8 NVCC_THREADS=1 \
pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3.post1
```

注意：`TORCH_CUDA_ARCH_LIST="12.0"` 对该版本的 FlashAttention 不生效；其实际读取的变量是 `FLASH_ATTN_CUDA_ARCHS`。使用以下命令监控编译目标：

```bash
watch -n 2 "ps -efww | grep '[n]vcc' | grep -oE 'arch=compute_[0-9]+,code=sm_[0-9]+' | sort -u"
```

重编译时仅出现：

```text
arch=compute_120,code=sm_120
```

最终结果：

```text
Successfully built flash-attn
Successfully installed flash-attn-2.8.3.post1
```

若后续增加多张同型号 RTX PRO 6000 Blackwell GPU，无需重新编译；该扩展按 GPU 架构编译，而非按 GPU 数量编译。

### 2.4 causal-conv1d 编译

执行：

```bash
pip install --no-cache-dir causal-conv1d==1.6.1 --no-build-isolation
```

该版本的 `causal-conv1d` 构建脚本会固定编译多个 CUDA 架构：`sm_62`、`sm_70`、`sm_72`、`sm_75`、`sm_80`、`sm_87`、`sm_90`、`sm_100`、`sm_120`。检查源码后确认它不使用 `TORCH_CUDA_ARCH_LIST`，因此无法仅通过该环境变量限制为 `sm_120`。

虽然编译过程无终端进度输出，但通过 `ninja -j 8`、`nvcc` 和 `cicc` 的活动状态确认构建正常推进，最终成功：

```text
Successfully built causal-conv1d
Successfully installed causal-conv1d-1.6.1
```

## 3. 遇到的问题与处理结论

| 问题 | 现象 | 处理与结论 |
| --- | --- | --- |
| Conda 镜像 TLS 失败 | `SSLEOFError` / `CondaHTTPError HTTP 000` | 检查代理/VPN，必要时临时改用默认 Conda channel；不关闭 SSL 校验。 |
| 依赖体积大 | 安装 PyTorch/CUDA/vLLM 时占用多 GB | `requirements.txt` 本身锁定完整 CUDA 运行库；使用独立环境，安装成功后清理 pip 缓存。 |
| 系统盘空间有限 | 系统盘总计 30 GB，安装中最低约剩 11 GB | 模型、数据、checkpoint 应放数据盘；安装扩展时使用 `--no-cache-dir`。 |
| FlashAttention 长时间无输出 | `Building wheel ...` 停留很久 | 通过 `nvcc`/`cicc` CPU 使用率确认仍在编译；用 `FLASH_ATTN_CUDA_ARCHS=120` 仅编译 Blackwell 架构。 |
| FlashAttention 架构变量错误 | `TORCH_CUDA_ARCH_LIST` 后仍出现 80/90/100/120 | 改用该项目构建脚本实际读取的 `FLASH_ATTN_CUDA_ARCHS=120`。 |
| causal-conv1d 多架构构建 | 仍出现从 `sm_62` 到 `sm_120` 的编译参数 | 1.6.1 的构建脚本硬编码这些架构，`TORCH_CUDA_ARCH_LIST` 无法覆盖；允许其并行完成。 |
| OpenMP 提示 | `libgomp: Invalid value for environment variable OMP_NUM_THREADS` | 当前仍可复现；每次运行 Python、vLLM 或训练前显式执行 `export OMP_NUM_THREADS=8`。 |

## 4. 当前可执行与待完成事项

现在可以执行：

- 进入 Day 3，完成玩具 JSD、Teacher `no_grad`、EMA 更新和训练代码映射；
- 需要补充推理样本时，继续使用已验证的本地 Qwen3.5-4B 单卡服务；
- Day 3 完成后进入 Day 4 的评测代码审计；
- 完整 Vision-OPD-9B 下载与训练不属于当前 21 天计划主线。

Qwen3.5-4B 已存放在数据盘。后续模型下载不使用 GPU，但需要网络和数据盘容量；模型、Hugging Face 缓存、数据集、rollout 和 checkpoint 应继续统一放在 `/root/autodl-tmp`，不要写入 30 GB 系统盘。

当前 Conda 环境实际位于系统盘的 `/root/miniconda3/envs/vision-opd`，与计划中“Conda 环境也放入数据盘”的目标不一致。为避免破坏已验证环境，本阶段不迁移，只将其作为已知偏差记录；开始准备数据与 checkpoint 前仍需将数据盘从当前 50 GB 扩容到计划的 150～200 GB。

建议的后续清理与检查命令：

```bash
pip cache purge
df -h / /root/autodl-tmp
export OMP_NUM_THREADS=8
```

## 5. 最终安装命令（本机修正后的记录）

以下命令用于新环境复现；已完成安装的当前环境无需重复执行：

```bash
conda create -n vision-opd python=3.12 -y
conda activate vision-opd
pip install --upgrade pip
pip install --no-deps -r requirements.txt
pip install -e . --no-deps

FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=8 NVCC_THREADS=1 \
pip install --no-cache-dir --no-build-isolation flash-attn==2.8.3.post1

pip install --no-cache-dir causal-conv1d==1.6.1 --no-build-isolation
```


## 6. 2026-08-13 每日记录：Qwen3.5-4B 单图推理与 Processor 验收

### 6.1 模型与服务启动

本地模型位置：

~~~text
/root/autodl-tmp/models/Qwen3.5-4B
~~~

模型目录约 8.8 GiB。model.safetensors.index.json 引用的 2 个权重分片均存在，权重总字节数与索引记录一致，且没有残留的 .incomplete 或 .part 文件。

本次仅使用 1 张 NVIDIA RTX PRO 6000 Blackwell Server Edition。实际启动参数：

~~~bash
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0 vllm serve /root/autodl-tmp/models/Qwen3.5-4B --served-model-name Qwen3.5-4B --tensor-parallel-size 1 --gpu-memory-utilization 0.80 --trust-remote-code
~~~

`--trust-remote-code` 是本次实际命令的一部分，但 vLLM 0.18.0 日志明确提示该参数对当前加载路径无效，后续启动可以省略。上述实际运行解析到 `max_model_len=262144`，首次编译与显存分析耗时较长。若仅进行单图 smoke，可使用下面的低成本配置；该命令是后续建议配置，不冒充本次实测参数：

~~~bash
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0 vllm serve /root/autodl-tmp/models/Qwen3.5-4B --served-model-name Qwen3.5-4B --tensor-parallel-size 1 --gpu-memory-utilization 0.50 --max-model-len 8192 --max-num-seqs 8 --enforce-eager
~~~

服务模型查询通过：

~~~bash
curl http://127.0.0.1:8000/v1/models
~~~

返回模型 ID Qwen3.5-4B、模型根目录 /root/autodl-tmp/models/Qwen3.5-4B 和 max_model_len=262144。首次启动包含权重加载、torch.compile、显存分析与 CUDA graph 准备；只有服务就绪后，8000 端口才可连接。

### 6.2 可复现推理设置

| 参数 | 实际值 |
| --- | --- |
| 图片 | figures/average_bar_chart.png |
| 模型 ID | Qwen3.5-4B |
| API Base | http://127.0.0.1:8000/v1 |
| enable_thinking | False |
| temperature | 0 |
| max_tokens | 512 |

请求中的关键采样设置：

~~~python
response = client.chat.completions.create(
    model="Qwen3.5-4B",
    messages=messages,
    temperature=0,
    max_tokens=512,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False,
        }
    },
)
~~~

本次问题：

~~~text
请描述这张图表的主要内容，并指出平均表现最好的模型。
~~~

模型原始回答（未改写）：

~~~text
这张图表是一个横向条形图，展示了多个视觉语言模型（Vision-Language Models）在某个评估任务中的得分。每个条形代表一个模型，其高度对应于该模型的得分，数值标注在条形上方。

---

✅ **图表主要内容：**

- **横轴（X轴）**：列出10个不同的视觉语言模型，包括：
  - Vision-OPD-9B
  - Gemini-3.1-Pro
  - Gemini-3.5-Flash
  - Qwen3.5-397B
  - Vision-OPD-4B
  - Qwen3-VL-235B
  - Kimi-K2.6
  - GPT-5.4
  - GPT-5.2
  - GLM-4.6V

- **纵轴（Y轴）**：表示得分范围从68到80，单位为分（可能是MMLU-VL或其他综合评估指标）。

- **每个条形顶部标注了具体得分**，并配有对应模型的官方Logo。

---

✅ **平均表现最好的模型：**

根据图表数据，**Vision-OPD-9B** 得分最高，为 **79.7分**，是所有模型中表现最好的。

紧随其后的是：
- Gemini-3.1-Pro：79.3
- Gemini-3.5-Flash：79.2

---

📌 **结论：**

该图表通过直观的方式比较了不同视觉语言模型的性能，其中 **Vision-OPD-9B** 以 **79.7分** 的成绩位居榜首，显示出在该评估任务中具有最强的综合表现。
~~~

链路验收结果：模型返回非空中文回答，并正确识别最高分模型 Vision-OPD-9B 及分数 79.7。回答同时存在可记录的轻微问题：对条形方向的描述前后不一致，并在图中没有明确依据时猜测评测指标“可能是 MMLU-VL”。因此本结果证明推理链路可用，但不将单条回答等同于准确率验证。

#### 第二张图片问答测试

图片信息：

| 项目 | 实际值 |
| --- | --- |
| 路径 | `figures/image.png` |
| 格式 | PNG，RGB |
| 分辨率 | `3024 × 4032` |
| 文件大小 | 约 12 MB |
| SHA-256 | `4180595b75c91e0547fe3174e31479e18e0c08df3c23ce6b330427dcd9a818ad` |

该测试沿用同一服务和采样设置：模型 `Qwen3.5-4B`、`enable_thinking=False`、`temperature=0`、`max_tokens=512`。

本次问题：

~~~text
请描述这张图表的主要内容，并指出图中有几粒什么颜色的糖豆。
~~~

模型原始回答（未改写）：

~~~text
这张图片展示了一只手掌中握着五粒彩色糖豆（类似M&M’s或Skittles），背景是浅色木纹桌面和一个粉色盒子。

图中糖豆的颜色分布如下：

- **绿色糖豆**：1粒（位于手掌下方，表面有模糊的黑色标记）
- **橙色糖豆**：2粒（一粒在左下，印有黑色叶子图案；另一粒在中间偏上，无图案）
- **青绿色/蓝绿色糖豆**：2粒（位于手掌右上方，其中一粒印有黑色叶子图案，另一粒表面有模糊标记）

✅ 总结：
- 共有 **5粒糖豆**
- 颜色包括：**绿色、橙色、青绿色**
- 其中 **3粒带有黑色叶子图案**（两粒橙色 + 一粒青绿色）

这些糖豆看起来可能是手工染色或定制款，因为部分表面有手写或印刷的标记，而非标准品牌包装上的字母。
~~~

链路验收结果：模型返回非空回答，并给出了糖豆总数、颜色和逐项计数，用户报告效果正常。回答中仍有两点需要与人工观察区分：品牌、手工染色或定制款属于推测；正文称只有一粒橙色糖豆带叶子图案，但总结括号写成“两粒橙色 + 一粒青绿色”，存在内部不一致。因此该样本计为可复现的第二次推理 PASS，但不把未经人工标注核对的细节作为准确率结论。

### 6.3 Processor 输出字段与 Shape

使用本地 AutoProcessor、同一图片和同一问题执行 CPU 预处理，未加载 4B 模型权重。实测组件：

~~~text
processor_class: Qwen3VLProcessor
image_processor_class: Qwen2VLImageProcessor
tokenizer_class: TokenizersBackend
image_original: (2807, 1426) RGB
~~~

enable_thinking=False 时，chat template 生成的内容为：

~~~text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>请描述这张图表的主要内容，并指出平均表现最好的模型。<|im_end|>
<|im_start|>assistant
<think>

</think>

~~~

Processor 返回字段和实际 tensor 信息：

| 字段 | Shape | Dtype | Device | 作用 |
| --- | --- | --- | --- | --- |
| input_ids | (1, 3987) | torch.int64 | CPU | 文本、特殊 token 和展开后的视觉占位 token |
| attention_mask | (1, 3987) | torch.int64 | CPU | 有效序列位置掩码 |
| mm_token_type_ids | (1, 3987) | torch.int64 | CPU | 区分多模态 token 类型 |
| pixel_values | (15840, 1536) | torch.float32 | CPU | 切块、归一化后的视觉 patch 输入 |
| image_grid_thw | (1, 3) | torch.int64 | CPU | 单张图片的时间、高度和宽度 patch 网格 |

本次 attention_mask 有效 token 数为 3987。原始图片经过 processor 转为视觉 patch，并通过视觉起始、占位和结束 token 对应的位置进入语言模型生成链路。

### 6.4 Day 2 结论

Day 2 验收通过：

- PyTorch、Transformers、vLLM、Ray、FlashAttention 和 causal-conv1d 可用；
- Qwen3.5-4B 权重完整存放在数据盘；
- 单卡 vLLM 服务成功启动，OpenAI-compatible API 可访问；
- 因租卡成本控制，双卡 NCCL/FSDP 验证明确延期至 Day 9；
- Conda 环境仍位于系统盘，该计划偏差已记录，模型与后续大体积产物继续放在数据盘；
- 普通图片问答生成非空回答，关键推理参数已冻结；
- processor、chat template、输入字段、dtype 和 shape 已完成实测记录。

下一步进入 Day 3：实现玩具 JSD、验证 Student 有梯度而 Teacher 使用 no_grad、执行 EMA 更新，并建立论文概念与训练代码位置对照表。
