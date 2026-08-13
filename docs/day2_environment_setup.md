# Vision-OPD Day 2：环境安装与 CUDA 扩展编译记录

## 1. 目标与结果

### 目标

在 AutoDL 的 RTX PRO 6000 Blackwell Server Edition 实例上，建立可运行 Vision-OPD 的独立 Conda 环境，并完成仓库依赖、`verl`、FlashAttention 和 `causal-conv1d` 的安装验证。

### 结果

环境已安装并验证通过。当前可进入模型下载、vLLM 部署、推理与后续训练冒烟阶段。

已验证的软件与硬件组合：

| 项目 | 实际值 | 状态 |
| --- | --- | --- |
| Conda 环境 | `vision-opd` | 已创建并激活 |
| Python | 3.12.13 | 已通过 |
| PyTorch | `2.10.0+cu128` | 已导入 |
| PyTorch CUDA | 12.8 | 已识别 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition | 已被 PyTorch 识别 |
| vLLM | 0.18.0 | 已导入 |
| FlashAttention | 2.8.3.post1 | 已导入，按 `sm_120` 编译 |
| causal-conv1d | 1.6.1 | 已导入 |
| 本地项目包 | `verl-0.7.0.dev0` | 已以 editable 模式安装 |

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
| OpenMP 提示 | `libgomp: Invalid value for environment variable OMP_NUM_THREADS` | 不影响 CUDA 与依赖导入；后续可运行 `unset OMP_NUM_THREADS` 或设为合法整数，例如 `export OMP_NUM_THREADS=8`。 |

## 4. 当前可执行与待完成事项

现在可以执行：

- 下载 Vision-OPD-9B 模型；
- 使用 vLLM 启动模型服务；
- 进行单张图片推理和评测链路冒烟；
- 按项目脚本准备数据并进行小规模训练验证。

模型下载不使用 GPU，但需要网络和数据盘容量。模型、Hugging Face 缓存、数据集、rollout 和 checkpoint 应统一放在 `/root/autodl-tmp`，不要存入 30 GB 系统盘。

建议的后续清理与检查命令：

```bash
pip cache purge
df -h /
unset OMP_NUM_THREADS
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

