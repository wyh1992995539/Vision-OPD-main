# Vision-OPD 环境记录

> 采集时间：2026-08-12 10:54:15 +08:00（2026-08-12 02:54:15 UTC）
> 采集原则：仅运行只读查询；未安装依赖、未下载模型或数据、未启动训练。

## 代码版本

| 项目 | 实际值 |
| --- | --- |
| 项目目录 | `/root/autodl-tmp/Vision-OPD`（存在） |
| Git commit | `c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471` |
| Git 分支 | `main` |
| 采集时工作树 | clean |

说明：以上 commit 是采集环境时的基线 commit；本文件及实验登记表将在后续 commit 中加入。

## 操作系统与计算资源

| 项目 | 实际值 |
| --- | --- |
| Linux | Ubuntu 22.04.5 LTS (Jammy Jellyfish) |
| Kernel | `Linux 5.15.0-78-generic x86_64` |
| CPU | Intel(R) Xeon(R) Platinum 8470Q；宿主拓扑可见 2 sockets / 104 cores / 208 threads |
| 容器 CPU 配额 | `cpu.max=50000 100000`（约 0.5 CPU）；`nproc=1` |
| 内存 | `free -h` 可见宿主总内存 1.0 TiB、可用 956 GiB；容器 `memory.max=2147483648`（2 GiB） |
| Swap | 0 B |

容器配额才是当前进程实际可用资源上限；宿主拓扑和 `free` 输出不能直接视为训练可用量。

## Python、Conda 与 PyTorch

| 项目 | 实际值 |
| --- | --- |
| Python 路径 | `/root/miniconda3/bin/python` |
| Python 版本 | 3.12.3（Anaconda；GCC 11.2.0） |
| Conda 路径 | `/root/miniconda3/bin/conda` |
| Conda 版本 | 24.4.0 |
| Conda 环境 | 仅发现 `base`：`/root/miniconda3`；当前 shell `active_prefix=None`，没有激活命名环境 |
| PyTorch | `2.8.0+cu128` |
| `torch.version.cuda` | `12.8` |
| `torch.cuda.is_available()` | `False` |

当前 Python 来自 Conda `base` 路径，但 shell 中没有激活 Conda 环境。该环境与仓库 `requirements.txt` 声明的版本并不一致，不能据此启动项目训练。

## GPU、驱动与 CUDA 工具链

| 项目 | 实际值 |
| --- | --- |
| PyTorch 可见 GPU 数 | 0 |
| GPU 型号 | 不可获取（当前没有 PyTorch 可见 GPU） |
| 单卡显存 | 不可获取（当前没有 PyTorch 可见 GPU） |
| CUDA 是否可用 | 否 |
| NVIDIA 驱动 | 580.95.05；来自 `/proc/driver/nvidia/version` |
| `nvidia-smi` | 不可执行：`/usr/bin/nvidia-smi: Permission denied`；该文件为 0 字节且无执行权限 |
| `nvcc` | 未找到：`nvcc: command not found` |
| PyTorch 编译 CUDA 版本 | 12.8（不代表运行时 CUDA/GPU 可用） |

因此，本记录不能给出 GPU 型号或显存；在获得可见 GPU 且 `nvidia-smi` 可执行后，应重新采集本节。

## 磁盘

采集命令为 `df -hT / /root/autodl-tmp /root/autodl-tmp/Vision-OPD`。

| 用途 | 文件系统 | 类型 | 容量 | 已用 | 可用 | 使用率 | 挂载点 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 系统盘 | `overlay` | overlay | 30 GiB | 2.6 GiB | 28 GiB | 9% | `/` |
| 数据盘 / 项目盘 | `/dev/md0` | xfs | 50 GiB | 9.4 MiB | 50 GiB | 1% | `/root/autodl-tmp` |

## 关键依赖

版本通过当前 Python 的 `importlib.metadata` 读取；“项目声明”来自 `requirements.txt`，未安装项没有被补装。

| 依赖 | 当前安装版本 | 项目声明/说明 |
| --- | --- | --- |
| torch | 2.8.0+cu128 | 2.10.0 |
| transformers | 未安装 | 5.5.0 |
| vllm | 未安装 | 0.18.0 |
| ray | 未安装 | 2.53.0 |
| flash-attn | 未安装 | README 要求单独安装，`requirements.txt` 未固定版本 |
| accelerate | 未安装 | 1.12.0 |
| datasets | 未安装 | 4.4.2 |
| deepspeed | 未安装 | `requirements.txt` 未声明 |
| peft | 未安装 | 0.18.0 |
| triton | 3.4.0 | 3.6.0 |
| xformers | 未安装 | 0.0.32.post1 |
| numpy | 2.3.2 | 1.26.4 |
| verl | 未作为 distribution 安装 | 仓库内含 `verl/` 源码 |
| pip | 24.0 | 当前 Python 的 pip |

## 缓存与临时目录

| 环境变量 | 显式值 | 当前生效/默认路径 |
| --- | --- | --- |
| `HF_HOME` | 未设置 | 默认约定为 `/root/.cache/huggingface`；目录当前不存在 |
| `TORCH_HOME` | 未设置 | `torch.hub.get_dir()` 为 `/root/.cache/torch/hub`；目录当前不存在 |
| `PIP_CACHE_DIR` | 未设置 | `python -m pip cache dir` 为 `/root/.cache/pip`；目录当前不存在 |
| `TMPDIR` | 未设置 | `tempfile.gettempdir()` 为 `/tmp`；目录存在 |

## 项目、数据、模型与产物目录

| 类型 | 实际目录/配置 | 当前状态 |
| --- | --- | --- |
| 项目 | `/root/autodl-tmp/Vision-OPD` | 存在 |
| 训练数据 | `/root/autodl-tmp/Vision-OPD/data` | 未创建；README 和脚本的默认目录 |
| 训练文件 | `/root/autodl-tmp/Vision-OPD/data/train.parquet` | 不存在 |
| 模型 | 无本地模型目录；`scripts/run_vision_opd.sh` 当前配置为 Hugging Face 标识 `Qwen/Qwen3.5-4B` | 未下载、未解析到本地目录 |
| Checkpoint | `/root/autodl-tmp/Vision-OPD/checkpoints` | 未创建；脚本默认在其下写入实验目录 |
| 默认训练 checkpoint | `/root/autodl-tmp/Vision-OPD/checkpoints/Vision-OPD-Qwen3.5-4B` | 不存在 |
| Rollout 产物 | `/root/autodl-tmp/Vision-OPD/rollouts/Vision-OPD-Qwen3.5-4B` | 不存在 |

## 复查命令

以下命令用于本次采集，可在机器资源或软件环境变化后重新执行：

```bash
date --iso-8601=seconds
git rev-parse HEAD
git branch --show-current
git status --short --branch
uname -a
sed -n '1,20p' /etc/os-release
command -v python
python --version
conda info --envs
conda info --json
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits
sed -n '1,5p' /proc/driver/nvidia/version
nvcc --version
lscpu
nproc
free -h
df -hT / /root/autodl-tmp /root/autodl-tmp/Vision-OPD
printenv HF_HOME TORCH_HOME PIP_CACHE_DIR TMPDIR
python -m pip cache dir
```
