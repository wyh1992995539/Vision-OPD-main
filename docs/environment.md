# Vision-OPD 历史初始环境快照

> 采集时间：2026-08-12 10:54:15 +08:00
> 文档定位：保留项目开始前的只读环境快照，不作为当前训练配置或硬件状态的依据。

## 1. 快照结论

采集时尚未创建 vision-opd 环境、未下载 Qwen3.5-4B、未准备数据，也没有可供 PyTorch 使用的 GPU。因此该快照只说明项目起点和后续环境改造的必要性，不能用于启动推理或训练。

## 2. 初始项目状态

| 项目 | 快照值 |
|---|---|
| 项目目录 | /root/autodl-tmp/Vision-OPD |
| Git commit | c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471 |
| Git 分支 | main |
| 工作树 | clean |
| 训练数据 | 未创建 |
| 本地模型 | 未下载 |
| checkpoint 与 rollout | 未创建 |

上述目录和 commit 为历史值；当前项目目录、代码版本和模型路径以 Day 1 preflight 为准。

## 3. 初始软件与硬件状态

| 项目 | 快照值 |
|---|---|
| Python | 3.12.3，Conda base |
| PyTorch | 2.8.0+cu128 |
| CUDA 编译版本 | 12.8 |
| torch.cuda.is_available() | false |
| PyTorch 可见 GPU 数 | 0 |
| Transformers、vLLM、Ray、FlashAttention | 未安装或未验证 |
| nvidia-smi | 无法执行 |
| nvcc | 未找到 |

当时的 base 环境与项目依赖不一致，不能运行 Vision-OPD。后续已创建独立 vision-opd 环境并完成单卡安装与推理验收，详见 [environment_setup_and_validation.md](environment_setup_and_validation.md)。

## 4. 初始存储状态

| 挂载点 | 容量 | 可用空间 |
|---|---:|---:|
| 系统盘 / | 30 GiB | 约 28 GiB |
| 数据盘 /root/autodl-tmp | 50 GiB | 约 50 GiB |

该数据仅代表项目开始前的磁盘情况。当前磁盘、模型清单与 SHA256 见 artifacts/runs/preflight/disk.txt、model_manifest.txt 和 model_sha256.txt。

## 5. 当前训练环境的权威来源

不要使用本文件判断当前训练条件。训练前应依次检查：

1. artifacts/runs/preflight/env.txt：核心包版本与 editable 项目路径。
2. artifacts/runs/preflight/pip_check.txt：依赖元数据风险。
3. artifacts/runs/preflight/hardware.txt：GPU 可见性与双卡验证状态。
4. configs/project_1024.yaml：固定 seed、模型、数据、存储和训练门槛。
5. docs/environment_setup_and_validation.md：历史安装与单卡推理验收过程。

当前双卡硬件验证仍为 DEFERRED；在任何 GPU Smoke 或正式训练前必须完成。
