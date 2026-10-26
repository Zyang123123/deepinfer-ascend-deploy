# DeepInfer 快速部署指南（Ascend 950DT）

目标：在 **8×Ascend950DT（28 AIC/56 AIV，每卡96 GiB HBM）** 上尽快跑通 DeepSeek-V4.1-Flash 离线推理。

使用附件：[部署补丁与脚本包](deepinfer-ascend-deploy.zip)。无需安装 Codex，解压后直接使用。本文适用于该硬件和指定源码版本；其他型号不能直接照搬补丁。

## 1. 准备环境

推荐使用已验证镜像：

```text
quay.io/ascend/vllm-ascend:v0.30.0rc2-a5-openeuler
```

| 项目 | 已验证配置 |
|---|---|
| 宿主驱动 | 25.6.rc1 |
| 容器系统 | openEuler 24.03 SP4，aarch64 |
| CANN / Python | 9.2.0 / 3.12.13 |
| 编译器 | GCC14，不能只使用镜像默认 GCC |
| PyTorch | torch 2.13.0+cpu + torch_npu 2.13.0rc1 |
| 模型目录 | `/home/weights/DeepSeek-V4.1-Flash-1017` |

容器需使用 **host 网络**，能访问8张卡、宿主驱动、模型目录和本机 `/etc/hccl_rootinfo.json`；历史配置使用500 GiB `/dev/shm`，宿主内存约1.5 TiB。镜像本身不包含本文后续补丁。

**已有 `dsv4.1_deepinfer` 容器的机器，直接进入：**

```bash
docker exec -it dsv4.1_deepinfer bash
npu-smi info
test -s /etc/hccl_rootinfo.json
test -s /home/weights/DeepSeek-V4.1-Flash-1017/config.json
```

8张卡必须空闲。没有上述容器时，先按镜像部署具备这些映射的容器；创建示例见附件 `references/environment.md`，不要复制其他机器的 EID 拓扑文件。

**以下命令均在容器内执行。**

## 2. 安装 GCC14，准备源码和补丁

```bash
dnf install -y gcc-toolset-14-gcc gcc-toolset-14-gcc-c++ \
  gcc-toolset-14-libstdc++-devel git cmake ninja-build unzip
source /opt/openEuler/gcc-toolset-14/enable
source /usr/local/Ascend/cann/set_env.sh
gcc --version
python --version
```

将部署包上传到容器可访问目录，例如 `/mnt/nfs_share/home/<你的目录>/`。下面的工作目录请自行替换：

```bash
export WORK=/mnt/nfs_share/home/<你的目录>
cd "$WORK"
unzip deepinfer-ascend-deploy.zip
export KIT="$WORK/deepinfer-ascend-deploy"

# 使用新目录，避免覆盖已有修改
git clone --recursive https://github.com/deepseek-ai/DeepInfer.git
cd DeepInfer
git checkout 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9
git submodule update --init --recursive
export DEEPINFER_ROOT="$PWD"

# 先检查，再应用：同时安装补丁、工具链包装器及测试脚本
python "$KIT/scripts/apply_bundle.py" --repo "$PWD"
python "$KIT/scripts/apply_bundle.py" --repo "$PWD" --apply
```

若提示版本不匹配或补丁冲突，停止处理，不强行忽略。已有修复完成的仓库不需要重新 clone 或重套补丁。

## 3. 构建

```bash
cd "$DEEPINFER_ROOT"
set -o pipefail
bash scripts/build_ascend_local.sh 2>&1 | tee build.log
```

该脚本会启用 CANN、GCC14和bisheng兼容包装器，安装依赖并构建子模块与主项目。**确认命令退出码为0** 后再继续。

如果 GitHub/pip 不通：在有7890代理的本机电脑保持以下 SSH 隧道：

```bash
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -R 127.0.0.1:17891:127.0.0.1:7890 root@<服务器IP>
```

然后在 host-network 容器内设置代理，重试下载/构建：

```bash
export http_proxy=http://127.0.0.1:17891
export https_proxy=$http_proxy
export no_proxy=localhost,127.0.0.1
```

## 4. 跑通首次离线推理

将 IP 和接口改为当前服务器的真实配置；`.36` 已验证值如下：

```bash
cd "$DEEPINFER_ROOT"
export HCCL_IF_IP=192.168.45.36
export HCCL_SOCKET_IFNAME=enp34s0f1
export GLOO_SOCKET_IFNAME=$HCCL_SOCKET_IFNAME
export DEEPINFER_HF_MODEL_DIR=/home/weights/DeepSeek-V4.1-Flash-1017
export DEEPINFER_USE_NEXTN_DECODE=5

bash scripts/run_offline_ascend_36.sh
```

启动器已包含：EP8/DP8、显存利用率0.95、Engram加载期CPU staging、GCC14 JIT配置和HCCL环境。首次加载、JIT编译和图捕获较慢，以日志是否继续推进判断，不仅看启动耗时。

**跑通标准：正常返回推理输出、进程退出码为0，日志没有 OOM、通信失败或算子异常。** 日志位于仓库 `logs/offline-*.log`。

已构建好的原 `.36` 容器，通常只需进入仓库后执行本节，无需重复前3节。

## 5. 跑通后再做数值验证和压测

数值验证建议先执行，相关脚本会使用NPU：

```bash
bash scripts/run_multi_endpoint_test_36.sh scripts/test_ascend_numerics.py prefill
bash scripts/run_multi_endpoint_test_36.sh scripts/test_ascend_numerics.py decode
bash scripts/run_multi_endpoint_test_36.sh scripts/test_sparse_mqa_compat.py
```

自定义性能测试：

```bash
bash scripts/run_configurable_36.sh \
  --input-len 4096 --output-len 2048 \
  --num-requests 400 --max-concurrency 200
```

一次运行预设三组：

```bash
python scripts/run_benchmark_suite.py
```

三组分别为 `4K+2K / 400请求 / 并发200`、`64K+1K / 48 / 12`、`128K+1K / 32 / 8`。结果保存为 `logs/suite-*/results.csv` 和 JSON。

**验证范围：**此前固定批200请求的4K+2K已跑通；上述新三组截至2026-10-26尚未上机完成，不代表长上下文已验收。首次部署优先完成第4节。

## 常见问题：直接按这张表处理

| 现象 | 原因及处理 |
|---|---|
| pip/GitHub下载失败 | 检查代理隧道和容器代理变量，再重试；不是编译器问题。 |
| GCC版本过低、bisheng报C++标准库错误 | 安装并启用GCC14，使用包内构建脚本；它使用局部兼容头和host/device编译分流。不要修改系统标准库。 |
| 构建通过，推理JIT仍报标准库错误 | 必须使用包内运行脚本，它会设置 `DG_BISHENG_GCC_TOOLCHAIN` 和 `DG_BISHENG_COMPAT_INCLUDE`；只改PATH不够。 |
| DeepEP报layer数量或多个local endpoint不支持 | 应用DeepEP/DeepGEMM补丁并重编；本机使用多端点通信，不伪造共享EID，也不要复制别机rootinfo。 |
| FlashMLA报AI core数不匹配 | 实机28核，上游路径假设32核；包内补丁同步修正启动核数和调度步长。 |
| MegaMoE失败 | 当前32核调度不适配28核；包内自动使用普通DeepEP＋grouped GEMM。此回退可能影响性能。 |
| sparse MQA编译器崩溃 | 包内补丁将有问题的SIMT循环改为32位并保留边界保护，重编后跑数值测试。 |
| Engram加载OOM | 包内CPU staging避免加载期双份NPU表；最终表仍占显存，不等于vLLM运行期CPU offload。 |
| 共享内存输入复制失败 | 包内 `try_concat` 修复通过torch pinned allocator中转，不直接异步DMA host-registered共享内存。 |
| 显存利用率0.95仍OOM | 先查其他进程、运行时可用容量和权重常驻量；不能靠调比例释放系统保留区。此前84 GiB机器只有68.25 GiB AI pool，未找到支持的在线扩容方法。 |
| 将max_model_len设成6208后断言 | 该基线内部索引容量有额外约束；4K+2K请求仍保留引擎容量32768，不需要把引擎容量压到输入输出之和。 |
| 性能突然停顿或偏低 | 完整预热至目标输出长度，检查测量期新JIT编译和其他NPU任务；不要用短预热结果对比。 |

补丁来源、精确镜像digest、数值证据及详细原因均保留在附件中。换机器时先核对型号/核数、驱动/CANN和拓扑，不能只根据“都是昇腾”判断兼容。



完整下载：[指南＋补丁脚本完整包](DeepInfer_快速部署包.zip)
