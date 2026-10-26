# DeepInfer 可配置离线压测脚本

适配此前 192.168.45.36 / dsv4.1_deepinfer / GCC14 / CANN 环境。
使用 GSM8K question 文本加两个换行，重复 token 并截断到配置输入长度；不含参考答案。
输出使用 temperature=0、ignore_eos=True，强制生成配置数量的 token。
普通 MoE、EP8/DP8/CP1、MTP5、显存利用率默认 0.95，延续此前运行配置。

## 安装和运行

将压缩包中的文件解压到服务器仓库的 scripts/：
`/mnt/nfs_share/home/j00930424/DeepInfer/scripts/`。
保留现有 scripts/toolchain/ 兼容性工具，启动脚本会调用其中的 prepare_headers.py。
脚本不自动停止其他任务；执行前确保 8 张卡可用。

运行全部三组：

```bash
docker exec -it -w /mnt/nfs_share/home/j00930424/DeepInfer dsv4.1_deepinfer \
  python scripts/run_benchmark_suite.py
```

| 用例 | input-len | output-len | num-requests | max-concurrency | 自动 max-model-len |
|---|---:|---:|---:|---:|---:|
| 4K+2K | 4096 | 2048 | 400 | 200 | 32768 |
| 64K+1K | 65536 | 1024 | 48 | 12 | 98304 |
| 128K+1K | 131072 | 1024 | 32 | 8 | 163840 |

K=1024。自动引擎容量为输入+输出+投机解码余量向上取整到 32768 的倍数，并至少为 32768。
128K 输入加 1K 输出超过 128K 总上下文，必须由模型和引擎支持更大的总长度。
64K/128K 可能遇到显存不足或长上下文实现限制；脚本不会自动缩短输入或擅自更改模型 RoPE。
2026-10-26 已在指定容器完成三组 NPU 实测，实际请求数和峰值并发均达到表中配置。详见 DeepInfer_suite_20261026_report.md 和 suite-20261026-v1/results.csv。

只运行一个用例：

```bash
docker exec -it -w /mnt/nfs_share/home/j00930424/DeepInfer dsv4.1_deepinfer \
  python scripts/run_benchmark_suite.py --case '64K+1K'
```

任意配置输入、输出、总请求数、最大并发数：

```bash
docker exec -it -w /mnt/nfs_share/home/j00930424/DeepInfer dsv4.1_deepinfer \
  bash scripts/run_configurable_36.sh \
  --input-len 4096 --output-len 2048 \
  --num-requests 400 --max-concurrency 200
```

可选参数：

```text
--model /home/weights/DeepSeek-V4.1-Flash-1017
--dataset /absolute/path/GSM8K.jsonl
--memory-utilization 0.95
--max-model-len 163840
--max-batched-tokens 4096
--warmup-requests 200
--tpot-slo-ms 50
--seed 1024
```

套件也可传入共同参数，例如 `python scripts/run_benchmark_suite.py --tpot-slo-ms 50`。
每组默认超时 14400 秒，可用套件参数 `--timeout-seconds` 修改。
默认预热请求数=min(最大并发,总请求数)，预热输入输出长度与正式测试相同；`--warmup-requests 0` 可关闭预热。
每组启动新的模型进程。套件首个失败即停止并保存失败状态，避免异常残留进程影响下一组。
排查并确认设备释放后，可通过 `--case` 单独继续剩余用例。
每次使用新结果目录；不复用已有 config.json 的目录。

## 请求调度

采用闭环负载：每 rank 达到本地在途上限后，等待某条请求完成，再提交下一条。
不是将全部请求提前放入引擎队列，也不是按固定 req/s 开环发压。
总请求数和最大并发数均为整机值，不是单卡值。
并发 12 在 8 rank 上分配为 2/2/2/2/1/1/1/1；48 条请求分配为 8/8/8/8/4/4/4/4。
不同 rank 的请求长度/完成速度可能不同，固定分配下尾部会自然降并发，不承诺全程恒定并发。
为保持 EP/DP 协同，局部无请求的 rank 仍参与全局 stepping。
当前脚本要求最大并发和总请求数均至少为 dp（默认 8）；输出长度至少为 2，便于定义 TPOT。

## 指标定义

令 N 为成功完成请求数，T 为首条提交至最后一条完成的秒数，I/O 为实际输入/输出 token 总数，D 为卡数（此 EP=DP 配置下等于 dp）。
对单请求 j：s_j=引擎提交时间，f_j=首 token 返回时间，e_j=完成时间，o_j=输出 token 数。
这是引擎 API 观察时间，不包含模型加载、tokenization、预热、HTTP 或客户端网络时间。

| CSV 列 | 公式或含义 |
|---|---|
| 实测总请求数 | 成功完成的 N；不直接填配置值 |
| 实测最大并发数 | 为对应 AISBench Max Concurrency，此列填写配置上限；另列“观测峰值在途并发数”保存事件扫描的真实峰值 |
| 实测系统并发数 | 时间加权平均在途数：Σ(e_j−s_j)/T，可能为小数 |
| 实测请求频率(req/s) | N/T，对应 AISBench Request Throughput，与 QPS 相同 |
| 实测TTFT平均(ms) | mean(f_j−s_j)×1000 |
| 实测TTFT P90(ms) | 请求 TTFT 的 P90，线性插值 |
| 实测TPOT平均(ms) | mean((e_j−f_j)/(o_j−1))×1000；请求等权 |
| 实测TPOT SLO_P90(ms) | 上述逐请求 TPOT 的 P90；不是相邻 token 间隔的 P90 |
| 实测输出吞吐(tps) | O/T，包含 prefill、decode 和排空时间 |
| 实测输入吞吐(tps) | I/T，对应 AISBench Input Token Throughput；分母同样包含整场 prefill、decode 和排空时间，不是纯 prefill 耗时 |
| Total Token Throughput(token/s) | (I+O)/T，即输入吞吐与输出吞吐之和；与“实测E2E吞吐(tps)”相同，显式保留 AISBench 名称以便对照 |
| 实测单卡输出吞吐(tps) | O/T/D，按卡数归一化，不是单卡独立实测 |
| 实测E2E吞吐(tps)/端到端时延(s) | (I+O)/T 与 mean(e_j−s_j)，以“吞吐/时延”显示；同时提供两个独立数值列 |
| 实测单卡E2E吞吐(tps) | (I+O)/T/D |
| 实测QPS | N/T，成功完成速率 |
| 实测QPM | 60×N/T |

2026-10-26 AISBench 对齐修订：请求频率与 QPS 均使用整场时长。此前 N/(最后提交−首次提交) 另存为“提交阶段请求频率(req/s)”，不再冒充 AISBench Request Throughput。原始历史 CSV 不覆盖，对齐重算结果位于 suite-20261026-v1/aisbench_aligned/。公式参考官方 DefaultPerfMetricCalculator 的 total 阶段，提交 7bcb6440da11bef020943b2af9ed7c1ac8362306；不对应 stable 阶段。当前 metrics() 面向全成功记录，不用于混合失败请求的 AISBench 汇总。
SLO_P90 按需求列名保留。如果指定 `--tpot-slo-ms`，另输出阈值和逐请求 TPOT 达标率；未指定时这两项为空，不虚构 SLO。
MTP 一次可能返回多个 token，TPOT 采用请求级平均公式，不把批量返回误认为逐 token 网络间隔。
失败用例不填写成功性能数字；部分完成记录保留在 JSONL，便于排查。

## 输出文件与验证

套件结果：`logs/suite-时间/results.csv` 和 `results.json`；CSV 使用 UTF-8 BOM，可用 Excel 打开。
每组包含 config.json、summary.csv/json、memory.json、jit_audit.json、run.log、各 rank 请求 JSONL/JSON、题目行号与 prompt 哈希、NPU 快照及显存采样。
memory.json 的 HBM 是采样峰值，不是保证捕获全部瞬时峰值；PyTorch 峰值仅覆盖其分配器。
`测量期间新增JIT编译数` 非零表示可能存在冷编译干扰，应保留当前结果并重新预热、用新目录再跑。
默认不启用 profiler，避免干扰正式测量。
题目不够时会循环使用数据集，config.json 标记 dataset_reused；三组默认数据均足够。

本地验证命令（不加载模型或占用 NPU）：

```bash
python -m unittest -v test_configurable_benchmark.py
python -m py_compile benchmark_configurable.py run_benchmark_suite.py
```

已通过指标公式、并发 12 非均分、请求数完整性、异步完成模拟、同时间完成/提交事件和失败记录校验。
三组闭环请求均已通过 NPU 实测，无 OOM 或推理报错；测量期均无新增 JIT 编译。逐请求长度和进程采样通过 audit_benchmark_results.py 核验。此结论针对本报告的模型、机器和配置，不代表任意模型或并发下均满足容量要求。
