# AISBench 指标公式对齐

核对版本：官方 AISBench/benchmark，提交 7bcb6440da11bef020943b2af9ed7c1ac8362306。
范围：DefaultPerfMetricCalculator，stage=total，流式、全成功请求。本次三组记录全成功；不能据此推断失败请求的过滤行为或 stable 阶段结果。

源代码：
- https://github.com/AISBench/benchmark/blob/7bcb6440da11bef020943b2af9ed7c1ac8362306/ais_bench/benchmark/summarizers/default_perf.py
- https://github.com/AISBench/benchmark/blob/7bcb6440da11bef020943b2af9ed7c1ac8362306/ais_bench/benchmark/calculators/default_perf_metric_calculator.py
- https://github.com/AISBench/benchmark/blob/7bcb6440da11bef020943b2af9ed7c1ac8362306/ais_bench/benchmark/calculators/base_perf_metric_calculator.py

记请求 i 的提交、首 token、结束时间为 s_i、f_i、e_i，单位秒；输入/输出数为 I_i/O_i。
N=成功请求数，D=8，T=max(e_i)-min(s_i)，L_i=e_i-s_i，F_i=f_i-s_i，P_i=(e_i-f_i)/(O_i-1)。

| 用户列 | AISBench 对应 | 公式 |
|---|---|---|
| 实测总请求数 | Total Requests / Success Requests | 总数与成功数应区分；本次全部成功，均为 N |
| 实测最大并发数 | Max Concurrency | 配置上限，不是观测峰值；真实峰值另存 |
| 实测系统并发数 | Concurrency | ΣL_i/T |
| 实测请求频率(req/s) | Request Throughput | N/T |
| 实测TTFT平均(ms) | TTFT Average | mean(F_i)×1000 |
| 实测TTFT P90(ms) | TTFT P90 | percentile(F_i,90)×1000 |
| 实测TPOT平均(ms) | TPOT Average | mean(P_i)×1000，每请求等权 |
| 实测TPOT SLO_P90(ms) | TPOT P90 | percentile(P_i,90)×1000；不是 ITL P90 或 SLO 阈值 |
| 实测输出吞吐(tps) | Output Token Throughput（Common Metric） | ΣO_i/T |
| 实测输入吞吐(tps) | Input Token Throughput（Common Metric） | ΣI_i/T |
| Total Token Throughput(token/s) | Total Token Throughput（Common Metric） | Σ(I_i+O_i)/T；与保留的“实测E2E吞吐(tps)”数值相同 |
| 实测单卡输出吞吐(tps) | 自定义派生列 | ΣO_i/T/D |
| 实测E2E吞吐(tps)/端到端时延(s) | Total Token Throughput / E2EL Average | 两个值：(ΣI_i+ΣO_i)/T，以及 mean(L_i)；不是再相除 |
| 实测单卡E2E吞吐(tps) | 自定义派生列 | (ΣI_i+ΣO_i)/T/D |
| 实测QPS | Request Throughput | N/T |
| 实测QPM | 自定义派生列 | 60N/T |

百分位使用 numpy.percentile 默认线性插值。数学计算值与显示舍入分开：核对的版本先将请求级统计的秒值 round(...,4)，再乘 1000 输出 ms，因此可出现约 0.05ms 的显示差；Common Metric 吞吐与平均并发保留 4 位小数。
本次每个输出长度均大于 1。AISBench 对输出数不超过 1 的 TPOT 赋 0；非流式场景 TTFT/TPOT 不作有效指标。

特别注意 OutputTokenThroughput（无空格，请求级）的 Average = mean(O_i/L_i)，不同于 Output Token Throughput（整场）=ΣO_i/T。用户要求整机输出吞吐，应选后者。
配置 request_rate 是发压设置，并不等于实测 Request Throughput。

## 已修正与重算

此前“实测请求频率”错误使用 N/(最后提交−首次提交)，本地 benchmark_configurable.py 已改为 N/T；旧公式改名为“提交阶段请求频率”保留。
Max Concurrency 改用配置上限，另保留“观测峰值在途并发数”。本次三组配置上限与真实峰值恰好相同，仍为 200/12/8。
没有重新运行 NPU 推理，也没有覆盖历史原始结果。基于逐请求记录生成：
- suite-20261026-v1/aisbench_aligned/results.csv：完整精度的中文映射表。
- suite-20261026-v1/aisbench_aligned/aisbench_display.csv：匹配上述官方显示舍入的对照表。
- export_aisbench_aligned.py：可复现导出脚本。

请求频率修正：4K 6.973228→4.860641 req/s；64K 0.385487→0.252233；128K 0.445102→0.330044。其余主要数学指标无需改变。
5 项本地测试通过，其中已覆盖配置并发与实际峰值不相等、请求频率与提交频率不相等的情况。

仅公式与显示口径对齐：本数据来自离线引擎提交/返回时间；AISBench 在线测试来自客户端请求/流式响应时间，包含服务和网络路径。不能把本表标成“由 AISBench 实测”，也不能把公式一致等同于在线性能一致。
仓库展开脚本与部署包已同步此修订；已运行服务器的历史日志保持原样。实际使用的 AISBench 安装版本若不同，应再按该版本及 summarizer 配置核对。
