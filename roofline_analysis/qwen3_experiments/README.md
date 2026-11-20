# Qwen3-8B 实测实验资料

Ascend 910B3 上的 Qwen3-8B BF16 推理测试，测量日期为 **2026-11-19**；软件配置为 `ge_graph` 图模式与 RMSNorm/AddRMSNorm 融合。

汇报正文见 [最终报告](../final_report/report.md)。实验资料按用途整理，原始运行 ID、配置文件名及数据内容保留。

```text
qwen3_experiments/
├── README.md
├── docs/          # 实测分析、时间线导读与 Profiling 报告
├── figures/       # 实测屋顶线图
├── results/       # 屋顶标定 JSON 与四组配置汇总 CSV
├── runs/          # 按原始 run_id 保存 summary.json 和 recipe_stdout.log
├── configs/       # 每次运行实际使用的 YAML
└── profiling/     # Perfetto JSON 与算子、API、步骤统计 CSV
```

## 阅读入口

| 文档 | 用途 |
|---|---|
| [实测屋顶线与硬件设计分析](docs/hardware_design.md) | 标定方法、逐配置性能、Profiling 归因与硬件设计讨论 |
| [时间线导读](docs/timeline_guide.md) | 对照 CPU Thread、NPU Stream 和算子阅读时间线 |
| [Prefill / Decode Profiling 分析](docs/profiling_report.md) | 设备任务、同步、空隙与统计口径 |

## 图表与原始结果

| 文件或目录 | 内容 |
|---|---|
| [实测屋顶线图](figures/roofline_validation_measured.png) | 最终报告中的实测 Roofline 配图 |
| [屋顶标定](results/roof_calibration.json) | 算力、带宽标定的原始结果 |
| [四组测试汇总](results/bench_runs.csv) | S=256/1024、B=1/4 的逐配置结果 |
| [运行记录](runs/) | 四次运行的完整派生指标与原始日志 |
| [运行配置](configs/) | 与运行 ID 对应的四份 YAML |
| [Prefill 时间线](profiling/prefill_trace_view_perfetto.json) | Perfetto 可打开的时间线 |
| [Decode 时间线](profiling/decode_trace_view_perfetto.json) | Decode step 3/4/5 时间线 |
| [Prefill kernel 明细](profiling/kernel_details_prefill.csv) | 逐 kernel 耗时、形状及 pipe 指标 |
| [Decode kernel 明细](profiling/kernel_details_decode.csv) | Decode 逐 kernel 记录 |
| [步骤统计](profiling/step_trace_time.csv) | Computing、Free、Stage 等统计 |
| [API 统计](profiling/api_statistic.csv) | Host API 耗时 |
| [算子统计](profiling/op_statistic.csv) | 算子聚合耗时 |

## 数据来源与使用

资料原先位于导出目录 `roofline_hardware_design_20261119_v2/` 和 `profiling_guide/`，现按上面的目录分类存放。`results/`、`runs/`、`configs/`、`profiling/` 保留实测原件；其中 JSON、YAML 和日志中的实验机路径属于原始记录，不是当前仓库的可点击路径。

本目录保存实验结果与说明。硬件设计报告中出现的 `roof_calibrate.py`、`roofline_measure.py`、`plot_validation.py` 等脚本属于原实验环境，当前仓库未包含；相关命令用于记录测量过程。理论模型的离线复算工具另见 [qwen3_theory](../qwen3_theory/qwen3_roofline/README.md)。
