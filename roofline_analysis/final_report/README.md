# 最终汇报报告

阅读 [report.md](report.md)。正文汇总理论 Roofline 分析、实测性能与 Profiling 解读。

```text
final_report/
├── README.md
├── report.md
└── assets/
    └── profiling/
        ├── prof-PREFILL.png
        └── prof-DECODE.png
```

| 内容 | 来源 |
|---|---|
| 理论图表、Prefill / Decode CSV | [理论计算工具](../qwen3_theory/qwen3_roofline/README.md) |
| 实测屋顶线图 | [实验图表](../qwen3_experiments/figures/roofline_validation_measured.png) |
| 实测分析与原始结果索引 | [实验资料](../qwen3_experiments/README.md) |
| Profiling 截图 | [Prefill](assets/profiling/prof-PREFILL.png) · [Decode](assets/profiling/prof-DECODE.png) |
| 时间线导读与数据入口 | [时间线导读](../qwen3_experiments/docs/timeline_guide.md) · [Profiling 分析](../qwen3_experiments/docs/profiling_report.md) |

所有仓库内链接使用相对路径和 `/` 分隔符。分享时请一并保留同级 `qwen3_theory/`、`qwen3_experiments/`，或直接分享整个 [roofline_analysis](../README.md) 目录。
