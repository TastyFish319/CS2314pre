# Qwen3-8B 理论 Roofline（Ascend 910B3）

这是一套完全离线运行的理论计算与绘图工具。无需 GPU/NPU、PyTorch、模型权重或 Profiling 数据；只使用 Python 3.10+、NumPy 和 Matplotlib。

推荐阅读 [最新版报告](docs/report.md)：第 3 节完整分析 Prefill，第 4 节完整分析 Decode，第 5、6 节将公式和数值对应到图表。若预览器不支持公式，可打开 [公式图片兼容版](docs/report_compatible.md)。用户原稿、历史报告和验证记录统一收在 [文档目录](docs/README.md)。

标准报告使用 `$$...$$` 块公式，需要预览器支持数学渲染。兼容版公式使用本地 PNG 显示，并附 SVG 矢量版链接。

## 安装和运行

在本文件所在的 `qwen3_roofline/` 目录执行：

```bash
python -m pip install -r requirements.txt
python roofline_qwen3.py
python -m unittest discover -s tests -v
```

脚本默认在自身目录生成 `figures/` 和 `results/`，从其他工作目录调用也能使用。代码不硬编码机器专属路径，报告使用相对链接。没有图形桌面时也可运行（Matplotlib Agg 后端）。

本次 Windows 工作区已创建父目录下的 `.venv_roofline/`。从项目根目录可直接复用：

```powershell
.\.venv_roofline\Scripts\python.exe qwen3_roofline\roofline_qwen3.py
.\.venv_roofline\Scripts\python.exe -m unittest discover -s qwen3_roofline\tests -v
```

如需建立自己的独立环境，在任意系统用 `python -m venv .venv` 创建后，用该环境的 Python 执行上面的安装与运行命令。虚拟环境不在 ZIP 内。

## 自定义场景

以下命令保留默认结果，分别写入另一个输出目录：

```bash
python roofline_qwen3.py --peak-tflops 294.912 --bandwidth-tb-s 1.6 --seq-lengths 128 512 1024 4096 8192 32768 --context-lengths 1024 4096 8192 32768 --batch-sizes 1 2 4 8 16 32 64 128 256 --output-dir custom_run
python roofline_qwen3.py --hardware hypothetical --output-dir hypothetical_run
python roofline_qwen3.py --lm-head-mode all --extra-mib-per-token 0 2 8 --output-dir all_logits_run
python roofline_qwen3.py --help
```

`--prefill-batch` 修改 Prefill Batch，`--capacity-gib` 修改容量参考。`--seq-lengths` 指输入长度 S；`--context-lengths` 指当前 Decode 步之前的历史 C。Decode 实际关注 C+1 个位置。`--extra-mib-per-token` 仅用于 Prefill，按整个模型每个输入 Token 的额外 HBM 流量定义；程序始终保留 0 作为基准。

默认值对应已交付报告。修改 CLI 只重算图与数据，**不会自动改写中文报告中的默认数值或结论**。建议始终用 `--output-dir` 保存比较实验，避免覆盖默认报告配图。重复运行同一输出目录会覆盖同名图与数据，不清理其他文件。

## 输出和数据定义

五种图各包含 300 DPI PNG 与可编辑文字的 SVG：Roofline、计算组成、Decode Batch 与 AI、每输出 Token 流量、Prefill 激活流量敏感性。图内使用英文字体，中文解释在报告中。

| 文件 | 含义 |
|---|---|
| `results/prefill_results.csv` | 默认 5 个 S × 3 种额外流量，共 15 行；每行对应完整 Prefill 调用 |
| `results/decode_results.csv` | 默认 4 个 C × 9 个 B，共 36 行；每行对应整个 Batch 单步 |
| `results/module_summary.csv` | 各模块参数、矩阵 FLOPs/Token 与基础权重流量 |
| `results/module_roofline_results.csv` | 各场景逐模块的 FLOPs、独立操作数流量、AI、性能上限及时间下界 |
| `results/hardware_config.json` | 实际硬件参数、两套参考配置、模型参数、假设、来源、版本与 CLI 参数 |
| `tests/test_calculations.py` | 标准库 unittest 数值与不变量测试 |
| `docs/report.md` | 最新分析报告 |
| `docs/report_compatible.md` | 最新报告的公式图片版本 |
| `docs/drafts/`、`docs/archive/` | 用户原稿及历史报告 |
| `docs/development/VALIDATION.md` | 本次运行与交付检查记录 |

CSV 原始 FLOPs 为 FLOP，所有 `_bytes` 为 Byte，`_s` 为秒，`roof_tflops` 为十进制 TFLOP/s。`arithmetic_intensity` 为 FLOP/Byte。各模块列如 `attention_qk_flops` 与 `mlp_gate_flops` 可直接求和得到 `flops`。

`weight_bytes` 是每次调用全部相关矩阵权重读一次；`kv_read_bytes`、`kv_write_bytes`、`activation_extra_bytes` 是调用级流量。四者之和为 `traffic_bytes`。Decode 的 `bytes_per_output_token=traffic_bytes/B`；Prefill 中此列除以 B，表示每条序列首个后续输出分布对应的整次 Prefill 流量，**并非每个输入 Token 流量**。`lm_head_mode=all` 也沿用这个输入请求口径。

`resident_lower_bound_bytes` 是全部 BF16 参数加最终 KV 的**驻留容量下限**，与流量不同。`fits_capacity_lower_bound=True` 仅意味着通过必要条件，尚未包含激活、临时工作区和运行时预留；未配置容量的假设硬件该列留空。超容量场景保留供理论讨论，不表示单卡可运行。

`module_roofline_results.csv` 额外采用**独立的未融合操作数模型**：每个矩阵算子各读一次权重与输入、写一次输出，Attention 假设 BF16 中间分数被物化。它用于讨论逐模块 Roofline，与五图使用的基础模型不同，不能把该表流量加到基础流量或要求两者相等。Embedding 仅计行读取和输出写入，其矩阵 AI 与性能上限留空。具体公式见报告。

## 计算口径

FMA 按 2 FLOPs；KV 用 8 个 KV heads，QK/AV 用 32 个 Query heads。默认 Prefill 仅最后位置计算 LM Head。Embedding 与 LM Head 不共享；Embedding 是查表，不计为稠密矩阵 FLOPs。非矩阵开销在报告单列解释。

所有性能点都是假设流量下的 Roofline 上限，时间是理论下界。不能据此推断实际吞吐、带宽利用率或实测 MFU。单位：TB=10¹² Byte，GB=10⁹ Byte，GiB=2³⁰ Byte，MiB=2²⁰ Byte。

依赖版本范围见 `requirements.txt`，本次实际版本见 JSON。离线复算不需要访问参考网页；源码中的默认参数已经集中保存。ZIP 是补充交付，原始代码、CSV、图和报告均独立保留。
