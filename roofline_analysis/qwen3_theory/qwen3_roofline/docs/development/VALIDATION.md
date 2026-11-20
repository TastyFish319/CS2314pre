# 本次生成与检查记录

2026-11-20 目录整理：当前报告统一位于 [docs/report.md](../report.md)，公式图片兼容版位于 [docs/report_compatible.md](../report_compatible.md)；用户原稿收在 `docs/drafts/`，历史报告收在 `docs/archive/`，本记录与实现计划收在 `docs/development/`。历史记录中的旧文件名表示当时的版本名称。五张主图、计算脚本和 CSV/JSON 内容保持不变，相对链接按新目录修正。19 项 unittest 全部通过，整理后的 CLI 成功生成五组图表及数据；临时实验、缓存及本地环境由工作区 `.gitignore` 排除。

日期：2026-11-17。计算仅在普通 CPU 上完成；没有加载 Qwen 权重、导入 PyTorch/torch_npu、查询 NPU 或启动 Profiling。

文档渲染修正：19 组块公式已统一为 `$$...$$`，并规范 `\mathrm` 与分母花括号。另提供 `qwen3_roofline_report_compatible.md`，19 组公式分别渲染为本地 PNG/SVG，供不支持数学公式的 Markdown 预览器使用。公式内容与计算结果保持一致，兼容版及公式图片一并打入 ZIP。

2026-11-19 文档重写：根据用户的 `report_simplified.md` 新建 `report_rewritten.md`，沿用六部分框架，将 Prefill 的计算、流量、算术强度集中在第 3 节，Decode 对应内容集中在第 4 节，并在每张图下注明数据来源。补充符号首次定义、两阶段分项示例、Batch 流量摊销及容量来源表；数值沿用现有默认 CSV，未修改计算脚本和原稿。新版同时提供公式图片兼容版，另存公式图并同步 ZIP。

## 环境和命令

Python 3.13.12、NumPy 2.4.6、Matplotlib 3.11.2。父目录下 `.venv_roofline/` 安装绘图依赖；环境不属于可分发成果。版本也保存在 `results/hardware_config.json`。

从项目根目录运行：

```powershell
.\.venv_roofline\Scripts\python.exe qwen3_roofline\roofline_qwen3.py
.\.venv_roofline\Scripts\python.exe -m unittest discover -s qwen3_roofline\tests -v
```

默认计算命令正常退出，生成 5 组 PNG/SVG、15 行 Prefill、36 行 Decode、11 行模块摘要，以及 451 行逐模块操作数 Roofline 数据。最后一张表对应 (5 个 Prefill 场景 + 36 个 Decode 场景) × 11 个模块，不重复扩展激活流量场景。

## 数值验证

19 项 unittest 全部通过。覆盖指定的 8 类数值要求以及 GQA Query heads、C+1、单 Token 等价边界、Embedding 查表、Norm 参数统计、Batch FLOPs 与权重摊销、激活敏感性、全序列 LM Head、自定义硬件、容量下限、无效输入及逐模块流量模型；还包含极端自定义硬件图表的回归验证。

测试先于计算模块创建；首次运行因模块尚不存在而失败。实现后一个额外总参数参考值的手工抄写错误被定位并更正：独立复算 `6,945,767,424 + 2×622,329,856 + 73×4096 + 72×128 = 8,190,735,360`；未改变公式以迎合参考数值。新增局部操作数功能同样先运行测试确认缺失，再实现并通过。

## 图表和数据检查

五张 PNG 均已目视检查：坐标、图例、脊点与场景标记可见，无观察到的标签重叠。图 2 同时给出绝对 FLOPs 和比例，图 4 的 KV 写入很薄，已补充精确量级说明。图 3 空心点、图 4 星号明确标记容量超限的理论外推。

检查 PNG 元数据约为 300 DPI，SVG 可被 XML 解析，五张图的两种格式均独立存在。报告中的十个图表相对路径及 README/VALIDATION 链接可解析；数值表取自同一计算函数与 CSV 的未舍入值。

自定义运行另存父目录 `.roofline_validation/custom/`，未覆盖默认成果：使用 hypothetical、120 TFLOP/s、2.4 TB/s、32 GiB、S=16/64、C=0/256、B=2/4、Prefill B=2、LM Head all、额外 0/1 MiB/Token。运行正常退出，验证非默认长度、Batch、带宽、算力、容量和 logits 模式能传到计算与绘图。

## 交付边界

独立只读审查再次通过全部主要公式，并独立核对 15+36 行 CSV。审查发现极端自定义硬件时，脊点可能落在横轴外，导致区域标签错误与空白画布；已先用实际生成 PNG 的测试复现失败，再修正坐标范围和按脊点放置标签，全套测试通过。README 对运行目录元数据的表述也已修正。

报告解释默认参数；自定义 CLI 不自动改写报告。ZIP 只包含 `qwen3_roofline/` 原始交付文件，不含虚拟环境、Python 缓存或审查用临时输出。所有流量为假设模型，所有性能为理论上限，时间为模型内下界。
