# Qwen3-8B 理论 Roofline 实现与验证计划

需求依据：本次用户给定的完整规格。用户已明确要求直接执行；采用 Python 3.10+、NumPy、Matplotlib 和 unittest，在当前独立目录内交付，不创建分支、不加载模型、不访问 NPU。

架构：集中保存模型与硬件常量；纯函数计算参数、逐模块 FLOPs、假设 HBM 流量和 Roofline；绘图与 CSV/JSON 导出消费同一组计算结果。Markdown 报告解释默认配置，命令行修改的结果以该次 CSV/JSON 为准。

- [x] 1. 先建立数值、维度、边界和参数变化测试，运行确认未实现时失败。
- [x] 2. 实现计算函数，逐项验证参考数值以及 GQA、最后位置 logits 和容量边界。
- [x] 3. 实现 CLI、五张 PNG/SVG、模块明细及运行配置导出；运行默认配置和独立目录的自定义配置。
- [x] 4. 根据核实来源与计算结果撰写中文报告、README；目视检查全部五张图，核对相对链接。
- [x] 5. 全套测试、数据与文件完整性检查，保存验证记录并打包 ZIP。

完成记录：19 项测试通过；默认 CSV/JSON、300 DPI PNG 与 SVG、报告相对链接均验证。独立审查确认主公式并复算全部 Prefill/Decode 行；其发现的自定义脊点越界问题已通过 RED→GREEN 回归测试修复。压缩包保留目录层级，排除缓存与虚拟环境。

审查重点：TB/GB 与 TiB/GiB；C 是执行前历史长度；注意力使用 Query Heads；Embedding 查表；未计激活容量不代表可部署；自定义 CLI 数据不能被误读为默认报告结果。

来源核实：官方 config.json；Transformers Qwen3 实现；Roofline 原始论文及 Berkeley 镜像；CSET 的带宽及 Cube/Vector 口径；华为 Cloud MFU 工具；FlashAttention 原始论文。所有权重流量与激活敏感性均为理论假设。
