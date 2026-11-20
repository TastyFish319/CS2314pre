# Qwen3-8B 理论 Roofline 分析

用于计算机系统结构课程讨论：根据 Qwen3-8B 结构和 Ascend 910B3 参考参数，计算理论 FLOPs、HBM 流量、算术强度与 Roofline 上限。仅使用 CPU 计算，无需模型权重或 NPU。

## 阅读入口

- [最新版报告](qwen3_roofline/docs/report.md)：第 3 节分析 Prefill，第 4 节分析 Decode，第 5、6 节解释图表。
- [公式图片兼容版](qwen3_roofline/docs/report_compatible.md)：不依赖 Markdown 数学渲染插件。
- [运行与数据说明](qwen3_roofline/README.md)：安装、CLI 参数及 CSV 字段。
- [文档目录](qwen3_roofline/docs/README.md)：原稿、历史报告和验证记录。
- [完整 ZIP](dist/qwen3_roofline.zip)：包含独立的 `qwen3_roofline/` 项目。

## 目录结构

```text
roofline_theory/
├── README.md
├── .gitignore
├── qwen3_roofline/              # 可独立运行、分发的项目
│   ├── README.md
│   ├── roofline_qwen3.py
│   ├── requirements.txt
│   ├── docs/
│   │   ├── report.md           # 当前报告
│   │   ├── report_compatible.md
│   │   ├── drafts/             # 用户 simplified 原稿
│   │   ├── archive/            # 历史分析报告
│   │   └── development/        # 实现计划与验证记录
│   ├── figures/                # 五张主图，PNG 与 SVG
│   │   ├── formulas/           # 当前报告的公式图片
│   │   └── archive/            # 历史报告的公式图片
│   ├── results/                # CSV 和硬件/模型配置
│   └── tests/
├── dist/                       # 分发 ZIP，Git 忽略
├── .work/                      # 临时实验和检查存档，Git 忽略
└── .venv_roofline/              # 本地 Python 环境，Git 忽略
```

## 重新运行

从当前工作区根目录复用已有的 Windows 环境：

```powershell
.\.venv_roofline\Scripts\python.exe qwen3_roofline\roofline_qwen3.py
.\.venv_roofline\Scripts\python.exe -m unittest discover -s qwen3_roofline\tests -v
```

在其他机器上，进入 `qwen3_roofline/`，安装依赖后运行：

```bash
python -m pip install -r requirements.txt
python roofline_qwen3.py
python -m unittest discover -s tests -v
```

默认输出仍写入项目内的 `figures/` 与 `results/`。自定义计算不会自动改写报告文字。`.work/` 保存开发期间的临时文件；正式成果均位于 `qwen3_roofline/`，ZIP 不包含本地环境与临时文件。
