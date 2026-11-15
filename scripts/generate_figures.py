"""Rebuild tutorial figures: Python 3 + numpy + matplotlib (+ cairosvg for diagrams).

Run from any directory: python scripts/generate_figures.py
All performance numbers are teaching assumptions, never benchmark results.
"""
from pathlib import Path
import json
import math
from html import escape
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)
available = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": next((f for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"] if f in available), "DejaVu Sans"),
    "axes.unicode_minus": False, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "path", "figure.facecolor": "white",
})
BLUE, GREEN, ORANGE, GRAY = "#2563eb", "#16a34a", "#ea580c", "#6b7280"
N, S, L, PEAK, BW = 7e9, 2, 1024, 100e12, 2e12
I_STAR = PEAK / BW


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(OUT / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def base(title):
    fig, ax = plt.subplots(figsize=(10.5, 6.2), layout="constrained")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(.25, 4096)
    ax.set_ylim(.35, 400)
    ax.set_xlabel("算术强度 I（FLOP/Byte；HBM 边界）")
    ax.set_ylabel("性能上限 P（TFLOP/s）")
    ax.set_title(title, loc="left", pad=18)
    ax.grid(which="major", alpha=.20)
    return fig, ax


def draw_roof(ax, peak=100, bw=2, color=BLUE, label=None):
    xs = np.unique(np.r_[np.geomspace(.25, 4096, 600), peak / bw])
    ax.plot(xs, np.minimum(peak, bw * xs), color=color, lw=2.6, label=label)


fig, ax = base("标准 Roofline：100 TFLOP/s，2 TB/s（假设硬件）")
draw_roof(ax)
ax.axvline(50, color=GRAY, ls="--", lw=1)
ax.text(.5, 23, "带宽受限区域\nP = BW × I\n双对数坐标斜率 = 1", color=BLUE)
ax.text(250, 185, "计算受限区域\nP = 100 TFLOP/s", color=BLUE)
for intensity, p, label, offset in [(2, 4, "I=2，P≤4", (18, -24)), (50, 100, "脊点 (50, 100)", (-130, 28)), (100, 100, "I=100，P≤100", (15, -34))]:
    ax.scatter(intensity, p, s=60, color=ORANGE, zorder=4)
    ax.annotate(label, (intensity, p), xytext=offset, textcoords="offset points", fontsize=10)
ax.scatter(2, 2, s=65, marker="x", color=GRAY)
ax.annotate("假想实际点：同一 I 也可能低于屋顶", (2, 2), xytext=(18, -28), textcoords="offset points", fontsize=10, color=GRAY)
save(fig, "roofline_standard")

fig, ax = base("Qwen-7B 级教学模型：仅计权重流量的理想上限")
draw_roof(ax)
ax.axvline(I_STAR, color=GRAY, ls="--", lw=1)
ax.text(55, .55, "I*=50", color=GRAY)
ax.text(.4, 60, "带宽受限区域", color=BLUE)
ax.text(350, 290, "计算受限区域", color=BLUE)
for x, y, label, offset, color in [
    (1, 2, "Decode B=1\nI=1，P≤2 TFLOP/s\nt≥7 ms/step", (18, -28), ORANGE),
    (1024, 100, "Prefill L=1024\nI=1024，P≤100 TFLOP/s\nt≥143.36 ms", (-140, -100), GREEN),
    (64, 100, "Decode B=64：仅权重模型\nI=64，P≤100", (24, 20), GRAY),
]:
    ax.scatter(x, y, color=color, s=70, zorder=5)
    ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=10, color=color,
                arrowprops={"arrowstyle": "-", "color": color, "lw": .9, "shrinkA": 5, "shrinkB": 8})
save(fig, "roofline_qwen")

configs = [("计算优先", 200, 1, BLUE), ("带宽优先", 50, 4, GREEN), ("相对均衡", 100, 2, ORANGE)]
fig, (ax, bar) = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained", gridspec_kw={"width_ratios": [1.4, 1]})
xs = np.geomspace(.25, 4096, 800)
for name, peak, bw, color in configs:
    ax.loglog(xs, np.minimum(peak, bw * xs), lw=2.5, color=color, label=f"{name}：{peak} TFLOP/s，{bw} TB/s")
    ax.scatter(peak / bw, peak, color=color, s=45)
ax.set(xlabel="I（FLOP/Byte）", ylabel="P 上限（TFLOP/s）", title="三种示意硬件的屋顶线", xlim=(.25, 4096), ylim=(.2, 400))
ax.grid(alpha=.2, which="major")
ax.legend(fontsize=9, loc="lower right")
categories = ["Decode\nB=1", "Decode\nB=64", "Prefill\nL=1024"]
for j, (name, peak, bw, color) in enumerate(configs):
    vals = [min(peak, bw * i) for i in [1, 64, 1024]]
    positions = np.arange(3) + (j - 1) * .25
    bar.bar(positions, vals, width=.23, color=color, label=name)
    for x, value in zip(positions, vals):
        bar.text(x, value * 1.15, str(value), ha="center", fontsize=9)
bar.set_yscale("log")
bar.set_ylim(.5, 500)
bar.set_xticks(np.arange(3), categories)
bar.set_ylabel("理论性能上限（TFLOP/s）")
bar.set_title("同一负载，资源配比改变上限")
bar.grid(axis="y", alpha=.2)
save(fig, "accelerator_comparison")

fig, ax = plt.subplots(figsize=(7, 5.7), layout="constrained")
mask = np.tril(np.ones((5, 5)))
from matplotlib.colors import ListedColormap
ax.imshow(mask, cmap=ListedColormap(["#f3f4f6", "#dbeafe"]), vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        ax.text(j, i, "可见" if j <= i else "屏蔽", ha="center", va="center", color="#1e40af" if j <= i else "#9ca3af")
ax.set_xticks(range(5), [f"K{j+1}" for j in range(5)])
ax.set_yticks(range(5), [f"Q{i+1}" for i in range(5)])
ax.set_xlabel("被查询的位置（Key）")
ax.set_ylabel("当前位置（Query）")
ax.set_title("因果 Attention：每行可并行计算，但不能读取未来位置", pad=18)
save(fig, "causal_attention")


# Technical diagrams use explicit SVG lines for predictable geometry and typography.
def svg_start(width, height, title, subtitle):
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    lines.append('<style>text{font-family:"Microsoft YaHei","SimHei",sans-serif;fill:#111827;font-size:16px}</style>')
    lines.append('<defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#2563eb"/></marker></defs>')
    lines.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    text_svg(lines, 40, 40, title, 22)
    text_svg(lines, 40, 72, subtitle, 15, GRAY)
    return lines


def text_svg(lines, x, y, label, size=16, color="#111827", anchor="start"):
    lines.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="font-size:{size}px;fill:{color}">{escape(label)}</text>')


def box(lines, x, y, w, h, label, fill="#eff6ff"):
    lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="#d1d5db" stroke-width="1.5"/>')
    text_svg(lines, x + w / 2, y + h / 2 + 6, label, anchor="middle")


def arrow(lines, x1, y1, x2, y2):
    lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow)"/>')


def export_diagram(lines, name):
    lines.append('</svg>')
    dest = OUT / f"{name}.svg"
    dest.write_text('\n'.join(lines), encoding="utf-8")
    ET.parse(dest)
    import cairosvg
    cairosvg.svg2png(url=str(dest), write_to=str(OUT / f"{name}.png"), scale=1.7)


lines = svg_start(1100, 510, "GEMM 分块：把数据搬入快存储后，多算几次再换块", "图示描述数据复用；不表示真实 Cache 一定能容纳全部矩阵。")
box(lines, 40, 110, 470, 330, "", "#f9fafb")
box(lines, 590, 110, 470, 330, "", "#f9fafb")
text_svg(lines, 65, 145, "缺少有效复用的示意", 18)
text_svg(lines, 615, 145, "Blocking 的示意", 18)
box(lines, 75, 185, 160, 65, "主存中的 A、B")
box(lines, 315, 185, 160, 65, "计算单元")
arrow(lines, 235, 217, 315, 217)
text_svg(lines, 75, 300, "重复需要某数据时，它可能已被驱逐。")
text_svg(lines, 75, 337, "相同数据反复跨越主存边界。")
text_svg(lines, 75, 395, "F 不变，主存传输量 D 可能很大。", 15, GRAY)
box(lines, 625, 185, 140, 65, "HBM / DRAM")
box(lines, 845, 185, 180, 65, "Cache / SRAM")
arrow(lines, 765, 217, 845, 217)
text_svg(lines, 795, 200, "分块搬入", 13, BLUE, "middle")
box(lines, 845, 320, 180, 65, "计算多个乘加")
arrow(lines, 935, 250, 935, 320)
text_svg(lines, 625, 304, "A、B 子块留驻", 16, GREEN)
text_svg(lines, 625, 341, "C 子块累加", 16, GREEN)
text_svg(lines, 615, 415, "F 不变，D 减少，I = F/D 增大。", 15, GRAY)
text_svg(lines, 40, 482, "注意：若原实现已充分命中 Cache，分块不一定进一步减少 HBM 流量。", 15, GRAY)
export_diagram(lines, "gemm_blocking")

lines = svg_start(1100, 520, "KV Cache 随 Decode 增长：每层均维护自己的 K、V", "每个色块代表一个 Token 的一对 K/V 向量；本图仅画一个请求、一个层。")
rows = [(130, "Prefill 后", ["x1", "x2", "x3", "x4"], "最后位置预测 y1"),
        (250, "Decode 第 1 步", ["x1", "x2", "x3", "x4", "y1"], "输入 y1，预测 y2"),
        (370, "Decode 第 2 步", ["x1", "x2", "x3", "x4", "y1", "y2"], "输入 y2，预测 y3")]
for y, label, tokens, note in rows:
    text_svg(lines, 40, y + 32, label, 18)
    for j, token in enumerate(tokens):
        box(lines, 230 + j * 90, y, 76, 62, token, "#dcfce7" if j == len(tokens)-1 and len(tokens)>4 else "#dbeafe")
    text_svg(lines, 810, y + 33, note, 16)
text_svg(lines, 230, 475, "旧 K/V 直接复用；当前 Q 读取可见 K/V；只追加当前输入 Token 的新 K/V。", 16, GRAY)
export_diagram(lines, "kv_cache_growth")

results = {
    "notice": "All performance values are theoretical teaching estimates, not measurements.",
    "ridge_flop_per_byte": I_STAR,
    "weight_GB": N*S/1e9,
    "prefill_FLOP": 2*N*L,
    "prefill_intensity": 2*L/S,
    "prefill_lower_bound_ms": max(2*N*L/PEAK, N*S/BW)*1e3,
    "decode_FLOP_per_step_B1": 2*N,
    "decode_intensity_B1": 2/S,
    "decode_lower_bound_ms_B1": max(2*N/PEAK, N*S/BW)*1e3,
    "kv_1024_bytes_MHA": 2*32*1024*32*128*2,
    "kv_8192_bytes_MHA": 2*32*8192*32*128*2,
    "qwen_config_derived_params": 32*(4*4096**2+3*4096*11008+3*4096+2*4096)+4096+2*151936*4096,
    "ideal_weight_bandwidth_fraction_prefill": (N*S)/(2*N*L/PEAK)/BW,
    "ideal_weight_bandwidth_fraction_decode_B64": (N*S)/(2*N*64/PEAK)/BW,
    "hardware": [{"name": name, "peak_TFLOPs": peak, "bandwidth_TBs": bw, "ridge": peak/bw,
                  "upper_bounds_TFLOPs": {str(i): min(peak,bw*i) for i in [1,64,1024]}}
                 for name,peak,bw,_ in configs],
}
assert math.isclose(results["prefill_lower_bound_ms"], 143.36)
assert math.isclose(results["decode_lower_bound_ms_B1"], 7)
assert results["kv_1024_bytes_MHA"] == 536870912
(OUT / "calculation_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n', encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=2))
