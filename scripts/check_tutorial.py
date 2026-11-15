"""Check tutorial structure, Markdown links/tables, assets and independent arithmetic.

python -m pip install markdown-it-py
python scripts/check_tutorial.py
This is document QA, not a GPU benchmark or validation of optional model execution.
"""
from pathlib import Path
import ast
import json
import math
import re
import xml.etree.ElementTree as ET

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "roofline_llm_tutorial.md"
doc = DOC_PATH.read_text(encoding="utf-8")
checks = []


def check(condition, label):
    if not condition:
        raise AssertionError(label)
    checks.append(label)


chapters = re.findall(r"^## (.+)$", doc, flags=re.M)
check(len(chapters) == 7, "exactly seven top-level chapters")
for i, number in enumerate("一二三四五六七"):
    check(chapters[i].startswith(f"第{number}章"), f"chapter {i+1} order")
check(doc.count("```" ) % 2 == 0, "code fences paired")
check(sum(line.strip() == "$$" for line in doc.splitlines()) % 2 == 0, "display math fences paired")
check("\ufffd" not in doc, "no Unicode replacement characters")

for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", doc):
    if not link.startswith(("http://", "https://", "#")):
        check((DOC_PATH.parent / link).is_file(), f"local link exists: {link}")
figures = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", doc)
check(len(figures) >= 6, "six or more referenced static diagrams")
for ref in ["1", "2", "3a", "3b", "3c", "4", "5", "6", "7", "7b", "8"]:
    check(f"**[{ref}]**" in doc, f"reference {ref} defined")

# Markdown-it parsing plus explicit source table widths: parsers may silently
# accept uneven rows, so checking only successful parsing would be insufficient.
tokens = MarkdownIt("commonmark").enable("table").parse(doc)
check(bool(tokens), "Markdown parsed")
check(sum(t.type == "table_open" for t in tokens) >= 15, "comparison tables parsed")
width = None
for line in doc.splitlines() + [""]:
    if line.startswith("|"):
        columns = len(re.split(r"(?<!\\)\|", line)) - 2
        if width is None:
            width = columns
        check(columns == width, "table row width consistent")
    else:
        width = None
for script in ROOT.glob("scripts/*.py"):
    ast.parse(script.read_text(encoding="utf-8"))
    check(True, f"Python syntax: {script.name}")
for svg in (ROOT / "assets").glob("*.svg"):
    ET.parse(svg)
    check(True, f"SVG XML: {svg.name}")

# Independent calculations, rather than taking printed manuscript values on trust.
peak, bandwidth, params, size, length = 100e12, 2e12, 7e9, 2, 1024
calc = json.loads((ROOT / "assets/calculation_results.json").read_text(encoding="utf-8"))
expected = {
    "ridge_flop_per_byte": peak/bandwidth,
    "weight_GB": params*size/1e9,
    "prefill_FLOP": 2*params*length,
    "prefill_intensity": 2*length/size,
    "prefill_lower_bound_ms": max(2*params*length/peak, params*size/bandwidth)*1000,
    "decode_FLOP_per_step_B1": 2*params,
    "decode_intensity_B1": 2/size,
    "decode_lower_bound_ms_B1": max(2*params/peak,params*size/bandwidth)*1000,
    "kv_1024_bytes_MHA": 2*32*1024*32*128*2,
    "kv_8192_bytes_MHA": 2*32*8192*32*128*2,
    "qwen_config_derived_params": 32*(4*4096**2+3*4096*11008+5*4096)+4096+2*151936*4096,
    "ideal_weight_bandwidth_fraction_prefill": 0.048828125,
    "ideal_weight_bandwidth_fraction_decode_B64": 0.78125,
}
for key, expected_value in expected.items():
    check(math.isclose(calc[key], expected_value, rel_tol=1e-12), f"arithmetic: {key}")
check(math.isclose(2*1024*4096**2/(2*(1024*4096+4096**2+1024*4096)), 682.6666666667), "linear GEMM full-traffic intensity")
check(math.isclose(4*32*1024**2*4096/1e12, .549755813888), "prefill full attention FLOPs")
kv_batch = 2*32*64*1024*4096*2
f_batch = 2*params*64 + 4*32*64*1024*4096
check(math.isclose(f_batch/(params*size+kv_batch), 19.238312070431423, rel_tol=1e-12), "B64 intensity after KV correction")
for name, peak_t, bw_t in [("计算优先",200,1),("带宽优先",50,4),("相对均衡",100,2)]:
    row = next(r for r in calc["hardware"] if r["name"] == name)
    check(row["ridge"] == peak_t/bw_t, f"hardware ridge: {name}")
    for intensity in [1,64,1024]:
        check(row["upper_bounds_TFLOPs"][str(intensity)] == min(peak_t,bw_t*intensity), f"hardware upper bound: {name}, I={intensity}")
print(f"PASS: {len(checks)} checks; {len(doc)} characters; {len(chapters)} chapters; {len(figures)} static figures.")
print("No GPU inference, FLOP counter or HBM measurement was performed by this checker.")
