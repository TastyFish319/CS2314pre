#!/usr/bin/env python3
"""Qwen3-8B analytical Roofline. CPU-only arithmetic; never loads a model.

FMA = 2 FLOPs. Rates use decimal SI; BF16 storage uses 2 bytes/element.
All plotted performance points are ceilings under assumed traffic, not observations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from numbers import Integral
from pathlib import Path
import platform

import numpy as np

MODEL = {
    'name': 'Qwen3-8B', 'num_layers': 36, 'hidden_size': 4096,
    'intermediate_size': 12288, 'num_attention_heads': 32,
    'num_key_value_heads': 8, 'head_dim': 128, 'vocab_size': 151936,
    'weight_dtype': 'bf16', 'bytes_per_element': 2, 'tie_word_embeddings': False,
}
HARDWARE = {
    'name': 'Ascend 910B3', 'peak_bf16_tflops': 294.912,
    'hbm_bandwidth_tb_s': 1.6, 'hbm_capacity_gib': 64,
}
HYPOTHETICAL_HARDWARE = {
    'name': 'Hypothetical Accelerator', 'peak_bf16_tflops': 100.0,
    'hbm_bandwidth_tb_s': 2.0,
}
DEFAULT_SEQUENCES = [128, 1024, 4096, 8192, 32768]
DEFAULT_CONTEXTS = [1024, 4096, 8192, 32768]
DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'mlp_gate', 'mlp_up', 'mlp_down']
SOURCES = {
    'model_config': 'https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json',
    'qwen3_implementation': 'https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py',
    'roofline': 'https://doi.org/10.1145/1498765.1498785',
    'cset': 'https://cset.georgetown.edu/publication/pushing-the-limits-huaweis-ai-chip-tests-u-s-export-controls/',
    'huawei_mfu': 'https://github.com/huaweicloud/huaweicloud-skills/blob/master/skills/ai/modelarts/huawei-cloud-ascend-op-mfu-calculator/references/mfu-calculation-methodology.md',
    'flashattention': 'https://arxiv.org/abs/2205.14135',
}


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')


def ridge_point(hardware=HARDWARE):
    """TFLOP/s divided by TB/s is directly FLOP/byte (both decimal)."""
    for key in ('peak_bf16_tflops', 'hbm_bandwidth_tb_s'):
        if not math.isfinite(hardware[key]) or hardware[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    return hardware['peak_bf16_tflops'] / hardware['hbm_bandwidth_tb_s']


def parameter_statistics(model=MODEL):
    """Per-module parameters across all layers, including learned norm scales.

    Q/K norm has one head_dim scale per norm per layer, shared across heads.
    No linear biases in Qwen3. Embedding and LM head are separate by default.
    """
    d, ff, layers = model['hidden_size'], model['intermediate_size'], model['num_layers']
    q = model['num_attention_heads'] * model['head_dim']
    kv = model['num_key_value_heads'] * model['head_dim']
    p = dict(q_proj=layers*d*q, k_proj=layers*d*kv, v_proj=layers*d*kv,
             o_proj=layers*q*d, mlp_gate=layers*d*ff,
             mlp_up=layers*d*ff, mlp_down=layers*ff*d)
    p['linear'] = sum(p.values())
    p['input_embedding'] = d * model['vocab_size']
    p['lm_head'] = d * model['vocab_size']
    p['rmsnorm'] = (2 * layers + 1) * d
    p['qk_norm'] = 2 * layers * model['head_dim']
    shared = p['input_embedding'] if model['tie_word_embeddings'] else 0
    p['total'] = p['linear'] + p['input_embedding'] + p['lm_head'] + p['rmsnorm'] + p['qk_norm'] - shared
    return p


def kv_bytes_per_token(model=MODEL):
    """Both K and V, all layers; GQA uses KV heads for storage."""
    return 2 * model['num_layers'] * model['num_key_value_heads'] * model['head_dim'] * model['bytes_per_element']


def prefill_flops(sequence, batch=1, lm_head_mode='last', model=MODEL):
    """Ideal triangular causal attention; last logits per sequence by default."""
    _integer(sequence, 'sequence')
    _integer(batch, 'batch')
    if lm_head_mode not in ('last', 'all'):
        raise ValueError('lm_head_mode must be last or all')
    p = parameter_statistics(model)
    f = {key: 2 * p[key] * batch * sequence for key in MODULES}
    pairs = sequence * (sequence + 1) // 2
    each = 2 * batch * model['num_layers'] * model['num_attention_heads'] * model['head_dim'] * pairs
    f.update(attention_qk=each, attention_av=each,
             lm_head=2*p['lm_head']*batch*(sequence if lm_head_mode == 'all' else 1))
    return f


def decode_flops(batch, context, model=MODEL):
    """One step for the WHOLE batch; context excludes the current token."""
    _integer(batch, 'batch')
    _integer(context, 'context', 0)
    p = parameter_statistics(model)
    f = {key: 2*p[key]*batch for key in MODULES}
    each = 2 * batch * model['num_layers'] * model['num_attention_heads'] * model['head_dim'] * (context + 1)
    f.update(attention_qk=each, attention_av=each, lm_head=2*p['lm_head']*batch)
    return f


def evaluate(phase, batch, length, hardware=HARDWARE, model=MODEL,
             extra_bytes_per_token=0, lm_head_mode='last'):
    """Return FLOPs, assumed HBM bytes, AI, ceiling and time lower bound.

    HBM weights: one read of linear and LM-head weights per invocation.
    Embedding row gathers and norm scales are omitted from BASE traffic.
    Resident storage includes all parameters + final KV, excludes workspace.
    """
    ridge = ridge_point(hardware)
    _nonnegative(extra_bytes_per_token, 'extra_bytes_per_token')
    _integer(batch, 'batch')
    p = parameter_statistics(model)
    kv = kv_bytes_per_token(model)
    weights = model['bytes_per_element'] * (p['linear'] + p['lm_head'])
    if phase == 'prefill':
        f = prefill_flops(length, batch, lm_head_mode, model)
        kv_read, kv_write = 0, batch * length * kv
        activation = batch * length * extra_bytes_per_token
        resident_kv = kv_write
        outputs = batch  # first next-token distribution per sequence
    elif phase == 'decode':
        f = decode_flops(batch, length, model)
        kv_read, kv_write = batch * length * kv, batch * kv
        activation = batch * extra_bytes_per_token
        resident_kv = kv_read + kv_write
        outputs = batch
    else:
        raise ValueError('phase must be prefill or decode')
    total_flops = sum(f.values())
    traffic = weights + kv_read + kv_write + activation
    ai = total_flops / traffic
    peak = hardware['peak_bf16_tflops'] * 1e12
    bandwidth = hardware['hbm_bandwidth_tb_s'] * 1e12
    compute_s, memory_s = total_flops / peak, traffic / bandwidth
    resident = p['total'] * model['bytes_per_element'] + resident_kv
    capacity = hardware.get('hbm_capacity_gib')
    if capacity is not None and (not math.isfinite(capacity) or capacity <= 0):
        raise ValueError('hbm_capacity_gib must be finite and positive')
    row = dict(phase=phase, batch_size=batch, sequence_length=length if phase == 'prefill' else 0,
               history_context=length if phase == 'decode' else 0,
               lm_head_mode=lm_head_mode if phase == 'prefill' else 'per_step',
               extra_bytes_per_token=extra_bytes_per_token, flops=total_flops,
               weight_bytes=weights, kv_read_bytes=kv_read, kv_write_bytes=kv_write,
               activation_extra_bytes=activation, traffic_bytes=traffic,
               arithmetic_intensity=ai, ridge_flop_per_byte=ridge,
               roof_tflops=min(peak, ai*bandwidth)/1e12,
               compute_time_lower_s=compute_s, memory_time_lower_s=memory_s,
               time_lower_s=max(compute_s, memory_s),
               bottleneck='memory' if ai < ridge else 'compute',
               bytes_per_output_token=traffic/outputs,
               resident_lower_bound_bytes=resident,
               fits_capacity_lower_bound='' if capacity is None else resident <= capacity*2**30)
    row.update({key+'_flops': value for key, value in f.items()})
    return row


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def module_rows(model=MODEL):
    p = parameter_statistics(model)
    rows = []
    for key in ['input_embedding'] + MODULES + ['lm_head', 'rmsnorm', 'qk_norm']:
        matrix = key in MODULES or key == 'lm_head'
        rows.append(dict(module=key, parameters_all_layers=p[key],
                         parameters_per_layer=p[key]//model['num_layers'] if key in MODULES or key == 'qk_norm' else '',
                         matrix_flops_per_token=2*p[key] if matrix else 0,
                         matrix_weight_bytes=p[key]*model['bytes_per_element'] if matrix else 0,
                         note='all transformer layers' if key in MODULES else
                         ('lookup only; base matrix FLOPs=0' if key == 'input_embedding' else
                          ('one vocabulary position' if key == 'lm_head' else 'nonmatrix; see report'))))
    return rows


def module_roofline_rows(phase, batch, length, hardware=HARDWARE, model=MODEL, lm_head_mode='last'):
    """Separate UNFUSED local operand model, NOT a decomposition of base bytes.

    Each linear operation reads weights, input, and writes output once to HBM.
    Attention materializes ideal triangular scores for prefill, dense scores for
    decode, with all operand storage BF16 and ideal GQA reuse. Fusion can remove
    these transfers. These local estimates must NOT be added to base traffic.
    """
    base = evaluate(phase, batch, length, hardware=hardware, model=model, lm_head_mode=lm_head_mode)
    p = parameter_statistics(model)
    d, ff, layers, s = model['hidden_size'], model['intermediate_size'], model['num_layers'], model['bytes_per_element']
    q, kv = model['num_attention_heads']*model['head_dim'], model['num_key_value_heads']*model['head_dim']
    tokens = batch*length if phase == 'prefill' else batch
    dims = {'q_proj': (d, q), 'k_proj': (d, kv), 'v_proj': (d, kv), 'o_proj': (q, d),
            'mlp_gate': (d, ff), 'mlp_up': (d, ff), 'mlp_down': (ff, d)}
    traffic = {'input_embedding': 2*s*tokens*d}
    for key, (din, dout) in dims.items():
        traffic[key] = s*(p[key] + layers*tokens*(din+dout))
    head_positions = tokens if phase == 'prefill' and lm_head_mode == 'all' else batch
    traffic['lm_head'] = s*(p['lm_head'] + head_positions*(d+model['vocab_size']))
    if phase == 'prefill':
        score_elements = batch*layers*model['num_attention_heads']*length*(length+1)//2
        each_attention = s*(layers*tokens*(q+kv) + score_elements)
    else:
        score_elements = batch*layers*model['num_attention_heads']*(length+1)
        each_attention = s*(batch*layers*(q+(length+1)*kv) + score_elements)
    traffic['attention_qk'] = each_attention
    traffic['attention_av'] = each_attention
    rows = []
    for key in ['input_embedding'] + MODULES + ['attention_qk', 'attention_av', 'lm_head']:
        f = base.get(key+'_flops', 0)
        byte_count = traffic[key]
        ai = f/byte_count if f else ''
        roof = min(hardware['peak_bf16_tflops'], ai*hardware['hbm_bandwidth_tb_s']) if f else ''
        rows.append(dict(phase=phase, batch_size=batch,
                         sequence_length=length if phase == 'prefill' else 0,
                         history_context=length if phase == 'decode' else 0,
                         lm_head_mode=base['lm_head_mode'], module=key,
                         parameters=p.get(key, 0), flops=f,
                         estimated_operand_bytes=byte_count, arithmetic_intensity=ai,
                         roof_tflops=roof,
                         time_lower_s=max(f/(hardware['peak_bf16_tflops']*1e12), byte_count/(hardware['hbm_bandwidth_tb_s']*1e12)),
                         traffic_model='unfused_local_operands_BF16; not additive to base',
                         note='lookup; matrix AI/roof not applicable' if not f else 'ideal operand reuse; score materialization assumed'))
    return rows


def _plot_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelsize': 12, 'axes.titlesize': 13,
                         'legend.fontsize': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none',
                         'savefig.facecolor': 'white'})
    return plt


def _save(figure, directory, stem):
    for suffix in ('png', 'svg'):
        figure.savefig(directory / f'{stem}.{suffix}', dpi=300, bbox_inches='tight')


def _grid(ax):
    ax.grid(True, which='major', alpha=0.25)
    ax.set_axisbelow(True)


def plot_roofline(plt, directory, prefill, decode, hardware):
    fig, ax = plt.subplots(figsize=(12, 6.5), layout='constrained')
    ridge = ridge_point(hardware)
    peak = hardware['peak_bf16_tflops']
    points = prefill + decode
    # Hardware overrides may move the ridge far outside the scenario points.
    # Keep both regions and the ridge inside the canvas for every valid peak/BW.
    xmax = max(1000, max(p['arithmetic_intensity'] for p in points)*3, ridge*30)
    xmin = min(0.65, min(p['arithmetic_intensity'] for p in points)*0.6, ridge/30)
    x = np.geomspace(xmin, xmax, 500)
    ax.loglog(x, hardware['hbm_bandwidth_tb_s']*x, '--', color='#9b9b9b', label='Memory bandwidth roof')
    ax.axhline(peak, color='#888888', linestyle=':', label='Compute roof')
    ax.loglog(x, np.minimum(peak, hardware['hbm_bandwidth_tb_s']*x), color='#222222', lw=2.3, label='Theoretical Roofline ceiling')
    ax.axvline(ridge, color='#7765a5', linestyle='--', lw=1.2)
    region_transform = ax.get_xaxis_transform()  # data x, axes-fraction y
    ax.text(ridge*1.13, 0.20, f'Ridge = {ridge:.2f}\nFLOP/Byte',
            color='#68518f', transform=region_transform)
    ax.text(math.sqrt(xmin*ridge), 0.92, 'Memory-bound',
            transform=region_transform, ha='center', weight='bold')
    ax.text(math.sqrt(ridge*xmax), 0.92, 'Compute-bound',
            transform=region_transform, ha='center', weight='bold')
    for i, row in enumerate(prefill):
        ai, perf = row['arithmetic_intensity'], row['roof_tflops']
        ax.scatter(ai, perf, s=65, marker='o', color='#0072B2', zorder=5,
                   label=f'Prefill B={row["batch_size"]}' if i == 0 else None)
        offset = -35 if i % 2 == 0 else -64
        ax.annotate(f'S={row["sequence_length"]:,}', (ai, perf), xytext=(0, offset),
                    textcoords='offset points', ha='center', color='#005f93', fontsize=10,
                    arrowprops={'arrowstyle': '-', 'color': '#0072B2', 'lw': 0.8})
    for i, row in enumerate(decode):
        ai, perf = row['arithmetic_intensity'], row['roof_tflops']
        ax.scatter(ai, perf, s=65, marker='s', color='#D55E00', zorder=5,
                   label=f'Decode C={row["history_context"]:,}' if i == 0 else None)
        ax.annotate(f'B={row["batch_size"]}', (ai, perf), xytext=(-9, 9), ha='right',
                    textcoords='offset points', fontsize=10, color='#a74700')
    ax.set(xlim=(xmin, xmax), ylim=(min(p['roof_tflops'] for p in points)*0.5, peak*3.4),
           xlabel='Arithmetic Intensity (FLOP/Byte)', ylabel='Performance ceiling (TFLOP/s)',
           title=f'Qwen3-8B theoretical Roofline | {hardware["name"]}')
    _grid(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc='outside upper center', ncol=3)
    fig.supxlabel('Every marker is an analytical ceiling under assumed traffic; no performance measurements.', fontsize=10)
    _save(fig, directory, '01_roofline')
    plt.close(fig)


def plot_compute_mix(plt, directory, prefill, decode, lm_head_mode):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), layout='constrained',
                             gridspec_kw={'height_ratios': [1.2, 1]})
    groups = [('Q/K/V/O projections', [x+'_flops' for x in MODULES[:4]], '#56B4E9'),
              ('MLP gate/up/down', [x+'_flops' for x in MODULES[4:]], '#009E73'),
              ('Attention QK + AV', ['attention_qk_flops', 'attention_av_flops'], '#E69F00'),
              ('LM head', ['lm_head_flops'], '#CC79A7')]
    for column, rows, scale, ylabel, phase in [(0, prefill, 1e12, 'Major matrix FLOPs (TFLOP)', 'Prefill'),
                                               (1, decode, 1e9, 'Major matrix FLOPs (GFLOP)', 'Decode')]:
        ax, share_ax = axes[0, column], axes[1, column]
        indices = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        totals = np.array([r['flops']/scale for r in rows])
        share_bottom = np.zeros(len(rows))
        for name, keys, color in groups:
            values = np.array([sum(r[k] for k in keys)/scale for r in rows])
            ax.bar(indices, values, bottom=bottom, width=0.65, label=name, color=color)
            shares = values/totals*100
            share_ax.bar(indices, shares, bottom=share_bottom, width=0.65, color=color)
            bottom += values
            share_bottom += shares
        for i, total in enumerate(totals):
            ax.annotate(f'{total:.2f}', (i, total), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
        ax.set_ylim(0, max(totals)*1.17)
        share_ax.set_ylim(0, 105)
        labels = [f'{r["sequence_length"] or r["history_context"]:,}' for r in rows]
        for current in (ax, share_ax):
            current.set_xticks(indices, labels, rotation=20)
            _grid(current)
        ax.set(ylabel=ylabel, title=f'{phase}, B={rows[0]["batch_size"]}')
        share_ax.set(xlabel='Input sequence length S' if phase == 'Prefill' else 'Historical context C',
                     ylabel='Share of matrix FLOPs (%)')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside upper center', ncol=4)
    note = 'last position only' if lm_head_mode == 'last' else 'all sequence positions'
    fig.supxlabel(f'Prefill LM head: {note}. Decode: one step. C excludes the current token.', fontsize=10)
    _save(fig, directory, '02_compute_mix')
    plt.close(fig)


def plot_decode_batch(plt, directory, decode, contexts, hardware):
    fig, ax = plt.subplots(figsize=(10.8, 6), layout='constrained')
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
    for index, context in enumerate(contexts):
        rows = [r for r in decode if r['history_context'] == context]
        color = colors[index % len(colors)]
        ax.plot([r['batch_size'] for r in rows], [r['arithmetic_intensity'] for r in rows],
                color=color, label=f'C={context:,}', lw=2)
        for r in rows:
            ax.scatter(r['batch_size'], r['arithmetic_intensity'], s=38, edgecolors=color,
                       facecolors='white' if r['fits_capacity_lower_bound'] is False else color, zorder=4)
    ax.axhline(ridge_point(hardware), ls='--', color='#555555', label=f'Ridge = {ridge_point(hardware):.2f}')
    ax.set_xscale('log', base=2)
    ax.set(xlabel='Batch size B', ylabel='Arithmetic Intensity (FLOP/Byte)',
           title=f'Decode batching and KV traffic | {hardware["name"]}')
    ax.legend(ncol=2)
    _grid(ax)
    fig.supxlabel('Hollow markers: weights + KV alone exceed HBM capacity (analytical what-if).\nFilled markers only pass this necessary capacity check; workspace is excluded.', fontsize=10)
    _save(fig, directory, '03_decode_batch_ai')
    plt.close(fig)


def plot_decode_traffic(plt, directory, decode, contexts):
    ncols = min(2, len(contexts))
    nrows = math.ceil(len(contexts)/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.6*nrows+1), squeeze=False, layout='constrained')
    for ax, context in zip(axes.flat, contexts):
        rows = [r for r in decode if r['history_context'] == context]
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        for key, label, color in [('weight_bytes', 'Amortized weights', '#56B4E9'),
                                   ('kv_read_bytes', 'Historical KV reads', '#E69F00'),
                                   ('kv_write_bytes', 'Current KV writes', '#009E73')]:
            values = np.array([r[key]/r['batch_size']/1e9 for r in rows])
            ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.72)
            bottom += values
        ax.set_ylim(0, max(bottom)*1.12)
        ax.set_xticks(x, [str(r['batch_size'])+('*' if r['fits_capacity_lower_bound'] is False else '') for r in rows])
        ax.set(xlabel='Batch size B', ylabel='HBM traffic / output token (GB)', title=f'C={context:,}')
        ax.text(0.97, 0.94, 'KV write = 144 KiB/token\n(too small to see at this scale)',
                transform=ax.transAxes, ha='right', va='top', fontsize=9)
        _grid(ax)
    for ax in list(axes.flat)[len(contexts):]:
        ax.set_visible(False)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='outside upper center', ncol=3)
    fig.supxlabel('Ideal assumed HBM traffic, decimal GB. *: weights + KV exceed capacity. KV is private to each request.', fontsize=10)
    _save(fig, directory, '04_decode_traffic')
    plt.close(fig)


def plot_prefill_sensitivity(plt, directory, prefill, extras, hardware):
    fig, ax = plt.subplots(figsize=(10.8, 6), layout='constrained')
    for i, extra in enumerate(extras):
        rows = [r for r in prefill if r['extra_bytes_per_token'] == extra*2**20]
        label = 'Base: weights + KV' if extra == 0 else f'+{extra:g} MiB/token (assumed)'
        ax.plot([r['sequence_length'] for r in rows], [r['arithmetic_intensity'] for r in rows],
                marker=['o', 's', '^', 'D'][i % 4], linestyle=['-', '--', '-.', ':'][i % 4], lw=2, label=label)
    ax.axhline(ridge_point(hardware), ls='--', color='#555555', label=f'Ridge = {ridge_point(hardware):.2f}')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set(xlabel='Input sequence length S', ylabel='Arithmetic Intensity (FLOP/Byte)',
           title=f'Prefill activation-traffic sensitivity, B={prefill[0]["batch_size"]}')
    ax.legend(loc='upper left')
    _grid(ax)
    fig.supxlabel('Extra activation bytes are sensitivity assumptions, not measured data.\nBase excludes activation intermediates, embedding gathers and norm traffic.', fontsize=10)
    _save(fig, directory, '05_prefill_sensitivity')
    plt.close(fig)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hardware', choices=['ascend', 'hypothetical'], default='ascend')
    parser.add_argument('--peak-tflops', type=float, help='Override BF16 matrix peak in decimal TFLOP/s')
    parser.add_argument('--bandwidth-tb-s', type=float, help='Override HBM peak in decimal TB/s')
    parser.add_argument('--capacity-gib', type=float, help='Override HBM capacity in GiB')
    parser.add_argument('--seq-lengths', nargs='+', type=int, default=DEFAULT_SEQUENCES)
    parser.add_argument('--context-lengths', nargs='+', type=int, default=DEFAULT_CONTEXTS)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=DEFAULT_BATCHES)
    parser.add_argument('--prefill-batch', type=int, default=1)
    parser.add_argument('--lm-head-mode', choices=['last', 'all'], default='last')
    parser.add_argument('--extra-mib-per-token', nargs='+', type=float, default=[0, 2, 8])
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent,
                        help='Directory for figures/ and results/; default is script directory')
    args = parser.parse_args(argv)
    try:
        _integer(args.prefill_batch, 'prefill_batch')
        for name in ('seq_lengths', 'context_lengths', 'batch_sizes'):
            for value in getattr(args, name):
                _integer(value, name, 0 if name == 'context_lengths' else 1)
            setattr(args, name, sorted(set(getattr(args, name))))
        for value in args.extra_mib_per_token:
            _nonnegative(value, 'extra_mib_per_token')
        args.extra_mib_per_token = sorted(set([0., *args.extra_mib_per_token]))
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_arguments(argv)
    hardware = dict(HARDWARE if args.hardware == 'ascend' else HYPOTHETICAL_HARDWARE)
    overridden = False
    for arg, key in [('peak_tflops', 'peak_bf16_tflops'), ('bandwidth_tb_s', 'hbm_bandwidth_tb_s'), ('capacity_gib', 'hbm_capacity_gib')]:
        if getattr(args, arg) is not None:
            hardware[key] = getattr(args, arg)
            overridden = True
    if overridden:
        hardware['name'] += ' (custom parameters)'
    try:
        ridge_point(hardware)
        evaluate('decode', 1, 0, hardware=hardware)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    results, figures = args.output_dir/'results', args.output_dir/'figures'
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    prefill = [evaluate('prefill', args.prefill_batch, s, hardware=hardware,
                        extra_bytes_per_token=extra*2**20, lm_head_mode=args.lm_head_mode)
               for extra in args.extra_mib_per_token for s in args.seq_lengths]
    decode = [evaluate('decode', b, c, hardware=hardware)
              for c in args.context_lengths for b in args.batch_sizes]
    write_csv(results/'prefill_results.csv', prefill)
    write_csv(results/'decode_results.csv', decode)
    write_csv(results/'module_summary.csv', module_rows())
    local_rows = []
    for sequence in args.seq_lengths:
        local_rows.extend(module_roofline_rows('prefill', args.prefill_batch, sequence,
                                               hardware=hardware, lm_head_mode=args.lm_head_mode))
    for context in args.context_lengths:
        for batch in args.batch_sizes:
            local_rows.extend(module_roofline_rows('decode', batch, context, hardware=hardware))
    write_csv(results/'module_roofline_results.csv', local_rows)
    import matplotlib
    manifest = dict(hardware=hardware, reference_ascend_hardware=HARDWARE,
                    hypothetical_hardware=HYPOTHETICAL_HARDWARE, model=MODEL,
                    ridge_flop_per_byte=ridge_point(hardware),
                    parameters=parameter_statistics(), kv_bytes_per_token=kv_bytes_per_token(),
                    assumptions={'weight_read_count_per_invocation': 1, 'fma_flops': 2,
                                 'attention': 'ideal causal triangular prefill; C+1 decode',
                                 'capacity_check': 'BF16 all parameters + final KV only; workspace excluded',
                                 'base_traffic_excludes': 'embedding gathers, norms, logits writes, activation intermediates',
                                 'module_roofline_traffic': 'separate unfused BF16 operand model; not additive to base',
                                 'measurement_status': 'theoretical only; no hardware or model execution'},
                    user_reported_device={'chip': 'Ascend 910B3', 'npu_id': 4,
                                          'hbm_display_mb': 65536, 'npu_smi_version': '25.5.0',
                                          'verification': 'provided by user; not queried by this script'},
                    run_arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    units={'TFLOP': 1e12, 'TB': 1e12, 'GB': 1e9, 'GiB': 2**30, 'MiB': 2**20, 'KiB': 2**10},
                    environment={'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
                    source_urls=SOURCES)
    (results/'hardware_config.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    plt = _plot_setup()
    base_prefill = [r for r in prefill if r['extra_bytes_per_token'] == 0]
    # Requested roofline decode points when available; custom CLI grids fall back to their own values.
    c = 1024 if 1024 in args.context_lengths else args.context_lengths[0]
    selected_batches = [b for b in [1, 8, 32, 64] if b in args.batch_sizes] or args.batch_sizes
    selected_decode = [r for r in decode if r['history_context'] == c and r['batch_size'] in selected_batches]
    b = 1 if 1 in args.batch_sizes else args.batch_sizes[0]
    mix_decode = [r for r in decode if r['batch_size'] == b]
    plot_roofline(plt, figures, base_prefill, selected_decode, hardware)
    plot_compute_mix(plt, figures, base_prefill, mix_decode, args.lm_head_mode)
    plot_decode_batch(plt, figures, decode, args.context_lengths, hardware)
    plot_decode_traffic(plt, figures, decode, args.context_lengths)
    plot_prefill_sensitivity(plt, figures, prefill, args.extra_mib_per_token, hardware)
    p = parameter_statistics()
    print(f'{hardware["name"]}: ridge = {ridge_point(hardware):.6f} FLOP/Byte')
    print(f'Linear = {2*p["linear"]/1e9:.9f} GFLOP/token; LM head = {2*p["lm_head"]/1e9:.9f} GFLOP/position')
    print(f'KV = {kv_bytes_per_token()/1024:g} KiB/token; parameters = {p["total"]:,}')
    for row in module_rows():
        print(f'{row["module"]:16s} {row["parameters_all_layers"]:>13,d} params; matrix FLOPs/token = {row["matrix_flops_per_token"]:,}')
    print(f'Wrote {len(prefill)} prefill rows, {len(decode)} decode rows, module CSV, config JSON and 5 PNG/SVG pairs to {args.output_dir}')
    print('All performance values are theoretical ceilings; time values are lower bounds.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
