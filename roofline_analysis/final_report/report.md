# Final Report

[目录与附件说明](README.md) · [理论计算工具](../qwen3_theory/qwen3_roofline/README.md) · [实测实验资料](../qwen3_experiments/README.md)

***为保留报告完整性，其中还是会有废话，但每一个 section 开头我都会加以说明重点怎么看，请关注粗斜体的 comments***

# 一、理论分析

本文根据模型结构估算 Prefill 与 Decode 的主要矩阵计算量和高带宽内存（HBM）流量，再计算算术强度、性能上限与时间下界。

## 1. Roofline 与硬件配置

***把 [1.2](#theory-hardware) 的配置说明一下即可***

### 1.1 Roofline 模型

Roofline 用计算吞吐和内存带宽两个约束，分析一次计算的理论性能上限。本文使用以下符号：

| 符号 | 含义 | 单位 |
|---|---|---|
| `F` | 一次 Prefill 调用或一个 Decode Batch 步骤的主要矩阵计算量 | FLOP |
| `D` | 对应调用或步骤中估算的 HBM 数据流量 | Byte |
| `I` | 算术强度，即每搬运一字节所完成的浮点运算数 | FLOP/Byte |
| `P_peak`、`P_roof` | 硬件矩阵计算峰值、Roofline 性能上限 | FLOP/s |
| `BW_peak` | HBM 峰值带宽 | Byte/s |
| `t_roof` | 给定计算量与流量假设下的时间下界 | s |

算术强度由计算量除以访存量得到：

$$
I=\frac{F}{D}.
$$

一次计算同时受矩阵计算峰值和带宽约束，因此其性能上限取两者的较小值：

$$
P_{\mathrm{roof}}=\min\left(P_{\mathrm{peak}},I\,BW_{\mathrm{peak}}\right).
$$

同理，计算和搬运数据各需要一定时间，整个过程不可能快于其中较慢的一项：

$$
t_{\mathrm{roof}}=\max\left(\frac{F}{P_{\mathrm{peak}}},\frac{D}{BW_{\mathrm{peak}}}\right).
$$

[Roofline 原始论文](https://doi.org/10.1145/1498765.1498785)

<a id="theory-hardware"></a>

### 1.2 Ascend 910B3 参数与单位

| 项目 | 采用值 |
|---|---|
| 芯片 | Ascend 910B3 |
| NPU ID / npu-smi 版本 | 4 / 25.5.0 |
| HBM 容量 | 64 GiB |
| BF16/FP16 矩阵峰值 `P_peak` | 294.912 TFLOP/s |
| HBM 带宽 `BW_peak` | 1.6 TB/s |

[华为 Cloud MFU 资料](https://github.com/huaweicloud/huaweicloud-skills/blob/master/skills/ai/modelarts/huawei-cloud-ascend-op-mfu-calculator/references/mfu-calculation-methodology.md)

记脊点为 `I*`，即计算上限与带宽上限相等时的算术强度：

$$
I^*=\frac{P_{\mathrm{peak}}}{BW_{\mathrm{peak}}}
=\frac{294.912\times10^{12}}{1.6\times10^{12}}
=184.32\ \mathrm{FLOP/Byte}.
$$

`I<I*` 时整体模型受带宽限制；`I>I*` 时受矩阵计算峰值限制。

本文按乘加一次计 2 FLOPs；速率和流量使用十进制：1 GFLOP=10⁹ FLOP、1 TFLOP=10¹² FLOP、1 GB=10⁹ Byte、1 TB=10¹² Byte。容量单位使用二进制：1 GiB=2³⁰ Byte、1 MiB=2²⁰ Byte、1 KiB=2¹⁰ Byte。

## 2. Qwen3-8B 模型配置

***对 Qwen 模型结构的说明，看有没有必要说明***

### 2.1 结构与符号

分析对象是原始 Qwen/Qwen3-8B，[官方配置](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)

| 符号 | 含义 | 数值 |
|---|---|---:|
| `L` | Transformer 层数 | 36 |
| `d` | 隐藏向量维度 | 4096 |
| `d_ff` | 门控 MLP 的中间维度 | 12288 |
| `H_q` | Query heads 数量 | 32 |
| `H_kv` | Key/Value heads 数量 | 8 |
| `d_h` | 每个注意力头的维度 | 128 |
| `V` | 词表大小 | 151936 |
| `s` | 一个 BF16 元素的存储字节数 | 2 Byte |

Q、K、V 分别表示查询、键和值张量，O Projection 将注意力输出映射回隐藏维度。

Qwen3-8B 采用分组查询注意力（GQA）：每 4 个 Query heads 共享一组 K/V，但仍有 32 个 Query heads 参与注意力计算。因此，**注意力 FLOPs 使用 `H_q`，KV 存储使用 `H_kv`**。

## 3. Prefill：从模型结构到图表数据

***后续两部分都旨在给出最终图表中相应数据的计算过程，为便于理解此处全部保留，但汇报时可以列公式与数据表格，然后一笔带过，重点呈现图表***

Prefill 处理输入序列并建立 KV Cache。记 `B` 为同时处理的请求数（Batch Size），`S` 为每条请求的输入 Token 数；本报告的 Prefill 配图均取 **B=1**。

### 3.1 计算量：逐 Token 线性计算与因果注意力

全部 B×S 个 Token 都需要经过 Transformer 线性层，因此线性计算量为 `2BSN_linear`。

注意力包含两次矩阵运算：QKᵀ 计算查询与键的相关分数，AV 根据注意力权重对值向量加权求和，其中 A 表示经过 Softmax 的注意力权重。理想因果注意力中，位置 i 只能关注第 1 到第 i 个位置，故有效位置对共有 `1+2+…+S=S(S+1)/2` 个。QKᵀ 与 AV 对每个位置对、每个 Query head 分别需要 `2d_h` FLOPs，因此两项各为：

$$
F_{\mathrm{QK,prefill}}=F_{\mathrm{AV,prefill}}
=2BLH_qd_h\frac{S(S+1)}{2}.
$$

这里的 `F_QK,prefill`、`F_AV,prefill` 分别表示整个 Prefill 调用的 QKᵀ 和 AV 计算量。它们随 S 近似二次增长，线性层则只随 S 一次增长。

记 Prefill 总主要矩阵计算量为 `F_prefill`。默认每条序列仅对最后一个位置执行 LM Head，故总量为：

$$
F_{\mathrm{prefill}}
=2BSN_{\mathrm{linear}}
+4BLH_qd_h\frac{S(S+1)}{2}
+2BN_{\mathrm{head}}.
$$

三项依次对应 Transformer 线性层、QKᵀ 与 AV、LM Head。最后位置的词表分数足以生成首个后续 Token；前面的所有位置仍需完成 Transformer 计算。如果框架对整段序列计算词表分数，最后一项应改为 `2BSN_head`。

以 **B=1、S=1024** 为例，图 5.1 中 Prefill 柱的分项来自下表。

| 计算组成 | 计算依据 | 计算量（GFLOP） |
|---|---|---:|
| Q/K/V/O Projection | 第 2.2 节投影 FLOPs/Token × 1024 | 3092.376453 |
| MLP Gate/Up/Down | 第 2.2 节 MLP FLOPs/Token × 1024 | 11132.555231 |
| Attention QKᵀ | 代入因果位置对数及 32 个 Query heads | 154.769818 |
| Attention AV | 与 QKᵀ 的主要矩阵 FLOPs 相同 | 154.769818 |
| LM Head | 只计算最后一个位置 | 1.244660 |
| **合计** | 各项相加 | **14535.715979** |

合计即 **14.535716 TFLOP**。图 5.1 将 QKᵀ 和 AV 合为 Attention，将上表的 GFLOP 除以 1000 后画成 Prefill 的 TFLOP 柱；组成比例则由各项除以总量得到。

### 3.2 访存量：一次权重读取与 KV 写入

记 `D_weights` 为一次调用读取矩阵权重的流量。基础模型假设 Transformer 主要线性权重和 LM Head 权重各从 HBM 读取一次：

$$
D_{\mathrm{weights}}=s\left(N_{\mathrm{linear}}+N_{\mathrm{head}}\right)
=15.136195\ \mathrm{GB}.
$$

该流量不随本次调用中的 Token 数重复计数，体现了理想权重复用。记 `D_KV/token` 为一个 Token 在全部层中保存 K 与 V 所需的缓存大小：

$$
D_{\mathrm{KV/token}}=2LH_{kv}d_hs
=147456\ \mathrm{Byte}=144\ \mathrm{KiB}.
$$

系数 2 对应 K、V 两份缓存，其余因素分别对应层数、KV heads、每头维度及每元素字节数。

Prefill 需要为全部 B×S 个输入 Token 写入缓存。记其基础流量为 `D_prefill`，则：

$$
D_{\mathrm{prefill}}=D_{\mathrm{weights}}+BS D_{\mathrm{KV/token}}.
$$

两项分别是权重读取和 KV 写入。B=1、S=1024 时，KV 写入为 0.150995 GB，总流量为 **15.287190 GB**。此基础模型没有单列 Attention 分块重读、Embedding 行读取、logits 写回和激活中间张量，属于理想化流量估计。

### 3.3 算术强度与 Roofline 点

将同一次 Prefill 的计算量与流量相除，就得到图中的横坐标。S=1024 时：

$$
I_{\mathrm{prefill}}=\frac{F_{\mathrm{prefill}}}{D_{\mathrm{prefill}}}
=\frac{14.535715979264\times10^{12}}{15.287189504\times10^9}
=950.842925\ \mathrm{FLOP/Byte}.
$$

这里 `I_prefill` 表示 Prefill 算术强度。它高于脊点 184.32，因此理论性能上限取 294.912 TFLOP/s，时间下界约为 49.288 ms。其他 S 的结果使用完全相同的计算过程，完整场景表见第 6 节。

### 3.4 额外 Activation 流量的敏感性

为了分析基础模型对中间数据流量的敏感程度，记 `e` 为**整个模型每个输入 Token 的额外 HBM 流量**，`D_prefill,ext` 为扩展后的流量：

$$
D_{\mathrm{prefill,ext}}=D_{\mathrm{prefill}}+BS e.
$$

图 5.4 分别取 e=0、2 MiB/Token、8 MiB/Token，保持计算量不变，再计算 `F_prefill / D_prefill,ext`。例如 S=1024 时，三种算术强度依次为 **950.843、833.725、608.772 FLOP/Byte**。

## 4. Decode：从单步计算到 Batch 与上下文影响

Decode 每步为每条请求处理一个新 Token。沿用第 3 节的 Batch Size `B`，记 `C` 为**执行当前 Decode 步之前**每条请求已有的历史 Token 数；计入当前 Token 后，注意力长度为 C+1。下面的 F、D 默认对应整个 Batch 的一次步骤。

### 4.1 计算量：固定线性计算与长度相关注意力

每条请求每步只处理一个新 Token，所以 Transformer 线性层和 LM Head 的计算量分别为 `2N_linear` 与 `2N_head`。QKᵀ、AV 各需对 C+1 个位置计算，每项为 `2LH_qd_h(C+1)`。

记整个 Batch 的主要矩阵计算量为 `F_decode`，则：

$$
F_{\mathrm{decode}}
=B\left[2N_{\mathrm{linear}}+2N_{\mathrm{head}}+4LH_qd_h(C+1)\right].
$$

括号内是**每个输出 Token**的计算量，外乘 B 才得到整个 Batch 的单步计算量。注意力仍使用 32 个 Query heads，而非 8 个 KV heads。

以 **B=1、C=1024** 为例，图 5.1 中 Decode 柱的分项如下。

| 计算组成 | 计算依据 | 计算量（GFLOP） |
|---|---|---:|
| Q/K/V/O Projection | 第 2.2 节投影 FLOPs/Token | 3.019899 |
| MLP Gate/Up/Down | 第 2.2 节 MLP FLOPs/Token | 10.871636 |
| Attention QKᵀ | 对每个 Query head 计算 1025 个位置 | 0.302285 |
| Attention AV | 与 QKᵀ 的主要矩阵 FLOPs 相同 | 0.302285 |
| LM Head | 当前 Token 的一个词表分布 | 1.244660 |
| **合计** | 各项相加 | **15.740764** |

固定 B 时，只有 Attention 部分随 C 增长；固定 C 时，整个 Batch 的各项 FLOPs 均随 B 成正比。表中各项由未舍入值求和，因此显示值相加可能有末位差异。

### 4.2 访存量：共享权重与独立 KV

Decode 沿用第 3.2 节的权重流量与每 Token KV 大小。记其基础总流量为 `D_decode`：

$$
D_{\mathrm{decode}}=D_{\mathrm{weights}}
+BC D_{\mathrm{KV/token}}+B D_{\mathrm{KV/token}}.
$$

三项依次对应一次权重读取、B 条请求各自读取 C 个历史位置的 KV、写入 B 个当前 Token 的 KV。当前 K/V 假设可以直接供本步注意力使用，不额外计一次 HBM 回读。

B=1、C=1024 时，三项为 **15.136195、0.150995、0.000147 GB**，总量为 **15.287337 GB**。不同请求共享相同模型权重，但历史 KV 通常属于各自的序列，不能按同样方式跨请求复用。

### 4.3 每输出 Token 的流量与算术强度

一个 Decode Batch 步骤产生 B 个输出 Token。将流量除以 B，可直接得到图 5.3 中堆叠柱的三项：

$$
\frac{D_{\mathrm{decode}}}{B}
=\frac{D_{\mathrm{weights}}}{B}+C D_{\mathrm{KV/token}}+D_{\mathrm{KV/token}}.
$$

只有权重项被 B 摊销；历史 KV 读取随 C 增长，当前 KV 写入恒为 144 KiB/输出 Token。再用每输出 Token 的计算量除以这个流量，可得图 5.2 的算术强度：

$$
I_{\mathrm{decode}}(B,C)
=\frac{2N_{\mathrm{linear}}+2N_{\mathrm{head}}+4LH_qd_h(C+1)}
{D_{\mathrm{weights}}/B+(C+1)D_{\mathrm{KV/token}}}.
$$

`I_decode(B,C)` 表示给定 B、C 时的 Decode 算术强度。此式与“整个 Batch 的 F 除以整个 Batch 的 D”等价，避免混用 Batch 总量和每 Token 平均量。

以 C=1024 为例，第 5.2、5.3 节的相应数据为：

| B | 平均权重读取（GB/Token） | 历史 KV 读取（GB/Token） | 当前 KV 写入（GB/Token） | 总流量（GB/Token） | 算术强度（FLOP/Byte） |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.136195 | 0.150995 | 0.000147 | 15.287337 | 1.029660 |
| 8 | 1.892024 | 0.150995 | 0.000147 | 2.043167 | 7.704102 |
| 32 | 0.473006 | 0.150995 | 0.000147 | 0.624148 | 25.219583 |
| 64 | 0.236503 | 0.150995 | 0.000147 | 0.387645 | 40.606086 |

例如 B=1 时，15.740764 GFLOP 除以 15.287337 GB 得到 1.029660 FLOP/Byte。相应理论性能上限为 **1.647457 TFLOP/s**，整个步骤的时间下界为 **9.555 ms**。

### 4.4 上下文增长为何限制 Batch 收益

当 B 持续增大，平均权重流量趋近于零，但每条请求的 KV 流量仍然存在。因此算术强度趋于有限值：

$$
\lim_{B\to\infty}I_{\mathrm{decode}}(B,C)
=\frac{2N_{\mathrm{linear}}+2N_{\mathrm{head}}+4LH_qd_h(C+1)}
{(C+1)D_{\mathrm{KV/token}}}.
$$

此外，容量是同时驻留的数据量，不能与一次步骤的累计流量混为一谈。记权重加最终 KV 的驻留容量下限为 `M_lower`，则必须满足：

$$
M_{\mathrm{lower}}=sN_{\mathrm{total}}+B(C+1)D_{\mathrm{KV/token}}
\leq64\times2^{30}\ \mathrm{Byte}.
$$

第一项包含全部 BF16 参数，第二项是当前步骤结束后的 KV Cache。该条件尚未计入激活、临时工作区和运行时预留，因此仅是必要条件。

| 历史长度 C | 历史 KV 读取（GB/输出 Token） | B→∞ 时的算术强度（FLOP/Byte） | 仅按权重+KV 算出的 Batch 容量上界 |
|---:|---:|---:|---:|
| 1024 | 0.150995 | 104.15 | 346 |
| 4096 | 0.603980 | 29.05 | 86 |
| 8192 | 1.207960 | 16.53 | 43 |
| 32768 | 4.831838 | 7.13 | 10 |

此表解释图 5.2 中曲线的饱和趋势，以及图 5.2、5.3 的容量超限标记。四种上下文对应的理论极限均低于脊点 184.32，因此在此模型内，仅增加 Batch 不能使它们进入整体计算受限区；容量还会限制实际可选择的 Batch。

## 5. 图示分析

***汇报重点：slides 上重点放以下几张图表，并进行解释***

***[5.1](#theory-compute-mix), [5.3](#theory-decode-traffic), [6](#theory-roofline)***

***其他两张图视内容量决定，可以不放***

<a id="theory-compute-mix"></a>

### 5.1 Prefill 与 Decode 的计算组成

![Prefill and decode compute mix](../qwen3_theory/qwen3_roofline/figures/02_compute_mix.png)

**数据来源：第 3.1、4.1 节的计算分项。** 上排柱高是各模块 FLOPs，下排是各模块占总 FLOPs 的比例。Prefill 用 TFLOP，Decode 用 GFLOP，两侧都取 B=1；Prefill 的 LM Head 仅计算最后位置。

短输入下，MLP 占主要计算量；随着 S 增大，Prefill Attention 的二次项逐渐显著，其占比由 S=1024 时的 **2.13%** 增至 S=32768 时的 **41.03%**。Decode 的线性部分保持不变，Attention 则随 C+1 增长。

### 5.2 Batch Size 对 Decode 算术强度的影响

![Decode arithmetic intensity versus batch](../qwen3_theory/qwen3_roofline/figures/03_decode_batch_ai.png)

[SVG 矢量版](../qwen3_theory/qwen3_roofline/figures/03_decode_batch_ai.svg)

**数据来源：第 4.3 节的 `I_decode(B,C)`，容量标记来自第 4.4 节。** 对每个固定 C，将 B=1、2、4、8、16、32、64、128、256 依次代入公式，得到一条曲线。水平虚线是硬件脊点；空心点表示权重加 KV 已超过 64 GiB，保留这些点仅作理论外推。

短上下文下，权重摊销能明显提高算术强度；长上下文下，KV 占比增加，曲线更早趋平。小 Batch 时，更长上下文也增加了 Attention 计算，算术强度不一定更低，因此曲线可能交叉。所有曲线仍低于本硬件的脊点。

<a id="theory-decode-traffic"></a>

### 5.3 每个输出 Token 的 HBM 流量

![Decode traffic per output token](../qwen3_theory/qwen3_roofline/figures/04_decode_traffic.png)

[SVG 矢量版](../qwen3_theory/qwen3_roofline/figures/04_decode_traffic.svg)

**数据来源：第 4.3 节每输出 Token 流量的三个加项。** 蓝色是 `D_weights/B`，橙色是 `C × D_KV/token`，绿色是 `D_KV/token`。绿色恒为 144 KiB，在 GB 尺度下很薄；星号表示容量必要条件不满足。

例如 C=1024 时，B 从 1 增至 64，平均总流量从 **15.287337 GB** 降为 **0.387645 GB**，原因是权重读取被摊销；历史 KV 仍为 0.150995 GB/Token。C=32768 时，仅历史 KV 就达到 **4.831838 GB/Token**，因而 Batch 增大的收益明显受限。

### 5.4 Prefill 对额外激活流量的敏感性

![Prefill activation traffic sensitivity](../qwen3_theory/qwen3_roofline/figures/05_prefill_sensitivity.png)

[SVG 矢量版](../qwen3_theory/qwen3_roofline/figures/05_prefill_sensitivity.svg)

**数据来源：第 3.4 节的扩展流量，分子仍使用第 3.1 节的 Prefill FLOPs。** 额外流量越大，算术强度越低；2 MiB、8 MiB 均是人为设置的敏感性参数。图中较长序列在三种假设下仍位于脊点上方，只能说明该整体模型仍判为计算受限，不能推断实际内核没有访存代价。

<a id="theory-roofline"></a>

## 6. Roofline 上的理论场景

***重中之重，不过经过前文的说明，这张图应该不需要太多解释，是最终的 Roofline 建模***

![Qwen3-8B theoretical Roofline](../qwen3_theory/qwen3_roofline/figures/01_roofline.png)

[SVG 矢量版](../qwen3_theory/qwen3_roofline/figures/01_roofline.svg)

图中横坐标由第 3、4 节计算的 `I=F/D` 得到，纵坐标由第 1.1 节的 `P_roof` 得到。蓝色圆点是 Prefill，橙色方点是 C=1024 的 Decode。

| Prefill 输入长度 S（B=1） | 计算量（TFLOP） | 基础流量（GB） | 算术强度（FLOP/Byte） | 性能上限（TFLOP/s） | 时间下界（ms） |
|---:|---:|---:|---:|---:|---:|
| 128 | 1.784231 | 15.155069 | 117.732 | 188.371 | 9.472 |
| 1024 | 14.535716 | 15.287190 | 950.843 | 294.912 | 49.288 |
| 4096 | 61.849982 | 15.740174 | 3929.434 | 294.912 | 209.724 |
| 8192 | 133.594323 | 16.344154 | 8173.829 | 294.912 | 452.997 |
| 32768 | 771.868071 | 19.968033 | 38655.189 | 294.912 | 2617.283 |

| Decode Batch B（C=1024） | 单步计算量（GFLOP） | 单步总流量（GB） | 算术强度（FLOP/Byte） | 性能上限（TFLOP/s） | 单步时间下界（ms） |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.741 | 15.287 | 1.030 | 1.647 | 9.555 |
| 8 | 125.926 | 16.345 | 7.704 | 12.327 | 10.216 |
| 32 | 503.704 | 19.973 | 25.220 | 40.351 | 12.483 |
| 64 | 1007.409 | 24.809 | 40.606 | 64.970 | 15.506 |

Prefill 通过序列内权重复用提高算术强度，S=128 时仍受带宽限制，S≥1024 的这些场景则受计算峰值限制。高算术强度不代表延迟更短：长输入的总计算量也更大。

Decode 的较小 Batch 受到权重流量限制；Batch 增大后，KV 流量逐步成为主要约束。Batching 提高潜在总吞吐，但不保证单请求延迟降低：表中的时间对应整个 Batch 的同一步骤，不能除以 B 后当作每条请求的步骤延迟。

以上结论依赖理想权重复用、BF16 KV 和简化激活流量。整体 Roofline 下界没有逐项累加串行算子及非矩阵操作的耗时，也没有覆盖真实内存峰值。完整数值见 [Prefill CSV](../qwen3_theory/qwen3_roofline/results/prefill_results.csv) 与 [Decode CSV](../qwen3_theory/qwen3_roofline/results/decode_results.csv)；复算方法见 [README](../qwen3_theory/qwen3_roofline/README.md)。




# 二、实测数据

***这部分主要是我自己操刀写的，可以都看看，都有采用价值***

## 1. 实验配置

- 设备：Ascend 910B3
- 模型：Qwen3-8B，BF16
- 模型性能优化：`ge_graph` 图模式 + RMSNorm/AddRMSNorm 融合

## 2. 实机性能在 Roofline 下的呈现

![实测屋顶线](../qwen3_experiments/figures/roofline_validation_measured.png)

[实测分析报告](../qwen3_experiments/docs/hardware_design.md) · [原始测试汇总](../qwen3_experiments/results/bench_runs.csv) · [屋顶标定结果](../qwen3_experiments/results/roof_calibration.json)

## 3. 图片解读

### 3.1 超参数矩阵

| 配置 | 输入 token | decode batch | max_new_tokens |
|---|---|---|---|
| tok256_b1 | 256 | 1 | 64 |
| tok256_b4 | 256 | 4 | 64 |
| tok1024_b1 | 1024 | 1 | 64 |
| tok1024_b4 | 1024 | 4 | 64 |

### 3.2 Roofline 绘制

图中的 roofline 是根据实机标定的算力与带宽峰值，按照理论分析的计算过程绘制的

## 4. 数据分析

### 4.1 Decode 分析

- **几乎将带宽吃满**
- decode 的算力用量只有 0.057~0.236 ms，占峰值的 0.02%~0.09%
- 带宽下限约 12.3 ms（权重 15.1 GB 每步流一遍），实测 16.3~18.2 ms
- 达成率为实测带宽屋顶的 **70.0%~75.4%**，即 `0.93~3.48 TF/s` 对应 `1.25~4.95 TF/s`
- 结论：decode 已经没有数量级级别的软件空间；除非减少字节，否则上限约 1.3~1.4x

### 4.2 Prefill 分析

- 整体算力利用率：

| 配置 | 主要矩阵运算量（TFLOP） | Prefill 耗时（ms） | 有效矩阵吞吐（TFLOP/s） | 实测峰值利用率 | 标称参考峰值利用率 |
|---|---:|---:|---:|---:|---:|
| tok256_b1 | 3.576880 | 70.29 | 50.89 | 19.05% | 17.26% |
| tok256_b4 | 14.307522 | 111.19 | 128.68 | 48.17% | 43.63% |
| tok1024_b1 | 14.535716 | 101.66 | 142.98 | 53.53% | 48.48% |
| tok1024_b4 | 58.142864 | 325.91 | 178.40 | 66.79% | 60.49% |

- 矩阵乘法算力利用率：

| 统计范围 | 矩阵吞吐（TFLOP/s） | 实测峰值利用率 | 标称参考峰值利用率 |
|---|---:|---:|---:|
| 全部 MatMul 的执行时间范围 | 223.60 | 83.71% | 75.82% |
| gate_up 单类算子的执行时间范围 | 243.60 | 91.19% | 82.60% |

- 整体而言尚未遇到算力瓶颈，但矩阵乘法阶段已经几乎将算力用到极致
- 因为某些原因，矩阵乘法阶段的算力利用率不高
- 较大规模 Prefill 的权重复用较充分，整体更偏向计算受限，提高带宽的收益通常小于提高有效计算吞吐
- 短输入则需要同时考虑带宽、小矩阵效率与调度开销

### 4.3 权重流量与 KV 流量（在搬什么数据？有没有可能少搬？）

KV = 147456 B/token，权重 = 15.136 GB：

| 场景 | KV 字节 | 权重 : KV |
|---|---|---|
| 1K ctx, B1 | 0.15 GB | 99 : 1 |
| 8K ctx, B8 | 9.7 GB | 61 : 39 |
| 32K ctx, B8 | 38.7 GB | 28 : 72 |
| 32K ctx, B16 | 77.3 GB | 16 : 84 |
| 128K ctx, B8 | 154.6 GB | 9 : 91 |

- 都有带宽压力，提高带宽需求明显，但难以满足如此巨大的需求
- 短上下文，带宽压力主要来自权重搬运，考虑量化权重等手段（权重无法一直在片上）
- 长上下文，KV Cache 的压力显著增加，优化策略可以转向量化 KV Cache 等，让其尽可能多地留在片上，减少搬运量

短上下文优先保障权重的连续高速读取；长上下文、大 Batch 则要把更多存储容量、缓存和访存调度能力用于 KV。具体比例随 Batch 与上下文长度动态调整。

***后两部分是 section 4 的附录***

### 4.4 下限模型

对任一阶段，理论最短时间取“算力下限”和“带宽下限”的较大者：

```
T_floor = max( FLOPs / 267.12e12 , Bytes / 1.2364e12 )
```

注意这里的分母是**实测可达值**，因此得到的是“在这台机器上、这套软件栈下”的下限，而不是数据手册下限。

### 4.5 下限与实测对比

| 场景 | 计算下限 | 带宽下限 | T_floor | 实测 | 实测/T_floor |
|---|---|---|---|---|---|
| prefill 256 tok B1 | 13.39 ms | 12.27 ms | 13.39 ms | 70.29 ms | **5.25x** |
| prefill 256 tok B4 | 53.56 ms | 12.36 ms | 53.56 ms | 111.19 ms | 2.08x |
| prefill 1024 tok B1 | 54.42 ms | 12.36 ms | 54.42 ms | 101.66 ms | 1.87x |
| prefill 1024 tok B4 | 217.66 ms | 12.73 ms | 217.66 ms | 325.91 ms | 1.50x |
| decode B1 | 0.057 ms | 12.28 ms | 12.28 ms | 16.28 ms | **1.33x** |
| decode B4 | 0.229 ms | 12.38 ms | 12.38 ms | 17.61 ms | 1.42x |

## 5. Profiling 分析

### 5.1 Prefill Profiling

![prof-PREFILL](assets/profiling/prof-PREFILL.png)

- 矩阵乘法占据多数时间
- 较长的 FREE 时段，是 NPU 在等待 CPU 完成 Attention 的输入准备和算子下发，调度层面可以优化
- 仍存在独立的小算子，其启动与调度也会影响效率
- 规模越大，调度、启动层面的开销占比会变小（Matmul 的占比会升高）

### 5.2 Decode Profiling

![prof-DECODE](assets/profiling/prof-DECODE.png)

- 由屋顶线分析，其优化空间相对较小
- 大段 FREE 时段在搬运数据
- 小操作较多，能否融合（？）
- 通过 Batching，共享权重读取，提高吞吐

## 6. 总结

**Prefill：**

- 算力吃紧，提高其配比
- 调度层面优化空间
- 考虑算子融合，减少小算子启动开销
- 根据规模大小，优化效果会有区别

**Decode：**

- 带宽吃紧，可以提升
- 物理层面上，带宽永远无法满足如此大的需求，考虑量化
- 根据上下文长度不同，量化侧重点有所不同，权重 / KV Cache
- 适度提高 Batch Size，一次读取权重计算多个token（不能无限提高）
