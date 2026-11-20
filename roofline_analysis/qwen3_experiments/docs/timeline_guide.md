# Qwen3-8B 推理时间线导读：谁在干活，每个算子在干什么

[实验资料导航](../README.md) · [实测分析报告](hardware_design.md) · [Profiling 分析](profiling_report.md)

- 数据来源：[Prefill 时间线](../profiling/prefill_trace_view_perfetto.json)（16022 个事件，1 个 prefill step）与 [Decode 时间线](../profiling/decode_trace_view_perfetto.json)（10354 个事件，decode step 3/4/5）
- 用例：Qwen3-8B，BF16，`ge_graph` + 04 章 RMSNorm 融合，输入 1024 token，decode batch 1，`max_new_tokens=64`
- 模型结构：36 层，hidden 4096，32 Q head / 8 KV head（GQA），head_dim 128，MLP 中间维 12288，词表 151936
- 建议配合 [实测分析报告](hardware_design.md) 第 3.7 节一起看：那份报告给结论，这份文档解释“时间线为什么长这样”

---

## 0. 一句话版本

prefill 是**一条流、714 个 kernel、串行跑完**，时间主要花在 4 个 MatMul 和 1 个 attention 上，但另外有 **79 次 host 同步**把 23 ms 浪费在等设备上；decode 是**图回放、三条流协作**，MatMul 把 15 GB 权重流一遍就占掉 71.5% 的时间，attention 只搬 151 MB 却占了 18.7%，剩下的采样小算子串在控制流上。

---

## 1. 先认识时间线的“楼层”

打开 Perfetto 后，左侧会看到 5 个进程。它们不是 5 个程序，而是同一进程的不同观测视角：

```
┌─ Python (pid 43125) ─────────── CPU：Python 前向、aten::*、npu::*、.item() 同步
│    ├─ Thread 43125            主线程
│    └─ Thread 43575            dequeue 线程（等算子出队）
├─ CANN (pid 44160255) ────────── CPU：ACL 接口调用、算子下发
│    ├─ Thread 43125            主线程发起的 ModelLoad / Synchronize / Memcpy
│    └─ Thread 43575            算子下发线程：Node@launch、aclrtLaunchKernelWithHostArgs
├─ Ascend Hardware (pid 44160416) NPU 0：真正的 device 流
│    ├─ Stream 46               prefill 唯一计算流；decode 的模型执行/控制流
│    ├─ Stream 0 / Stream 1     decode 的两条图内计算流（lane 名，不是 stream id）
│    ├─ Stream 49               profiling 控制流（只有 ENABLE / DISABLE）
│    └─ Step Trace(Model ID:2)  decode 的 Iteration 1/2/3（仅 decode 有）
├─ Overlap Analysis (pid 44160960) 汇总泳道：Computing / Free / Communication
└─ AI Core Freq (pid 44160448)     计数器：1800 MHz
```

### 1.1 五个“楼层”分别回答什么问题

| 楼层 | 你能在这里看到 | 典型用途 |
|---|---|---|
| Python | `aten::matmul`、`npu::npu_rms_norm`、`ProfilerStep#N`、`aten::item` | Python/框架层有没有多余拷贝、同步、断图 |
| CANN（Thread 43575） | `Node@launch`、`aclrtLaunchKernelWithHostArgs`、`aclnnXxx` | 下发一个算子要多久、host 是否跟不上 |
| CANN（Thread 43125） | `ModelLoad`、`aclrtSynchronize*`、`Memcpy` | 一次性开销、显式同步点 |
| Ascend Hardware | 真正的 kernel（`MatMul`、`RmsNorm`、`FusedInferAttentionScore`…） | kernel 耗时、并行度、流间关系 |
| Overlap Analysis | `Computing` / `Free` / `Communication` | 设备是“忙”还是“空”，空档有多长（`NOP` 在设备流 lane 上） |

### 1.2 三个容易踩的坑

1. **`Stream 0` / `Stream 1` 是 Perfetto 的 lane 编号，不是真实 stream id。** 真实 id 在 `kernel_details.csv` 的 `Stream ID` 列：prefill 全部是 `46`；decode 是 `43`（主流）、`42`（attention 的 AI_CPU 流）、`46`（控制流）。
2. **`Computing` / `Free` / `NOP` 不是算子**，是 profiler 插进去的“设备忙/空闲”标记。`NOP` 在 decode 里有 864 个，是图回放的占位节点，不要算进算子数。
3. **`kernel_details.csv` 里同一层会出现两条 attention 记录**（一条 `AI_CPU`、一条 `MIX_AIC`），它们是同一个融合算子的两个执行部分，不是两个不同的 attention。

---

## 2. 每条流/线程负责什么

### 2.1 prefill：单流串行

```
Python 主线程 ──► CANN 下发线程 ──► Stream 46（唯一计算流，714 个 kernel 串行）
      ▲                                      │
      └────────── 79 次 aten::item 同步 ◄─────┘
```

| 角色 | 证据 | 说明 |
|---|---|---|
| Python 主线程 | `ProfilerStep#0` 113.18 ms；`aten::matmul` 145 次；`npu::npu_add_rms_norm` 72 次 | 逐层调用算子、拼 mask、取 token |
| CANN 下发线程 | `Node@launch` 714 次 / 10.86 ms；`aclrtLaunchKernelWithHostArgs` 714 次 / 7.03 ms | 每个 kernel 一次下发，平均 9.8 us |
| Stream 46 | 714 条 kernel，83.19 ms | 全部计算都在这条流上，**没有任何跨流并发** |
| Stream 49 | `PROFILING_ENABLE` / `DISABLE` 各 1 次 | profiler 开关，不参与计算 |
| 同步点 | `aten::item` 79 次 / 21.06 ms；`aclrtSynchronizeStreamWithTimeout` 75 次 / 14.93 ms | **每取一次标量就等一次设备**，是 23 ms gap 的主要嫌疑 |

### 2.2 decode：图回放，三流协作

```
                   ┌─ Stream 43（主流）: MatMul、MIX_AIC attention、RmsNorm、Scatter、Swish…
Stream 46 ─────────┼─ Stream 42（AI_CPU 流）: attention 的 AI_CPU 任务
（模型执行/控制）   └─ Stream 46（自身）: 采样小算子（Cumsum/ArgMax/Tril/Gather…）

Stream 46 上还有一个 NOTIFY_WAIT：每步 15.27 ms，就是“等整张图跑完”
```

| 角色 | 证据 | 说明 |
|---|---|---|
| Stream 43（主流） | 1638 条记录 / 3 步；`AI_CORE` 435、`MIX_AIC` 108、`AI_VECTOR_CORE` 876、`MIX_AIV` 219（另 57 个向量算子在 Stream 46） | 图里的绝大部分节点：145 个 MatMul/步 + 36 个 attention 计算部分 + 归一化/写 cache/激活 |
| Stream 42（AI_CPU 流） | 108 条 `FusedInferAttentionScore`，`Accelerator Core=AI_CPU`，5.5 ms / 3 步 | attention 的调度/控制部分；与 Stream 43 的 MIX 部分**时间上重叠 5.06 / 5.18 ms** |
| Stream 46（控制流） | `NOTIFY_WAIT` 3 次 / 45.81 ms；`MODEL_EXECUTE` 3 次；57 个采样向量算子 + 6 个 `Cumsum(AI_CPU)` | 调图、等图、做贪心采样与 mask 更新 |
| Step Trace | `Iteration 1/2/3` = 15.36 / 15.29 / 15.28 ms | 图本身的执行跨度，不含采样 |
| CANN 侧 | `Node@launch` 609 次（3 步）/ 22.17 ms | 图回放内部仍是逐节点下发，只是省掉了 Python 层开销 |

> 为什么 kernel 名带 `_1`、`_42`、`_144`？这是图编译给每个节点固化的唯一名字，`_N` 只是编号。`MatMul` 和 `MatMul_2` 是**不同层的同一个算子**，不是两种算子。

---

## 3. 算子字典：每个算子在 Qwen3-8B 里干什么

### 3.1 矩阵乘（AI_CORE / Cube）

形状列以 prefill（M=1024）为例；decode 时 M=1，K/N 不变。

| 算子 | 形状（M×K→N） | 模型语义 | prefill | decode/步 |
|---|---|---|---|---|
| `MatMulV3` | 1024×4096→6144 | **QKV 投影**（Q 4096 + K 1024 + V 1024） | 36 次，9.02 ms | 36 次，1.71 ms |
| `MatMulV3` | 1024×4096→4096 | **o_proj**（attention 输出投影） | 36 次，5.47 ms | 36 次，1.10 ms |
| `MatMulV2` | 1024×4096→24576 | **MLP gate+up**（24576 = 2×12288） | 36 次，30.47 ms | 36 次，5.36 ms |
| `MatMulV2` | 1024×12288→4096 | **MLP down** | 36 次，17.61 ms | 36 次，3.07 ms |
| `MatMulV2` | 1×4096→151936 | **lm_head**（输出词表投影） | 1 次，1.02 ms | 1 次，0.85 ms |

> prefill 的 lm_head 在 `kernel_details` 里形状字段显示为 `256x1->1`，这是 WeightNz 布局下的元数据，实际就是 4096→151936 的词表投影；它前面那两个 `TransData`（0.009 / 0.011 ms）就是给 lm_head 权重做布局转换。

### 3.2 Attention 与归一化

| 算子 | 单元 | 模型语义 | prefill | decode/步 |
|---|---|---|---|---|
| `FusedInferAttentionScore` | MIX_AIC（+AI_CPU） | **FlashAttention**：读整个 paged KV cache，做 QKᵀ→softmax→AV；GQA 8 个 KV head | 36 次，6.40 ms | 36 次，1.46 ms（+AI_CPU 1.83 ms） |
| `AddRmsNorm` / `InplaceAddRmsNorm` | AI_VECTOR | **残差相加 + RMSNorm**；每层 2 次（post-attention、post-MLP），decode 用 inplace 版本 | 72 次，2.93 ms | 72 次，0.19 ms |
| `RmsNorm` | AI_VECTOR | **Q/K 归一化**（每层 2 次）+ 第 0 层输入归一化（36×2+1 = 73） | 73 次，1.92 ms | 73 次，0.59 ms |

> 一个容易看漏的融合：`AddRmsNorm` 出现在 `down_proj` 之后，它同时做了“MLP 输出加回残差”和“下一层输入归一化”两件事；最后一层的这个 kernel 顺带就是**最终归一化**。所以每层只有 2 个 `AddRmsNorm`，而不是 3 个。

### 3.3 KV cache、位置编码、激活

| 算子 | 单元 | 模型语义 | prefill | decode/步 |
|---|---|---|---|---|
| `ScatterNdUpdate` | MIX_AIV | **把新的 K/V 写进 paged KV cache**，每层 2 次（K、V） | 72 次，1.56 ms | 72 次，0.27 ms |
| `ApplyRotaryPosEmb` | AI_VECTOR | **RoPE** 旋转位置编码，每层 1 次 | 36 次，1.26 ms | 36 次，0.11 ms |
| `Silu` + `Mul` | AI_VECTOR | **SwiGLU**：`silu(gate) * up`；prefill 里每个还会多一个 `Slice` 子 kernel | 各 72 次，合计 4.23 ms | 合并成 `SwishMul`，36 次，0.13 ms |
| `InplaceCopy` / `Slice` | AI_VECTOR | Q/K/V 拆分、cache 切片、视图整理 | 72 次，0.61 ms | — |
| `StridedSliceV2` | AI_VECTOR | decode 图里的切片/取数 | — | 72 次，0.13 ms |

### 3.4 每步一次的前处理与采样（不在 36 层循环里）

| 算子 | 语义 |
|---|---|
| `Embedding` / `GatherV2` | 查词表，得到 token 的 hidden |
| `Index` / `IndexByTensor` / `IndexSelect` | 取 position id、cache slot、block table |
| `Tril` | 构造**因果 mask**（下三角） |
| `Cumsum`（AI_CPU） | 累积和：算 position / attention mask 的前缀和，prefill 0.144 ms 是纯 CPU 串行 |
| `FloorDiv` / `Remainder` / `BroadcastTo` | 由 position 算 block 编号、页内偏移 |
| `BitwiseNot` / `OnesLike` / `ZerosLike` / `Fill` / `ReduceMax` | mask 的填充、取反、边界处理 |
| `ArgMax` / `Cast` | **贪心采样**：取 logits 最大值作为下一个 token |
| `Gather` / `GatherElements` | 按 index 取数（采样、cache 索引） |
| `Add` / `Adds` / `Muls` / `Sub` / `Max` | 标量/张量小运算（温度、位置、mask 修正） |

---

## 4. prefill：一层是怎么跑的

下面是一层的真实顺序（第 1 层，单位 ms，来自 `kernel_details.csv` 的 Start Time 排序）：

| # | 算子 | 单元 | 耗时 | 在干什么 |
|---|---|---|---|---|
| 1 | `MatMulV3` | AI_CORE ×20 | 0.253 | QKV 投影，1024×4096→6144 |
| 2 | `InplaceCopy` ×2 | AI_VECTOR ×40 | 0.017 | 把 QKV 结果切成 Q、K、V 三块 |
| 3 | `RmsNorm` ×2 | AI_VECTOR ×40 | 0.052 | Qwen3 的 q_norm、k_norm |
| 4 | `ApplyRotaryPosEmb` | AI_VECTOR ×40 | 0.034 | 对 Q、K 加 RoPE |
| 5 | `ScatterNdUpdate` ×2 + `Slice` | MIX_AIV ×40 | 0.052 | 把 K、V 写进 paged KV cache |
| 6 | `FusedInferAttentionScore` | MIX_AIC ×20 | 0.176 | 读全部 KV，做 flash attention |
| 7 | `MatMulV3` | AI_CORE ×20 | 0.154 | o_proj，4096→4096 |
| 8 | `AddRmsNorm` | AI_VECTOR ×40 | 0.043 | 残差 + post-attention RMSNorm |
| 9 | `MatMulV2` | AI_CORE ×20 | 0.846 | **MLP gate+up，4096→24576（单层最大头）** |
| 10 | `Silu` ×2 + `Mul` ×2 | AI_VECTOR ×40 | 0.118 | SwiGLU（各含一个 Slice 子 kernel） |
| 11 | `MatMulV2` | AI_CORE ×20 | 0.489 | MLP down，12288→4096 |
| 12 | `AddRmsNorm` | AI_VECTOR ×40 | 0.041 | 残差 + 下一层输入 RMSNorm（融合） |

合计约 **2.31 ms/层 × 36 层 ≈ 83 ms**，与 kernel 求和 83.19 ms 一致。

### 4.1 这层的三个观察

1. **只有 4 个 MatMul 是“大活”**：gate_up 0.846 + down 0.489 + qkv 0.253 + o_proj 0.154 = 1.74 ms，占单层 75%。
2. **其余 15 个都是向量小算子**：单个 0.008~0.18 ms，但数量多，合计约 0.57 ms（25%），而且每个都要一次 host 下发。
3. **同一时刻只有一个 kernel 在跑**：所有 kernel 的 `Stream ID` 都是 46，时间上首尾相接。prefill 没有 decode 那种“两条流重叠”的机会。

### 4.2 第 0 层前面还有一段“一次性”代码

在 `Embedding` → `Index` ×2 → 第 0 层输入 `RmsNorm`（0.026）之前，还有 17 个 mask/position 小 kernel：`OnesLike`、`Tril`、`LogicalNot`、`ReduceSum`、`Arange` ×2、`Cumsum`（0.144，AI_CPU）、`ZerosLike`、`FloorDiv`、`BroadcastTo`、`FloorMod`、`ReduceMax`、`Cast`、`GatherElements`、`Mul`、`Add`。这些是**每步一次的 mask/position 构造**，不在 36 层循环里，但同样各占一次下发。

---

## 5. decode：一层是怎么跑的

decode 的一层（来自 step 4 的时间线，单位 ms）：

| 顺序 | 算子 | 单元/Block | 耗时 | 说明 |
|---|---|---|---|---|
| 1 | `GatherV2` + `IndexByTensor` ×2 | AI_VECTOR | 0.016 | 取 position / cache slot |
| 2 | `MatMul` | AI_CORE ×12 | 0.049 | QKV，1×4096→6144 |
| 3 | `RmsNorm` ×2 | AI_VECTOR ×32 / ×8 | 0.017 | q_norm、k_norm |
| 4 | `ApplyRotaryPosEmb` | AI_VECTOR | 0.003 | RoPE |
| 5 | `ScatterNdUpdate` ×2 | MIX_AIV | 0.007 | 写 K/V cache（1 个 token） |
| 6 | `FusedInferAttentionScore` | **AI_CPU + MIX_AIC ×17** | 0.088 | 读 1024 个历史 token 的 KV |
| 7 | `MatMul` | AI_CORE ×20 | 0.027 | o_proj |
| 8 | `InplaceAddRmsNorm` | AI_VECTOR | 0.003 | 残差 + norm |
| 9 | `MatMul` | AI_CORE ×20 | 0.164 | **gate_up（单层最大头）** |
| 10 | `SwishMul` | AI_VECTOR | 0.004 | SwiGLU（图里已融合成一个 kernel） |
| 11 | `MatMul` | AI_CORE ×20 | 0.085 | down |
| 12 | `InplaceAddRmsNorm` | AI_VECTOR | 0.003 | 残差 + 下一层输入 norm |

合计约 **0.44 ms/层（36 层 ≈ 15.8 ms）**，再加 lm_head 0.85 ms 与采样约 0.15 ms ≈ 16.8 ms，与实测 16.89 ms 吻合。

### 5.1 prefill 与 decode 的结构差异

| 维度 | prefill | decode |
|---|---|---|
| 流 | 1 条（Stream 46） | 3 条（43 主流 / 42 AI_CPU / 46 控制） |
| 每层 kernel 数 | ~20 | ~17 |
| MatMul 形状 | M=1024 | M=1 |
| attention 单元 | MIX_AIC ×20 | AI_CPU + MIX_AIC ×17 |
| SwiGLU | `Silu` + `Mul` + 2 个 Slice（4 个 kernel） | `SwishMul`（1 个 kernel） |
| 归一化 | `AddRmsNorm` | `InplaceAddRmsNorm`（省一次拷贝） |
| 采样 | 末尾 1 次 ArgMax | 每步 1 次 ArgMax + 一串 mask/采样小算子 |

> 图模式对 decode 的优化是实打实的：SwiGLU 从 4 个 kernel 变 1 个、norm 从 out-of-place 变 in-place、整个 forward 只在 `TorchNpuGraphBase::Run` 里回放一次。

---

## 6. host 侧在忙什么（为什么 prefill 会空 23 ms）

### 6.1 prefill 的 host 时间线

| 位置 | 事件 | 次数 | 总耗时 |
|---|---|---|---|
| Python 主线程 | `ProfilerStep#0` | 1 | 113.18 ms |
| Python 主线程 | **`aten::item`** | **79** | **21.06 ms** |
| Python 主线程 | `aten::_local_scalar_dense` | 79 | 20.72 ms |
| Python 主线程 | `empty_tensor` | 828 | 9.41 ms |
| Python 主线程 | `aten::view` | 690 | 5.33 ms |
| CANN 主线程 | `aclrtSynchronizeStreamWithTimeout` | 75 | 14.93 ms |
| CANN 下发线程 | `Node@launch` | 714 | 10.86 ms |
| CANN 下发线程 | `aclrtLaunchKernelWithHostArgs` | 714 | 7.03 ms |
| Python dequeue 线程 | `Dequeue@aclnnXxx` | 606 | 34.0 ms |

- **79 次 `aten::item` 是重点**：每次 `.item()` 都要把标量从 device 拷回 host，必须等前面的 kernel 跑完，等于给 714 个 kernel 插了 79 个“必须等设备”的同步点。它与 `SynchronizeStream` 是同一批等待的两种记录，21.06 ms 的量级与 23.13 ms 的 gap 相当。
- 714 次下发 × 9.8 us ≈ 7 ms 是纯 launch 成本；剩下的是同步等待与框架开销。
- 对比 decode：`aten::item` 只有 21 次（7 次/步），所以 decode 的 gap 只有 0.15 ms。

### 6.2 decode 的 host 时间线

| 位置 | 事件 | 次数 | 总耗时 |
|---|---|---|---|
| Python | `ProfilerStep#3/4/5` | 3 | 22.21 / 21.35 / 20.83 ms |
| Python | `TorchDynamo Cache Lookup` | 3 | 4.52 ms |
| Python | `TorchNpuGraphBase::Run` | 3 | 1.02 ms |
| Python | `aten::item` | 21 | 1.08 ms |
| CANN 主线程 | `aclrtSynchronizeDeviceWithTimeout` | 7 | 44.98 ms |
| CANN 主线程 | `Node@launch` | 549 | 21.17 ms |
| CANN 主线程 | `Model@InputCopy` | 3 | 0.28 ms |

- `aclrtSynchronizeDeviceWithTimeout` 的 44.98 ms 是**等待**，不是计算：7 次 × 6.4 ms，等的是图跑完 + 采样结果。
- `Node@launch` 549 次说明图回放内部仍然逐节点下发，但已经不再经过 Python/框架层，所以 host 能跟上设备（gap 0.15 ms）。

---

## 7. 自己读这份 trace 的步骤

1. **先看 `Ascend Hardware` 的 Stream 43 / Stream 46**：确认 kernel 是不是首尾相接（prefill）还是有重叠（decode）。
2. **再对照 `kernel_details.csv` 的 `Stream ID` 和 `Accelerator Core`**：`AI_CORE` = Cube，`AI_VECTOR_CORE` = Vector，`MIX_AIC`/`MIX_AIV` = 混合核，`AI_CPU` = CPU 侧任务。
3. **找 host 瓶颈**：在 CANN lane 上看 `Node@launch` 的密度；在 Python lane 上搜 `aten::item`、`synchronize`、`copy_`。
4. **判断空档**：`Computing` 和 `Free` 是 Overlap Analysis 泳道；`Free` 在带 profiler 的运行里包含打点开销，真正的 gap 用“kernel 求和 vs 墙钟”算。
5. **数算子**：按 `Name` 去掉 `_数字` 后缀再聚合，否则会把同一层的 36 个 `MatMul_7`/`MatMul_42` 当成不同算子。

### 7.1 常用对照表

| 你在 trace 里看到 | 对应到模型 |
|---|---|
| `MatMulV3` 4096→6144 | QKV 投影 |
| `MatMulV3` 4096→4096 | o_proj |
| `MatMulV2` 4096→24576 | MLP gate+up |
| `MatMulV2` 12288→4096 | MLP down |
| `MatMulV2` 4096→151936 | lm_head |
| `FusedInferAttentionScore` | FlashAttention（读 KV cache） |
| `AddRmsNorm` / `InplaceAddRmsNorm` | 残差 + RMSNorm |
| `RmsNorm` | q_norm / k_norm / 第 0 层输入 norm |
| `ScatterNdUpdate` | KV cache 写入 |
| `ApplyRotaryPosEmb` | RoPE |
| `Silu`+`Mul` / `SwishMul` | SwiGLU |
| `Tril` / `Cumsum` / `FloorDiv` | 因果 mask / position / block table |
| `ArgMax` | 贪心采样 |

---

## 8. 从时间线直接读出的 5 条结论

1. **prefill 只有一条计算流**：714 个 kernel 全部在 Stream 46 上串行，没有并发可利用；想快只能减少 kernel 数或让每条 kernel 更快。
2. **prefill 的 23 ms gap 有明确的 host 侧证据**：79 次 `aten::item`（21.06 ms）与 75 次 `SynchronizeStream`（14.93 ms）是同一批等待的两种记录，再加上 714 次下发（7.03 ms），量级上正好覆盖 23 ms 的 gap。
3. **prefill 每层 20 个 kernel 里只有 4 个是 MatMul**：其余 16 个是 norm/rope/scatter/silu/mul/slice，单看每个都不到 0.05 ms，但合计占单层 25%，且每个都要一次下发。
4. **decode 的三条流是真的在协作**：attention 的 AI_CPU 部分（Stream 42，5.18 ms）与主流（Stream 43）重叠 5.06 ms，几乎完全被藏住；`NOTIFY_WAIT` 每步 15.27 ms 才是真正的串行等待。
5. **decode 的采样链路很轻但很“碎”**：Stream 46 上每步约 20 个采样/mask 小算子，总耗时不到 0.1 ms，却带来 7 次 `aten::item`；如果以后要压 decode 的 host 开销，这里是第一站。

---

## 9. 附：本文用到的原始文件

| 文件 | 内容 |
|---|---|
| [prefill_trace_view_perfetto.json](../profiling/prefill_trace_view_perfetto.json) | prefill 时间线（Perfetto 可直接拖入） |
| [decode_trace_view_perfetto.json](../profiling/decode_trace_view_perfetto.json) | decode step 3/4/5 时间线 |
| [kernel_details_prefill.csv](../profiling/kernel_details_prefill.csv) | 714 条 kernel 明细（Stream ID、Accelerator Core、Block Num、Input Shapes、pipe 占比） |
| [kernel_details_decode.csv](../profiling/kernel_details_decode.csv) | 1809 条 kernel 明细 |
| [step_trace_time.csv](../profiling/step_trace_time.csv) | Computing / Free / Stage / Preparing / Bubble |
| [api_statistic.csv](../profiling/api_statistic.csv) | host API 耗时（含 `aten::item`、`Node@launch`） |
| [op_statistic.csv](../profiling/op_statistic.csv) | 按算子聚合的耗时 |
