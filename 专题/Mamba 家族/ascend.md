---
date: "2026-11-03"
title: "硬件：昇腾 NPU 上跑 Mamba 有什么特殊点"
type: "topic"
topic: "Mamba 家族与混合架构"
topic_id: "mamba-family"
part: "ascend"
part_title: "硬件：昇腾 NPU 上跑 Mamba 有什么特殊点"
order: 3
tags: ["昇腾 NPU", "Mamba", "GDN", "roofline", "vllm-ascend", "混合架构"]
---

# 硬件：昇腾 NPU 上跑 Mamba 有什么特殊点

> 读完这篇，你能：说清 scan 为什么喂不饱 Cube；复算 chunk 的算术强度，判断瓶颈在带宽还是算力；分清「总搬运量」与「单位字节算术强度」，不再把「省显存」当成「一定更快」。

## 为什么需要它

attention 的 prefill 是几个大矩阵乘，M 有几千行，Cube 吃得满。Mamba / GDN（Gated DeltaNet：Mamba-2 的选择性衰减 + delta rule 定向写入，[ICLR 2025](https://arxiv.org/abs/2412.06464)）换了一种算法：不存 KV，只维护固定大小的循环状态，每来一个 token 更新一次。它在 GPU 上有成熟的 Triton kernel，搬到昇腾上就冒出三个问题：为什么「能跑但不快」；为什么为 Mamba-2 prefill 写六个算子；混合模型里哪一半拖后腿。

主线只有一条：**scan 的计算形状和 Cube 的形状不匹配**。判断瓶颈用 roofline，能并行的数学依据是结合律。

## 核心概念

### 达芬奇 Cube / Vector

Cube 是昇腾 AI Core 里的矩阵单元，一条指令完成一个 16×16×16 块乘加（FP16 下 4096 次乘加）；Vector 做加法、乘、指数、归一化这类逐元素运算。Cube 像必须整托盘进出的压铸机，吞吐高但要来料成块；Vector 像手工工位，什么都能干，一次只动一小把。

要让 Cube 跑满，M、N、K 都得切成 16 的倍数，数据还得在片上反复复用。scan 每步只做「128×128 状态矩阵 × 128 维向量」，M=1，16 行里 15 行闲着，只能退回 Vector。不是昇腾特别差，是 scan 天生喂不饱矩阵单元。

### 逐元素算子与 roofline 脊点

算术强度 = 计算量（FLOP）÷ 搬运字节数；roofline 给的上界是 `min(峰值算力, 带宽 × 算术强度)`，两条线的交点是脊点：左侧带宽受限，右侧算力受限。

手算 decode 的 scan（单头，Dk=Dv=128，bf16）：状态 S 是 128×128×2 B = 32 KiB，读+写 64 KiB；计算是三次「矩阵 × 向量」（S·k、S·q、外积更新），约 3×2×128² ≈ 9.8 万 FLOP。算术强度 ≈ **1.5 FLOP/Byte**。910B 量级取 FP16 峰值 3×10¹⁴ FLOP/s、HBM 1.2–1.6 TB/s（厂商与评测口径，**存疑**），脊点 ≈ 3×10¹⁴ ÷ (1.2–1.6×10¹²) ≈ **190–250 FLOP/Byte**——差了约两个数量级。所以这类算子的方向是**少搬数据**（融合、块化、让状态留在片上），不是堆算力。

### 仿射映射复合

GDN / Mamba-2 的状态更新可以写成仿射映射 `S ↦ A_t·S + b_t`；两个仿射映射复合仍是仿射映射：`(A₂,b₂)∘(A₁,b₁) = (A₂A₁, A₂b₁+b₂)`，满足结合律 `(a∘b)∘c = a∘(b∘c)`——这就是并行扫描（O(log n) 步）的数学依据。注意 A 是 128×128 矩阵，扫的是矩阵；只有衰减为标量时才退化成 (α, β) 二元组。

### WY / UT 变换与 chunk 化

块内一串 rank-1 修正要先用 WY / UT 变换合成一个矩阵，才变成一次块级 GEMM；序列切成长 L 的块后，块内一次算完，块间只传一个 O(1) 的状态矩阵。

块内是 `qkᵀ`、`attn@v` 这类 L×L×D 的 GEMM，Cube 吃得上。下面这组是**纸面推算、非实测**，但可逐步复算：计算量按 **4L²D + 4LD²** FLOP；口径 A 只算读入 q/k/v/o 四个张量（bf16 每元素 2 字节，共 **8LD 字节**），不含状态读写、也不含输出写回。于是 AI = (4L²D + 4LD²)/(8LD) = **(L + D)/2**。取 D=128：

| L | 口径 A（8LD 字节） | 口径 B（10LD + 4D² 字节） |
| --- | --- | --- |
| 128 | 128 | ≈73 |
| 256 | 192 | 128 |
| 512 | 320 | ≈233 |

口径 B 把 o 写回（2LD 字节）和状态 S 的读+写（4D² 字节）也算进分母，AI = 2L(L+D)/(5L+2D)，同样可代入。

**结论要用区间，不要用单一阈值**：脊点 190–250 本身是**存疑**值。口径 A 下，越过脊点需要 L ≈ 2×脊点 − D，即 L ≈ 252–372：L=256 贴着下沿，L=512 已越过。口径 B 下要 L ≈ 407–555 才越过，L=512 只是刚进入脊点区间。所以「L=256 贴着脊点下沿、仍偏带宽」是稳的，「L=512 一定算力受限」则取决于口径与脊点。另外，块内串行、UB 吃紧，L 不能无限加大：MindSpeed 的 Mamba-3 反向算子在 `N*P > 8192` 时状态装不进 UB，只能放回 HBM 分块扫（[MindSpeed 文档](https://raw.gitcode.com/Ascend/MindSpeed-Ops/raw/485c0c5b7e820b871a8d155a00be181a778f2e83/docs/triton/mamba3_mimo_bwd_bwd.md)）。

### chunk_state / chunk_state_passing

chunk_state 算「本块自身贡献的状态」，chunk_state_passing 把上一块状态按块衰减后接力。CANN 的 ops-transformer 为 Nemotron-H 写了六个 prefill 算子：`mamba2_causal_conv1d`、`mamba2_chunk_cumsum`、`mamba2_chunk_state`、`mamba2_chunk_state_passing`、`mamba2_chunk_scan`、`mamba2_rmsnormgated`。卷积、cumsum、gated RMSNorm 走 Vector；含矩阵乘的三个走 Cube。文档写明仅 prefill、910B、FP32/FP16、BSND、chunk_size 固定 256、S 需先 pad 到 chunk 整数倍（**已确认**，见[算子目录说明](https://raw.gitcode.com/cann/ops-transformer/raw/6e58fff992f58811c28a075db589eb81367065d0/experimental/mamba/Readme.md)）。

### CVCV 融合与 VC 并行

官方原话是 chunk_scan「包含两个 matmul 和两部分 vector 计算，因此实现为 CVCV 融合算子，通过 VC 并行提升计算性能」（[原文](https://raw.gitcode.com/cann/ops-transformer/raw/9.1.0/experimental/mamba/mamba2_chunk_scan/README.md)）。本质是**不硬算 scan**：改写成 Cube 的 GEMM + Vector 的逐元素，再让两条流水重叠掩盖空闲——这是昇腾路线的核心。

### Ascend C 自定义算子与 aclnn

Ascend C 是写昇腾算子的框架，按 op_host（原型、tiling、shape 推导）+ op_kernel（核函数）组织，编成 CANN 的 aclnn 接口，再由 torch-npu 包成 Python 函数。vllm-ascend PR #12607 是完整样例：在 `csrc/attention/chunk_gated_delta_rule/` 写核函数，注册到 `ascend910b`/`ascend910_93`/`ascend950`，用 `EXEC_NPU_CMD(aclnnChunkGatedDeltaRule, ...)` 调用，输入是 TND、bf16（[PR #12607](https://github.com/vllm-project/vllm-ascend/pull/12607)）。看清这条流水线，就知道「支持一个模型」要补哪几层。

### TND 布局与算子 fallback

TND 把变长序列首尾相接拼成 `(T=ΣLᵢ, 头数, 头维)`，不做 padding，用 `actual_seq_lengths` 记边界。fallback 指布局、精度、维度任一不满足就退回 Triton/FLA 或直接报错——「有算子 ≠ 能跑」。`torch_npu.npu_chunk_gated_delta_rule` 的约束是硬清单（**已确认**）：仅推理、TND、q/k/v/beta 全 bf16、`Nk/Nv ≤ 64` 且 `Nv % Nk == 0`、`Dk/Dv ≤ 128`、q/k 归一化到 [−1,1]、g∈[−1,0]、beta∈(0,1)、TND 下 beta/initial_state/actual_seq_lengths 必传、fp32 状态仅 950 支持（[API 文档](https://raw.githubusercontent.com/ascend/op-plugin/abd599370b927e60f98669456d8655725ba8c974/docs/zh/custom_APIs/torch_npu/torch_npu-npu_chunk_gated_delta_rule.md)）。

### 支持矩阵

| 栈 | 支持什么 | 限制 |
| --- | --- | --- |
| CANN ops-transformer | Mamba-2 prefill 六算子，910B | 仅 prefill、BSND、chunk 固定 256 |
| torch-npu | `npu_chunk_gated_delta_rule`，950/A3/A2 | 仅推理、TND、bf16、头维与头数上限 |
| vllm-ascend | Qwen3-Next（GDN 混合，80B-A3B）自 v0.10.2rc1 支持，走 Triton Ascend（实验）；AscendC 版 chunk GDR（PR #12607）；KDA 有 `kimi_kda.py`（prefill `chunk_kda_fwd` + decode `recurrent_kda`） | 见下方版本脚注 |
| MindSpore / MindSpeed | 未见官方一等公民的 GDN 文档（**未核实**）；MindSpeed-Ops 有 Triton 版 Mamba-3 反向算子 | 仅 arch32，chunk 16 |
| 上游 vLLM | V1 有 `mamba_utils`、Jamba 模型、hybrid memory allocator（RFC #11382） | 进 V1 的版本时间线**未核实** |

**版本脚注**：表中前缀缓存 / PD 分离的判断按 **v0.18.0 支持矩阵**口径，只代表该时点。**此后 PR #8829 为 GDN 增加了前缀缓存，支持 `mamba_cache_mode` 的 align / all 两种模式；all 模式下投机解码会回退到 align**（据本专题姊妹篇，本篇未独立复核）。所以「空白」「已实现」都能成立，取决于版本与模式——读矩阵先问这两件事。

## 一步一步理解

拿 Qwen3-Next（GDN + gated attention 的混合 MoE，80B-A3B）走一遍。官方口径：A3-752T 两节点、TP4、2k/2k、并发 32，**580 tokens/s，TPOT 54 ms**（[官方文档](https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3-Next.html)，未独立复核）。

**第一步，先判断哪半好跑。** attention 那半已经成熟：FlashAttention/MLA、KV 分页、前缀缓存、PD 分离在矩阵里都是 ✅。GDN 那半要弱一档，但必须按版本拆：v0.18.0 的支持矩阵里，Qwen3-Next 的 chunked prefill、前缀缓存、投机解码、PD 分离都是空白；其中**前缀缓存**在 v0.18.0 之后由 PR #8829 补上了 GDN 版本（align/all 模式），本篇未独立复核；**投机解码**在 all 模式下会回退到 align；**chunked prefill 与 PD 分离**仍属「没验证过」，不是「不行」。先分清你要的功能落在哪半、哪个版本。

**第二步，缺口还是带宽。** 80B-A3B 每 token 激活约 3B，bf16 约 6 GB，TP4 每卡约 1.5 GB；按单卡 1.6 TB/s 量级，读权重的时间下限在 1 ms 量级。实测 TPOT 54 ms，差了约 50 倍（**推算，未实测**）。瓶颈不在读权重，而在调度、MoE 通信、GDN 那半的算子与布局。这是 roofline 的正确用法：先排除，再 profile。

**第三步**：先跑通（`vllm serve`，TP4，`--max-model-len 32768`）→ 对齐输出（官方 gsm8k 95.53，AISBench gen 口径）→ 压 TTFT/TPOT → 最后 profile。

**第四步，profiler**：算子耗时 Top（`causal_conv1d` + `chunk_scan` + recurrent GDN 占比）、AI Core 利用率（低且耗时高 = 在等数据）、每轮 HBM 读写字节（远超权重 = 状态/KV 反复搬）。scan 系占比不到 10%，优化收益上限就是 10%；Vector 高而 Cube 空转，查 fallback。

**值得**：长上下文（状态与长度无关）、decode 带宽受限且 batch 起不来、显存吃紧，或模型本就是混合架构。**不值得**：短 prompt（GEMM 本来就跑得好）、强精确检索与逐字复制（状态不是无损记忆）、依赖成熟的投机解码/PD 分离全家桶、布局精度落在约束外（非 bf16、头维超 128）。

**三天验证清单**：第 1 天跑通 + 对齐输出（差 1 点以内）；第 2 天压 TTFT/TPOT（固定长度，并发 1/8/32，记口径）；第 3 天 profile，出「算子占比 + 利用率 + HBM 流量」三张表，再决定换算子、改布局还是换模型。数字都标来源与口径，没实测写「未核实」。

## 常见误区

1. **「有算子就等于能跑」**：bf16、TND、`Dk/Dv ≤ 128`、`Nv % Nk == 0`、q/k 归一化，少一条就 fallback 到 Triton 或直接报错。矩阵里一个 ✅ 背后是一整串条件。
2. **「Mamba 更省，所以在 NPU 上也一定更快」**：要拆成两件事。**总搬运量**上 SSM 确实大赢：decode 每步不再读随长度线性增长的 KV，只读固定大小的状态（姊妹篇的例子：每步 512 MiB KV → 8.3 MiB 状态），长上下文下差数量级，这是实打实的带宽收益。但**单位字节的算术强度**只有约 1.5 FLOP/Byte，仍比脊点低两个数量级，所以它依旧带宽受限。正确说法：SSM 减少的是搬运字节总数，不是每字节的计算密度；优化只能继续减少搬运（融合、块化、状态留片上），不能把算术强度提到算力受限区。快不快，取决于长度、batch 与 kernel 是否兑现。
3. **「chunk 越大越好，把 FLA 的 chunk size 直接搬过来」**：CANN 的 Mamba-2 算子固定 256、vllm-ascend 的 GDR 核用 64、MindSpeed 的 Mamba-3 用 16。chunk 越大算术强度越高，但块内串行变长、UB 装不下就得落 HBM——这是超参，不是常数。

## 一句话总结

昇腾上跑 Mamba 的特殊点只有一条主线：scan 是低算术强度的逐元素/串行计算，喂不饱 Cube；解法是 chunk 化——块内 GEMM 走 Cube、块间传状态走 Vector，再用 CVCV 与 VC 并行掩盖空闲。SSM 赢在总搬运量小，输在每字节算术强度低；值不值得，看瓶颈是不是带宽和显存。

## 自测题

### 1. L=256、D=128，只读 q/k/v/o（bf16），chunk 的算术强度是多少？脊点取 190–250 时它越过脊点了吗？

<details><summary>查看答案</summary>AI = (4L²D + 4LD²)/(8LD) = (L+D)/2 = (256+128)/2 = **192**。脊点 190–250 时，192 贴着下沿，仍偏带宽侧；若把状态读写和输出写回也算进分母（口径 B），AI 降到 128，更明确地在带宽侧。L=512 时口径 A 为 320、口径 B 约 233——是否「越过」取决于口径与脊点，不要用一个阈值下结论。</details>

### 2. 为什么 decode 的 scan 算术强度只有 1.5，而 chunk 内能到 100 以上？

<details><summary>查看答案</summary>decode 每步 M=1，只有三次「矩阵×向量」，FLOP 是 O(D²)，字节是状态读+写 4D²，AI ≈ 1.5；chunk 内是 L×L×D 的 GEMM，FLOP 含 L² 项，计算量按 L² 涨、搬运按 L 涨，AI 线性上升（(L+D)/2）。这就是 chunk 化的意义：把串行 scan 改写成 Cube 吃得下的块级 GEMM。</details>

### 3. 同事说「vllm-ascend 的 Qwen3-Next 前缀缓存是空白，所以不能用」。怎么纠正？

<details><summary>查看答案</summary>先问版本与模式。v0.18.0 的支持矩阵里该行确实空白；但此后 PR #8829 增加了 GDN 前缀缓存，支持 `mamba_cache_mode` 的 align/all 两种模式（本篇未独立复核）；all 模式下投机解码会回退到 align；chunked prefill 与 PD 分离仍属未验证。正确说法是「v0.18.0 时点空白，之后前缀缓存在 align/all 下可用，但投机解码与 PD 分离另有边界」，而不是一句「空白」或「已实现」。</details>

### 4. 「Mamba 显存省，所以在 NPU 上更快」这句话错在哪？什么情况下它对？

<details><summary>查看答案</summary>省的是显存容量和总搬运量，不是单位字节的算术强度。长上下文 decode 时每步只读固定大小的状态，而不是线性增长的 KV，总字节大幅下降，这是真实的带宽收益；但这些字节上的 AI 只有约 1.5，远低于脊点，所以仍然带宽受限，kernel 优化只能减少搬运、不能把它推进算力受限区。长度长、batch 起得来、kernel 融合到位时，「更省」才兑现成「更快」；短 prompt 或强依赖投机解码/PD 分离时未必。</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| 达芬奇 Cube / Vector | Cube 一拍一个 16×16×16 乘加，要凑满块；Vector 做逐元素，灵活但吞吐低 |
| 算术强度 / roofline 脊点 | FLOP÷字节；上界 min(算力, 带宽×算术强度)，交点左侧带宽受限 |
| 仿射映射复合 | (A₂,b₂)∘(A₁,b₁)=(A₂A₁, A₂b₁+b₂)，结合律是并行扫描的依据 |
| WY / UT 变换 | 把块内一串 rank-1 修正合成一个矩阵，变成块级 GEMM |
| chunk_state / state_passing | 块内算自身状态、块间只传状态矩阵，六算子按此分工 |
| CVCV 融合 / VC 并行 | 算子内 Cube→Vector→Cube→Vector 交替，两条流水并行掩盖空闲 |
| Ascend C / aclnn | 写核函数、注册进 CANN 成 aclnn 接口，torch-npu 再包成 Python 函数 |
| TND 布局 | 变长序列拼成（总 token, 头, 头维）不 padding，很多昇腾算子的硬约束 |
| 算子 fallback | 布局/精度/维度不满足就退回慢路径或报错，即「有算子 ≠ 能跑」 |
| 支持矩阵 | 各栈在哪个版本、哪个阶段支持什么、限制在哪 |
| GDN / KDA | GDN：ICLR 2025（arXiv:2412.06464）；KDA：Kimi Linear 的线性注意力（arXiv:2510.26692） |

## 还没想清楚

- **已确认**：六个 Mamba-2 算子只覆盖 prefill、chunk 256、BSND；torch-npu 的 GDR 只覆盖 TND/bf16。
- **存疑**：910B 的 FP16 峰值与 HBM 带宽没有官方精确值，脊点 190–250 只是量级值；文中算术强度是纸面推算（口径 A/B 都已写明），不是实测。
- **未核实**：六个算子是否被 vllm-ascend 默认路径调用；PR #8829 的 align/all 语义与投机回退（转述姊妹篇，未独立复核）；纯 Mamba 与同规模 Transformer 的端到端对比；KDA 的端到端性能；MindSpore 原生 GDN 算子是否存在。

## 来源

- [CANN ops-transformer · MambaV2 算子目录说明](https://raw.gitcode.com/cann/ops-transformer/raw/6e58fff992f58811c28a075db589eb81367065d0/experimental/mamba/Readme.md)
- [mamba2_chunk_scan 算子说明（CVCV 融合、VC 并行）](https://raw.gitcode.com/cann/ops-transformer/raw/9.1.0/experimental/mamba/mamba2_chunk_scan/README.md)
- [torch_npu.npu_chunk_gated_delta_rule API 文档](https://raw.githubusercontent.com/ascend/op-plugin/abd599370b927e60f98669456d8655725ba8c974/docs/zh/custom_APIs/torch_npu/torch_npu-npu_chunk_gated_delta_rule.md)
- [vllm-ascend 支持矩阵（v0.18.0）](https://docs.vllm.ai/projects/ascend/en/v0.18.0/user_guide/support_matrix/supported_models.html)
- [vllm-ascend Qwen3-Next 教程](https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3-Next.html)
- [vllm-ascend PR #12607：GDN prefill 换用 aclnn chunk_gated_delta_rule](https://github.com/vllm-project/vllm-ascend/pull/12607)
- [vllm-ascend `kimi_kda.py`（KDA 的 AscendC prefill/decode 路径）](https://github.com/vllm-project/vllm-ascend/blob/8d838ee8/vllm_ascend/ops/kimi_kda.py)
- [MindSpeed-Ops · mamba3_mimo_bwd_bwd 算子迁移说明](https://raw.gitcode.com/Ascend/MindSpeed-Ops/raw/485c0c5b7e820b871a8d155a00be181a778f2e83/docs/triton/mamba3_mimo_bwd_bwd.md)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule (ICLR 2025)](https://arxiv.org/abs/2412.06464)
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- [vLLM RFC #11382：Hybrid Memory Allocator](https://github.com/vllm-project/vllm/issues/11382)
- [vLLM PR #20016：Enable V1 for Hybrid SSM/Attention Models](https://github.com/vllm-project/vllm/pull/20016)
