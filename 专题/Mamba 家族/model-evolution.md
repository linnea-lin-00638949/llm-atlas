---
date: "2026-11-03"
title: "模型演进：从 S4 到 Mamba-2，再到 GDN / KDA 等混合模型"
type: "topic"
topic: "Mamba 家族与混合架构"
topic_id: "mamba-family"
part: "model-evolution"
part_title: "模型演进：从 S4 到 Mamba-2，再到 GDN / KDA 等混合模型"
order: 1
tags: ["SSM", "Mamba", "混合架构", "线性注意力", "长上下文"]
---

# 模型演进：从 S4 到 Mamba-2，再到 GDN / KDA 等混合模型

> 读完这篇，你能：
> - 说清 SSM 用「定长状态」换掉了什么，以及它为什么不是「压缩版 KV cache」；
> - 自己手算离散化、chunked scan 复杂度、显存账这三笔账；
> - 分清 SSD 的对偶究竟对偶了什么，GDN / KDA 各改动了哪一处。

## 为什么需要它

第 3 课的 attention 有一笔隐含账单：把历史全留成 KV。一个 32 层、8 个 KV 头、head_dim 128、bf16 的模型，32K 上下文时 KV 约 4 GiB（算法见「一步一步理解」），而 decode 每生成一个 token 还要把这段 KV 整个读一遍。

还有一条路：不留历史，只维护一个定长状态 h，每读一个 token 就「更新状态、读出结果」。这就是状态空间模型（SSM）。账本完全不同：显存不随上下文涨、每 token 计算量恒定，代价是状态只记摘要。

这条路从 S4 到 Mamba（选择性）、Mamba-2（SSD）、GDN（delta rule + 门控）、KDA（逐通道门控）一路演化，落成今天最常见的形态——混合架构：多数层用 SSM，少数层保留 attention。本文是另两篇的「记号表」。

## 核心概念

### 状态空间模型（SSM）

定义：用定长隐状态 h_t ∈ R^N 概括 x_1…x_t 的序列模型，由连续时间线性系统离散化而来：ẋ = Ax + Bu，y = Cx + Du。

直觉：attention 把每句话录音存档、需要时回放；SSM 像只维护一页摘要的笔记员，每读一句改一页。摘要页大小固定、永不爆内存，但也没法逐字回放第 37 句。

机制：A 管状态自己怎么衰减（特征值实部为负＝遗忘），B 管当前输入怎么写进来，C 管怎么读出去，D 是直通项（常省略）。S4 / Mamba 取 A 为对角阵，各通道独立。

### 状态维度 N

h 的宽度，常见 16–128。它不表示「能记住多少 token」，而是「摘要用多少个数字写」；N 决定单层能存多少线性无关的 key，也是状态缓存的成本因子（每序列每层约 d_inner × N 个数）。

它不是 KV 的压缩版：KV 可寻址、可回滚、随长度增长；状态不可寻址、只进不出、与长度无关。语义不同，不能互换。

### 离散化（Δ、Ā、B̄）

定义：零阶保持下，标量情形有精确式 Ā = e^{Δa}、B̄ = ((e^{Δa} − 1)/a)·B（矩阵形式是 B̄ = (ΔA)⁻¹(e^{ΔA} − I)·ΔB）。递推为 h_t = Ā h_{t−1} + B̄ x_t，y_t = C h_t。

注意「B̄ ≈ ΔB」只是近似：把 e^{Δa} 展开成 1 + Δa 才得到它，所以只在 |Δa| ≪ 1 时成立。Δa = −0.1 时精确值 0.0952、近似值 0.1，已差 5%——下面手算用精确值。

直觉：Δ 是「这一步积分多长」。Δ 大 → Ā 小 → 旧状态被冲掉，等于重置；Δ 小 → 状态几乎不动，历史保住。这是 Mamba 表达「该不该记住这个 token」的旋钮。

### HiPPO 初始化

给 A 一组特定初值（正交多项式基底，如 Legendre），使状态成为历史的最优投影。它回答「A 随便设会怎样」：a = −1000 时 e^{−1000Δ} ≈ 0，一步清零；a = 0 时永不衰减、越加越乱。HiPPO 把 a 排成多尺度衰减，只影响初始化，训练中 A 仍可学。

### 卷积视图

递推展开：y_t = Σ_{s≤t} C_t (Π_{r=s+1..t} Ā_r) B̄_s x_s。若 Ā、B̄、C 不随输入变，核只依赖 t−s，就是一个因果长卷积——训练可一次卷积 / FFT 算完，推理仍可递推，每步 O(1) 次状态读写。

### 选择性（S6）

B、C、Δ 由当前 token 经线性层算出，A 仍与输入无关。S4 的核固定，「该记什么」写死在参数里；Mamba 让模型按内容决定记还是忘——见到分隔符调大 Δ（重置），见到关键词调小 Δ（多写一点）。代价：核随时间变，不再是固定卷积、不能 FFT，只能沿时间真递推。

### 并行扫描与 chunked scan

并行扫描：元素取 (a, b) 表示 h_out = a·h_in + b，复合 (a₂,b₂)∘(a₁,b₁) = (a₂a₁, a₂b₁+b₂)，满足结合律，可做树形归约：O(n) 工作量、O(log n) 深度。

chunked scan：把序列切成长度 L 的块。块内用一次带因果掩码的矩阵乘算完，块间只传一个状态 h，因此块间要串行 n/L 次——串行步数是块数，不是块长 L。总计算约 O(n(L+N)D)：L 决定块内的矩阵乘规模与块间串行跳数，D 是通道数（每个通道各做一次），N 是块间低秩传递的秩。L 取大，块内矩阵更大、串行跳数更少，矩阵乘占比更高——这是硬件喜欢它的原因，也是 Mamba-2 / GDN / KDA 在 GPU、NPU 上的共同地基。

### SSD（状态空间对偶）

SSM 一侧：y = M·x，其中 M_{t,s} = C_t (Π_{r=s+1..t} Ā_r) B̄_s（s ≤ t，否则为 0）。M 下三角且「半可分」：对角块之外秩 ≤ N。

attention 一侧要说准：给定 Q、K 之后，输出是 y = A·V，A = softmax(QKᵀ/√d) ⊙ mask（下三角）。所以 attention 对 V 线性，但 A 由输入经 Q、K 现算——对输入整体不是线性映射，别读成「固定 M 的线性层」。

对偶指：两者都能写成「一个下三角矩阵乘一个序列」——SSM 乘 x，attention 乘 V；区别在矩阵怎么造：A 稠密、来自 QKᵀ；M 半可分（对角块外秩 ≤ N）、来自 A/B/C 的递推。N 大 → M 接近满秩 → 像 attention；分块后对角块用低秩 (C B) matmul、块间只传状态 → 像 SSM。买到的是同层可选 matmul 形态，主计算变成大矩阵乘——昇腾能参与的前提。

### Delta rule、门控与混合架构

Delta rule：状态 S 是 key-value 关联矩阵，写入前先按 key 擦旧值：S_t = S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ，等价于对 ‖S k_t − v_t‖² 走一步梯度下降（学习率 β_t）。不是「再写一条」，而是「先擦旧答案再写新的」，所以能改错、检索更准。

GDN（Gated DeltaNet，ICLR 2025，arXiv:2412.06464）把 Mamba-2 的门控衰减与 delta rule 合起来：S_t = α_t S_{t−1}(I − β k kᵀ) + β v kᵀ。门控管「整体快速清空」，delta 管「定点改写」，二者互补。

KDA（Kimi Delta Attention，arXiv:2510.26692）把 α 从每头一个标量细化到逐通道（对角），并用 Diag-Plus-Low-Rank 转移矩阵的块算法维持效率。官方称 3B 激活 / 48B 总参、KDA + MLA 分层混合，1M 上下文 KV 最多降 75%、解码吞吐最高 6×——厂商自报，未见第三方复现。

混合架构：attention 层与 SSM 层按比例交错，成本只剩一条公式——KV 只按 attention 层数付。

- Jamba：32 层 = 4 block × 8 层，block 内 attention:Mamba = 1:7，故只有 4 层 attention；MoE 每 2 层一次、16 专家 top-2；52B 总参 / 12B 激活；256K 上下文 KV 约 4 GB。排法理由：attention 保检索，Mamba 给吞吐，MoE 保容量。
- Zamba：Mamba 主干 + 一个共享 GSA 块，每 6 个 Mamba 块调一次、参数共享；attention 参数与 KV 只付一份，代价是表达受限。
- Hymba：不交错而是并行——同一层 attention 头与 SSM 头各算再融合，另有可学习 meta token 存全局信息；attention 头管高分辨率召回，SSM 头管摘要，代价是结构复杂。
- Samba：逐层 Mamba + 滑动窗口 attention；近期精确召回交给窗口，远期压缩交给 Mamba，可无限流式，代价是窗口外没有精确召回。

工程现状：vLLM 主线已列出 Mamba、Mamba2、Jamba、Zamba2、FalconMamba、KimiLinear 与 Qwen3.5 的 GDN 混合；V1 早期要求「强制 eager + 关前缀缓存」，放开后才用上 CUDA graph，prefix caching 由 mamba_cache_mode 控制 checkpoint 粒度。

昇腾侧（vllm-ascend）的口径是 aclnn 优先、Triton 回落，以 0.11 线为参照（PR #12607 接入 AscendC 的 ChunkGatedDeltaRule，PR #19875 把 prefill 的 chunk 扫描切到该融合算子）：

| 阶段 | 默认算子 | 回落 |
| --- | --- | --- |
| prefill | torch_npu.npu_chunk_gated_delta_rule（aclnn） | Triton chunk_gated_delta_rule（FLA） |
| decode | torch_npu.npu_recurrent_gated_delta_rule（AscendC） | Triton recurrent 实现 |

约束以 torch_npu 算子文档为准：仅推理、TND 布局、q/k/v 全 bf16、Dk/Dv ≤ 128、Nk/Nv ≤ 64 且 Nv % Nk == 0，任一不满足即回落。换版本时默认路径与阈值都可能变，落地前按自己版本查源码。另外 ssm_state 与 conv_state 各一个 pool、与 attention 的 block table 分开索引；all 模式的 GDN 前缀缓存已实现，但该模式下投机解码会回退到 align——「混合 + 投机 + 前缀缓存」同开仍是坑。

## 一步一步理解

第一步，单通道手算。取 a = −1、Δ = 0.1、B = 1：Ā = e^{−0.1} ≈ 0.9048，B̄ = 1 − e^{−0.1} = 0.0952。x 恒为 1，则 h₁ = 0.0952，h₂ = 0.9048 × 0.0952 + 0.0952 = 0.1813，稳态 h* = B̄/(1 − Ā) = 1.000（正好是 −B/a）。用近似 B̄ ≈ ΔB 则稳态 1.05，偏高 5%。记忆地平线约 1/(1 − Ā) ≈ 10.5 步，更早的输入基本读不出来——这就是「定长状态」的含义。

第二步，看核。令 C = 1、x = [1, 0, 0, …]，输出是 0.0952、0.0861、0.0779…，一条每步乘 0.9048 的衰减核。S4 里它固定，所以能一次卷积算完。

第三步，看选择性。一旦 Δ、B、C 随 token 变，核随时间变，卷积失效，只能 scan。分块后设 n = 4096、L = 128、N = 64、D = 2048：序列切成 4096/128 = 32 个块，块间串行 32 次；总计算约 n(L+N)D = 4096 × 192 × 2048 ≈ 1.6 × 10⁹ 次乘加，矩阵乘占绝对主体。

第四步，算两笔账。KV：每 token 每层 2 × 8 × 128 = 2048 个数、bf16 约 4 KiB，32K 上下文 = 128 MiB/层，32 层 = 4 GiB；1:7 混合只剩 4 层 = 512 MiB（1/8，与 Jamba 的 8× 一致）。状态：N = 128、宽度 2048，每层 2048 × 128 = 262144 个数 = 512 KiB，28 个 Mamba 层约 14 MiB/序列，且与上下文无关——小两三个数量级，只是「只读摘要」。

| 状态形式 | 每 token 计算量 | 训练复杂度 | 推理复杂度 | 有无 KV | 典型场景 |
| --- | --- | --- | --- | --- | --- |
| attention（GQA） | 读全部 KV | O(n²) | 每步 O(n)，KV 随 n 涨 | 有 | 通用对话、精确检索 |
| S4 | O(N·d) | O(n log n)（FFT） | O(1)/token | 无 | 长序列、音频、时序 |
| Mamba（S6） | O(N·d) | O(n)（scan） | O(1)/token | 无 | 长文本预训练、边缘推理 |
| Mamba-2 / SSD | O(N·d)，主体是 matmul | O(n) 分块 | O(1)/token | 无 | 训练效率优先 |
| GDN / KDA | 同上 + delta / 逐通道门控 | O(n) 分块 | O(1)/token | 无 | 检索要求高、超长上下文 |
| 混合（Jamba 类） | attention 层 O(n)，其余 O(N·d) | 介于两者之间 | 每步 O(n_attn) | 有，只按 attention 层 | 长上下文 + 高并发服务 |

## 常见误区

一、「SSM 状态就是压缩过的 KV cache」。不是：KV 能按位置取回、能回滚、能前缀复用；状态只进不出、不可寻址。所以混合模型的 prefix caching 要额外做 checkpoint，不能按 token 切块。

二、「有卷积视图，所以训练不需要 scan」。卷积只在 B、C、Δ 与输入无关时成立（S4）；S6 之后核时变，FFT 路线作废，必须 scan 或分块 matmul。

三、「选择性 = 门控」。门控 α_t 只是按通道整体缩放状态；选择性是让 B、C、Δ 随内容变。GDN 是两者相加，不是同一件事。

四、「混合模型长上下文免费」。SSM 层不付 KV，但要付状态缓存与每 token 的 O(N·d) 计算；状态不可寻址，针尖检索仍靠那几层 attention。

## 一句话总结

attention 把历史全留成 KV，SSM 把历史压进定长状态；S6 让「记什么」随内容变，SSD 说明两者是同一个下三角矩阵乘法的两种算法（attention 乘 V、SSM 乘 x，矩阵一个稠密一个半可分），混合架构则按「KV 只按 attention 层数付」这条公式把两者拼起来。

## 自测题

### 1. 取 a = −1、Δ = 0.1、B = 1、x 恒为 1，求 Ā、B̄、h₂ 与稳态 h*；用 B̄ ≈ ΔB 会差多少？

<details><summary>查看答案</summary>

精确式 B̄ = ((e^{Δa} − 1)/a)·B = 1 − e^{−0.1} ≈ 0.0952，Ā = e^{−0.1} ≈ 0.9048。h₁ = 0.0952，h₂ = 0.9048 × 0.0952 + 0.0952 = 0.1813，稳态 h* = B̄/(1 − Ā) = 1.000（即 −B/a）。若用 B̄ ≈ ΔB = 0.1，h* 会算成 1.05，偏高 5%——「≈」必须带条件 |Δa| ≪ 1，这里 Δa = −0.1 已不算小。

</details>

### 2. chunked scan 块间串行几次？总计算为什么是 O(n(L+N)D)？

<details><summary>查看答案</summary>

串行次数是块数 n/L，不是块长 L；n = 4096、L = 128 时是 32 次。总计算 n(L+N)D 各项分工：块内 L×L 因果矩阵乘贡献 n·L·D；块间只传秩 ≤ N 的状态，贡献 n·N·D；D 是通道数，每个通道各自独立做一遍，所以乘上去。L 大则块内矩阵大、串行跳数少。这也说明 chunk 化后主体是矩阵乘——硬件吞吐在 matmul 上。

</details>

### 3. 「attention 也是 y = M·x」这句话哪里不严谨？

<details><summary>查看答案</summary>

不严谨在 x 和 M 的来源。准确说法：给定 Q、K 后 attention 对 V 的映射是 y = A·V，A = softmax(QKᵀ/√d) ⊙ 掩码；但 Q、K 由输入算出，所以 attention 对输入整体不是线性映射，A 也不是固定矩阵。对偶只说两者都能写成一个下三角矩阵乘一个序列（SSM 乘 x、attention 乘 V），且矩阵结构不同。把「同族」读成「attention 是线性层」就错了。

</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| 状态空间模型（SSM） | 用定长隐状态 h_t 概括全部历史；连续形式 ẋ=Ax+Bu、y=Cx+Du，离散后 h_t=Āh_{t−1}+B̄x_t |
| 离散化（Δ / Ā / B̄） | Ā=e^{ΔA}，标量下 B̄=((e^{Δa}−1)/a)·B，仅 |Δa|≪1 时约等于 ΔB |
| HiPPO 初始化 | 给 A 一组正交多项式基底初值，使状态成为历史的最优投影，避免 A 随便设导致瞬间遗忘 |
| 卷积视图 | 递推展开等于一个因果长卷积核；固定核时训练可并行、推理仍 O(1) |
| 选择性（S6） | B、C、Δ 由当前 token 算出，模型按内容决定记住还是重置；代价是核时变、必须 scan |
| chunked scan | 序列分块，块内并行 matmul、块间串行传状态（串行 n/L 次）；总计算约 O(n(L+N)D) |
| SSD（状态空间对偶） | SSM 写成 y=M·x（M 半可分）；attention 写作 y=A·V（A=softmax(QKᵀ)⊙mask）；同族指两者都是下三角矩阵乘 |
| Delta rule | 写入前按 key 擦旧值再写新值，等价于对 ‖Sk−v‖² 走一步梯度下降；状态是可改写的关联 |
| 门控衰减 α_t | 状态各通道按输入决定衰减多少；GDN 用标量 α，KDA 细化到逐通道对角 |
| 混合架构（hybrid） | attention 与 SSM 层按比例交错（Jamba 1:7、Zamba 共享层、Hymba 并行头、Samba 加滑窗），KV 只按 attention 层计 |

## 还没想清楚

已确认：S4→Mamba→Mamba-2 的公式与动机、四个混合模型的排布、GDN / KDA 的机制、vLLM 的模型支持与 V1 eager 限制解除，都有论文或代码为据。

存疑（未能核实）：
一、Mamba-1 默认 d_state=16 / expand=2、Mamba-2 常用 N=64–128，属社区常见说法，未逐字核对官方 config；
二、Samba 的滑窗宽度、Hymba 跨层 KV 共享的层数，未查到确定值；
三、Kimi Linear 的 75% / 6× 是厂商自报，缺第三方复现；
四、昇腾上 GDN 的实测吞吐无公开可复现数据；vllm-ascend 从哪一版起把 GDN prefill 默认切到 aclnn（本文按 0.11 线与 PR #12607 / #19875 描述），未逐版核对源码。

## 来源

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces — https://arxiv.org/abs/2312.00752
- Transformers are SSMs（Mamba-2 / SSD） — https://arxiv.org/abs/2405.21060
- Gated Delta Networks: Improving Mamba2 with Delta Rule（ICLR 2025） — https://arxiv.org/abs/2412.06464
- Kimi Linear: An Expressive, Efficient Attention Architecture（KDA） — https://arxiv.org/abs/2510.26692
- Parallelizing Linear Transformers with the Delta Rule over Sequence Length — https://arxiv.org/abs/2406.06484
- Jamba: A Hybrid Transformer-Mamba Language Model — https://arxiv.org/abs/2403.19887
- Zamba: A Compact 7B SSM Hybrid Model — https://arxiv.org/abs/2405.16712
- Hymba: A Hybrid-head Architecture for Small Language Models — https://arxiv.org/abs/2411.13676
- Samba: Simple Hybrid State Space Models for Unlimited Context — https://arxiv.org/abs/2406.07522
- S4: Efficiently Modeling Long Sequences with Structured State Spaces — https://arxiv.org/abs/2110.13985
- vLLM Supported Models（Mamba / Jamba / Zamba2 / KimiLinear / Qwen3.5） — https://docs.vllm.ai/en/latest/models/supported_models/
- vLLM PR #21233：V1 混合模型移除 enforce_eager 限制 — https://github.com/vllm-project/vllm/pull/21233
- vllm-ascend PR #8829：GDN all 模式前缀缓存 — https://github.com/vllm-project/vllm-ascend/pull/8829
- vllm-ascend PR #12607：AscendC ChunkGatedDeltaRule（aclnn 算子） — https://github.com/vllm-project/vllm-ascend/pull/12607
- vllm-ascend PR #19875：GDN prefill 切换到 torch_npu 融合算子 — https://github.com/vllm-project/vllm-ascend/pull/19875
- torch_npu.npu_chunk_gated_delta_rule API 文档（维度与精度约束） — https://raw.githubusercontent.com/ascend/op-plugin/abd599370b927e60f98669456d8655725ba8c974/docs/zh/custom_APIs/torch_npu/torch_npu-npu_chunk_gated_delta_rule.md
- vLLM-Ascend 支持模型矩阵 — https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_models.html
- Hybrid Models as First-Class Citizens in vLLM（PyTorch 博客） — https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/
