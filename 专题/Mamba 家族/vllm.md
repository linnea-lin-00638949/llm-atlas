---
date: "2026-11-03"
title: "推理框架：vLLM 怎么处理 Mamba 与混合模型"
type: "topic"
topic: "Mamba 家族与混合架构"
topic_id: "mamba-family"
part: "vllm"
part_title: "推理框架：vLLM 怎么处理 Mamba 与混合模型"
order: 2
tags: ["Mamba", "vLLM", "混合架构", "推理框架", "昇腾"]
---

# 推理框架：vLLM 怎么处理 Mamba 与混合模型

> 读完这篇，你能：
> 1. 说清 conv_state / ssm_state 与 KV cache 在容量、切块、复用上的差别；
> 2. 解释 vLLM 为什么把状态也当分页缓存管，以及 chunked prefill 怎么「接力」；
> 3. 自己按 config 估一笔混合模型的 KV/状态账，不被「省 8 倍」带偏。

## 为什么需要它

你熟悉的推理服务默认三件事：prefill 一次算完、每层 K/V 写进 KV cache；decode 每步只读历史；prefix caching 因为每个 token 都有独立副本，能在任意 token 边界命中。前提是历史可随机访问、随长度线性增长。

Mamba 换掉了前提：历史被压成一个定长状态，每读一个 token 更新一次。prefill/decode 还在，但「缓存什么、按什么粒度切、能否跨请求复用、投机被拒怎么回滚」全要重做。vLLM 在 0.10 一线集中改这件事，也踩过版本口径不一致的坑——这篇只写能对上的部分。

## 核心概念

### 缓存换成了 conv_state + ssm_state

**定义**：Mamba 是一种 SSM（状态空间模型）；一层的前向是 in_proj → causal-conv1d → 选择性 scan（沿时间轴递推）→ 门控 → out_proj。它不存 KV，只存两样：
- **conv_state**：短卷积的窗口尾巴，每层 (conv_dim, K−1)，K=4 就是 3 步；
- **ssm_state**：循环状态，Mamba-2 形状 (heads, head_dim, d_state)，与上下文长度无关。

**直觉**：KV 是逐字记账的流水账，越读越厚；ssm_state 是固定厚度的一页摘要，读多少页都不变厚。conv_state 则是传送带尽头留的最后 3 格样品，下一个 token 来了才拼得出完整窗口。

**机制**：Mamba-2 的递推是 h_t = Ā_t·h_{t−1} + B̄_t·x_t，y_t = C_t·h_t + D·x_t；Ā、B̄、C 由当前输入算出，所以叫「选择性」。decode 每步只更新一次状态，prefill 按块算、只在块末落一份状态。O(1) 不等于小：Mamba-Codestral-7B 64 层、128 头、head_dim 64、d_state 128、bf16，每层 128×64×128×2 B = 2 MiB，64 层约 128 MiB/序列。

**三家模型的账放同一张表**（避免配置打架）：

| 模型（config）         | 结构                                                                        | KV/token | 32K KV  | 状态/序列     |
| ------------------ | ------------------------------------------------------------------------- | -------- | ------- | --------- |
| Llama-3-8B         | 32 层全 attention，8 KV 头，128 维，fp16                                         | 128 KiB  | 4 GiB   | 0         |
| Jamba-v0.1         | 32 层，4 层 attention（1:7），8 KV 头，128 维；28 层 Mamba-1，d_state 16、d_inner 8192 | 16 KiB   | 512 MiB | 约 8.3 MiB |
| Mamba-Codestral-7B | 64 层纯 Mamba-2，128 头，head_dim 64，d_state 128，bf16                          | 0        | 0       | 约 128 MiB |

按公开 config 手算（fp16/bf16 每元素 2 字节），未计分页对齐与 padding，实际略高。Jamba 的 512 MiB 是 4 GiB 的 1/8，因为 attention 只占 4/32 层——「省 8 倍」是层比的倒数，不是 Mamba 的魔法。状态与 KV 的交叉点在 8.3 MiB ÷ 16 KiB ≈ 530 token：更短的上下文，状态反而更贵。

### 状态也要分页：block table 与页大小

**定义**：block table 是「逻辑块 → 物理块」的映射表；vLLM 把状态按 block 分配，与 KV 共用一套分页内存池。

**直觉**：KV 像活页纸，每页一个 token，可任意抽换；状态像白板，只能整块换，换下的旧内容不留。一个序列任一时刻只有一份「当前状态」，不能按 token 切。

**机制**：混合模型里 state 每 token 的字节数远大于 attention 每 token 的 KV 字节数，两边页大小对不上。vLLM 调大 attention 的 block_size，再给 state 补齐（padding，见[设计文档](https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md) case 4；可能超过 400，Nemotron-H 例子里到 4224）。这也是 PD 限制的来源：HMA（Hybrid Memory Allocator，混合内存分配器）要求两端 block size 一致。

### 跑通一个模型要三件套

**定义**：模型类（认识权重）+ 专用 attention backend（mamba1/mamba2/short_conv/linear/gdn，知道这层怎么算）+ 缓存规格 MambaSpec（知道留多大、留几种状态）。

**机制**：缺任一环就退回 eager（逐算子跑、不图捕获；CUDA graph 则是把一串算子录成一张可重放的图）。V1 早期把 Mamba 标成 V0-only（V1 是新引擎，V0 是旧引擎），根子就在这里：缓存管理器和 backend 都假设「一个 token 一格 KV」，没有 MambaSpec 这种规格可写。以主干[支持模型列表](https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md)为准：Mamba/Mamba2（含 Codestral）、Jamba、NemotronH、FalconMamba、GraniteMoeHybrid、KimiLinear 在列；Zamba 初代不在；Hymba 在列表和注册表里都查不到（未能核实，可能走 Transformers 后端回退）。

### chunked prefill：块间要接力

**定义**：长 prompt 切块算，块与块之间必须显式传初始状态。

**直觉**：Transformer 的块只通过 KV 通信，谁先谁后无所谓；Mamba 的块是流水线工序，上一块的出口状态就是下一块的入口。

**机制**：当前接口传 cu_chunk_seqlen 与 last_chunk_indices；老接口叫 block_idx_first_scheduled_token / initial_state_idx。v0.10.2 还有一条 bugfix「Fix mamba2 prefill chunking」（#23279），说明这条路是踩过坑才通的。

这件事三篇各讲一面：数学在[[模型演进：从 S4 到 Mamba-2，再到 GDN / KDA 等混合模型|模型演进篇]]「并行扫描与 chunked scan」——仿射映射 (a,b) 的复合是 (a₂a₁, a₂b₁+b₂)，常数项记 b_t，别和 delta rule 的学习率 β_t 混用；框架在本篇；昇腾上 chunk_state / chunk_state_passing 怎么落成算子，见[[硬件：昇腾 NPU 上跑 Mamba 有什么特殊点|硬件篇]]「分段扫描与六个算子」。

### APC：只能从「状态点」恢复

**定义**：APC（自动前缀缓存）按 token 块 hash 复用前缀；Mamba 状态只在 block 边界写成实际张量，所以命中只能落在 block 边界。

**机制**：`--enable-mamba-shared-prefix-checkpoint` 在共享前缀结束处额外存一个检查点，默认关。要生效需同时满足：APC 已开；Mamba 组上有 EAGLE/MTP；`--prefix-match-unit` 小于 block size；模型不用多模块 MTP。EAGLE 是用小模型在特征层先猜几个 token，MTP（Multi-Token Prediction）是模型自带的多 token 预测头，两者都是投机解码。

**存疑**：v0.10.2 默认对 hybrid/Mamba 关掉 APC（#23716），v0.11.0 的 v1_guide 仍写「prefix caching is not yet supported」，而主干 APC 文档又在讲 checkpoint 开关——两份官方文档口径不一。只信自己版本的 `--help` 与启动日志。

### 投机回滚：num_accepted_tokens

**定义**：投机解码里被拒 token 对应的状态推进要撤销，把状态退到第 j 个被接受 token 的位置。

**机制**：KV 是只读的，验证完丢掉多余位置就行；状态是「已经走过」的，走错了得倒带。selective_state_update 的 num_accepted_tokens 就是「上一步接受几个」，用它选初始状态；causal_conv1d_update 也有同名参数，卷积窗口按接受数滑动。

### 量化与 PD 的边界

- **W8A8**：只压 Linear（in_proj/out_proj/x_proj 可以量化），A、Δ、状态更新通常保持高精度。Mamba2 的 TP（张量并行）+量化要求 n_groups（B/C 按组共享的组数）能被 tp_size 整除，见 [#24593](https://github.com/vllm-project/vllm/pull/24593)。
- **PD 分离**：prefill 实例交给 decode 的不是纯 KV，而是装着 KV + conv/ssm state 的 block。NIXL（PD 间传 KV 的库）兼容表把 Hybrid SSM/Mamba 列为基础可用，但 HMA 要求两端 block size 一致，PP（流水线并行）+hybrid 不支持，投机解码一栏是「未验证」。

## 版本时间线：把「哪版有什么」钉住

下表只覆盖 2025 年 0.10–0.11；之后的版本以自己 pin 的 tag 为准。

| 版本 / 时间 | PR（合并） | 改了什么 | 可核对引文 |
| --- | --- | --- | --- |
| v0.10.0（07-24） | #21194（07-18） | Mamba 层支持分段 CUDA graph（piecewise） | release notes 列 "Enable piecewise CUDA Graph for mamba layers" |
| v0.10 文档线（07-19） | #21233（仅文档） | 去掉 **hybrid** 的 enforce_eager 限制，不涉及 Mamba-1 | PR 标题 "remove enforce_eager limitation for hybrid models"；v0.10.0 文档已无此要求 |
| v0.10.1（08-18） | #21249 | Mamba-1/Jamba 进 V1，但没有 CUDA graph | release notes "... (without CUDA graphs)"；v1_guide "Mamba-1 models require `enforce_eager=True`" |
| v0.10.1 | #21401 | Mamba 层 decode-only 全图 | release notes |
| v0.10.2（09-13） | #22594 / #23035 / #23650 / #23716 | hybrid 默认全图 / Mamba1 全图 / V1 设为 Mamba 默认 / 默认关 hybrid-Mamba APC | release notes 四条 |
| v0.10.2 | — | v1_guide 删除 "Mamba-1 ... enforce_eager=True" | 对比 v0.10.1 与 v0.10.2 同文件 |
| v0.11.0（10-02） | #24593 | Mamba2 TP+量化 | "n_groups % tp_size == 0" |
| v0.11.0 | 多个 V0 deprecation PR | V0 引擎移除 | "completes the removal of V0 engine" |

**#21233 与 #22594/#23035 的关系**：前者只是文档改动（v0.10 文档线），改的是 hybrid（Mamba-2 + attention），那时 Mamba-1 还没进 V1；Mamba-1 在 v0.10.1 才进来且没有 CUDA graph，所以文档保留 enforce_eager；v0.10.2 的 #23035 让 Mamba-1 也能全图捕获，#22594 让 hybrid 默认全图，这句话才被删。第三篇说「时间线未核实」，本篇只钉到 tag 文档、PR 标题、release notes 能对上的粒度；更细的 kernel 行为仍标存疑。

## 昇腾入口

Qwen3.5-27B 是 dense hybrid（GDN + full attention；GDN 即 Gated DeltaNet），教程写明最低 vllm-ascend v0.17.0rc1；Qwen3.6-27B 才是 v0.18.0rc1——第三篇引的 v0.18.0 对应后者。Qwen3-Next 在支持矩阵里标 🔴 Unmaintained，教程原文「discontinued since vLLM Ascend v0.30.0」：这是已发布的停止维护，不是计划；v0.30.0 release notes 未单列这条，依据是模型教程与支持矩阵。GDN 的 prefill 分块算、decode 逐步更新状态；昇腾侧的前缀缓存适配见 [PR #8829](https://github.com/vllm-project/vllm-ascend/pull/8829)，融合算子与 Triton 回落的差异无公开数字。

## 一步一步理解

Jamba-v0.1，prompt 8000 token，生成 500：

1. **进调度器**：两套缓存分开管，attention 要 KV、Mamba 要 state，按缓存组（cache group）分配。
2. **chunked prefill**：切成两块 4096。第一块算完把块末状态写进状态 block，第二块读回来当初值——这就是接力。
3. **decode 第一步**：attention 层从 KV 取 8000 个位置；Mamba 层不读历史，只读 conv_state 与 ssm_state，跑一次 selective_state_update。
4. **显存账**：KV 16 KiB × 8000 ≈ 125 MiB；32K 时 512 MiB；状态 8.3 MiB，涨到 128K 也不动。
5. **并发账**：状态随并发线性涨（8.3 MiB × 并发），KV 随「并发 × 长度」涨；纯 Mamba 没有 KV，高并发时状态会成为新瓶颈，混合模型则看层比与上下文长度。
6. **开投机**：一次验证 5 个草稿 token，接受 3 个 → num_accepted_tokens=3，状态回到第 3 个被接受 token 的位置。

## 常见误区

1. **「Mamba 没有缓存，显存不用管。」** 错。有 conv_state + ssm_state，只是 O(1)；Mamba-Codestral 约 128 MiB/序列，200 并发就约 25 GiB。
2. **「省 8 倍是 Mamba 的魔法。」** 错。8 = 总层数 ÷ attention 层数，Jamba 恰好 1:7；层比不同结果就变，短上下文甚至不省。
3. **「V1 早就支持 Mamba。」** 错。2025-04 论坛还在问、v0.10.0 文档写 Mamba-1 不支持；v0.10.1 才进 V1（无图），v0.10.2 才补上 Mamba-1 全图、让 hybrid 默认全图并把 V1 设为 Mamba 默认，v0.11.0 才删掉 V0。
4. **「vLLM 支持 Mamba = 全家都能跑。」** 错。Hymba 未查到原生支持，Zamba 初代不在，Kimi Linear 走 KDA（Kimi Delta Attention）自己的 backend。

## 一句话总结

vLLM 把 Mamba 当成「另一种缓存」接进 V1：O(1) 状态替 O(L) KV，prefill/decode 不变，但分页、接力、边界复用、回滚都得围绕「状态点」重做；省下的是显存与带宽，付出的是精度与工程复杂度。

## 自测题

### 1. 为什么 Mamba 的 prefix cache 默认只能从 block 边界恢复？

<details><summary>查看答案</summary>

KV 每个 token 位置都有独立副本，切块粒度可以很细；Mamba 状态是「整段前缀揉成的一个点」，没存过就恢复不出来。vLLM 把状态存在 block 网格上，所以默认命中只能落在 block 边界。要让「前缀在 block 中间结束」也能复用，得开 `--enable-mamba-shared-prefix-checkpoint` 在共享前缀结束处补一个 checkpoint，代价是额外存储，而且要求 APC 已开、EAGLE/MTP 在 Mamba 组上、`--prefix-match-unit` 小于 block size，且不是多模块 MTP。

</details>

### 2. Jamba-v0.1 有 32 层、其中 4 层 attention，每层每 token 的 KV 约 4 KiB；32K 上下文的 KV 是多大？

<details><summary>查看答案</summary>

每 token = 4 层 × 4 KiB = 16 KiB；32K = 32768 token，16 KiB × 32768 = 512 MiB。同规模全 attention 是 32 层 × 4 KiB = 128 KiB/token，32K 约 4 GiB。所以 Jamba 的 KV 是全 attention 的 1/8，不是 1/2。状态侧约 8.3 MiB，与长度无关；交叉点约 530 token。

</details>

### 3. 投机解码在 Mamba 上为什么不能照搬 Transformer 的验证？

<details><summary>查看答案</summary>

Transformer 验证 k 个 draft token 时只读 KV，被拒的位置直接丢掉即可；Mamba 的状态是递推出来的，验证时状态已经「走过」k 个位置，只接受 j 个就必须把状态推进到第 j 个被接受 token 的位置、丢弃被拒 token 的改动。所以 decode 算子要接受 `num_accepted_tokens`，按接受数选初始状态。

</details>

### 4. chunked prefill 对 Mamba 多做了什么？三篇各讲哪一面？

<details><summary>查看答案</summary>

多了一次状态接力：每个 chunk 算完要把块末状态写进 cache，下一个 chunk 读回来当初始状态；调度器还要告诉算子这次覆盖哪些 block、从哪个 block 读初值（`cu_chunk_seqlen` / `last_chunk_indices`，老接口 `block_idx_*` 与 `initial_state_idx`）。模型演进篇讲数学（仿射复合、块内矩阵合成），本篇讲调度与缓存，硬件篇讲 chunk_state / chunk_state_passing 怎么在昇腾上落成算子。

</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| conv_state | 短因果卷积的窗口尾巴（K−1 步），Mamba 缓存的一半，形状 (conv_dim, K−1) |
| ssm_state | 每层定长循环状态 (heads, head_dim, d_state)，不随上下文变长 |
| 状态分页 / block table | 状态与 KV 共用分页内存池、按 block id 索引；block table 是逻辑块到物理块的映射；一个序列任一时刻只有一份当前状态 |
| HMA（Hybrid Memory Allocator） | 混合内存分配器：按层类型分组、共用物理页，把 full attention（FA）的 block_size 抬到不小于 SSM 页，再给 SSM 补齐（padding） |
| MambaSpec | 混合模型的缓存规格：声明每层要几种状态、各多大，缺它缓存管理器只能按 KV 假设分配 |
| chunked prefill 状态接力 | 块间显式传初始状态（cu_chunk_seqlen / 老接口 initial_state_idx） |
| selective_scan / selective_state_update | prefill 的分块扫描 / decode 的单步状态更新，两个 Triton 算子 |
| causal-conv1d | 把短因果卷积融合成一个算子，否则滑窗小算子会拖垮长序列 |
| APC / mamba prefix checkpoint | 状态只能从 block 边界恢复，故可额外存检查点；默认关，且要求 EAGLE/MTP |
| EAGLE / MTP | 两类投机解码：EAGLE 用小模型在特征层先猜 token；MTP 是模型自带的多 token 预测头 |
| num_accepted_tokens | 投机解码「上一步接受几个」，用它把状态回滚到第 j 个被接受 token |
| n_groups | Mamba2 中 B/C 按组共享的组数；TP+量化要求 n_groups % tp_size == 0 |
| W8A8 | 权重与激活都压到 8 bit，只作用在 Linear；A/Δ/状态更新通常保高精度 |
| GDN / KDA | Gated DeltaNet（Mamba-2 + delta rule，写入前先按 key 擦旧值）与 Kimi Delta Attention（逐通道门控），补精确召回 |
| d_state / d_inner | 每通道的状态维度 / 内部通道维度；状态大小 ≈ d_inner × d_state 个数，是显存账的乘数 |
| head_dim / KV 头 | 每个头的维度 / K、V 的头数；KV 每 token ≈ 2 × KV 头 × head_dim × 字节 |
| 仿射复合 (a, b) | h_out = a·h_in + b；复合 (a₂,b₂)∘(a₁,b₁) = (a₂a₁, a₂b₁+b₂)。b 是常数项，不是 delta rule 的学习率 β |

## 还没想清楚

- **未能核实｜Hymba**：主干注册表与支持列表里都查不到 `HymbaForCausalLM`，Hugging Face 模型卡片却提到 vLLM 用法，可能走 Transformers 后端回退；没跑通，不下结论。
- **存疑｜Mamba 前缀缓存的当前默认**：APC 文档讲 checkpoint 开关，v0.11.0 的 v1_guide 仍写「not yet supported」，代码默认值没核对；只信自己版本的 `--help` 与启动日志。
- **未能核实｜昇腾 GDN 融合算子与 Triton 路径的数值/性能差异**：没找到公开数字，不写成结论。

## 来源

- [vLLM v1_guide @ v0.10.0](https://github.com/vllm-project/vllm/blob/v0.10.0/docs/usage/v1_guide.md)
- [vLLM v1_guide @ v0.10.1](https://github.com/vllm-project/vllm/blob/v0.10.1/docs/usage/v1_guide.md)
- [vLLM v1_guide @ v0.10.2](https://github.com/vllm-project/vllm/blob/v0.10.2/docs/usage/v1_guide.md)
- [vLLM v1_guide @ v0.11.0](https://github.com/vllm-project/vllm/blob/v0.11.0/docs/usage/v1_guide.md)
- [vLLM v0.10.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.10.0)
- [vLLM v0.10.1 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.10.1)
- [vLLM v0.10.2 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.10.2)
- [vLLM v0.11.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.11.0)
- [PR #21233（hybrid 文档去掉 enforce_eager）](https://github.com/vllm-project/vllm/pull/21233)
- [PR #22541（mamba_ssm 移出核心依赖）](https://github.com/vllm-project/vllm/pull/22541)
- [PR #23035（Mamba1 全图）](https://github.com/vllm-project/vllm/pull/23035)
- [PR #23716（默认关 hybrid/Mamba APC）](https://github.com/vllm-project/vllm/pull/23716)
- [PR #24593（Mamba2 TP+量化）](https://github.com/vllm-project/vllm/pull/24593)
- [vLLM 源码 ops/mamba_ssm.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/mamba_ssm.py)
- [vLLM 源码 ops/causal_conv1d.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py)
- [Hybrid KV cache manager 设计文档](https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md)
- [APC 文档（Hybrid Mamba models）](https://github.com/vllm-project/vllm/blob/main/docs/features/automatic_prefix_caching.md)
- [NIXL PD 兼容表](https://github.com/vllm-project/vllm/blob/main/docs/features/nixl_connector_compatibility.md)
- [vLLM 博客：Hybrid SSM 的 PD 分离（HMA）](https://vllm-project.github.io/2026/04/21/hybrid-ssm-disagg.html)
- [Jamba-v0.1 config](https://huggingface.co/ai21labs/Jamba-v0.1/blob/main/config.json)
- [Mamba-Codestral-7B config](https://huggingface.co/mistralai/Mamba-Codestral-7B-v0.1/blob/main/config.json)
- [vllm-ascend Qwen3.5-27B / Qwen3.6-27B 教程](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md)
- [vllm-ascend Qwen3-Next 教程（停止维护说明）](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/Qwen3-Next.md)
- [vllm-ascend v0.30.0 release notes](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0)
- [vllm-ascend 支持矩阵](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/support_matrix/supported_models.md)
- [vllm-ascend PR #8829（GDN 前缀缓存）](https://github.com/vllm-project/vllm-ascend/pull/8829)
- [vLLM 论坛：Why V1 not support Mamba Models?（2025-04-21）](https://discuss.vllm.ai/t/why-v1-not-support-mamba-models/496)
