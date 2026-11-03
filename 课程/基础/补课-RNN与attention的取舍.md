---
date: 2026-11-03
title: 补课 · RNN 与 attention 的取舍
type: supplement
tags: [RNN, attention, 序列模型, 并行, Mamba]
---

# 补课 · RNN 与 attention 的取舍

> 读完这篇，你能：
> 1. 说清 RNN 的 h_t 为什么天然串行；
> 2. 用「并行度 / 显存 / 长程记忆」三个维度比较 RNN、attention、SSM；
> 3. 理解 Mamba 为什么是「线性 RNN + 并行 scan」，以及它换走了什么。

## 为什么需要它

序列模型要回答一个问题：读到第 100 个 token 时，怎么记住前 99 个？

- **RNN 的答案**：维护一个递推状态 h，每读一个 token 更新一次：`h_t = tanh(W_h·h_{t-1} + W_x·x_t)`；
- **attention 的答案**：把历史全部留着（K/V），每个新 token 直接「回看」所有历史。

2017 年 Transformer 用 attention 取代 RNN，主因不是「更聪明」，而是**能并行**。但 attention 留下了 KV cache 越写越长的账单；Mamba 又把 RNN 捡回来，用「线性化 + 并行 scan」把并行度买回来。不理解 RNN 的串行，就理解不了 Mamba 在解决什么。

## 核心概念

### RNN 的串行：链式数据依赖

递推式 `h_t = tanh(W_h·h_{t-1} + W_x·x_t)` 里，h_t 直接依赖 h_{t-1}。要算第 100 步，必须先算完第 1…99 步。GPU 有几千个核，但这里**没有可同时做的独立工作**——只有沿时间轴的串行链。

直觉：接力赛，下一棒必须等上一棒交棒；而 attention 像把所有运动员叫到同一块场地，各自同时跑。

### 线性 RNN 为什么能并行：结合律 + scan

如果递推是**线性**的：`h_t = a_t·h_{t-1} + b_t`，就把它看成一次仿射变换 `(a_t, b_t)`。两个仿射变换复合仍是仿射变换：

`(a₂,b₂) ∘ (a₁,b₁) = (a₂a₁, a₂b₁+b₂)`

复合满足**结合律**，于是可以两两合并、树形归约——这就是并行前缀和（scan）：O(n) 总计算、O(log n) 深度。

**关键**：scan 要求「线性 + 结合律」。RNN 的 tanh 是非线性的，破坏了结合律，只能串行。Mamba 的「选择」让 a_t、b_t 随输入变（仍保持线性递推），所以训练能 scan、推理能递推——这是它同时拿到并行和 O(1) 推理的原因。

### attention 的并行：没有时间依赖

`QK^T` 一次算出所有 token 对的关系，序列维上没有依赖（因果 mask 只是把未来遮掉，不改变可并行性）。所以 prefill 能把整段 prompt 一次并行算完，GPU 吃满。

### 三者的账

| 维度 | RNN | attention | 线性 SSM（Mamba） |
| --- | --- | --- | --- |
| 训练并行度 | O(n) 串行（非线性时） | O(1) 深度，全并行 | O(log n) 深度（scan） |
| 训练显存（激活） | O(n) | O(n²)（attention 矩阵） | O(n) |
| 推理每 token 成本 | O(1) 计算 + O(1) 状态 | 读全部 KV，O(n) 带宽 | O(1) 计算 + O(1) 状态 |
| 长程记忆 | 压缩进状态，会衰减 | 精确检索，不丢 | 压缩进状态，可学习遗忘 |
| 可解释 / 可编辑 | 弱 | 强（KV 可查、可改） | 弱 |

一句话：**并行度、显存、长程记忆，三者不可兼得**。attention 用显存换并行与精确记忆；RNN 用串行换 O(1) 成本；Mamba 想用「线性 + scan」同时拿并行与 O(1)，代价是记忆被压缩、只能近似检索。

## 一步一步理解

设序列长 n = 4096，隐藏维 d = 512。

1. **RNN 训练**：4096 个时间步串行。每一步是一次 512×512 的矩阵乘（外加 batch 维并行）。GPU 每个时间步只有很小的并行度，利用率低，墙钟时间长。
2. **attention 训练**：QK^T 是 4096×4096×512 的矩阵乘，切成几万个 128×128 的小块，GPU 满负荷；代价是 4096² 的 attention 矩阵要进显存（bf16 约 32 MiB/头，层数一多就爆）。
3. **推理**：attention 每生成一个 token，要把历史 KV 全读一遍——32 层、8 个 KV 头、128 维、fp16，每 token 128 KiB；4096 上下文就是 512 MiB，且每步都要读。RNN / Mamba 每步只读写一个定长状态（Mamba-2 约 128 MiB/序列，与长度无关）。
4. **长上下文**：n 从 4K 涨到 128K，attention 的 KV 涨 32 倍，RNN/SSM 的状态不变。这就是长上下文场景里 SSM 的吸引力。

## 常见误区

1. **「RNN 不能并行」**：线性 RNN 训练可以 scan 并行；串行的是**推理时**的逐步递推（每步仍 O(1)）。
2. **「attention 没有代价」**：prefill 是 O(n²) 计算，KV 随长度线性涨、decode 每步读全量 KV；长上下文时显存和带宽才是大头。
3. **「Mamba 会取代 attention」**：现实是混合架构（Jamba 1:7、Zamba、Hymba），attention 保留精确检索，SSM 承担长程压缩与吞吐。
4. **「状态就是压缩版 KV」**：状态不可寻址、只进不出；KV 可随机访问、可回滚。语义不同。

## 一句话总结

RNN 输在「时间步必须排队」，attention 赢在「所有位置同时看」，代价是 KV 越写越长；Mamba 把 RNN 线性化后用 scan 排队变并行，但记忆只能压缩——这就是三者的取舍。

## 自测题

### 1. 为什么 tanh RNN 不能 scan，而线性 RNN 可以？
<details><summary>查看答案</summary>scan 依赖「复合运算满足结合律」做树形归约。线性递推 h_t = a_t·h_{t−1} + b_t 是仿射变换，复合仍是仿射变换、满足结合律；tanh 是非线性函数，复合后不满足结合律，只能逐步串行。</details>

### 2. attention 的「并行」代价具体花在哪？
<details><summary>查看答案</summary>两处：训练时 QK^T 产生 O(n²) 的 attention 矩阵（显存与计算）；推理时 KV cache 随上下文线性增长，decode 每步都要把全部 KV 读一遍（带宽）。</details>

### 3. 为什么 Mamba 推理是 O(1) 而训练能并行？
<details><summary>查看答案</summary>推理沿时间递推，每步只更新定长状态（O(1) 计算与显存）；训练时因为递推是线性的，可用并行 scan 把 O(n) 串行链变成 O(log n) 深度的树形归约。</details>

### 4. 混合架构为什么保留少量 attention 层？
<details><summary>查看答案</summary>SSM 把历史压成定长状态，适合长程与吞吐，但精确检索（比如「第 37 句原文」）会丢；少量 attention 层保留可寻址的 KV，负责精确召回，其余层用 SSM 省成本。Jamba 就是 4/32 层 attention。</details>

## 来源

- Vaswani et al., *Attention Is All You Need* (2017) — https://arxiv.org/abs/1706.03762
- Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate* (2014) — https://arxiv.org/abs/1409.0473
- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023) — https://arxiv.org/abs/2312.00752
- 3Blue1Brown, *But what is a neural network?* / *Attention in transformers* 系列 — https://www.youtube.com/@3blue1brown
