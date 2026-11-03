---
date: "2026-11-03"
title: "第 5 课 · 推理服务基础：prefill/decode 与 KV cache"
domain: "推理框架"
module_id: "serving-basics"
module: "推理服务基础：prefill/decode 与 KV cache"
status: "已学"
lesson_no: 5
goals: ["说清一个请求在服务端经历的 prefill 和 decode 两个阶段", "解释 KV cache 为什么让 decode 变快、代价是什么", "解释 continuous batching 与 PagedAttention 解决什么问题，并区分 TTFT 与 TPOT"]
tags: ["推理服务", "KV cache", "PagedAttention", "性能指标", "vLLM", "昇腾"]
---

# 第 5 课 · 推理服务基础：prefill/decode 与 KV cache

> 学完今天，你能：
> 1. 说清一个请求在服务端经历的 prefill 和 decode 两个阶段
> 2. 解释 KV cache 为什么让 decode 变快、代价是什么
> 3. 解释 continuous batching 与 PagedAttention 解决什么问题，并区分 TTFT 与 TPOT

## 课前自测

先回顾前置模块「Transformer 基础」的一句话：这类模型是自回归（autoregressive）生成——根据已生成的词元（token，文本切分后的最小单元），逐个预测下一个词元。带着它回答：

1. 生成第 100 个 token 时，前 99 个 token 的中间结果能复用吗？靠什么？
2. 服务端某一刻只有一个请求在跑，「批处理」就没有意义了吗？
3. 为什么首 token 的时延通常远大于之后每个 token？背后是哪两个阶段？

## 为什么需要它

vLLM（开源的大语言模型推理服务框架）跑得快的秘密不在模型，而在管好推理的两个阶段和显存：KV cache、continuous batching、PagedAttention 是三块基石。对昇腾工程师，这直接对应 `--max-model-len`、`--gpu-memory-utilization` 调什么、`/metrics` 里 TTFT/TPOT/吞吐在说什么——不懂它们，调参只能靠猜。

## 核心概念

### prefill

**定义**：把整个 prompt（用户输入的提示文本）一次性并行送入模型、产出首个新 token 的阶段（未启用 chunked prefill 时一次算完）。
**直觉**：考试前先通读题目，才能开始答题。
**机制**：prompt 的 n 个 token 拼成矩阵一次算完，每层注意力的 K（键）、V（值）全部存入 KV cache。这是「算力密集」阶段：吃满 NPU（昇腾 AI 处理器）矩阵单元，prompt 越长越久。
**例子**：8 token 的 prompt 一次算完，产出第 9 个 token。

### decode

**定义**：逐 token 生成阶段：每步只算一个新 token，直到输出结束标记。
**直觉**：成语接龙——每步只能接一个词，必须等上一步。
**机制**：每步只为新 token 计算 Q、K、V（新 K/V 写入缓存），注意力中的历史 K/V 直接查 KV cache，真正新算的只有一个 token 的量。这是「访存密集」阶段：单步计算量小，但 200 token 要串行走 200 步，瓶颈是读 KV cache 的带宽而非算力。
**例子**：生成 200 token = 200 次「接下一个词」。

### KV cache

**定义**：把每个历史 token 每层的 K、V 向量缓存在显存，避免每步重算历史。
**直觉**：演草纸——中间结果记下来，下一步直接查。
**机制**：注意力公式 softmax(QK^T/√d)·V（d 为每头维度）中，decode 时只有新 token 的 Q 是新的，历史 K/V 可完全复用；缓存与否数值结果一模一样，省的只是重复计算。代价是显存（量级见下文）。昇腾实践：共享长系统提示词时开 `--enable-prefix-caching`（相同前缀的 KV 块跨请求复用），省显存也省 prefill。
**例子**：生成第 100 个 token 时，前 99 个的 K/V 直接查表、不重算。

### continuous batching

**定义**：在「迭代」（一批 token 提交给 NPU 计算的一轮）粒度批处理：新请求随时进批、完成的随时退批，不等整批结束。
**直觉**：公交车随时上下客，而不是包车坐满才发车。
**机制**：每轮迭代把正在 prefill 和 decode 的请求混在同一批提交 NPU，谁生成结束标记谁退出。它治的是静态批处理「短请求陪长请求等到底」的浪费；vLLM 还支持 chunked prefill（长 prompt 切块、与 decode 交错），避免一个长请求卡住整批。
**例子**：短请求不必等长请求结束：随到随进批，出完即退批。

### PagedAttention

**定义**：把 KV cache 切成固定大小（默认 16 token）的块，从显存池按需分配，用一张「块表」记录每个请求用到哪些块。
**直觉**：像操作系统的虚拟内存分页：散页拼起来用，不求连续整块。
**机制**：各请求不再按最大长度连续预留显存，内部碎片被消除；并行采样（同一 prompt 生成多个结果）、束搜索（一次尝试多条候选路径、取概率最高者）时还能共享物理块。论文报告 KV cache 显存浪费从 60%–80% 降到 4% 以下。昇腾侧还有官方特性 KV Cache CPU Offload（KV 卸载到主机内存）。
**例子**：300 token 的请求领 19 个 16-token 块，而非预留一段连续显存。

### TTFT、TPOT 与吞吐

**定义**：TTFT（Time To First Token）：请求到达服务端到首个输出 token 出现的时延；TPOT（Time Per Output Token）：其后每个输出 token 的平均时延；吞吐：单位时间产出的 token 数（或完成请求数）。
**直觉**：TTFT 像第一道菜上桌时间，TPOT 像之后每道菜的间隔，吞吐像一天出菜量。
**机制**：TTFT ≈ 排队时间 + prefill 时间——prefill 时间随 prompt 长度与可用算力变化，排队时间随系统负载变化；TPOT 主要由 decode 单步时长（同批请求数、KV 读取带宽）决定；吞吐 ≈ 同批请求数 ÷ 单步时长（同批都在 decode 时成立，混入 prefill 后只能当数量级估计）。三者互相拉扯，是调参的核心矛盾；`--max-num-seqs`（同批最大请求数）调的正是它。
**例子**：聊天看重 TTFT，流式输出看重 TPOT，压测看重吞吐。

## 一步一步理解

拿 Llama-2-7B 算账：32 层、每层 32 头、每头 128 维，FP16（每数 2 字节）。请求：prompt 100 token、生成 200 token。

**KV cache 占多少显存？** 每个 token 每层存 K、V 两组向量：2 × 32 头 × 128 维 × 2 字节 = 16 KB，乘 32 层 = 512 KB/token；300 token（100+200）≈ 150 MB。一个请求约 150 MB，几百个并发就能吃满 HBM（昇腾 910B 单卡 64 GB）。`--max-model-len`（单请求最长 prompt+输出）与 `--gpu-memory-utilization`（引擎允许占用的显存总比例；KV cache 实际分到的是扣除模型权重与激活后的剩余部分，调大它间接扩大 KV 池）正因此存在。

**时延怎么拆？** 假设 prefill 100 token 耗时 300 ms，decode 每步 20 ms：

- TTFT ≈ 300 ms（假设排队时间为 0：等 prefill 算完出首个 token）；
- TPOT =（总时延 − TTFT）÷（200 − 1）≈ 20 ms，与单步 decode 一致；
- 总时延 ≈ 300 + 20 × 200 = 4.3 s。

**并发时吞吐怎么算？** 8 个请求同批 decode、每步 30 ms：每 30 ms 产出 8 个 token，吞吐 ≈ 8 ÷ 0.03 ≈ 267 token/秒，但每个人的 TPOT 恶化到 30 ms——并发拉高吞吐、也拉高每个人的 TPOT，调同批请求数必须接受这个交换。

## 常见误区

**误区 1：「KV cache 是缓存答案的缓存。」** 错。它只缓存中间向量 K/V，数值上与不缓存完全一致，省的是重复计算。

**误区 2：「decode 慢是因为它每步都要把整个序列重新算一遍。」** 错。有 KV cache，decode 每步真正新算的只有一个 token；慢在串行——200 步必须一步一步出。

**误区 3：「TTFT 慢说明 decode 慢。」** 错。TTFT 产生于 prefill 与排队：先查 prompt 是否过长、是否排队，而不是先查 decode。

**误区 4：「TPOT 是固定值，优化它等于优化吞吐。」** 错。TPOT 是单个用户的体感时延，随并发变差；吞吐是系统整体产出，两者可同时上升，优化目标不同。

## 一句话总结

推理 = prefill 并行吃进 prompt，decode 串行吐出 token；KV cache 用显存换掉历史重算，continuous batching 与 PagedAttention 把算力和显存塞满；TTFT、TPOT、吞吐是三支温度计。

## 自测题

### 1. 一个请求 prompt 50 token、生成 100 token：为什么这 100 个输出必须 100 步串行算，不能像 prompt 那样一次并行算完？

<details><summary>查看答案</summary>
自回归有严格串行依赖：第 i+1 个 token 以「前 i 个 token」为条件，最后一个必须等第 i 步算完才知道，只能一步步出。KV cache 让每步不重算历史 K/V，但改变不了依赖链——这正是 TPOT 主要由单步 decode 时长决定的原因。</details>

### 2. 「显存给得越多 KV cache 越好，把 `--gpu-memory-utilization` 调到 1.0 就够用」——错在哪？

<details><summary>查看答案</summary>
`--gpu-memory-utilization` 是引擎允许占用的显存总比例，权重、激活与 KV cache 都从这份配额里出，KV 池是扣除权重与激活后的剩余；调大它只是间接扩大 KV 池，调到 1.0 会挤占系统缓冲、有 OOM 风险。KV cache 也不是越大越有用：只有活跃请求用到它，更好的做法是 PagedAttention 按需分块，配合 `--max-model-len`、prefix caching 控制总量。</details>

### 3. 请求 A（prompt 2000、输出 20）与 B（prompt 10、输出 1000）同时到达：静态批处理浪费了什么？continuous batching 如何改善？

<details><summary>查看答案</summary>
静态批处理整批同进同出：A 的 prefill 长，B 陪等；之后 B 的 decode 长，A 又陪等——快的陪慢的耗到底。continuous batching 迭代粒度随时进出：B 在 A 仍 prefill 时进批、生成完即退批；chunked prefill 把 A 的长 prompt 切块与 decode 交错，避免独占整批。</details>

### 4. 同批最大请求数从 8 调到 64，TTFT、TPOT、吞吐分别怎么变？为什么？

<details><summary>查看答案</summary>
吞吐先升后平：同批 decode 请求越多，每步产出 token 越多，但 KV 带宽与算力打满后不再上升。TPOT 变差：每步要读的 KV 更多、单步更长。TTFT 也可能变差：新请求要等当前迭代结束才能进批，排队变长。这正是 `--max-num-seqs` 的权衡。</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| prefill | 一次性并行处理整个 prompt、产出首个 token 的阶段，算力密集 |
| decode | 逐 token 生成的阶段，访存密集、必须串行 |
| KV cache | 缓存每个 token 每层的 K/V 向量，用显存换掉对历史的重复计算 |
| continuous batching | 迭代粒度随时进、退请求的批处理，解决静态批处理互相等待 |
| PagedAttention | 把 KV cache 切成固定大小的块、按需分配的显存分页管理 |
| TTFT | 首 token 时延，约等于排队时间加 prefill 时间 |
| TPOT | 每个输出 token 的平均时延，主要由 decode 单步时长（并发）决定 |
| 吞吐 | 单位时间产出的 token 数或完成请求数，随并发先升后平 |

## 下一步

第 6 课「vLLM 引擎架构」进入引擎内部：调度器、执行器与 KV cache 管理器如何把 prefill/decode 跑成流水线。

## 来源

- Architecture Overview（vLLM 官方文档）— https://docs.vllm.ai/en/latest/design/arch_overview/
- PagedAttention 论文（arXiv:2309.06180，SOSP 2023）— https://arxiv.org/abs/2309.06180
- LLM Inference Series（KV caching 章节，Hugging Face 博客）— https://huggingface.co/blog/llm-inference-series
- vLLM Ascend 插件官方文档 — https://docs.vllm.ai/projects/ascend/en/latest/
