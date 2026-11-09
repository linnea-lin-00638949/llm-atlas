---
date: "2026-11-09"
title: "第 17 课 · 切状态、切矩阵、切层：DP/TP/PP/ZeRO 到底切了什么（上）"
domain: "训练"
module_id: "distributed"
module: "分布式并行：DP/TP/PP/ZeRO"
status: "进行中"
lesson_no: 17
goals: ["学完今天，你能说清 DP/TP/PP 各自切的是什么（数据、单层矩阵、层），并指出哪一种真正降低单卡显存、哪一种只是把状态复制了一份。", "学完今天，你能按 all-reduce / all-gather / reduce-scatter 把 7B 模型在 8 卡与 512 卡上的通信字节数和耗时算出来，说清加卡为什么从来不是线性加速。", "学完今天，你能解释 ZeRO-1/2/3 与 FSDP 的分片差异，并针对「模型多大、几卡、什么互联」给出一个有数字支撑的并行组合。"]
tags: ["分布式训练", "DP/TP/PP", "ZeRO", "FSDP", "集合通信", "显存与通信账"]
---

# 第 17 课 · 切状态、切矩阵、切层：DP/TP/PP/ZeRO 到底切了什么（上）

> 学完今天，你能：
> 1. 说清 DP/TP/PP 各自切什么——数据、单层矩阵、层——并指出哪一种真正降低单卡显存、哪一种只是把状态复制了一份。
> 2. 按 all-reduce / all-gather / reduce-scatter 把 7B 在 8 卡与 512 卡上的通信字节和耗时算出来，说清加卡为什么从来不是线性加速。
> 3. 解释 ZeRO-1/2/3 与 FSDP 的分片差异，并针对「模型多大、几卡、什么互联」给出一个有数字支撑的并行组合。

## 课前自测

1. 8 卡纯 DP，单卡显存会降到 1/8 吗？为什么？
2. TP=8 切开一层，这一层前向要几次 all-reduce？反向呢？
3. 7B 的梯度 14 GB，8 卡 ring all-reduce 一圈，每卡发出多少字节？512 卡呢？
4. ZeRO-3 比 ZeRO-2 多出哪种集合通信？通信量是几倍？
5. 8 段流水，每个 stage 每步只有 8 个 micro-batch，气泡占多少？

## 为什么需要它

上一课的账停在：7B 按 16 B/参数 算是 112 GB 训练状态，一张 64 GB 卡差 50 GB 上下，于是有了 ZeRO-1/2/3 那三行。但那三行只回答「装得下吗」，没回答「划不划算」——ZeRO-3 省下的每一 GB 显存，都是拿通信买的。另两条路上一课没展开：切开一层内的矩阵（TP）、切开层（PP）。

今天把四个旋钮放上同一张坐标纸。判据只有一句：**只有切模型或切状态才省单卡显存，切数据只换吞吐。**后面所有选择都从这一句长出来。

## 核心概念

### DP：切数据，复制状态

**定义**：N 卡各持一份完整模型与优化器状态，喂不同数据分片，各自前向反向得局部梯度，all-reduce 求平均后各卡用同一份梯度更新，参数永远一致。

**直觉**：N 个学生用同一本教材做不同习题，做完对答案取平均，再一起改笔记。笔记人手一份，纸一点没省，省的是时间。

**机制**：每步一次梯度 all-reduce，对象是 14 GB 级梯度，量只与参数量有关、与 token 数无关。所以 DP 是「用通信换吞吐」：卡越多同一全局 batch 越快，单卡占用纹丝不动。

### TP：切单层内的矩阵

**定义**：把一层的大矩阵按列/按行切成 N 份，每卡算一份，算完拼成完整向量再进下一层。Megatron 的切法：Q/K/V 与 MLP 第一层按列切，输出投影与 MLP 第二层按行切，于是每层前向 2 次 all-reduce（attention 后、MLP 后），反向再 2 次。

**直觉**：八个人各算几列；下一层要吃完整向量，每次算完都得拼齐——拼的动作就是 all-reduce，一层都省不掉。

**机制**：通信对象是激活（token × hidden），不是权重，所以随 token 数线性涨、每层都发生。两条硬约束：hidden 与 head 数要能被 TP 整除；TP 组 ≤ 8、留在机内。

### PP：切层

**定义**：L 层按顺序切 p 段，每段一组卡，micro-batch 从第 1 段流到第 p 段，反向再流回来。

**直觉**：工厂流水线，灌满才有产能；开头灌入、结尾排空时总有人在等，那就是气泡。

**机制**：段边界只传激活（点对点 send/recv），量小、能跨机，这是它比 TP 更适合跨机的根本原因。代价是气泡：(p−1)/(m+p−1)，m 是每个 stage 每步的 micro-batch 数；经验底线 m ≥ 4p，气泡才压到 20% 以下。

### ZeRO 与 FSDP：切状态

**定义**：ZeRO 不切模型，切「每卡都存一份」的冗余状态：1 切优化器状态（7B 是 84 GB），2 再切梯度，3 连参数一起切。FSDP 是 PyTorch 原生的全分片数据并行，按层 all-gather 参数，约等于 ZeRO-3。

**直觉**：三人共用一本参考书。DP 是人手一本；ZeRO-1 把最厚的答案册撕成三份，谁用谁借；ZeRO-3 连课本都撕了，每节课前借齐、下课还回——借书就是通信。

**机制**：分片对象每降一级，省一截显存、多一截通信。同口径下 ZeRO-1/2 通信约 2× 参数量（与 DP 同量），ZeRO-3 是 3×（DP 的 1.5 倍）。ZeRO-2 用 reduce-scatter + all-gather，一半加一半正好等于一次 all-reduce。

### 通信量算术

**定义**：一次集合通信里单卡真正发出的字节，只由张量大小 S 与卡数 N 决定。

**直觉**：ring 上每卡每圈收发变少，圈数却变多，两者抵消，总量收敛到 2S。

**机制**：all-reduce 是 2(N−1)/N × S；reduce-scatter 与 all-gather 是 (N−1)/N × S。(N−1)/N 从 8 卡 0.875 到 512 卡 0.998，再涨也涨不动。7B 梯度 14 GB：8 卡 2×0.875×14 = **24.5 GB**；512 卡 2×0.998×14 ≈ **28 GB**。卡数涨 64 倍，通信只涨 14%，这是 DP 能扩到几千卡的算术基础。

计算通信比要拿同一段时间比。7B 一层 0.4375 GB 参数、4096 token 的 micro-batch：一层计算约 5.4 TFLOP，TP=8 每卡 0.67 TFLOP，400 TFLOPS 峰值、MFU 40% 下 4.2 ms；通信 4 次 all-reduce × 58.6 MB = 234 MB，机内 392 GB/s 下 0.6 ms，比值 7:1。换跨机 IB 400 Gbps（50 GB/s）是 4.7 ms，比值 0.9:1——通信比计算还长。机内 392 GB/s、机间 50 GB/s，差约 8 倍，接近一个数量级，这就是「TP 留机内、PP/DP 敢跨机」的定量理由。

### 选择规则

按顺序问四个问题：**容量**——16 B × 参数量 ÷ 卡数（全分片下限）≤ 单卡显存 × 0.6，过不了先加卡；**装得下只是慢**——DP + ZeRO-1/2，不改模型代码；**一层都放不下或单层时延太长**——TP，且 ≤ 8、留机内；**还要更大**——PP 补容量、DP 补吞吐，叠成 TP × PP × DP 的 3D 并行。

### 失效模式

| 失效 | 可观测信号 | 第一步 |
| --- | --- | --- |
| TP 跨机 | all-reduce 时间接近计算时间；TP 4→8 吞吐反降 | 查 rank 映射，把 TP 收回单机 |
| PP 气泡 | 首末段以外卡空闲率高；step time 不随 p 线性降 | 加 m：降 micro-batch 或加梯度累积，使 m ≥ 4p |
| ZeRO-3 放大 | all-gather 随层数线性涨；扩卡 step time 不降 | 先降到 ZeRO-2，再分桶、加 micro-batch 摊薄 |
| 超参没跟 | 改并行度后 loss 出台阶，或前几千步发抖 | 先固定全局 batch，再按变比重设 lr/warmup |

只记一条：**改并行度等于改有效 batch，改完必须回头对一遍 lr、warmup、梯度裁剪。**

## 一步一步理解

7B、8 张 64 GB 卡、micro-batch 4096 token。数字都能手算复核。

**第 1 步 容量**：112 ÷ 8 = 14 GB/卡，这是全分片的理论下限；纯 DP 每卡 112 GB 装不下，所以 8 卡 7B 至少要 ZeRO-1。加卡不是为了装下，是为了时间：140B token、8 卡 400 TFLOPS、MFU 40%，要 53 天。

**第 2 步 DP 的账**：DP=8 + ZeRO-1，每卡状态 38.5 GB；每步一次梯度 all-reduce，2 × 7/8 × 14 = **24.5 GB/卡**，可与反向重叠，通常不是瓶颈，但一分显存不省。

**第 3 步 ZeRO-3 展开**：32 层，每层 14 ÷ 32 = 0.4375 GB，每卡只存 1/8：

| 时机 | 集合通信 | 次数/层 | 每卡字节 | 32 层合计 |
| --- | --- | --- | --- | --- |
| 前向 | all-gather 参数 | 1 | 0.4375 × 7/8 = 0.383 GB | 12.25 GB |
| 反向 | all-gather 参数（重算用） | 1 | 0.383 GB | 12.25 GB |
| 反向 | reduce-scatter 梯度 | 1 | 0.383 GB | 12.25 GB |
| 合计 | — | 3 | 1.15 GB | **36.75 GB** |

对账：上一课那三行 40.5 / 28.3 / 16 GB 含约 2 GB 激活与 workspace，只算状态项是 38.5 / 26.3 / 14 GB，下文用后者。ZeRO-1/2 是 2× = 24.5 GB，ZeRO-3 是 3× = 36.75 GB。状态从 26.3 降到 14 GB 省 12.25，通信从 24.5 涨到 36.75 付 12.25，接近 1:1。（口径：每步 1 个 micro-batch；梯度累积 m 次则前向、反向 all-gather 各 m 次，ZeRO-3 变成 (2m+1)/3 × 36.75 GB。）

**第 4 步 TP 的账**：TP=8，每层 all-reduce 对象是 4096 × 4096 × 2 B = 33.5 MB 激活，ring 后每卡 58.6 MB，一层 4 次 = 234 MB，32 层 = **7.5 GB / 每个 micro-batch**。一层计算 4.2 ms、通信 0.6 ms，14% 开销，机内付得起。

关键在分母：DP 的 24.5 GB 是每步一次，TP 的 7.5 GB 是每个 micro-batch 一次。同一个 4.2M token 的全局 batch，DP=8 只付 24.5 GB；TP=8 时 DP 只能是 1，要 1024 个 micro-batch 才凑够，付 7.7 TB，差两个数量级。跨机 7.7 TB ÷ 50 GB/s = 154 s，整步计算才 137 s——通信比计算还长。

**第 5 步 PP 的账**：换成 64 卡（TP1·PP8·DP8），32 层切 8 段、每段 4 层，气泡 7/(m+7)；全局 batch 4.2M、micro-batch 4096 → m = 4.2M ÷ (4096×8) = 128，气泡 5%，划算。固定 PP=8，DP 扩到 64（512 卡）时 m 掉到 16，气泡 30%；DP=128 时 m=8，气泡 47%。PP 段数不变，DP 越大、每副本 token 越少，气泡越贵。

**第 6 步 定组合**：8 卡 7B 取 TP=1、PP=1、DP=8 + ZeRO-1，38.5 GB/卡。TP=1 因单层放得下，切它纯付通信税；不选 ZeRO-3，因余量够放激活，没必要为 12 GB 显存多付 50% 通信。

**第 7 步 换规模**：

| 模型 | 状态总量 | 容量下限 | 8 卡 | 64 卡 | 512 卡 |
| --- | --- | --- | --- | --- | --- |
| 7B | 112 GB | 2 | DP8·Z1 / 38.5 GB | DP64·Z1 / 29.3 GB | DP512·Z1 / 28.2 GB |
| 70B | 1120 GB | 18 | 放不下 | TP8·DP8·Z1 / 48.1 GB | TP8·DP64·Z1 / 36.6 GB |
| 405B | 6480 GB | 102 | 放不下 | 放不下 | TP8·PP8·DP8 / 34.8 GB |

读法：7B 全程 TP=1，单层小，开 TP 只加税；70B 在 8 卡上 140 GB/卡 装不下；到 64 卡，要么 TP8+ZeRO-1（48.1 GB，2× 通信、TP 留机内），要么 DP64+ZeRO-3（17.5 GB，但 all-gather 跨机），取前者；405B 到 512 卡才装得下，靠 PP 把每卡压到 16 层。

反例：405B 在 512 卡上用 TP1·DP512 + ZeRO-3 也装得下（12.7 GB/卡），但每步 3 × 810 × 511/512 ≈ 2.4 TB/卡，50 GB/s 下 48 s，而整步计算约 124 s——四成时间在等网。3D 把跨机字节压到 DP 梯度的 177 GB/卡（3.5 s），TP 走机内、PP 只付 5% 气泡，这才是 512 卡的解。

**第 8 步 昇腾上的同一件事**：vLLM 起 TP=8 的 70B 服务，TP 组就是单机 8 卡，走 HCCS 做 all-reduce，与训练前向同构，只是没有反向；训练侧对应 HCCL，原则相同：TP 组留同一机框。

## 常见误区

**一、「加卡做 DP 就能训更大的模型。」** DP 复制状态，单卡占用不动。70B 的 1120 GB 除以 8 卡是 140 GB，远超 64 GB 的卡，连 ZeRO-3 下限都装不下——除了加卡，只剩 CPU/NVMe offload。

**二、「ZeRO-3 最省显存，无脑开 3。」** 3 号位是 3× 通信，2 号位是 2×；省下的显存与多付的通信大致 1:1。机间只有 50 GB/s 时这笔交易常亏，能用 ZeRO-1 解决的（7B/8 卡）别上 3。

**三、「PP 段数越多越省显存，切细总没错。」** p 从 4 加到 8，显存省一半，气泡从 3/(m+3) 变成 7/(m+7)；m=16 时从 16% 涨到 30%。切之前先数 micro-batch，m ≥ 4p 是底线。

**四、「并行度调完就开跑，超参不用动。」** 改 TP/PP/DP 都会改有效 batch 与梯度噪声：全局 batch 从 4.2M 变成 1M 而 lr 没动，loss 会先掉一台阶再翘尾，容易被误判成数据问题。

## 一句话总结

DP 切数据、复制状态，只换吞吐；TP 切层内矩阵，每层 4 次 all-reduce、通信随 token 涨，只能留机内；PP 切层，省显存但按 (p−1)/(m+p−1) 收气泡税；ZeRO/FSDP 切状态，1、2 级通信等于 DP，3 级是 1.5 倍。选组合的顺序：先算容量下限，单卡放不下上 ZeRO，单层放不下上 TP（≤8），再大就 TP×PP×DP。

## 自测题

### 1. 7B 的梯度 all-reduce，8 卡与 512 卡各多少字节？为什么加卡不是线性加速？

<details><summary>查看答案</summary>all-reduce 每卡发出 2(N−1)/N × S，S = 14 GB。8 卡：2 × 7/8 × 14 = 24.5 GB；512 卡：2 × 511/512 × 14 = 27.95 GB，只多 14%。ring 上每卡发送量随 N 变小、圈数随 N 变多，两者抵消，总量收敛到 2S。于是每步时间 ≈ 计算/N + 近乎不变的通信，卡越多通信占比越大，加速比饱和——这就是加卡不线性的算术原因。</details>

### 2. 8 卡上 ZeRO-2 与 ZeRO-3 的通信各是多少？为什么说 ZeRO-3 是 1.5 倍？

<details><summary>查看答案</summary>ZeRO-2 = reduce-scatter 梯度 12.25 + all-gather 参数 12.25 = 24.5 GB（2Ψ）；ZeRO-3 = 前向 all-gather + 反向 all-gather + 反向 reduce-scatter = 3 × 12.25 = 36.75 GB（3Ψ），3Ψ/2Ψ = 1.5。省下的显存（26.3 → 14 GB）与多付的通信几乎 1:1，不是免费的。</details>

### 3. TP=8 一层：计算通信比在机内和跨机各是多少？

<details><summary>查看答案</summary>一层计算 5.4 TFLOP，每卡 0.67 TFLOP，按 400 TFLOPS × 40% 算 4.2 ms。通信 4 次 all-reduce，每次 33.5 MB 激活、ring 后 58.6 MB，共 234 MB：机内 392 GB/s → 0.6 ms，比值 7:1；跨机 50 GB/s → 4.7 ms，比值 0.9:1，通信比计算还长。所以 TP 只能留机内。</details>

### 4. PP 的气泡怎么算？为什么 DP 越大气泡越贵？

<details><summary>查看答案</summary>气泡占比 (p−1)/(m+p−1)，m 是每个 stage 每步的 micro-batch 数。p=8、m=16 时 7/23 ≈ 30%，m=8 时 7/15 ≈ 47%。m = 全局 batch ÷ (micro-batch × DP)：全局 batch 固定时 DP 越大、每副本 token 越少，m 越小，气泡越贵。所以 PP 段数与 DP 度要一起定，底线 m ≥ 4p。</details>

### 5. 70B、64 张 64 GB 卡，给一个装得下的组合并算单卡状态。

<details><summary>查看答案</summary>TP=8、DP=8、ZeRO-1：每卡权重 2 × 70/8 = 17.5 GB，梯度 17.5 GB，优化器 12 × 70/64 = 13.1 GB，合计 48.1 GB。另一条路 TP=1、DP=64、ZeRO-3 只有 17.5 GB/卡，但 all-gather 跨 64 卡、通信 3Ψ，通常不如前者。判据：先看容量，再看谁的通信跨机。</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| 数据并行（DP） | 每卡一份完整模型、喂不同数据，梯度 all-reduce 后同步更新；不省单卡显存。 |
| 张量并行（TP） | 把一层内的权重矩阵按行或按列切开，前向反向各有 2 次 all-reduce。 |
| 流水线并行（PP） | 把层按顺序切成若干段，micro-batch 像流水线一样流过；省显存，代价是气泡。 |
| 流水线气泡 | 灌入与排空时的空闲，占比约 (p−1)/(m+p−1)。 |
| ZeRO | 把每卡重复存储的状态分片：1 切优化器状态、2 切梯度、3 切参数。 |
| FSDP | PyTorch 的全分片数据并行，按层 all-gather 参数，约等于 ZeRO-3。 |
| all-reduce | 全卡求和后人人拿到完整结果；ring 下每卡发出 2(N−1)/N × S。 |
| reduce-scatter | 求和后每卡只留自己那一片，量是 all-reduce 的一半。 |
| all-gather | 把各卡分片拼成完整张量，量也是 all-reduce 的一半。 |
| 计算通信比 | 同一段时间里计算耗时与通信耗时之比；低于 1 就该换切法。 |
| 3D 并行 | TP × PP × DP 三维同时开，总卡数等于三者乘积。 |
| micro-batch | 一次前向反向真正喂进去的小批；梯度累积把它攒成全局 batch。 |
| 全局 batch | 一个优化器步吃掉的 token 总数；改并行度时它必须保持不变。 |

## 下一步

下一课（下）把通信本身拆开：ring all-reduce 怎么走、为什么 8 卡之后加卡收益骤减、拓扑感知的 rank 映射怎么写、昇腾 HCCL 与 vLLM 的 TP 组怎么对齐。之后进入「后训练：SFT 与强化学习」——SFT 的数据配比、loss mask 与多轮样本拼接，RLHF/DPO/GRPO 的目标函数与显存账（策略、参考、奖励、评论四个模型怎么摆），以及并行度怎么重算：SFT 是标准 DP/ZeRO 问题，RL 还多一份采样与参考模型常驻。

## 来源

- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism — https://arxiv.org/abs/1909.08053
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models — https://arxiv.org/abs/1910.02054
- Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM — https://arxiv.org/abs/2104.04473
- DeepSpeed 官方教程：ZeRO — https://www.deepspeed.ai/tutorials/zero/
- PyTorch 官方文档：FSDP — https://docs.pytorch.org/docs/stable/fsdp.html
- Megatron-LM GitHub（张量并行与流水线并行实现） — https://github.com/NVIDIA/Megatron-LM
- GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism — https://arxiv.org/abs/1811.06965
- PipeDream: Generalized Pipeline Parallelism for DNN Training — https://arxiv.org/abs/1806.03377
- NVIDIA NCCL 官方文档（ring all-reduce 与集合通信） — https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
- 昇腾社区 CANN 文档：HCCL 集合通信库用户指南 — https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/devaids/hcclug/hcclug_0001.html