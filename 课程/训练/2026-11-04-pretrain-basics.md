---
date: "2026-11-04"
title: "第 15 课 · 预训练在学什么：从下一个 token 到 scaling law"
domain: "训练"
module_id: "pretrain-basics"
module: "预训练基础与 scaling law"
status: "进行中"
lesson_no: 15
goals: ["学完今天，你能写出语言模型的训练目标（下一个 token 的交叉熵），并说清 loss 数值在直觉上意味着什么", "学完今天，你能用 scaling law 解释「给定算力，模型该多大、数据该多少」，并独立算出 Chinchilla 式的 20 token/参数配比", "学完今天，你能读懂一条 loss 曲线，区分正常收敛、学习率过大导致的尖刺/发散、以及数据或实现 bug 造成的异常"]
tags: ["预训练", "scaling-law", "交叉熵", "困惑度", "训练稳定性", "Chinchilla"]
---

# 第 15 课 · 预训练在学什么：从下一个 token 到 scaling law

> 学完今天，你能：
> 1. 写出语言模型的训练目标（逐位置交叉熵），把 loss 数值换成「平均预测概率」与困惑度。
> 2. 用 scaling law 回答「给定算力，模型多大、数据多少」，并独立算出 Chinchilla 式的 20 token/参数配比。
> 3. 读懂一条 loss 曲线，区分正常收敛、学习率过大导致的尖刺/发散、数据或实现 bug 造成的异常。

## 课前自测

1. 词表 4 万、正确 token 概率 0.5，这一位交叉熵多少？4 个位置平均 0.8，困惑度多少？
2. GPT-3 是 175B、300B token；Chinchilla 是 70B、1.4T。各自每参数分到多少 token？
3. loss 2.0→1.9，困惑度从多少降到多少？算大变化吗？
4. 第 3 万步 loss 从 2.1 跳到 3.6、两步后回落，第一件事查什么？

## 为什么需要它

第 14 课把推理成本压成「机器成本 ÷ 有效 token 数 ÷ 利用率」。那个 7B 模型 8 卡输出 1,500 token/s 是压测来的，没人问过为什么是这个量级。答案在预训练：每步搬多少字节、算多少浮点，由 N 与 batch 决定，而 N 在训练开始那天就定死了。分两个 regime：

- **低 batch：memory-bound（访存受限）**。每吐一个 token 要把约 2N 字节权重（bf16 下 7B 是 14 GB）读一遍，单流上限 ≈ 带宽 ÷ 2N 字节；7B、3 TB/s 下约 214 token/s。
- **高 batch：compute-bound（算力受限）**。权重每步只读一次、摊到 B 个请求上，上限 ≈ 有效 FLOPS ÷ 2N；单卡 160 TFLOPS 有效算力下约 1.1 万 token/s。

所以 `--max-num-seqs` 与量化不是「救一点点」，而是把你从第一个 regime 推向第二个；但两个上限都由 N 定。预训练又最贵，几百万到几千万一次，钱怎么花、何时停由 scaling law 回答。

## 核心概念

### 训练目标：逐位置预测下一个 token

**定义**：给定前 t−1 个 token，模型输出词表 V 上的分布，目标是平均负对数似然 L = −(1/T) Σ log p(x_t | x_<t)，即与 one-hot 真实分布的交叉熵。

**直觉**：−log p 押错重罚：p=0.9 罚 0.11，p=0.1 罚 2.30；平均即「每 token 付多少意外」。

**机制**：不能拿模型自生成的前缀（一步错、无法并行），要永远喂真实前缀（teacher forcing），标签右移一位：

```
input_ids = tokens[:-1]   # BOS 我 爱 学
labels    = tokens[1:]    # 我 爱 学 习  ← 右移一位
loss      = cross_entropy(model(input_ids).view(-1, V), labels.view(-1))
```

第 t 位输出对应第 t+1 个 token。并行与不泄漏未来能同时成立，靠 attention 的 causal mask（因果掩码，下三角）：位置 t 只看 ≤t。分清两种 mask：causal 防泄漏，padding 防 pad 参与 attention 与 loss。

### 困惑度：把 loss 换成「猜几个」

**定义**：PPL = e^L，等于各位置正确概率倒数的几何平均——「相当于在 PPL 个候选里均匀乱猜」。

**直觉**：PPL=10 就是每步像在 10 个词里瞎猜，词表 4 万，候选已缩小 4,000 倍。

**机制**：换算 2.0→7.39、2.3→9.97、3.0→20.1、1.0→2.72。loss 是对数尺度：降 0.1 在任何位置都降 9.5%——相对变化只由 ΔL 决定。

### scaling law：把 loss 写成 N 与 D 的函数

**定义**：L(N,D) = E + A/N^α + B/D^β。N 是参数量、D 是训练 token 数；E 是外推到 N,D→∞ 的截距（不可约损失），后两项可约。

**直觉**：两个水桶漏水，E 是地板，水永远流不到地板以下。Chinchilla 拟合 E=1.69、A=406.4、B=410.7、α≈0.34、β≈0.28；指数远小于 1，投入翻倍只降一点点。

**E 的口径**：1.69 只是 Chinchilla 在其数据配比与分词器上、以 nats/token 拟合的截距，不是「文本」的普适熵底，换数据或分词器就变。别拿「loss < 1.69」当记忆判据，要看 train/val gap、去重后评测或同分布对照。

### 20 token/参数：经验点，非常数

固定算力 C ≈ 6ND（前向 2ND、反向 4ND）。对 N、D 求极小，拉格朗日条件给出 αA N^−α = βB D^−β：参数再降的边际收益 = 数据再降的边际收益。

Chinchilla 的实际训练点是 70B/1.4T（20 token/参数），配合 Approach 1–2 的扫描得出口诀 D ≈ 20N。但 20 是**经验点**，不是这组常数的推论：联立上式与 6ND = C 得 D/N ∝ C^((α−β)/(α+β)) ≈ C^0.097，随算力缓升而非常数——Chinchilla 算力（C≈5.9×10²³）处解出 N*≈32B、D*≈3.0T，D/N≈93，本课算例约 41。但 70B/1.4T 与 32B/3T 的 loss 只差约 0.006：面在 N–D 方向极平，这才是 20 好用的原因。同理「各约 ∝ C^0.5」也只是近似，严格解是 N∝C^0.45、D∝C^0.55。

修正对象是 GPT-3 时代：按 Kaplan 等 2020 年（N∝C^0.73、D∝C^0.27），算力几乎全堆参数，175B 只喂 300B token，1.7 token/参数；Chinchilla 用同等算力训 70B/1.4T，多数评测超过 4 倍参数量的对手。而 20 只是训练算力最优：要服务几十亿次请求时，参数每多一份推理就多付一份，业界于是故意超配数据——Llama 3 的 8B 用了约 15T token（约 1,875 token/参数），这叫推理最优。

### 数据配比、质量与重复 epoch

**定义与直觉**：配比是网页、代码、数学、多语等来源的 token 占比，是课程表不是营养成分表：代码教严格语法与长程依赖，数学教分步推理，网页给世界知识；某类涨，那类能力上来，也挤掉别的。

**机制**：Llama 3 公开配比约通用知识 50%、数学与推理 25%、代码 17%、多语 8%。质量侧最划算的是去重（重复文档让模型把「背下来」当「学会了」）与小模型打分的质量分类——「高质量」指干净、信息密度高，非「难」。

**重复训练**：Muennighoff 等 2023（最多 900B token、9B 模型）：数据受限、算力固定时，重复到约 4 个 epoch 与全新数据的 loss 差异很小，再多则边际收益按幂律衰减。由此外推（推断，非实测）：极少量顶级数据刷很多遍，多半不如把面铺开；低质数据稀释每 token 信息量，可粗略理解为「有效 D 变小」——类比，非替换。

### 训练稳定性

**定义**：loss 不炸、不出 NaN、持续下降。

- **warmup + cosine**：前几百到几千步把学习率从 0 升到峰值（Adam 二阶矩估计刚初始化不可靠，迈大步易踩坏区域），之后按余弦降到峰值约 10%。
- **梯度裁剪**：全局梯度范数超阈值就等比缩小，防坏 batch 踢飞参数。
- **bf16 而非 fp16**：fp16 只有 5 位指数、上限 ±65504，梯度小到 1e-8 就下溢成 0，得靠 loss scaling（放大 loss、反传后除回）绕开——这套机制本身就不稳。bf16 有 8 位指数、范围同 fp32，免 loss scaling，代价是尾数 7 位（约 2–3 位十进制精度）。
- **qk-layernorm**：attention logits = QKᵀ/√d 一大，softmax 就饱和成 one-hot、梯度趋 0 又极易爆；qk-norm 在算 logits 前归一化 Q、K 钉住尺度。
- **z-loss**：作用在**输出层**，不在 attention。lm_head 打出词表 logits，其 softmax 分母（配分函数 Z = Σexp）整体漂移时训练会抖；z-loss 加 1e-4·(log Z)² 即 1e-4·(logsumexp(logits))²，锚住 logits 的共同偏移。PaLM 之后常用，非人人标配。

**spike 成因**：坏数据 batch、学习率过高或 warmup 太短、fp16 溢出、attention 熵崩塌、MoE 路由塌缩。恢复顺序：跳过该 batch → 回滚 checkpoint 降学习率 → 查过滤与梯度裁剪是否生效。z-loss 与 qk-norm 是提前免疫，非事后吃药。

### loss 曲线怎么读

横轴步数、纵轴 loss，两条线：train 与 val。

- **双降且贴合**：正常；train 降而 val 平：数据太少或验证集太远。
- **val 先降后翘尾**：过拟合或分布偏移；前者加数据、正则、早停，后者先查数据。
- **早期尖刺后自愈**：warmup 期常见，几十步回到原轨迹就不管；连续多次且更高才动手。
- **卡在台阶上不动**：多半是 bug——mask 写错（padding 位参与 loss）、标签没移位（在学抄输入，loss 异常低）、学习率设 0、重复率 100%。

**能力上限就写在这条曲线上**：收尾 val loss 是 2.0 还是 2.4，决定 SFT、RL 能推多远；且有平台期，1.8 附近降 0.05 可能换不来一个点。

## 一步一步理解

**第 1 步 移位标签**。[BOS, 我, 爱, 学, 习] 取前 4 个当输入、后 4 个当标签：位置 1 见 BOS 猜「我」，位置 2 见 BOS 我 猜「爱」，余同。位置 2 看不到位置 3 的「学」（causal mask），输入却永远是真实的「我」——这就是 teacher forcing。

**第 2 步 算交叉熵**。V=8，4 个位置给正确 token 的概率是 0.8、0.5、0.4、0.25，对应 −ln p = 0.223、0.693、0.916、1.386。L = 3.218/4 = **0.805**，PPL = e^0.805 ≈ 2.24。校验：倒数的几何平均 = (1.25×2×2.5×4)^(1/4) = 25^(1/4) = 2.24。

**第 3 步 套 scaling law**。N=10⁹、D=2×10¹⁰（正好 20 token/参数）：A/N^α = 406.4/1148 = 0.354，B/D^β = 410.7/766 = 0.536，L = 1.69+0.354+0.536 = **2.580**，PPL ≈ 13.2。地板 1.69 占 65%——算力全在攻那 0.89。

**第 4 步 算力加在哪**。C = 6ND = 1.2×10²⁰，两条路都翻倍：N→2×10⁹ 得 L=2.506（ΔL=−0.074）；D→4×10¹⁰ 得 L=2.486（ΔL=−0.095）。同样的钱，加数据多降 28%。

那最优点在哪？令 αA N^−α = βB D^−β，代入 D = C/(6N) 取对数得 (α+β) ln N = ln(αA/βB) + β ln(C/6)，解出 N≈7×10⁸、D≈2.9×10¹⁰、D/N≈41，L=2.575——只比 20 配比好 0.005。用「翻倍谁划算」做离散比较会得到约 44N。

**第 5 步 多少钱、几次请求回本**。C=1.2×10²⁰ FLOP。8 卡 910B，单卡 bf16 峰值 400 TFLOPS（假设值）、MFU 40%：有效 1.28×10¹⁵ FLOP/s，时间 ≈ 9.4×10⁴ s ≈ 26 小时，按 32 元/小时 ≈ **833 元**。

回本要用**同一个模型**的单价。第 14 课那 0.00193 元是 7B 的价，这里只有 1B：低 batch、同请求形状时单请求成本大致 ∝ N，折算 ≈ **0.00028 元**，833 ÷ 0.00028 ≈ **300 万次**请求回本。换成 7B/1.4×10¹¹：C 涨 49 倍、训练约 4.1 万元、单请求回到 0.00193 元。小模型总成本永远更便宜，唯一短板是 loss 更高——真问题不是「哪个回本快」，而是「把省下的算力加到 D 上能不能补回 loss」，Chinchilla 说能。若同一模型 N→2N，训练费与单请求成本都翻倍、回本数不变；∝N 只在低 batch 成立，高 batch 成本 ∝ N ÷ 有效 FLOPS。

## 常见误区

**一、loss 越低模型越强。** loss 有拟合截距 E；判断「变强」要同数据、同分词器对比并配下游评测。反向的坑更危险——loss 异常低（如 0.3）几乎总是 bug：标签没右移，在抄输入。

**二、参数量是能力的唯一货币。** GPT-3 是 1.7 token/参数，Chinchilla 是 20，同等算力下后者更强。说一个模型「多大」得同时给 N 和 D。

**三、数据不够就把 epoch 刷上去。** 4 个 epoch 内与全新数据几乎等价，之后趋零、还加速过拟合；而多找 10% 新数据通常是线性的。

**四、训练不稳就是学习率太大。** 坏数据 batch 与 attention logits 爆炸同样常见；先看 spike 前那步样本，再动超参。

**五、z-loss 能治 attention 的数值问题。** z-loss 管输出层 logits 漂移，attention 那端归 qk-layernorm；混着记，出事会调错地方。

## 一句话总结

预训练就是把「给定前文、下一个 token 是什么」的平均负对数似然压到最低；scaling law 告诉你每多花一倍算力能降多少、该加在参数还是数据上（经验点 20 token/参数，loss 面平坦所以好用）；稳定性工程保证曲线不掉，曲线终点同时是能力上限与推理成本下限。

## 自测题

### 1. 正确 token 概率 0.5 时这一位的交叉熵是多少？4 个位置平均 0.8，困惑度多少？
<details><summary>查看答案</summary>−ln 0.5 ≈ 0.693。0.8 nats ⇒ PPL = e^0.8 ≈ 2.23，相当于在 2.23 个候选里乱猜（词表 4 万，缩窄约 1.8 万倍）。交叉熵吃对数平均：几何平均正确概率是 e^−0.8 ≈ 0.449。</details>

### 2. GPT-3 175B/300B 与 Chinchilla 70B/1.4T 各多少 token/参数？7B 按 20 配比要多少 token，显存花在哪？
<details><summary>查看答案</summary>3×10¹¹÷1.75×10¹¹ ≈ 1.71；1.4×10¹²÷7×10¹⁰ = 20，差约 12 倍。7B 要 1.4×10¹¹ token，fp16 约 280 GB，装不进 64 GB 卡，得流式读盘；显存花在权重、梯度、优化器状态（16 字节/参数 ≈ 112 GB）与激活。</details>

### 3. 第 3 万步 loss 从 2.1 跳到 3.6、两步后回落，第一件事查什么？
<details><summary>查看答案</summary>先查那条 batch：乱码、超长无分隔、重复段，以及梯度裁剪是否真生效（看 grad norm）；正常再怀疑学习率/warmup 与 attention 数值。只跳一次先记账，连续多次且更高才回滚。</details>

### 4. 为什么说 20 token/参数是经验点？loss 从 2.0 降到 1.9 算大变化吗？
<details><summary>查看答案</summary>它来自 Chinchilla 的实际训练点（70B/1.4T）与 Approach 1–2 的扫描；用同一组常数严格求解，D/N 随算力缓升（∝C^0.097）：本课算例约 41、Chinchilla 算力处约 93，但这两点的 loss 只差约 0.006——面极平，所以 20 是好起点而非物理常数。PPL 则 7.39→6.69（降 9.5%）：ΔL=0.1 处处降 9.5%，故盯 ΔL 并在平台期配评测。</details>

### 5. 现在训练为什么默认 bf16 而不是 fp16？
<details><summary>查看答案</summary>fp16 只有 5 位指数、上限 ±65504，梯度小到 1e-8 就下溢成 0，必须靠 loss scaling 绕开，而这套机制本身就引入不稳定。bf16 有 8 位指数，范围同 fp32，不需 loss scaling；代价是尾数 7 位（约 2–3 位十进制精度），训练够用。</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| 交叉熵 | 平均负对数似然，衡量预测与真实 token 的差距。 |
| teacher forcing | 每步喂真实前缀，一次前向并行算所有位置。 |
| 移位标签 | 输入 x[:T]、标签 x[1:T+1]，右移一位。 |
| causal mask | 下三角掩码，位置 t 只看 ≤t，并行不泄漏未来。 |
| 困惑度 | PPL = e^loss，等于在几个候选里均匀乱猜。 |
| scaling law | L(N,D)=E+A/N^α+B/D^β，N 与 D 的幂律。 |
| 不可约损失 | E，N,D→∞ 的外推截距，随数据与分词器变化。 |
| token/参数 | 训练 token 数 ÷ 参数量；Chinchilla ≈20，GPT-3 ≈1.7。 |
| epoch | 数据集过一遍；4 个 epoch 内与全新数据近似等价。 |
| 梯度裁剪 | 梯度范数超阈值就等比缩小，防 batch 踢飞。 |
| bf16 | 8 位指数、7 位尾数，范围同 fp32，免 loss scaling。 |
| z-loss | 惩罚输出层 logits 的 logsumexp（log 配分），防漂移。 |
| qk-layernorm | 算 attention logits 前归一化 Q、K，钉住尺度。 |
| loss spike | loss 突然上跳，多因坏 batch 或 attention 数值爆炸。 |

## 下一步

下一课「分布式并行：DP/TP/PP/ZeRO」回答：第 5 步那 1,280 TFLOPS 怎么跑出来，112 GB 的「参数+梯度+优化器状态」怎么切到 8 张 64 GB 卡上。预习：切参数、切梯度、切优化器状态各带来什么通信开销？Chinchilla 那 5.9×10²³ FLOP 为何在同样 8 卡上要 14 年？

## 来源

- Training Compute-Optimal Large Language Models（Chinchilla） — https://arxiv.org/abs/2203.15556
- Language Models are Few-Shot Learners（GPT-3） — https://arxiv.org/abs/2005.14165
- Scaling Laws for Neural Language Models（Kaplan 等） — https://arxiv.org/abs/2001.08361
- The Llama 3 Herd of Models（配比与超配数据） — https://arxiv.org/abs/2407.21783
- Scaling Data-Constrained Language Models（重复 epoch 的收益衰减） — https://arxiv.org/abs/2305.16264
- PaLM: Scaling Language Modeling with Pathways（z-loss） — https://arxiv.org/abs/2204.02311
- Query-Key Normalization for Transformers（qk-layernorm） — https://arxiv.org/abs/2010.04245
- Mixed Precision Training（fp16 的 loss scaling） — https://arxiv.org/abs/1710.03740
- PyTorch 文档：CrossEntropyLoss — https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- PyTorch 文档：自动混合精度（amp） — https://pytorch.org/docs/stable/amp.html
- nanoGPT（移位标签与训练循环的最小实现） — https://github.com/karpathy/nanoGPT
