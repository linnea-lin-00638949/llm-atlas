---
date: "2026-11-03"
title: "第 3 课 · Transformer 与 decoder-only 范式（上）"
domain: "模型演进"
module_id: "transformer-foundation"
module: "Transformer 与 decoder-only 范式"
status: "已学"
lesson_no: 3
goals: [
  "解释 attention 在做什么，以及它如何解决 RNN 的串行与长距离短板",
  "写出 scaled dot-product attention 公式并逐项说明每一项，能手算一个注意力输出",
  "说清多头注意力与残差连接各自解决什么问题，并正确读出注意力权重矩阵的行/列语义"
]
tags: [
  "transformer",
  "attention",
  "QKV",
  "multi-head attention",
  "残差连接",
  "模型演进"
]
---

# 第 3 课 · Transformer 与 decoder-only 范式（上）

> 学完今天，你能：
> 1. 解释 attention 在做什么，以及它如何解决 RNN 的串行与长距离短板；
> 2. 写出 scaled dot-product attention 公式并逐项说明每一项，能手算一个注意力输出；
> 3. 说清多头注意力与残差连接各自解决什么问题，并正确读出注意力权重矩阵的行/列语义。

## 课前自测

先回忆前两课（nn-basics-1/2）的工具：张量与矩阵乘法；线性层 y = xW + b；softmax 把实数向量变成「非负、和为 1」的概率分布；损失函数与梯度下降。自测 4 题：

1. RNN 处理长度为 n 的序列为什么必须走 n 步？这会带来什么问题？
2. softmax([2, 0, 1]) 的三个输出有什么共同性质？（不必算出精确值）
3. 两个向量点积越大，几何上说明什么？
4. 若某层输出定义为 F(x) + x，而 F 完全没学到东西（F(x) = 0），此时层的输出是什么？

（本集覆盖模块目标 1–2：attention 在做什么 / 公式逐项；目标 3–4 由下集覆盖。）

## 为什么需要它

前两课的网络吃「固定长度的一条输入」。但句子长度不固定、词与词有依赖：「他拿起___喝了」里，空格处要知道前文指的是「水杯」。序列模型要建模这种跨位置依赖。

最朴素的做法是 RNN（循环神经网络）：按顺序读入 token（文本切分出的最小处理单元，词或子词），每一步把「当前 token + 上一步的隐藏状态（hidden state，压缩历史的向量）」算成新状态。像逐字读信，历史全压进一个不断更新的记忆。两个致命短板：

- 串行：第 t 步依赖第 t-1 步，n 步只能一步步算，长文本又慢又无法并行；
- 长距离：第 1 个词的信息传到第 1000 个词要经过 999 次「压缩—传递」，容易丢；训练时还伴随梯度消失（反向传播时靠前位置几乎收不到梯度信号，参数学不动）。

Transformer（Vaswani 等，2017）的答案：放弃一步步传，让每个位置直接和所有位置比一次「相关程度」，一步到位。直觉类比：找资料不是逐页读完一本书，而是拿关键词查索引、直接翻到相关页——这就是 attention（注意力）。attention 任意两位置的计算路径长度是 1，没有中间传递损耗；而且全是矩阵运算，天生适合并行硬件。

工程上这直接决定推理形态：attention 是推理中最重的计算之一——vLLM 怎么调度它，下一模块再讲。

## 核心概念

### attention（注意力）

直觉：在图书馆找资料。你手里有「我要找什么」（query），每本书贴着「标签」（key）并有「正文」（value）。你拿 query 和每本书的标签比对，按匹配程度决定读多少。

定义：给定一个 query 与一组 key–value 对，输出是全部 value 的加权和；第 i 个 value 的权重由「query 与第 i 个 key 的相似度」经 softmax 归一得到。

机制：序列里每个位置同时扮演三个角色，用三套参数矩阵投影出来（下一节），再按公式批量计算——本质是「可学习的加权检索」。

### Q / K / V

直觉：一句话里，每个 token 同时有三种身份：「我想找什么信息」（Query）、「别人拿我当什么检索标签」（Key）、「我真正携带的内容」（Value）。

定义：输入 X 形状 (n, d_model)：n 个 token、每个 d_model 维。Q = XW_Q、K = XW_K、V = XW_V，W_Q/W_K/W_V 是训练中学习的投影矩阵，把每个 token 投成 d_k 维（Q、K）或 d_v 维（V）。注意：Q、K、V 不是三份不同数据，是同一份数据的三种投影。

机制：每个 token 用自己的 Q 和所有 token 的 K 算相似度，再按权重从所有 token 的 V 里抄信息。W 在训练中学会「按什么标准相关、抄什么内容」。

### scaled dot-product attention（缩放点积注意力）

直觉：「打分、归一、加权求和」三步，外加一个防数值爆炸的缩放。

定义与公式：

Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

逐项拆解（记 n 个 token）：

1. QKᵀ：形状 (n, n)，第 i 行第 j 列 = 第 i 个 query 与第 j 个 key 的点积。点积几何意义：两向量越同向值越大，这里当「相关分数」用。分数还没归一，可以是任意实数。
2. ÷√d_k：缩放。论文论证：若 q、k 各维独立且均值 0、方差 1，则点积的方差是 d_k——d_k 越大分数越大，softmax 会被推入「饱和区」，该处梯度几乎为 0。除以 √d_k 把方差拉回 1，训练才稳定。
3. softmax：按行把分数变成非负、和为 1 的权重（nn-basics-2 学过）。
4. ×V：每个位置的输出 = 所有 token 的 V 按权重加权平均，形状 (n, d_v)。输出向量「汇集」了全序列的信息。

（在只按从左到右逐 token 生成文本的 decoder（解码器）模型中，还会加因果 mask——把「未来」token 的分数置为负无穷再 softmax，使权重为 0；decoder 与 decoder-only 是什么，下集展开。）

### 多头注意力（multi-head attention）

直觉：只找一个「专家」看句子有盲区——有的头学会盯前一个词，有的头看指代关系。多找几个专家从不同侧面同时看。

定义：每头用一套小投影矩阵把 Q/K/V 投成 d_model/h 维，独立做 scaled dot-product attention（等价于把一个大头切成 h 个小头），h 个输出拼接后经 W^O 投影：

head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
MultiHead(Q, K, V) = Concat(head_1, …, head_h) W^O

机制：总参数量和单头同阶，但模型能在 h 个子空间里学 h 种不同的「相关方式」。论文可视化观察到不同头确实呈现不同模式（有的盯相邻位置、有的看远距离指代）。主流模型都沿用（如 LLaMA-7B 有 32 个头）。

### 残差连接（residual connection）

直觉：高速公路的应急车道——主路（新子层）没修好，车流仍沿原路通过。

定义：子层输出 = F(x) + x，其中 x 是本层输入，F 是注意力或前馈网络。

机制：反向传播时，恒等路径 x 的梯度恒为 1，不随层数加深衰减，深层网络因此训得动（He 等 2015 年在 ResNet 中提出，Transformer 每个子层都套用）。Transformer 的完整子层是「F(x) + x → 归一化」，归一化是 LayerNorm（层归一化），下集讲它的省算力变体均方根归一化（RMSNorm）。

## 一步一步理解

用 3 个 token「我 / 爱 / 昇腾」，取 d_model = 4、d_k = d_v = 2、单头，手算一遍。token 先经 embedding（把离散 token 查表变成稠密向量的层）得到输入 X。设 X（形状 (3, 4)）：

```
我：  [1, 0, 1, 0]
爱：  [0, 1, 0, 1]
昇腾：[1, 1, 0, 0]
```

第 1 步：投影。为便于手算取 W_Q = W_K = W_V = [[1,0],[0,1],[1,0],[0,1]]（真实模型三者不同且可学习）。按「X 第 i 行与 W 第 j 列点积」得：

```
Q = K = V：
我：  [2, 0]
爱：  [0, 2]
昇腾：[1, 1]
```

第 2 步：分数矩阵 QKᵀ（第 i 行第 j 列 = Q_i · K_j）：

```
[[4, 0, 2],
 [0, 4, 2],
 [2, 2, 2]]
```

第 3 步：除以 √d_k = √2 ≈ 1.414：

```
[[2.828, 0,     1.414],
 [0,     2.828, 1.414],
 [1.414, 1.414, 1.414]]
```

第 4 步：逐行 softmax（e^2.828 ≈ 16.9，e^1.414 ≈ 4.11，e^0 = 1）：

```
第 1 行：[16.9, 1, 4.11] / 22.0   ≈ [0.768, 0.045, 0.187]
第 2 行：[1, 16.9, 4.11] / 22.0   ≈ [0.045, 0.768, 0.187]
第 3 行：[4.11, 4.11, 4.11] / 12.3 ≈ [0.333, 0.333, 0.333]
```

第 5 步：输出 = 权重 × V：

```
「我」：  0.768·[2,0] + 0.045·[0,2] + 0.187·[1,1] ≈ [1.72, 0.28]
「爱」：  0.045·[2,0] + 0.768·[0,2] + 0.187·[1,1] ≈ [0.28, 1.72]
「昇腾」：0.333·[2,0] + 0.333·[0,2] + 0.333·[1,1] = [1.0, 1.0]
```

怎么读结果：「我」的输出 76.8% 来自自己（第 1 行权重大头压在自己身上）；「昇腾」所在第 3 行的权重对三个 token 都是 1/3——它从每个位置等量取信息，所以它的输出是三个 V 的平均，和谁都没有突出的「专属相关」。记住行/列语义：第 i 行 = 第 i 个 token 的 query 把注意力分给各 key 的权重；第 j 列 = 第 j 个 token 作为 key 被各 query 读取的权重——例如第 3 列 [0.187, 0.187, 0.333] 表示「昇腾」作为 key 被三个 query 读取的比例，并不均匀。真实模型的权重由训练出的 W 决定，但流程就是这 5 步；随后做残差（输出 + 原输入 X）再归一化。

工程视角（落到昇腾 + vLLM）：vLLM 把 Q、K、V 三个线性层合并成一次大矩阵乘法（GEMM，NPU/GPU 上最核心的算子），attention 部分则优先用融合算子——中间的 (n, n) 分数矩阵不写回显存，省带宽；昇腾后端（vllm-ascend 开源项目）通过 CANN（Compute Architecture for Neural Networks，昇腾计算架构）的融合算子实现同样效果。记住：上面 5 步是「数学定义」，工程实现会重排、融合这些计算；下一模块再展开 vLLM 怎么调度。

## 常见误区

1. 「attention 像人一样读懂了语义」。错。机制只是「算相似度 → 加权平均」，是可学习的检索，不是理解；有意义的注意力来自 W_Q/W_K/W_V 在海量数据上的训练。把机制当智能，是新手最常见的错觉。
2. 「Q、K、V 是三种不同的输入数据」。错。三者都来自同一份 X，只是乘了三套参数矩阵；「谁查、谁被查、取什么内容」这些角色完全由训练决定，不是先验语义。
3. 「÷√d_k 只是调温度的玄学，去掉也没事」。错。不缩放时点积方差随 d_k 线性变大，softmax 输出趋近 one-hot（只有一个位置权重接近 1、其余接近 0），该处梯度趋近 0，训练不稳。这是论文的数值稳定性设计，不是可选项。
4. 「注意力权重大 = 两个词语义相似」。不全对。权重表示「当前这一步该抄谁的信息」，受位置、层数、头的功能影响，还随上下文变化；注意力权重能否当「解释」用学界仍有争论，别直接当语义相似度读。

## 一句话总结

Attention 用「QK 点积打分 → softmax 归一 → 对 V 加权平均」取代 RNN 的串行传递，让每个 token 一步直达全体 token；√d_k 缩放保数值稳定，多头提供多个并行视角，残差连接让深层网络训得动。

## 自测题

### 1. attention 相对 RNN 解决了哪两个核心问题？
<details><summary>查看答案</summary>串行：RNN 第 t 步依赖第 t-1 步，n 个 token 必须走 n 步、无法并行；attention 用矩阵运算一次算完所有两两相关，适合并行硬件。长距离：RNN 第 1 个词传到第 1000 个词要经 999 次压缩传递，易丢信息、伴随梯度消失；attention 任意两位置路径长度是 1，没有中间损耗。</details>

### 2. 写出 scaled dot-product attention 的公式，并说明 √d_k 缩放的作用。
<details><summary>查看答案</summary>Attention(Q, K, V) = softmax(QKᵀ/√d_k)V。若 q、k 各维独立且均值 0、方差 1，点积的方差是 d_k——d_k 越大分数越大，softmax 被推入饱和区、该处梯度几乎为 0；除以 √d_k 把方差拉回 1，训练才稳定。</details>

### 3. 注意力权重矩阵某一行是 [0.1, 0.8, 0.1]（序列 A/B/C），这一行表示什么？对应 token 的输出向量怎么算？
<details><summary>查看答案</summary>按行读：第 i 行 = 第 i 个 token 的 query 分给各 key 的权重。这一行表示「我这个 token」把 80% 注意力放在 B 上。它的输出 = 0.1·V_A + 0.8·V_B + 0.1·V_C。列语义不同：第 j 列是第 j 个 token 作为 key 被各 query 读取的比例，两者不要混。</details>

### 4. Q、K、V 是三份不同数据吗？多头相比单头参数量为什么没有数量级差异？
<details><summary>查看答案</summary>不是。Q = XW_Q、K = XW_K、V = XW_V，三者都来自同一份 X 的三套投影，角色由训练决定。多头把每头维度缩到 d_model/h，h 个头拼接后总参数量与单头同阶，却能在 h 个子空间学 h 种「相关方式」。</details>

### 5. 残差连接 F(x) + x 为什么能让深层网络训得动？
<details><summary>查看答案</summary>反向传播时恒等路径 x 的梯度恒为 1，不随层数衰减，梯度能一路传回浅层；即使 F 没学好（F(x) = 0），层输出仍等于输入，网络不会立刻崩。Transformer 每个子层都套用「F(x) + x → 归一化」。</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| token | 文本切分出的最小处理单元（词或子词），先经 embedding 变成向量 |
| RNN（循环神经网络） | 按顺序读序列、用隐藏状态逐字传历史的网络；串行、长距离易丢信息 |
| attention（注意力） | 用 query 与各 key 的相似度做权重、对 values 加权求和，让每个位置一步直达全部位置 |
| Q / K / V | 同一输入 X 分别乘三套可学习矩阵得到的 Query / Key / Value 投影 |
| 点积 | 两向量对应位置相乘再求和，几何上度量「同向程度」 |
| scaled dot-product attention | softmax(QKᵀ/√d_k)V：点积打分、√d_k 缩放、softmax 归一、加权求和 |
| softmax | 把实数向量变成非负、和为 1 的概率分布（nn-basics-2 已学） |
| 多头注意力 | 每头用一套小投影矩阵把 Q/K/V 投成 d_model/h 维独立做 attention，h 个输出拼接后经 W^O 投影，学多种相关模式 |
| 残差连接 | 子层输出 F(x) + x，恒等捷径保住信息与梯度，深层可训 |
| decoder（解码器） | 按从左到右逐 token 生成文本的模型结构；decoder-only 是什么，下集展开 |
| d_model / d_k / d_v | 模型隐藏维度 / Q、K 的维度 / V 的维度 |
| GEMM | General Matrix Multiply，通用矩阵乘，NPU/GPU 上最核心的算子 |
| embedding | 把离散 token 查表变成稠密向量的层（nn-basics 已学） |

## 下一步

明天（第 3 课下集）先补齐三件事：位置信息从哪来（旋转位置编码 RoPE）、归一化怎么做省（均方根归一化 RMSNorm）、前馈层用什么激活（SwiGLU 激活函数），并解释分组查询注意力（GQA）与「为什么主流都做 decoder-only」；之后进入下一模块「推理服务基础：prefill/decode 与 KV cache」。

## 来源

- Attention Is All You Need — https://arxiv.org/abs/1706.03762
- The Illustrated Transformer — https://jalammar.github.io/illustrated-transformer/
- LLaMA: Open and Efficient Foundation Language Models — https://arxiv.org/abs/2302.13971
- Deep Residual Learning for Image Recognition — https://arxiv.org/abs/1512.03385
- Dive into Deep Learning，第 11 章 Attention Mechanisms and Transformers — https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html
- vLLM 官方文档 — https://docs.vllm.ai/en/stable/
- vLLM Ascend 后端文档 — https://docs.vllm.ai/projects/ascend/en/latest/
