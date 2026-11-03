---
date: "2026-11-03"
title: "第 4 课 · Transformer 与 decoder-only 范式（下）"
domain: "模型演进"
module_id: "transformer-foundation"
module: "Transformer 与 decoder-only 范式"
status: "已学"
lesson_no: 4
goals: ["能解释 attention 在做什么（对比 RNN），写出缩放点积注意力公式并说明每一项", "能说清 RoPE/RMSNorm/SwiGLU/GQA 各自解决什么问题", "能理解主流为何收敛到 decoder-only，并会估算 KV cache 大小"]
tags: ["transformer", "decoder-only", "RoPE", "GQA", "KV cache", "LLM 架构"]
---

# 第 4 课 · Transformer 与 decoder-only 范式（下）

> 学完今天，你能：
> 1. 解释 attention 在做什么（对比 RNN），写出缩放点积注意力公式并说明每一项
> 2. 说清 RoPE / RMSNorm / SwiGLU / GQA 各自解决什么问题
> 3. 理解主流为何收敛到 decoder-only，并会估算 KV cache 大小

## 课前自测

1. 默写缩放点积注意力公式，并说明分母 √d 的作用。
2. token 的顺序信息（「A 打 B」≠「B 打 A」）要怎样进入模型？说一种你知道的办法。
3. 生成第 100 个 token 时，前 99 步算出的 K、V 不缓存会怎样？缓存大概多大？
4. GQA 的 g 取 1、取 n_heads 时各是什么？它为什么能省显存？

## 为什么需要它

上集搭好了注意力骨架（Q/K/V、softmax 缩放、多头、残差），但要高效、稳定、低成本地训出并部署 LLM（大语言模型），还差四块：位置信息、深层训练稳定、前馈表达、KV 缓存成本。RoPE、RMSNorm、SwiGLU、GQA 各补一块，最后讨论「主流为何收敛到 decoder-only」。注意：这四项是效率/稳定性/质量改进，不是硬性前提——GPT-2/GPT-3 用学习式位置编码 + LayerNorm + GELU 同样训出了可用模型。

对比 RNN 看 attention 在做什么：RNN 靠隐藏状态逐步传递，长距离易衰减、只能串行；attention 用 Q·K 直接算任意两位置的关联，路径短、可并行。Q 是「我想找什么」、K 是「我有什么」、V 是「实际取值」，点积大即相关，softmax 后按权重从 V 取值。你在昇腾 NPU（华为 AI 加速芯片）上跑的每个 vLLM（开源大模型推理引擎）服务，内部就是这些组件。

## 核心概念

### RoPE（旋转位置编码）
定义：按 token 位置 m，把每头 Q、K 的相邻两维 (x₂ᵢ, x₂ᵢ₊₁) 当平面坐标旋转 m·θᵢ，θᵢ=10000^(−2i/d)（i 为维度对编号，d 为头维度）：x′₂ᵢ=x₂ᵢcos(mθᵢ)−x₂ᵢ₊₁sin(mθᵢ)；x′₂ᵢ₊₁=x₂ᵢsin(mθᵢ)+x₂ᵢ₊₁cos(mθᵢ)。
直觉：不往 embedding（token 映射成的连续向量）里「加」位置，而是让 Q、K 随位置「转」一个角度——长度不变（内容不变），方向变。
机制：把旋转前每对 (x₂ᵢ, x₂ᵢ₊₁) 写成极坐标 (r, α)（幅长、幅角），旋转后 q_m·k_n=Σᵢ r_q·r_k·cos(αᵢ+(m−n)θᵢ)，只依赖相对位置 m−n，绝对位置被消掉：「第 2 词看第 1 词」与「第 100 词看第 99 词」用同一套角度关系，也有利于长度外推。相比往 embedding 上加正弦值的绝对编码，RoPE 让分数显式携带相对距离。例子：下文 worked example 中 q₂ 转 90° 后与 k₀ 夹角 90°、点积为 0。

### RMSNorm（均方根归一化）
定义：LayerNorm（层归一化）是减均值、除标准差，再乘 γ 加 β。RMSNorm 只留后半：除 RMS(x)=√(mean(x²)+ε) 再乘 γ，不减均值、无 β；y=x/RMS(x)·γ（逐元素，ε 是防除零小常数）。
直觉：训练不稳定主要来自「尺度失控」；把每个向量尺度压回 1（二阶矩）就解决大半，「减均值」（一阶矩）贡献小，可省。
机制：少一次均值统计与减法、参数减半（无 β），更快、更省显存，稳定性与 LayerNorm 相当，LLaMA 全系都用它。例子：t0=[2,2,2,2] 的 RMS 是 2，归一后变 [1,1,1,1]。

### SwiGLU（Swish 门控前馈）
定义：传统 FFN（前馈网络）是 x→W₂·max(0, W₁x)。SwiGLU 用两个并行投影加逐元素乘 ⊙：SwiGLU(x)=(Swish(x·W_gate)⊙(x·W_up))·W_down，Swish(z)=z·σ(z)，σ 是 sigmoid（把实数压到 0~1 的 S 形函数），整体是 GLU（门控线性单元）的一种。
直觉：「up 路」给候选值，「gate 路」给 0~1 软开关，决定每个候选通过多少；ReLU 是「负数直接掐掉」的硬开关，Swish 平滑。
机制：负值不再硬置 0，保留小梯度、表达更强；代价是多一个权重矩阵（共 3 个），LLaMA 1/2 把中间维度从 4d 缩到约 8/3·d（取整到 256 的倍数），参数量与 ReLU 版持平（Llama 3 系比例不同）。例子：gate=[1,2] 经 Swish 变 ≈[0.73, 1.76]。

### GQA（分组查询注意力）
定义：MHA（多头注意力）每头各一套 K、V。GQA 把 n_heads 个 Q 头分成 g 组，每组共享一套 K、V：g=n_heads 即 MHA，g=1 即 MQA（全部共享）。LLaMA2-70B：64 个 Q 头、8 套 KV。
直觉：书（K/V）只放 g 个书架，每个读者（Q 头）在本组书架里找；组数不太少时找书质量几乎不受影响（g=1 的 MQA 有可见损失），书的总量却从 64 套减到 8 套。
机制：decode（逐个生成 token）阶段每步都读整个 KV cache（缓存的历史 K/V，免重算），且访存受限（卡在内存带宽而非算力）。KV cache 大小≈2×层数×n_kv_heads×head_dim×序列长×字节数×batch：2 是 K、V 各一份；head_dim 是每头维度；字节数是每元素字节数（fp16/bf16=2、fp32=4）；batch 是同时服务的请求数。例：64 层×8 套 KV×头维 128×序列 1024×2 字节×batch 1≈256MB。GQA 把 n_kv_heads 降到 g，省下的显存可换更大 batch。落到昇腾：vLLM 里组数写在模型配置 num_key_value_heads，由权重决定、不能运行时改；规划 NPU 内存与 batch 时按它而非 n_heads 估算。

### decoder-only
定义：只用 Transformer 的解码器堆叠：每层「因果注意力+FFN」，位置 i 只能看 ≤i（因果掩码），逐 token 自回归（把已生成输出当作下一步输入）预测下一个词。对比：encoder-decoder（如 T5）另有双向 encoder；encoder-only（如 BERT）只双向理解、不生成。
直觉：语言建模就是「根据上文猜下一个词」，恰好就是因果结构本身；encoder 的「看全文」对生成没有直接收益，砍掉还省参数。
机制：一个统一目标（next-token 预测）训练整张网，算力全服务生成；因果注意力与 KV cache 天然契合（每步只追加一个新 token 的 K/V）。GPT、LLaMA 用越来越大的规模证明这条路有效。注意：这是经验收敛而非理论必然，encoder 在理解类任务仍有用。例子：输入「今天天气」，模型按因果序先猜「很」再猜「好」。

## 一步一步理解

玩具模型：3 个 token（位置 0/1/2），d_model=4，2 个头（head_dim=2，每头 1 个旋转对），θ=π/4，d_ff=2（FFN 中间宽度）。为手算方便取 γ=1、W_Q=W_K=W_V=单位阵 I。输入 X（每行一个 token）：t0:[2,2,2,2]；t1:[0,2,2,0]；t2:[0,0,2,2]。

① RMSNorm（γ=1）：t0:√(16/4)=2→[1,1,1,1]；t1:√(8/4)=√2→[0,√2,√2,0]；t2 同 t1→[0,0,√2,√2]。记作 X̂。
② QKV：q=k=v=X̂；头 1 取维度 (0,1)，头 2 取 (2,3)。
③ RoPE（只旋转 Q、K；v 保持 X̂ 不旋转）：θ=π/4，位置 0/1/2 转 0°/45°/90°。头 2 的 k：k₀=[1,1]；k₁=R(45°)[√2,0]=[1,1]；k₂=R(90°)[√2,√2]=[−√2,√2]（q 同 k；v 不旋转：v₀=[1,1]、v₁=[√2,0]、v₂=[√2,√2]）。
④ 注意力（头 2 的 token 2，缩放 1/√2，因果掩码只看 ≤2）：s₀=q₂·k₀/√2=0；s₁=0；s₂=q₂·k₂/√2=4/√2≈2.83。softmax（把分数变成非负、和为 1 的权重）：e⁰=1、e⁰=1、e^2.83≈16.9 → 权重≈[0.05,0.05,0.89]（精确 0.053/0.053/0.894，用它算）：o₂=0.053·[1,1]+0.053·[√2,0]+0.894·[√2,√2]≈[1.39,1.32]。几何解释：q₂ 方向 135°（45° 再转 90°），与 k₀、k₁（45° 方向）夹角 90°→点积 0；与自己同向→点积 |q₂||k₂|=4。位置只以「角度差」进入分数——这就是 RoPE。其余行同法，注意力输出 O（每行=头1‖头2）：

| token | 头 1 | 头 2 |
| --- | --- | --- |
| t0 | [1,1] | [1,1] |
| t1 | [0.20,1.33] | [1.21,0.50] |
| t2 | [0.33,0.80] | [1.39,1.32] |

⑤ 残差（y=x+f(x)，梯度直通）再二次 RMSNorm：A=X̂+O：t0:[2,2,2,2]；t1:[0.20,2.74,2.62,0.50]；t2:[0.33,0.80,2.81,2.73]；再归一（γ=1）：t0→[1,1,1,1]；t1≈[0.10,1.43,1.37,0.26]；t2≈[0.16,0.40,1.40,1.36]。
⑥ SwiGLU（t0 的 x=[1,1,1,1]）：W_gate 两列=[1,0,0,0]ᵀ、[0,0,0,2]ᵀ → gate=[1,2]，Swish(1)≈0.73、Swish(2)≈1.76；W_up 两列=[2,0,0,0]ᵀ、[0,0,0,1]ᵀ → up=[2,1]；门控相乘 y=[0.73×2, 1.76×1]=[1.46,1.76]；W_down 两行=[1,0,0,0]、[0,0,1,0] → FFN 输出=1.46·[1,0,0,0]+1.76·[0,0,1,0]=[1.46,0,1.76,0]。
⑦ 二次残差：t0 最终输出=[1,1,1,1]+[1.46,0,1.76,0]=[2.46,1,2.76,1]。这就是一层 decoder 的完整数据流：RMSNorm→因果注意力(+RoPE)→残差→RMSNorm→SwiGLU→残差；输出送下一层，最后一层后接输出头映射到词表。

## 常见误区

1. 「RoPE 和正弦位置编码一样，只是往 embedding 上加随位置变化的数。」错。RoPE 不改 embedding，而是把 Q、K 旋转 mθ 角；它编码相对位置——q_m·k_n 只依赖 m−n。
2. 「GQA 把 8 个头合并成 1 个，输出变少。」错。Q 头数不变，输出仍是 n_heads 个头拼接；共享的只是 K/V 投影与缓存，改变的是内存与带宽成本。
3. 「RMSNorm 就是 LayerNorm 删掉 β。」不准确。它还删了「减均值」：只压尺度、不中心化到零均值，参数更少、更快，但数学上是不同运算。

## 一句话总结

现代 decoder-only LLM = 因果注意力（QKV·缩放·多头）+ RoPE 相对位置 + RMSNorm/SwiGLU 高效稳定组件 + GQA 省 KV 缓存，一环扣一环解决「位置、稳定、质量、成本」四个问题。

## 自测题

### 1. 默写缩放点积注意力公式，并说明分母 √d 的作用

<details><summary>查看答案</summary>

Attention(Q,K,V)=softmax(QKᵀ/√d)V。逐项：Q 查询（我想找什么）、K 键（我有什么），QKᵀ 逐对算相关性；d 是 Q、K 的维度，维度越大内积越大，÷√d 防止 softmax 饱和（权重被最大项垄断）；softmax 把分数变成非负、和为 1 的权重；再加权求和 V（值）。对比 RNN：RNN 靠隐藏状态逐步传递、长距离易衰减、只能串行；attention 用 Q·K 直接算任意两位置关联，路径短、可并行。

</details>

### 2. token 的顺序信息（「A 打 B」≠「B 打 A」）要怎样进入模型？

<details><summary>查看答案</summary>

三种办法：a) 绝对位置编码——把随位置变化的正弦值或可学习向量加到 embedding；b) 学习式位置向量（GPT-2/GPT-3 用）；c) 本课的 RoPE——不「加」，而是按位置把 Q、K 旋转 mθ 角。旋转后 q_m·k_n=Σᵢ r_q·r_k·cos(αᵢ+(m−n)θᵢ)，只依赖相对位置 m−n：绝对位置被消掉，也有利于长度外推。

</details>

### 3. 生成第 100 个 token 时，前 99 步的 K、V 不缓存会怎样？缓存多大？

<details><summary>查看答案</summary>

不缓存则每步从头重算全部历史 K、V，计算量随序列长平方增长。KV cache≈2×层数×n_kv_heads×head_dim×序列长×字节数×batch：2 是 K、V 各一份；head_dim 是每头维度；字节数按 dtype（fp16/bf16=2、fp32=4）；batch 是同时服务的请求数。例：64 层×8 套 KV×头维 128×序列 1024×2 字节×batch 1≈256MB。序列变长、batch 变大时线性增长，是显存规划的关键量。

</details>

### 4. GQA 的 g=1 与 g=n_heads 各是什么？它为什么能省显存？

<details><summary>查看答案</summary>

g=n_heads 是标准 MHA（每头一套 K/V）；g=1 是 MQA（全部共享一套），GQA 取中间（LLaMA2-70B：64 个 Q 头、8 套 KV）。decode 每步读整个 KV cache 且访存受限，cache 正比于 n_kv_heads；分组共享把显存省到 n_heads/g 分之一，可换更大 batch。代价：组数不太少时质量几乎不受影响，g=1 的 MQA 有可见损失。vLLM 中组数写死在配置 num_key_value_heads，运行时不能改。

</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| Q/K/V | 查询/键/值：Q·K 算相关性，softmax 后加权求和 V |
| 缩放点积注意力 | softmax(QKᵀ/√d)V；÷√d 防内积过大、权重被最大项垄断 |
| 多头注意力 | d_model 切 h 份并行算注意力再拼接，不同子空间学不同关系 |
| 残差连接 | y=x+f(x)：子层输出加回输入，梯度直通 |
| RoPE | 按位置把 Q/K 旋转 mθ 角，分数只依赖相对位置 |
| RMSNorm | 只除 RMS(x) 再乘 γ，不中心化，替代 LayerNorm |
| SwiGLU | Swish(z)=z·σ(z) 作门控的 GLU 前馈，替代 ReLU 的平滑非线性 |
| GQA | 多个 Q 头分组合享 K/V，KV cache 缩小 n_heads/g 倍 |
| KV cache | 缓存历史 K/V 免重算；大小估算与符号说明见 GQA 节 |
| decoder-only | 只用带因果掩码的解码器堆叠，逐 token 自回归预测下一个词 |

## 下一步

下节课「推理服务基础：prefill/decode 与 KV cache」展开：prefill（一次性处理输入提示）与 decode 为什么一个算力受限、一个访存受限，vLLM 的 PagedAttention（按页管理 KV cache）怎么省碎片，以及这些如何映射到昇腾 NPU 上的 vLLM 参数。

## 来源

- 《Attention Is All You Need》 — https://arxiv.org/abs/1706.03762
- 《RoFormer: Enhanced Transformer with Rotary Position Embedding》 — https://arxiv.org/abs/2104.09864
- 《Root Mean Square Layer Normalization》 — https://arxiv.org/abs/1910.07467
- 《GLU Variants Improve Transformer》 — https://arxiv.org/abs/2002.05202
- 《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》 — https://arxiv.org/abs/2305.13245
- 《LLaMA: Open and Efficient Foundation Language Models》 — https://arxiv.org/abs/2302.13971
- 《Llama 2: Open Foundation and Fine-Tuned Chat Models》 — https://arxiv.org/abs/2307.09288
- vLLM 官方文档 — https://docs.vllm.ai/en/latest/
