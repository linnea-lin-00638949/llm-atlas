---
date: "2026-11-18"
title: "第 19 课 · 从「续写」到「听话」：SFT 的 loss mask 与 RLHF 四阶段（上）"
domain: "训练"
module_id: "post-training"
module: "后训练：SFT 与强化学习"
status: "进行中"
lesson_no: 19
goals: ["学完今天，你能说清预训练与后训练在目标函数、数据规模和算力占比上的真实差别，并算出「SFT 只花预训练百分之几的算力，却决定了模型听不听话」这笔账。", "学完今天，你能手工写出 SFT 的 loss——只在回答 token 上算、prompt 位被 mask 掉、多轮样本怎么拼、packing 后边界怎么靠 attention mask 与 position id 隔开，并说出某处 mask 写错时模型会表现出什么症状。", "学完今天，你能把 RLHF 四个阶段（SFT → 偏好数据 → 奖励模型 → PPO）的输入输出、每阶段在优化什么、以及 PPO 阶段为什么要同时常驻策略/参考/奖励/评论四个模型讲清楚，并粗算 7B 的显存账。"]
tags: ["SFT", "loss mask", "packing", "RLHF", "PPO", "后训练"]
---

# 第 19 课 · 从「续写」到「听话」：SFT 的 loss mask 与 RLHF 四阶段（上）

> 学完今天，你能：
> 1. 说清预训练与后训练在目标函数、数据规模、算力占比上的真实差别，并算出「SFT 只花预训练不到 1% 的算力，却决定了模型听不听话」这笔账。
> 2. 手工写出 SFT 的 loss——只在回答 token 上算、prompt 位 mask 成 -100、多轮样本怎么拼、packing 后边界怎么靠 attention mask 与 position id 隔开，并说出某处 mask 写错时模型会有什么症状。
> 3. 讲清 RLHF 四阶段（SFT → 偏好数据 → 奖励模型 → PPO）各交换什么，PPO 为什么要同时常驻四个模型，并粗算 7B 的显存账。

## 课前自测

1. 交叉熵的 label 填 `-100` 是什么意思？为什么 prompt 段要填它？
2. 一条 2000 token 的三轮对话，回答只占 1200 token。忘了 mask，模型在这条样本上被训练去预测哪些位置？上线后会看到什么？
3. 平均 2000 token 的样本 pad 到 4096，多少算力花在 pad 上？packing 之后呢？
4. 奖励模型输出一个标量，它为什么不能替代人？
5. 7B 全参 SFT 在 8 卡上跑得动，换成 PPO 为什么可能跑不动？

答案落点：1、2 → 核心概念一；3 → 一步一步第 3 步；4 → 核心概念四；5 → 核心概念五。

## 为什么需要它

上一课把 ring all-reduce 的账算到了字节：14 GB 梯度、8 卡每卡 24.5 GB、机内 62.5 ms。那套账是给预训练算的——模型从零学「下一个 token 是什么」，喂几万亿 token 的网页。

今天反过来：模型已经会说话，你要教它「听话」——按指令、按格式、按人的偏好回答。数据从 1e13 token 掉到 1e9，事故率反而更高，因为每一步都在动一个已经很脆的东西。

SFT 的 loss mask 是 90% 的 SFT 事故现场；RLHF 四阶段则决定你该买几张卡、每张卡上放什么。这一课把这两笔账算清，DPO/GRPO 留到（下）。

## 核心概念

### 一、SFT 没换目标函数，只换了数据分布与 loss 作用域

**定义**：SFT（Supervised Fine-Tuning，监督微调，也叫指令微调）仍是 next-token 交叉熵：L = −(1/K)·Σ_{t∈S} log p_θ(x_{t+1} | x_{≤t})，S 是「要计 loss 的位置集合」，K = |S|。

**直觉**：预训练让模型读完整张互联网学「话怎么说」；SFT 给它看几万份标准问答，只给「回答」那一段打分。同一张考卷，只改评分范围。

**机制**：落到代码就一行——labels 里 prompt 段全填 `-100`（PyTorch `CrossEntropyLoss` 的 `ignore_index` 默认值），只有回答段保留真实 token id。prompt 位不贡献梯度，反向传播也就不会把「预测用户会问什么」当成学习目标。两个易踩点：loss 分母是 K 还是序列长，各框架不同（PyTorch 默认按非 ignore 位平均），换框架数值会跳，别拿绝对值横向比；mask 切口按 token id 算，不按字符数截——中文一个字常对多个 token，按字符切必偏。

**两种故障，别混成一个**：

- **不 mask prompt**：K 从 1200 变 2000，回答位权重被稀释到 60%；模型同时在学用户轮，于是学会「替用户说话」。症状是抢话（用户没说完就接）、自问自答、复读用户的问题。
- **packing 的 label 错位**：拼包后 labels 是 input_ids 右移一位，A 的最后一个 token 的 label 恰好是 B 的首 token，「说完 A 就接 B 的答案」成了学习目标。这才是「串味」——上一轮的问题被当成下一轮的回答。修法是每条样本首 token 的 label 置 -100，外加 attention 分块。

两者修法完全不同：前者查 mask 与 chat template，后者查拼包边界。混成一个，排障就会查错方向。

### 二、SFT 数据的四个硬约束

**定义**：量级、质量、模板一致性、epoch 数，四个约束互相独立，破一个就废。

**直觉**：SFT 不是喂知识，是改口音。改口音不需要听一万小时，需要听标准音。

**机制**：① 量级——几万到百万条，1e8–1e9 token，比 1e13 的预训练小 4–5 个数量级（1e9 差 4 阶、1e8 差 5 阶）；这个量级下单条脏数据在梯度里的占比被放大得厉害。② 质量远胜数量——LIMA 用约 1000 条精选样本做出可用的对齐模型，说明格式与风格由少量高质量样本决定；但「1000 条够用」依赖任务多样性，别当定律。③ 模板逐字节一致——训练用的 chat template 必须与推理（vLLM 的 `--chat-template` 或 tokenizer 内置模板）逐字节相同，BOS（序列开头的特殊标记）、角色标记、换行、空格都算。验法：同一段对话分别用训练脚本的 `apply_chat_template` 和线上服务实际收到的 prompt 渲染，`diff` 两份字符串，不一致就先修模板再谈训练。④ 1–3 个 epoch——再往后训练 loss 继续降，降的是记忆量。判据是留出对话集加人看，不是 loss。

### 三、多轮拼接与 packing：三个边界

**定义**：packing 是把多条短样本首尾相接填满定长序列（如 4096），让每个位置都在干活。多轮拼接则是把一轮对话按「历史 + 当前问题 + 当前回答」拼成一条样本，历史位 mask、当前回答位监督。

**直觉**：一车货，每箱只装半车就走，运费翻倍。

**机制**：三件事必须同时对。① attention 分块——样本之间不能互相看见，否则 B 的 token attend 到 A，两条无关对话被混成一条；靠块对角 mask 或 varlen attention 的 `cu_seqlens`。② position id——RoPE 是按位置旋转 Q/K 的编码方式，位置决定旋转角；`cu_seqlens` 只保证「样本之间不可见」，不解决相位。position id 必须每条样本从 0 重数（`reset_position_ids`），才能与推理时「每条样本都从 0 开始」的分布对齐。部分框架/版本默认不重置，这是已知会掉效果的坑，不是等价方案——在自己版本上打印 `position_ids`，确认第二条样本确实从 0 开始。③ 跨样本 loss mask——每条样本的 prompt 段各自 mask，且每条样本首 token 的 label 置 -100。

**例子**：第一条 500 token、第二条 3500 token 拼进 4096 的窗口；不重置时第二条从第 500 位开始数位置，训练看到的相位分布与推理时（从 0 开始）就不一致。

### 四、RLHF 四阶段：每一环交换什么

**定义**（InstructGPT 的流程，原文写三步，把偏好收集并进奖励模型阶段；拆成四段更好看清交换物）：① SFT——给出格式与起点，产出策略 π_SFT；② 偏好数据——同一 prompt 采样多条回答，让人标「A 比 B 好」，交换的是相对信号；③ 奖励模型（reward model, RM）——用 Bradley-Terry 把比较压成标量打分器 r(x,y)，损失 −log σ(r(x,y_w) − r(x,y_l))，其中 σ(x)=1/(1+e^{−x}) 把分数差压成 0–1 的「A 更好」概率；④ PPO——目标 max E[r(x,y)] − β·KL(π_θ‖π_ref)，KL 惩罚把策略拴在参考模型（通常就是 π_SFT）附近。

**直觉**：SFT 教「怎么答」，偏好教「哪种答法更好」，RM 把偏好固化成能批量调用的评分器，PPO 拿评分器当教练反复练。

**机制**：PPO 用裁剪的比率目标 min(ρA, clip(ρ,1−ε,1+ε)A) 限制每步更新幅度，A 是优势（这次回答比该 prompt 的平均水平好多少）。它是 on-policy 的——只能用当前策略刚采的样本更新，所以 rollout（在线采样）是必经的一段，不是可选优化。

**为什么 RM 不能替代人**：RM 只是有限偏好数据上的代理，输出的是「在这个数据集上更像被选中的那条」，分布外没有定义；更关键是它自己就是个可被优化的目标，策略会朝它的漏洞走而不是朝人的偏好走。RLHF 的闭环里人只给相对判断，RM 把判断放大到百万次调用——RM 能压缩偏好，不能产生偏好。

**失败沿链路传导**：SFT 太差 → RM 在垃圾分布上学偏好，分数没有意义；标注不一致 → RM 学到标注员的噪声；RM 过拟合 → reward hacking（变长、谄媚、堆格式）；β 太小 → 语言崩坏（重复、乱码），β 太大 → 等于没训。

### 五、显存与并行：SFT 是裸奔的 DP/ZeRO，PPO 要四份模型、两个阶段

**定义**：SFT 的并行就是上一课的 DP/ZeRO。PPO 在同一批卡上常驻四份模型，且 rollout 与 update 两个阶段的每卡占用不是同一个数。

**直觉**：SFT 像单人跑；PPO 像四个人接力，还要在跑道上换鞋。

**机制**：账在「一步一步」第 6 步里算。要点三条：rollout 阶段策略、参考、奖励三份都按 TP 切分，KV cache（缓存已生成 token 的 K/V 以免重算）也跟着切；update 阶段策略与 critic（价值模型，估基线算优势）走 ZeRO-2，梯度与优化器状态被切走；每卡峰值取两阶段的 max，不是把 196 GB 平摊。

**例子**：64 GB 卡上 update 阶段约 45.5 GB/卡、rollout 约 5.8 GB/卡——峰值由 update 定。

## 一步一步理解

7B、bf16、8 卡一机、序列长 4096。

**第 1 步 数据账**：50 万条多轮对话，平均 2000 token → 5e5 × 2000 = 1e9 token。预训练 10T = 1e13，差 4 个数量级；若 SFT 只做到 1e8，则差 5 个。

**第 2 步 step 数**：一条公式管到底——steps = 真实 token T ÷（卡数 × 窗口长 × 梯度累积 × 利用率 u）。每卡 micro-batch 1 条、累积 8 → 每步处理 8 × 4096 × 8 = 262,144 个位置。u = 100%（假设已 packing、零 pad）时，1 epoch = 1e9 ÷ 2.62e5 ≈ 3815 步，3 epoch ≈ 1.14 万步。预训练按全局 batch 约 4e6 token/步（1e13 ÷ 2.5e6 步，典型 7B 预训练的全局 batch 量级）算是 250 万步；SFT 3 epoch 占 1.14e4 ÷ 2.5e6 ≈ 0.46%，1 epoch 只占 0.15%。

**第 3 步 packing 利用率**：不 packing，平均 2000 token 的样本 pad 到 4096，u = 2000/4096 = 48.8%，一半算力在算 pad。代回公式：steps = 1e9 ÷ (2.62e5 × 0.488) ≈ 7820。packing 后 u 按 96% 算，steps ≈ 3975。3815 与 7820 不是有一个算错，而是 u 取了 100% 与 48.8%——同一份数据、同样的卡，差近 2 倍时间，这就是 packing 唯一但足够充分的理由。

**第 4 步 mask 的 token 账**：一条 2000 token 三轮对话：system 100、u1 300、a1 400、u2 200、a2 500、u3 200、a3 300，回答共 1200（60%），prompt 共 800（40%）。忘了 mask，K = 2000 而不是 1200：四成位置在教模型「预测用户会问什么」，回答位的有效权重掉到 60%。症状是抢话、扮演用户；「把问题当回答」不是这一处的问题，那是拼包 label 错位。

**第 5 步 梯度与通信占比**：mask 不改变梯度稠密的事实，7B 梯度仍是 14 GB。8 卡 ring all-reduce 每卡 2 × 7/8 × 14 = 24.5 GB，按机内 392 GB/s 是 62.5 ms。每步计算 = 262,144 × 6 × 7e9 = 1.1e16 FLOP = 11 PFLOP。按单卡 bf16 峰值 400 TFLOPS（量级估计，口径随机型，存疑）、MFU 40%（硬件峰值里真正用于有效计算的百分比），8 卡聚合 400 × 0.4 × 8 = 1280 TFLOPS，每步 1.1e16 ÷ 1.28e15 ≈ 8.6 s；等价地，每卡 32,768 token 除单卡 160 TFLOPS 也是 8.6 s。通信占比 62.5 ms ÷ 8.6 s ≈ 0.7%。结论不变：SFT 的瓶颈几乎从不在通信，而在 packing 做没做（3975 步还是 7820 步）。

**第 6 步 PPO 显存：分两阶段算**。四份模型总量：策略 84 GB（权重 14 + 梯度 14 + AdamW 的 m/v 各 4 字节/参数共 56）、参考 14、奖励 14、critic 84，合计 196 GB。

- **update 阶段**：策略 84 GB 经 ZeRO-2 切梯度与优化器状态，每卡 14 + 14/8 + 56/8 = 22.75 GB；critic 也在训，按同尺寸最坏情况再算一份 22.75，合计 45.5 GB/卡，激活另计。参考与奖励只在打分时用，可以在 update 前 offload（先挪出显存）；若坚持常驻，按 TP 切开再各加 1.75 GB/卡。
- **rollout 阶段**：策略、参考、奖励三份都按 TP=8 切开，各 1.75 GB/卡，合计 5.25 GB。KV cache 以 32 层、GQA 8 个 KV 头 × 128 维为例（GQA 指多个 Q 头共享一组 KV 头），每 token 每层 2 × 8 × 128 × 2 B = 4 KB，32 层即 128 KB/token；采样 32 条 × 1024 token = 32,768 token，全量 4.3 GB，按 TP 切成 0.54 GB/卡。合计约 5.8 GB/卡。

每卡峰值 = max(45.5, 5.8) ≈ 45.5 GB 加激活：64 GB 卡上能跑、余约 18 GB，不宽裕；32 GB 卡装不下。把 196 GB 平摊成 24.5 GB 再往上加 KV、或按全量算 KV，都会得出错结论。

**第 7 步 两套通信域**：rollout 阶段 TP=8 落在一台机内，每层 2 次集合通信、对象是激活、随 token 线性涨；update 阶段切回 DP=8，一次 24.5 GB 的 ring（62.5 ms）。上一课那句「TP 不出机框、DP 才跨机」在这里是硬约束：采样一跨机，rollout 时间翻倍，而 rollout 是 PPO 里最贵的一段。两套域轮流建，建链有开销，rank 映射也不能让两组撞车。

## 常见误区

**一、「SFT 是一种新算法、新 loss」。** 不是。还是 next-token 交叉熵，改的只有数据分布和 loss 作用域。SFT 效果不好时先查数据与模板，别去翻优化器。

**二、「不 mask prompt 只是让 loss 算得慢一点」。** mask 不省算力——前向仍是稠密矩阵乘，它只改 loss 与梯度来源。不 mask 的代价是学习目标错了：模型被训练去预测用户轮，学出抢话、扮演用户。

**三、「packing 只是性能优化，拼错最多慢一点」。** 拼错是正确性问题：attention 不按样本分块会跨样本泄漏，label 错位一格会串味。诊断分别看 attention mask 是否块对角、每条样本首 token 的 label 是不是 -100。

**四、「多训几个 epoch 更划算，反正 loss 还在降」。** 3 epoch 之后降的是训练集记忆量，不是能力。出现「benchmark 涨、真实对话变差」（换个问法就崩）时，第一反应该是减 epoch，不是加数据。

## 一句话总结

SFT 没换目标函数，只在 labels 里把 prompt 段填 -100，让梯度只从回答位回传，这一行就是「对齐」在工程上的全部真相；数据守四条：1e8–1e9 token（比预训练小 4–5 个数量级）、质量优先、模板与推理逐字节一致、1–3 epoch；工程上用 packing 把利用率从 48.8% 抬到 96%，但 attention 分块、position id 归零、跨样本 loss mask 三个边界必须同时对，错一格就串味；RLHF 四阶段是 SFT 给起点、偏好给相对信号、RM 压成标量、PPO 用 KL 拴住策略，每环的错都以特定形态传给下一环；账上 SFT 只是上一课的 DP/ZeRO，PPO 要四份模型常驻、rollout 用 TP、update 用 DP/ZeRO，峰值取两阶段的 max。

## 自测题

### 1. 一条 2000 token 样本，回答 1200 token。写出 loss，并说明 prompt 位填什么、分母是多少、忘了 mask 会怎样。
<details><summary>查看答案</summary>labels 里 prompt 段全填 -100（ignore_index），回答段保留真实 token id；L = −(1/K)Σ_{t∈S} log p_θ(x_{t+1}|x_{≤t})，S 是回答位，K = |S| = 1200（按 PyTorch 默认的非 ignore 平均口径；若框架按序列长归一，分母是 2000）。忘了 mask：K 变 2000，回答位权重被稀释到 60%，且模型同时被训练去预测用户轮，推理时表现为抢话、自问自答、复读问题。注意这与「上一轮问题被当下一轮回答」不是一回事——后者是拼包 label 错位，修法在拼包边界。</details>

### 2. packing 后必须同时做对哪三件事？各错一件分别看到什么？
<details><summary>查看答案</summary>① attention 分块（块对角 mask 或 varlen 的 cu_seqlens）：错则 B 的 token 能看见 A，两条样本混成一条，回答里出现别的对话的信息；② position id 每条样本从 0 重数：错则第二条样本接着前一条累加位置，与推理时「从 0 开始」的相位分布不一致，属已知掉效果的坑；验证方法是打印 position_ids 看第二条是否从 0 开始（cu_seqlens 只保证不可见，不解决相位）；③ 跨样本 loss mask，且每条样本首 token 的 label 置 -100：错则 A 的最后一个 token 被训练去预测 B 的首 token，表现为「上一轮问题被当成下一轮回答」的串味。</details>

### 3. 为什么说 SFT 只花预训练不到 1% 的算力？给出算式。
<details><summary>查看答案</summary>算力约等于 6 × 参数量 × token 数，参数量相同时直接比 token 数：SFT 1e9、预训练 1e13，差 4 个数量级即 1/10000。按步数看：SFT 每步 262,144 个位置（8 卡 × 4096 × 累积 8），3 epoch 约 1.14 万步；预训练按全局 batch 约 4e6 token/步是 250 万步，1.14e4 ÷ 2.5e6 ≈ 0.46%。但 SFT 改的是数据分布与 loss 作用域，格式与行为几乎全由这不到 1% 决定——这就是「小算力、大权重」。</details>

### 4. PPO 为什么要常驻四个模型？7B、bf16、8 卡、64 GB 卡，每卡峰值怎么算？
<details><summary>查看答案</summary>策略（被训练，也是采样器）、参考模型（冻结，只为算 KL 把策略拴住）、奖励模型（给 rollout 打分）、critic（估基线算优势）。总量 84 + 14 + 14 + 84 = 196 GB。update 阶段：策略 84 GB 经 ZeRO-2 切到 14 + 14/8 + 56/8 = 22.75 GB/卡，critic 再一份 22.75，合计 45.5 GB/卡；rollout 阶段：三份模型各按 TP=8 切成 1.75 GB/卡共 5.25，KV cache 全量 4.3 GB 切成 0.54 GB/卡，合计约 5.8 GB/卡。每卡峰值 = max(45.5, 5.8) + 激活 ≈ 45.5 GB + 激活：64 GB 上能跑、余约 18 GB，不宽裕。把 196 GB 平摊成 24.5 GB、或把 KV 全量记在每卡上，都会得出错误结论。</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| SFT（监督微调） | 用「指令-回答」示范继续做 next-token 交叉熵，教模型按指令与格式说话；目标函数没变，变的是数据分布与 loss 作用域。 |
| 指令微调 | SFT 的同义词，强调数据形态是指令-回答对，而非普通文本。 |
| next-token 交叉熵 | −log p(下一个正确 token) 的平均；预训练与 SFT 用的是同一个。 |
| loss 作用域 | 参与 loss 求和的位置集合 S；SFT 只取回答位，prompt 位被 mask。 |
| `-100` / ignore_index | PyTorch 交叉熵里「不计 loss、不回传梯度」的标签约定值，默认 ignore_index 就是 -100。 |
| prompt 位 | 样本中 system 与用户输入部分；SFT 里必须 mask。 |
| chat template | 把消息列表渲染成含角色标记的字符串的模板；训练与推理必须逐字节一致。 |
| packing | 多条短样本首尾相接填满定长序列，消除 pad 浪费。 |
| 利用率 u | 窗口里有效 token 的占比；steps = 真实 token ÷（卡数 × 窗口长 × 累积 × u）。 |
| attention 分块 | 让同一 pack 内的不同样本互不可见；靠块对角 mask 或 varlen 的 cu_seqlens。 |
| cu_seqlens | varlen attention 里记录每条样本起止位置的数组；只管「不可见」，不管位置相位。 |
| position id 重置 | 每条样本位置从 0 重新计数；不重置则 RoPE 相位跨样本累加，与推理分布不一致。 |
| RoPE | 旋转位置编码：按位置旋转 Q/K 向量来注入位置信息。 |
| 串味 | 拼包 label 错位导致样本间泄漏，典型表现是上一轮问题被当成下一轮回答。 |
| 多轮拼接 | 把一轮对话按「历史 + 当前问题 + 当前回答」拼成一条样本，历史位 mask、当前回答位监督。 |
| 偏好数据 | 同一 prompt 下多条回答的人工比较结果，只给相对好坏、不给绝对分。 |
| Bradley-Terry | 把成对比较转成概率 σ(r_w − r_l) 的模型，奖励模型损失的基础。 |
| σ（sigmoid） | σ(x)=1/(1+e^{−x})，把任意实数压到 0–1，这里用来把分数差变成「A 更好」的概率。 |
| 奖励模型（RM） | 给 (prompt, 回答) 打标量的模型；是有限偏好上的代理，分布外无定义、可被优化钻空子。 |
| PPO | 近端策略优化：用裁剪的比率目标做在线策略更新。 |
| on-policy | 只能用当前策略刚采的样本更新；所以 PPO 必须有 rollout 采样阶段。 |
| 优势 A | 这次回答比该 prompt 平均水平好多少，PPO 用它给每个 token 加权。 |
| KL 惩罚 | β·KL(π_θ‖π_ref)，衡量新策略与参考模型的分布差，把它拴在附近。 |
| 参考模型 | π_ref，通常是 SFT 模型，冻结不动，只用来算 KL。 |
| rollout | 用当前策略在线采样回答的阶段，相当于一段推理服务。 |
| reward hacking | 策略钻奖励模型的空子，分数涨、人评降（变长、谄媚、堆格式）。 |
| GQA | 分组查询注意力：多个 Q 头共享一组 KV 头，KV cache 按 KV 头数算而非 Q 头数。 |
| KV cache | 推理时缓存已生成 token 的 K/V；显存随 batch × 序列长 × 层数 × KV 头数线性涨。 |
| MFU | Model FLOPs Utilization：硬件峰值里真正用于有效计算的百分比，本课取 40%。 |
| 梯度累积 | 多个 micro-batch 累加梯度再更新一次，用来凑全局 batch。 |
| DP / ZeRO | 数据并行与状态分片；SFT 的标准并行方式，复用上一课的通信账。 |

## 下一步

下一课「数据工程」把今天埋的雷逐个拆：SFT 与偏好数据从哪来、怎么清洗去重、模板与标注一致性怎么在流水线上保证、packing 与 mask 的检查清单怎么写成自动化断言。今天那四条硬约束（量级、质量、模板、epoch）会在那一课变成可执行的验收标准。本模块（下）讲 DPO 与 GRPO 时，两阶段显存账直接复用今天这套 max(rollout, update) 的算法。

## 来源

- Training language models to follow instructions with human feedback（InstructGPT，arXiv:2203.02155） — https://arxiv.org/abs/2203.02155
- Proximal Policy Optimization Algorithms（PPO，arXiv:1707.06347） — https://arxiv.org/abs/1707.06347
- Deep Reinforcement Learning from Human Preferences（偏好学习起源，arXiv:1706.03741） — https://arxiv.org/abs/1706.03741
- Direct Preference Optimization（DPO，arXiv:2305.18290） — https://arxiv.org/abs/2305.18290
- LIMA: Less Is More for Alignment（数据质量，arXiv:2305.11206） — https://arxiv.org/abs/2305.11206
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models（arXiv:1910.02054） — https://arxiv.org/abs/1910.02054
- PyTorch 官方文档：CrossEntropyLoss 的 ignore_index — https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Hugging Face Transformers 官方文档：Chat templating — https://huggingface.co/docs/transformers/chat_templating
- Hugging Face TRL 官方文档：SFTTrainer 与 packing/padding-free — https://huggingface.co/docs/trl/sft_trainer
- vLLM 官方文档：张量并行与扩容 — https://docs.vllm.ai/en/latest/serving/parallelism_scaling/
- 昇腾 CANN 集合通信 HCCL 用户指南：通信域与 rank 配置 — https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/commlib/hcclug/hcclug_000009.html
