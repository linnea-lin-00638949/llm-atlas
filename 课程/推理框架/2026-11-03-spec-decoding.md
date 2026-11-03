---
date: "2026-11-03"
title: "第 12 课 · 投机解码"
domain: "推理框架"
module_id: "spec-decoding"
module: "投机解码"
status: "已学"
lesson_no: 12
goals:
  - "能解释投机解码为什么能加速 decode：把 K 个草稿接在前缀后一次前向并行核对，一次权重读取摊到 K+1 个位置"
  - "能说出 acceptance rate 的含义与收益条件：分清条件 α_i 与聚合 accepted/drafted，并用 L > 1+K·c_d/c_t 判断是否净赚"
  - "能比较 draft model / EAGLE / Medusa / n-gram 的异同：草稿来源、是否要训练、额外开销与擅长场景"
tags:
  - "投机解码"
  - "推理加速"
  - "接受率"
  - "EAGLE"
  - "vLLM"
  - "昇腾推理"
---

# 第 12 课 · 投机解码

> 学完今天，你能：
> 1. 解释投机解码为什么能把 decode 的访存开销摊薄
> 2. 分清 acceptance rate 的条件定义与聚合口径，判断收益条件
> 3. 比较 draft model / EAGLE / Medusa / n-gram 的异同

## 课前自测

回顾 serving-basics：推理分 prefill（prompt 一次算完）和 decode（每步只出一个 token）；decode 算得少却要读一遍全部权重，是访存受限（memory-bound）的。

1. 一步只产 1 个 token，矩阵单元大半时间在等数据——能不能一趟多算几个？
2. 小模型先猜 4 个、大模型一次核对，大模型的计算量涨多少？
3. 猜错了怎么办，质量会不会下降？
4. 小模型猜得准但很慢，还值得用吗？

## 为什么需要它

7B 模型 FP16 权重约 14 GB，带宽 1.6 TB/s 时读一遍约 8.75 ms，只换来 1 个 token；这一步只做约 14 GFLOP（2×7×10⁹）计算，折合 1.6 TFLOPS。FP16 算力 300 TFLOPS 量级的卡，利用率不到 1%（随卡型浮动）——这就是 TPOT（每个输出 token 的平均耗时）难降的根因。

batch 能把一次读权重摊给多个请求，却摊不到同一请求的多个 token：自回归（每出一个 token 都要接回输入再算下一个，天然串行）决定第 i+1 个必须等第 i 个。投机解码（speculative decoding）先便宜地猜出 K 个 token，再让目标模型一次前向并行核对，K+1 个位置共用一次权重读取。它不改主模型（部分方案额外训一个小头），在中等偏低 QPS、访存受限时收益最大，与 vLLM 文档定位一致。

## 核心概念

### draft model

**定义**：小得多的模型，自回归先猜出 K 个 token，再交目标模型裁决。

**直觉**：实习生起草、资深编辑一遍审完；快但会错，编辑只读一遍就能定稿。

**机制**：须与目标模型共享词表，否则 token 对不上。K 个草稿串行 K 步，成本 = K × 小模型单步耗时；参数量常是目标的 1/10 量级。输出只是提议。

### acceptance rate

**定义**：α_i = P(第 i 个草稿被接受 | 前 i−1 个都被接受)；单点严格式 α = E_{x~q}[min(1, p(x)/q(x))]，x 遍历词表、按草稿分布 q 加权平均，p 是目标模型给同一 token 的概率。

**直觉**：α=0.7 是「第 1 个草稿有 70% 通过」，不是「10 个里 7 个」——连过 4 个只有 0.7⁴≈24%；监控里的接受率多是聚合口径 accepted/drafted，本例 1.77/4≈44%。同名不同量。

**机制**：草稿给 0.5、目标给 0.2，就以 0.2/0.5=0.4 接受；拒绝时从残差分布 max(0, p−q) 归一化后重采，保证最终分布与目标模型一致。α 随位置递减（依赖链越长越易错）。聚合接受率 =(Σα^i)/K，接受长度 L = 1+Σα^i。

**验证为何能一次并行**：K 个草稿接在前缀后拼成 K+1 个位置的序列，因果掩码让第 j 个位置只看到「前缀 + 前 j−1 个草稿」，一次前向同时得出 K+1 组条件分布，正是逐位验证所需。随后从前往后比对，第一处不通过即停，用残差补出该位置 token，其后草稿连同 KV 全部丢弃——所以能接受的必是前 i 个，L 就是前缀长度。

### EAGLE

**定义**：在特征空间起草的薄草稿头：吃目标模型倒数第二层隐藏向量（hidden state，模型内部表示），预测下一个隐藏向量，再复用目标模型 LM head（隐藏向量→词表概率的最后一层）转成 token。全称 Extrapolation Algorithm for Greater Language-model Efficiency（arXiv:2401.15077）。

**直觉**：不重学一门语言，只学「大模型下一步会怎么想」；猜想法比猜字容易，输出层还能白借。

**机制**：输入 = 当前 token 的 embedding（词向量）+ 上一位置的隐藏向量；训练时把序列整体前移一步，缓解特征预测的不确定性。论文在 LLaMA2-Chat 70B 上报 2.7–3.5× 加速、吞吐翻倍且分布不变；EAGLE-2（arXiv:2406.16858）改用动态草稿树（一次提交多条候选路径），按置信度逐层分支，报 3.05–4.26×。自带 MTP（多 token 预测头）的模型可直接拿它当草稿。

### Medusa

**定义**：在冻结主干上并行挂 K 个解码头，第 i 个头直接预测「往后第 i 个 token」，再用树注意力一次验证多条候选。

**直觉**：有人猜下一句、有人猜下两句，同时交稿，主编一次审完。

**机制**：每个头只是接在主干最后一个隐藏向量上的小网络；Medusa-1 训练时主干冻结，不改原模型能力。K 个头的候选天然成树，验证时把所有路径拼进一个 batch、用掩码让每条路径只看自己的前缀，一次前向验完。论文报 Medusa-1 2.2×、Medusa-2（头与主干联合微调）2.3–3.6×，不损质量。差别：不用第二个模型，但每个目标模型都得单独训一套头。

### n-gram

**定义**：不用模型。拿序列结尾的 n 个 token（n-gram，连续 n 个 token 的片段）去 prompt 或已生成文本里找同样片段，命中就抄后面跟着的 token 当草稿。

**直觉**：复制粘贴：改函数名、答 RAG（检索增强生成）问题时大段引用原文，要写的东西前面常已出现过。

**机制**：vLLM 用 `prompt_lookup_min` / `prompt_lookup_max` 控制匹配窗口，最多抄 `num_speculative_tokens` 个。零参数、零额外显存、零训练，开销几乎只在验证；只在重复型文本上命中，自由创作时接受率很低。

| 方法 | 草稿从哪来 | 要训练吗 | 额外开销 | 擅长什么 |
| --- | --- | --- | --- | --- |
| draft model | 独立小模型逐 token 生成 | 不用，找现成小模型 | 小模型权重 + 它的 KV cache | 通用，前提是够小 |
| EAGLE | 目标模型特征喂小头，预测下一特征 | 要训草稿头 | 很小，复用 LM head | 通用，接受率高 |
| Medusa | 冻结主干 + 并行解码头 | 要训解码头 | 小，几个头 | 通用，一次验多路 |
| n-gram | 从 prompt / 已生成文本抄 | 不用 | 几乎为零 | 重复、引用型文本 |

四种里只有 n-gram 不新增任何模型参数。

## 一步一步理解

设定：目标单步 30 ms（c_t）、草稿 3 ms（c_d）、K=4、各位置取常数 α=0.7。**先划清限制**：常数 α 只为手算方便，真实 α_i 逐位下降，2.77 是乐观上界；按 0.7/0.6/0.5/0.4 递减时 L 只有 2.414，每 token 42÷2.414≈17.4 ms，加速 1.72× 而非 1.98×。

**第 1 步**：不投机时 30 ms/token。

**第 2 步**：第 1 个通过的概率 0.7，前 2 个 0.49，依此类推；加上验证必然多出的 1 个 token，L = 1+α+α²+α³+α⁴ = 1+0.7+0.49+0.343+0.2401 = 2.7731，一般式 (1−α^{K+1})/(1−α)。

**第 3 步**：一轮 = 草稿 4×3 = 12 ms + 验证 30 ms（5 个位置并行、只读一遍权重，K 不大时约等于一次 decode）= 42 ms。被拒位置的 KV cache 槽位由引擎按块回收，代价很小。

**第 4 步**：42÷2.7731≈15.1 ms/token，加速 30÷15.1≈1.98×。

**第 5 步（保本点）**：(K·c_d+c_t)/L < c_t，即 L > 1+K·c_d/c_t。本例门槛 1.4；试算 α=0.29：L = 1+0.29+0.0841+0.0244+0.0071 ≈ 1.406，略高于 1.4，保本 α≈0.288。草稿若有目标一半大（c_d=15 ms），门槛升到 L>3、保本 α≈0.74；α=0.7 时 90÷2.7731≈32.5 ms，比不开还慢。

**第 6 步（K 怎么选）**：

| K | L | 一轮成本 | ms/token | 加速比 |
| --- | --- | --- | --- | --- |
| 2 | 2.19 | 36 ms | 16.4 | 1.83× |
| 4 | 2.77 | 42 ms | 15.1 | 1.98× |
| 8 | 3.20 | 54 ms | 16.9 | 1.78× |

K 从 4 到 8 只多换 0.43 个 token，草稿成本却线性涨。最优 K 只能实测。

**第 7 步（昇腾上怎么开）**：`--speculative-config` 收一个 JSON 串统一传参：

```
vllm serve Qwen/Qwen3-8B --tensor-parallel-size 4 \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'
```

EAGLE 家族用 `{"method": "eagle", "model": "<草稿头>", "draft_tensor_parallel_size": 1, "num_speculative_tokens": 2}`。**method 必须与草稿头版本匹配**：EAGLE-1/2 头用 `eagle`、EAGLE-3 头才用 `eagle3`，混用会直接加载失败；草稿头还须与目标模型配套，EAGLE 通常还要 `--enforce-eager`，键名以所用版本文档为准。三条硬约束：昇腾融合注意力单次 decode 最多 16 个 token（`num_speculative_tokens+1≤16`）；草稿头不做张量并行，`draft_tensor_parallel_size` 只允许 1 或与主模型相同；开图模式时捕获尺寸取 (K+1) 的整数倍，如 K=4、并发 1–4 设 [5,10,15,20]。

上线判据分两步，别拍单一阈值：① 按实测 c_d/c_t 算保本 L* = 1+K·c_d/c_t；② 从 `/metrics` 取草稿 token 数与接受 token 数（指标名随版本变，先确认分子分母）换算实测 L，明显低于 L* 就调小 K 或换方法。本例 L*=1.4、实测 2.77 有得赚；聚合接受率 44% 不代表配置差——它和 α 不是同一个量。

## 常见误区

**误区 1：「投机解码降低输出质量，是一种近似。」** 错。按 min(1, p/q) 接受、拒绝时从 max(0, p−q) 重采，数学上保证输出分布与目标模型一致，草稿只影响速度。（故意放宽验证的近似变体才是有损的；浮点误差另计。）

**误区 2：「acceptance rate 越高越快。」** 不够。加速由 L 与成本比共同决定：c_d/c_t=0.5、K=4 时门槛 L>3，α 要接近 0.75；α=0.7 时每 token 32.5 ms 反而更慢。挑草稿第一标准是「够小」，第二才是「猜得准」。

**误区 3：「K 越大赚得越多。」** 收益按 α^K 衰减、成本按 K 线性涨，最优 K 常是个位数（K=4 优于 K=8）；昇腾还有 K+1≤16 的硬上限。

**误区 4：「开了投机解码吞吐必然涨。」** 中低 QPS 最有效；高 QPS 时算力已被 batch 占满，验证 K+1 个位置在抢算力，吞吐可能不升反降。n-gram 这类零成本方案在峰值流量下不至于明显拖慢，但收益同样有限。

## 一句话总结

便宜地猜几个 token、目标模型一次前向并行核对、拒绝采样保证分布不变；只有「草稿够便宜 + 接受率够高」，才把访存瓶颈换成真实的 TPOT 下降。

## 自测题

### 1. 为什么 decode 是访存受限的？投机解码怎样摊薄一次权重读取？

<details><summary>查看答案</summary>

14 GB 权重按 1.6 TB/s 读一遍要 8.75 ms，却只产出 1 个 token；单步约 14 GFLOP、折合 1.6 TFLOPS，算力大量闲置。batch 只能摊给不同请求，摊不到同一请求的串行 token。投机解码把 K 个草稿接在前缀后组成 K+1 个位置的序列，因果掩码让每个位置只看自己的前缀，一次前向同时得到 K+1 组分布，一次读权重服务最多 K+1 个 token。前提是草稿足够便宜、接受率足够高。

</details>

### 2. α=0.8、K=3、c_d/c_t=0.1 时，L、聚合接受率、加速比各是多少？

<details><summary>查看答案</summary>

L = 1+0.8+0.64+0.512 = 2.952；聚合接受率 = (0.8+0.64+0.512)/3 ≈ 65%，不是 80%。一轮成本 3×0.1+1 = 1.3 c_t，每 token 1.3/2.952≈0.44 c_t，加速约 2.27×。把 α 误读成「80% 的草稿被接受」会算出 L≈3.4、加速虚高——这正是要区分条件 α 与聚合口径的原因。

</details>

### 3. EAGLE 与 Medusa 都在「头」上做文章，差别在哪？

<details><summary>查看答案</summary>

EAGLE 的头在特征空间：输入是当前 token 的 embedding 加上一位置隐藏向量，输出「下一个隐藏向量」，再复用目标模型 LM head 转成 token，因此头很薄、可自回归连猜多个。Medusa 的 K 个头在 token 空间并行输出「往后第 i 个 token」，彼此独立、不串行，靠树注意力一次验证多路候选。代价上 Medusa 每个目标模型都要训一套头，EAGLE 只需一个薄头但要配套的草稿头文件。

</details>

### 4. 什么情况下开投机解码反而更慢？

<details><summary>查看答案</summary>

当 (K·c_d+c_t)/L > c_t 时。典型三种：草稿不够便宜（c_d/c_t=0.5 门槛 L>3，α=0.7 只有 2.77，每 token 32.5 ms > 30 ms）；K 过大（4→8 从 1.98× 掉到 1.78×）；高 QPS 下算力被 batch 占满，验证 K+1 个位置在抢算力。上线应按实测 c_d/c_t 算出保本 L*，与实测 L 对比再定。

</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| draft model | 草稿模型：先猜若干 token 的小模型，输出只是提议，最终由目标模型裁决 |
| acceptance rate | 单位置条件接受概率 α_i = P(第 i 个草稿通过 \| 前 i−1 个已通过) = E_{x~q}[min(1, p/q)]；与监控的聚合口径 accepted/drafted 不同，换算用 L=1+Σα^i |
| 拒绝采样 | 按 min(1, p/q) 接受，拒绝时从残差 max(0, p−q) 重采样，保证输出分布不变 |
| 验证 | 目标模型一次前向并行核对 K+1 个位置，接受前缀、拒绝后缀 |
| 树注意力 | 把多条候选路径拼成树、用掩码让每条路径只看自己前缀，一次前向验多条 |
| 草稿树 | 一次提交多条候选路径的树形结构，提高「至少接受一条」的概率 |
| EAGLE | 在特征空间自回归的薄草稿头，复用目标模型 LM head 即可起草 |
| Medusa | 在冻结主干上并行挂多个解码头，一次前向预测往后多个 token，配合树注意力验证 |
| n-gram | 连续 n 个 token 的片段；prompt lookup 匹配后复制后续 token 当草稿，不需要模型 |
| MTP | Multi-Token Prediction：模型自带的多 token 预测头，可直接充当草稿 |
| TPOT | Time Per Output Token：每个输出 token 的平均耗时，投机解码主要优化的指标 |
| 访存受限 | memory-bound：单步时间由读权重/KV 的数据量决定，算力闲置，是收益来源 |

## 下一步

第 13 课「Prefill/Decode 分离与长上下文」：本课优化 decode 单步；下一课把 prefill 与 decode 拆到不同实例，处理长上下文的显存与 KV cache 压力。带着「一次权重读取能摊到几个 token」去看分离式部署，会更容易理解它为什么能同时改善 TTFT（首 token 时延）和 TPOT。

## 来源

- vLLM 官方文档 · Speculative Decoding — https://docs.vllm.ai/en/latest/features/speculative_decoding/
- vLLM Ascend 官方文档 · Speculative Decoding Guide — https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/speculative_decoding.html
- Leviathan et al., Fast Inference from Transformers via Speculative Decoding (ICML 2023) — https://arxiv.org/abs/2211.17192
- Chen et al., Accelerating Large Language Model Decoding with Speculative Sampling — https://arxiv.org/abs/2302.01318
- Li et al., EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077) — https://arxiv.org/abs/2401.15077
- Li et al., EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858) — https://arxiv.org/abs/2406.16858
- Cai et al., Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads (arXiv:2401.10774) — https://arxiv.org/abs/2401.10774
