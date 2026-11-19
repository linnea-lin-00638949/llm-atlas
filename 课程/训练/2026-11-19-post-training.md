---
date: "2026-11-19"
title: "第 20 课 · 从 PPO 到 DPO 与 GRPO：把奖励模型和 critic 都省掉（下）"
domain: "训练"
module_id: "post-training"
module: "后训练：SFT 与强化学习"
status: "已学"
lesson_no: 20
goals: ["把 DPO 的损失从「最大化奖励减 KL」一步步推出来，说清隐式奖励 β·log(π_θ/π_ref) 怎么消掉奖励模型，以及推导依赖的三个前提：Bradley-Terry 偏好模型、最优策略有闭式解、π_ref 近似最优策略。", "算清一个偏好对在 DPO 里要过几次前向、β 从 0.1 调到 1 会改变什么，并诊断 chosen 与 rejected 的 logprob 一起下降——那是分布外偏好对造成的 likelihood displacement，不是学习率太大。", "写出 GRPO 的组内优势 A_i=(r_i−mean(r))/std(r)，说清组内均值替掉 critic 换来多少显存与多少额外采样，并能按奖励模型、在线 rollout、偏好数据复用三个维度给 DPO/PPO/GRPO 选型。"]
tags: ["DPO", "GRPO", "PPO", "RLHF", "RLVR", "后训练"]
---

# 第 20 课 · 从 PPO 到 DPO 与 GRPO：把奖励模型和 critic 都省掉（下）

> 学完今天，你能：
> 1. 把 DPO 的损失从「最大化奖励减 KL」一步步推出来，说清隐式奖励 β·log(π_θ/π_ref) 怎么消掉奖励模型，以及推导依赖的三个前提：Bradley-Terry 偏好模型、最优策略有闭式解、π_ref 近似最优策略。
> 2. 算清一个偏好对在 DPO 里要过几次前向、β 从 0.1 调到 1 会改变什么，并诊断 chosen 与 rejected 的 logprob 一起下降——那是分布外偏好对造成的 likelihood displacement，不是学习率太大。
> 3. 写出 GRPO 的组内优势 A_i=(r_i−mean(r))/std(r)，说清组内均值替掉 critic 换来多少显存与多少额外采样，并能按奖励模型、在线 rollout、偏好数据复用三个维度给 DPO/PPO/GRPO 选型。

## 课前自测

1. PPO 的 KL 惩罚项拴住的是哪两个分布？β 调大、调小各会出什么症状？
2. 奖励模型输出一个标量，为什么说它只是偏好的代理而不是偏好本身？
3. on-policy 意味着同一批 rollout 数据能拿来做几次梯度更新？为什么？
4. DPO 训一个偏好对要几次前向？参考模型要不要反传？
5. GRPO 一个 prompt 采 8 条全对，优势是多少？还有梯度吗？

## 为什么需要它

上一课把 PPO 的四份模型算到 196 GB、每卡峰值 45.5 GB。那笔账里两块最贵的零件都不是目标本身：奖励模型（14 GB，还得先单独训一遍）和 critic（84 GB，和策略一样大），它们只是估计目标的工具。

今天顺着 PPO 的目标 max_π E[r(x,y)] − β·KL(π_θ‖π_ref) 往下推：消掉奖励模型得到 DPO，消掉 critic 得到 GRPO。省下的不白给——DPO 用离线偏好对换掉在线探索，GRPO 用 G 倍采样换掉一份 critic。

## 核心概念

### 一、DPO 推导：把最优策略的闭式解代回去

**定义**：KL 约束下的目标 max_π E_{y∼π}[r(x,y)] − β·KL(π‖π_ref) 有闭式最优解 π*(y|x) = (1/Z(x))·π_ref(y|x)·exp(r(x,y)/β)，Z(x) = Σ_y π_ref(y|x)·exp(r(x,y)/β) 是配分函数。

**直觉**：最优策略就是「参考模型按奖励做指数加权」，奖励高的回答被 e^{r/β} 放大；β 是温度，β 大放大得温和，π* 离 π_ref 近。

**机制**：对闭式解取对数、移项，得 r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)。代进 Bradley-Terry 的 σ(r_w − r_l)，同一个 prompt 下 β·log Z(x) 出现两次、直接抵消，于是
L_DPO = −E log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x)) − β·log(π_θ(y_l|x)/π_ref(y_l|x)) )。
奖励模型被「隐式化」：不再需要网络输出标量，奖励就是 β 乘以策略与参考模型的对数比。Z(x) 原本要对所有 y 求和，DPO 让它在成对比较里抵消，所以不用真算；真正的代价是训练只有一批固定偏好对，在线探索没了，数据没覆盖的区域在目标里没有对应项。

前提三条：Bradley-Terry 假设偏好只由标量奖励差决定；最优策略有闭式解；π_ref 近似最优策略。前两条是数学，第三条是工程假设。

**例子**：固定当前 log ratio 差 34，β=0.1 时 Δ=3.4；把 β 调到 1，Δ 变成 34，σ 早已饱和、单对梯度趋零；同时 KL 约束更强，策略被更紧地拴在 π_ref 附近。β 不是学习率。

### 二、DPO 损失与计算账

**定义**：上式每项叫 log ratio，乘 β 就是隐式奖励 r̂(x,y) = β·log(π_θ(y|x)/π_ref(y|x))；损失只要求 r̂(y_w) 比 r̂(y_l) 高。

**直觉**：只要 chosen 比 rejected 高出一截，σ 就饱和、这一对的梯度趋零；它从不要求某个绝对分数。

**机制**：一个偏好对 4 次前向——chosen 与 rejected 各跑策略和参考；参考冻结，只有策略反传，所以是「2 次带反向 + 2 次纯前向」。参考模型通常从 SFT 模型初始化并冻结，它定义 log ratio 的零点；换参考要重算零点、重新解释 loss。β 是拴住策略的绳子，和 PPO 里同一个角色、不同位置：PPO 把 β·KL 加在奖励外面做 shaping，DPO 把 β 乘在 log 比里面，决定「跑多远算合格」。常用 β=0.1（0.01–0.5），越大越贴参考。

**例子**：初始点 π_θ=π_ref，所有 log ratio 为 0，Δ=0，L=−log σ(0)=log2≈0.693。

### 三、DPO 的三个真实故障

**似然位移（likelihood displacement）**：偏好对落在策略几乎不生成的区域时，损失只要求「w 比 l 相对更高」。最省力的解法是把 l 压得比 w 更狠，w 的绝对 logprob 跟着掉。症状是「训练 loss 降、模型变差」：该说的话不说了（arXiv:2404.10719）。机制是偏好对在策略低概率区，不是学习率太大。

**长度偏置**：隐式奖励是 token 级 log ratio 的**和**而非平均。每个 token 多 0.01 分，100 个 token 就多 1 分，于是「把 chosen 拉长、把 rejected 缩短」比「答得更好」便宜。SimPO（arXiv:2405.14734）改用平均 logprob 正是为此。方向由数据与归一化决定，必须实测，不能先验断言偏爱长答还是短答。

**覆盖不足**：偏好数据只覆盖被标注过的分布，prompt 与风格之外的行为没有约束，SFT 学到的东西可能被顺手压掉。这是第三类独立故障，也会放大似然位移；但长度偏置来自求和机制，覆盖充分也会出现，不能说三者都以覆盖为总根。

**诊断**：同时打印 chosen/rejected 的 logprob（总和与每 token 平均）和长度。健康形态是 w 升或持平、l 降、长度差不放大；w 与 l 同降 → 似然位移；长度差拉大 → 长度偏置。

### 四、GRPO：把 critic 换成组内基线

**定义**：同一 prompt 采 G 条回答 {o_i}、得奖励 {r_i}，组内标准化得到优势 A_i = (r_i − mean(r)) / std(r)，再代进 PPO 的裁剪目标，一条回答里每个 token 共用这个 A_i。

**直觉**：critic 估的是 V(s_t)——从某个前缀/已生成 token 出发的期望回报，逐 token 给基线；GRPO 用同组 G 条最终奖励的均值做 prompt 级基线，所有 token 共用，省掉逐 token 价值网络，但基线更粗、需要 G 倍采样压方差。

**机制**：省掉 critic 的 84 GB（7B 全参：权重 14 + 梯度 14 + AdamW 的 m/v 56）。代价两条：G 倍采样，rollout 变贵，而 rollout 本来就是 PPO 最贵的一段；组内全对或全错时 std=0、A 全 0，这条 prompt 没有梯度，RLVR 后期大量样本全对，训练会空转（组内与长度归一化本身也带偏置，见 arXiv:2503.20783）。KL 用 k3 估计直接写进 loss：D_KL ≈ π_ref/π_θ − log(π_ref/π_θ) − 1，逐 token 算，估计的是 KL(π_θ‖π_ref)；它只需 π_θ 的采样，不需从 π_ref 额外采样，逐 token 非负、方差低于 k1。

**例子**：G=8、奖励全对时 std=0，8 个优势全是 0，这条 prompt 一个梯度也不贡献。

### 五、三者选型与两阶段账

统一口径：7B、bf16、8 卡、奖励模型也按 7B 同尺寸（14 GB）、ZeRO-2 切策略（权重复制 14 GB，梯度+AdamW 70 GB 按 8 卡切 = 8.75 GB/卡）、参考模型 TP-8（14/8 = 1.75 GB/卡）；未计激活与临时 buffer，是粗算。

- **DPO**：离线，2 个模型（策略 84 + 参考 14 = 98 GB），无 rollout。每卡 14 + 8.75 + 1.75 = 24.5 GB。显存最低、最便宜，上限由偏好数据覆盖决定。
- **PPO**：4 个模型 196 GB，rollout + update 两阶段。update 阶段策略 22.75 + critic 22.75 = 45.5 GB/卡；参考与奖励通常在推理侧或另计，不含在这 45.5 里。最贵、最灵活，在线采样能探索数据里没有的回答。
- **GRPO**：3 个模型（策略 84 + 参考 14 + 奖励 14 = 112 GB），每卡 22.75 + 1.75 + 1.75 = 26.25 GB；奖励换规则验证器时 24.5 GB。RLVR 的主力。

R1 的边界要说清：DeepSeek-R1 在数学/代码/STEM 等可验证任务上用「答案对不对 + 格式对不对」的规则奖励，不需要奖励模型；通用任务仍依赖奖励模型/偏好数据，不能概括成「R1 完全不用奖励模型」。

工程上，rollout 是一段 vLLM 服务：昇腾用 vllm-ascend，开 prefix caching 复用同一 prompt 的 KV，省 G−1 次 prefill，贵的只是 decode。接回推理侧：GRPO 训出的长 CoT 直接变成线上输出长度；输出从 1000 token 涨到 4000，输出段 KV cache 与 decode 总时长约 ×4，TPOT（每 token 输出时间）不按输出长度线性增长、但长上下文下会略升——训练侧的选择最后记在推理账单上。

## 一步一步理解

设定：7B、bf16、β=0.1；一个偏好对，chosen 100 token、rejected 80 token；logprob 用自然对数。假设已训练若干步（初始点见概念二，L=0.693）。

**第 1 步 算 log ratio（和）**：chosen 策略平均 logprob −0.5/token → 总和 −50；参考 −0.8/token → −80；log ratio_w = −50 − (−80) = +30。rejected 策略 −0.9 → −72；参考 −0.85 → −68；log ratio_l = −72 + 68 = −4。

**第 2 步 算 Δ 与 loss**：Δ = β(30 − (−4)) = 0.1 × 34 = 3.4。σ(3.4) ≈ 0.967，L = −log 0.967 ≈ 0.033。这一对已分得很开，梯度权重 σ(−3.4) ≈ 0.033，几乎不学了。

**第 3 步 梯度往哪走**：∇L = −β·σ(−Δ)·(∇log π_θ(y_w) − ∇log π_θ(y_l))，系数约 0.0033，方向是「抬 w、压 l」；参考模型没有梯度项，只当常数。

**第 4 步 长度偏置怎么发生**：若每 token 的 log ratio 是 chosen +0.30、rejected −0.05，模型把 chosen 从 100 加到 120 token 而每 token 质量不变，Δ 变 0.1×(36 + 4) = 4.0，loss 从 0.033 掉到 0.018。加长度比提质量便宜——偏置就是这么来的。

**第 5 步 显存对比**：DPO 24.5 GB/卡；PPO 45.5 GB/卡（只含策略+critic）；GRPO 带奖励模型 26.25 GB/卡、带规则验证器 24.5 GB/卡。64 GB 卡上 DPO 余量充足；32 GB 卡 DPO 只剩约 7.5 GB 给激活，偏紧，PPO 放不下。未计激活，粗算，数值存疑。

**第 6 步 GRPO 的优势与空转**：G=8，奖励 [1,0,1,1,0,0,1,0]，mean=0.5，std=0.5，A=[+1,−1,+1,+1,−1,−1,+1,−1]；全对组 std=0，A 全 0，没有梯度。G=8、每条 1024 token，一个 prompt 就是 8192 token rollout；512 个 prompt 一轮约 4.2e6 token 生成，DPO 一轮是 0。

**第 7 步 k3 手算**：某 token 上 π_θ=0.4、π_ref=0.8，比值 2，k3 = 2 − log 2 − 1 = 1 − 0.693 = 0.307；策略等于参考时比值 1，k3 = 1 − 0 − 1 = 0。k3 在比值 1 处归零，两侧都非负。

## 常见误区

**一、「DPO 不要奖励模型，所以也不要参考模型」。** 参考模型是 log ratio 的分母，也是「起点」的默认定义。去掉它，β·log π_θ 失去零点，等于把绝对 logprob 当奖励，模型会发现「越短总和越高」，直接退化成越短越好。标准做法是从 SFT 模型初始化并冻结，不是数学上「必须」；换参考要重算零点。

**二、「长回答 logprob 天然更低，所以 DPO 偏爱短答」。** log π(y) 确实随长度下降，但 DPO 比的是 π_θ/π_ref，参考会抵消一部分公共长度项；真正的偏置来自「和」不是「平均」，方向由数据与归一化决定，必须实测。常见现象是 chosen 越训越长、rejected 越训越短，SimPO 因此改用平均 logprob。

**三、「GRPO 省了 critic，训练一定更快」。** 省的是显存和 critic 的训练时间，换来 G 倍 rollout。decode 是访存瓶颈，生成一个 token 比一次训练前向贵得多，G 从 4 加到 64，rollout 时间线性上涨。

**四、「DPO 的 loss 足够低就是训好了」。** loss 只看相对差，绝对 logprob 掉下去它照样降。必须并看 logprob、长度与留出集人评。

## 一句话总结

DPO 把 KL 约束下最优策略的闭式解代回 Bradley-Terry，奖励就成了 β·log(π_θ/π_ref)，奖励模型被隐式化，代价是离线偏好对取代在线探索。三个故障各有独立机制：似然位移来自偏好对落在策略低概率区，长度偏置来自 token 求和而非平均，覆盖不足是离线数据的天花板；离线数据会放大前两者，但不是三者的总根。GRPO 把 critic 换成组内标准化优势，省 84 GB、花 G 倍采样，全对全错组梯度归零；三张账：DPO 98 GB 无 rollout、PPO 196 GB 两阶段（45.5 只含策略+critic）、GRPO 112 GB 大 G（26.25/卡），后者是 RLVR 主力，长 CoT 最终以输出长度和 TPOT 回到推理账单。

## 自测题

### 1. 从 KL 约束目标推到 DPO loss，写出关键三步，并说明 β·log Z(x) 为什么消失。
<details><summary>查看答案</summary>① 对目标 max_π E[r]−β·KL(π‖π_ref) 求闭式解：π*(y|x)=π_ref(y|x)exp(r/β)/Z(x)，Z(x)=Σ_y π_ref(y|x)exp(r(x,y)/β)。② 取对数移项：r(x,y)=β·log(π*/π_ref)+β·log Z(x)。③ 代入 Bradley-Terry：P(y_w≻y_l)=σ(r_w−r_l)，同一个 prompt 下 β·log Z(x) 在 r_w 与 r_l 里各出现一次、相减抵消，再把 π* 换成待训的 π_θ，得到 L_DPO=−E log σ(β log(π_θ(y_w)/π_ref(y_w))−β log(π_θ(y_l)/π_ref(y_l)))。三个前提：BT 偏好模型、闭式解存在、π_ref 近似最优策略。</details>

### 2. 初始点一个偏好对的 DPO loss 是多少？训练到 log ratio_w=+30、log ratio_l=−4、β=0.1 时，Δ、loss、梯度权重各是多少？把 chosen 从 100 token 拉到 120 token（每 token log ratio 不变）后 loss 怎么变？
<details><summary>查看答案</summary>初始点 π_θ=π_ref，两个 log ratio 都是 0，Δ=0，L=−log σ(0)=log2≈0.693。中途 Δ=0.1×(30−(−4))=3.4，σ(3.4)≈0.967，L≈0.033，梯度权重 σ(−3.4)≈0.033；乘 β 后梯度系数约 0.0033。chosen 拉到 120 token：每 token +0.30，总 log ratio 变 36，rejected 仍是 −4，Δ=0.1×40=4.0，σ(4.0)≈0.982，L≈0.018。每 token 质量没变，loss 却降了，这就是「和不是平均」带来的长度偏置。</details>

### 3. G=8，奖励 [1,0,1,1,0,0,1,0]，算 mean、std 与每个 A_i；全对组呢？GRPO 的组均值与 critic 的 V(s_t) 差在哪？
<details><summary>查看答案</summary>mean=4/8=0.5；每个奖励离均值 ±0.5，方差 0.25，std=0.5；A=[+1,−1,+1,+1,−1,−1,+1,−1]。全对组 8 个 1：mean=1、std=0，A 全 0，这条 prompt 没有梯度。critic 估的是 V(s_t)，即从某个前缀/已生成 token 出发的期望回报，逐 token 给基线；GRPO 用同组最终奖励的均值做 prompt 级基线，所有 token 共用，省掉逐 token 价值网络，但基线更粗，需要 G 倍采样来压方差。</details>

### 4. 日志：chosen logprob 从 −50 降到 −70，rejected 从 −40 降到 −65，loss 从 0.6 降到 0.2（β=0.3），长度不变。诊断是什么？还要打印什么、怎么处理？
<details><summary>查看答案</summary>两者绝对 logprob 同时下降，w 相对 l 的差从 −10 收窄到 −5（相对变好 5），但绝对概率都掉了 20/25；β=0.3 时 5 nat 的相对改善足以让 Δ 升约 1.5、loss 从 0.6 降到 0.2 左右。这是似然位移，不是学习率太大。偏好对落在策略低概率区，损失只要求 w 比 l 相对更高，最省力的解法是把 l 压得更狠，w 跟着掉。再打印 logprob 总和与每 token 平均、长度、参考模型下的概率；处理方向是补在线/同分布偏好对、过滤参考概率过低的样本、混入 SFT loss 或调小 β。</details>

### 5. 8 卡、ZeRO-2、参考 TP-8，算 DPO、PPO、GRPO（带奖励模型）每卡显存；PPO 的 45.5 为什么不能和 DPO 的 24.5 直接比？
<details><summary>查看答案</summary>策略 84 GB：权重 14 复制、梯度+AdamW 70/8=8.75，得 22.75；参考 14/8=1.75；奖励按 7B 同尺寸 14/8=1.75。DPO=24.5；GRPO=26.25（规则验证器 24.5）；PPO update=22.75+22.75=45.5。45.5 只含策略与 critic，参考/奖励在推理侧或另计；DPO 的 24.5 含参考，口径不同，不能直接相减。以上未计激活与临时 buffer，是粗算。</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| 隐式奖励 | DPO 里替代奖励模型的分数 r̂ = β·log(π_θ(y∣x)/π_ref(y∣x))，由策略与参考模型的对数比直接给出。 |
| 配分函数 Z(x) | 归一化常数 Σ_y π_ref·exp(r/β)，保证 π* 是合法概率分布；在成对比较里恰好抵消。 |
| 对数比（log ratio） | log(π_θ(y∣x)/π_ref(y∣x))，衡量策略相对起点对这条回答「抬了多少」。 |
| Bradley-Terry | 把成对比较变成 σ(分数差) 的模型，奖励模型与 DPO 共同的概率假设。 |
| σ（sigmoid） | σ(x)=1/(1+e^−x)，把任意实数压到 0–1，这里当「A 更好」的概率。 |
| β | 拴住策略的绳子：PPO 里是 KL 惩罚系数，DPO 里乘在 log ratio 上；越大越贴参考。 |
| 参考模型（π_ref） | 冻结的起点模型（通常是 SFT 模型），DPO 的分母、PPO 的 KL 锚点；换它要重算隐式奖励零点。 |
| 似然位移 | chosen 与 rejected 的 logprob 同时下降、loss 却下降的故障，偏好对落在策略低概率区时出现。 |
| 长度偏置 | 隐式奖励对 token 求和而非平均，长度直接进入分数；方向由数据决定，需实测。 |
| 覆盖 | 偏好数据覆盖的输入空间；范围外目标里没有对应项，是第三类故障（覆盖不足），会放大似然位移，但不是长度偏置的成因。 |
| 离线偏好对 | 训练前就固定好的 (x, y_w, y_l)，不再由当前策略采样，DPO 的信号来源。 |
| GRPO | 组相对策略优化：同一 prompt 采 G 条、组内标准化当基线，省掉 critic。 |
| 组内标准化 | A_i = (r_i − mean)/std，用同组 G 条的均值与标准差把奖励变成优势。 |
| critic（价值模型） | 估计 V(s_t)（从某前缀出发的期望回报）的模型，逐 token 给基线；GRPO 省掉的那 84 GB 就是它。 |
| rollout | 用当前策略在线采样回答的阶段，等价于一段推理服务；GRPO 的 G 倍成本在这里。 |
| k3 估计 | KL(π_θ‖π_ref) 的逐 token 估计 π_ref/π_θ−log(π_ref/π_θ)−1；只需 π_θ 采样、不需从 π_ref 额外采样，恒非负、方差低于 k1。 |
| RLVR | 可验证奖励的强化学习：奖励来自规则校验（如数学答案对错），不需要奖励模型；通用任务仍可能需要奖励模型/偏好数据。 |
| ZeRO-2 | 数据并行的一种显存优化：切梯度与优化器状态，权重在每卡复制一份。 |
| TP（张量并行） | 把单层权重切到多卡、层内做集合通信的并行方式；参考模型常用 TP 摊薄显存。 |
| prefix caching | 复用同一 prompt 已算好的 KV cache，避免 G 条回答重复 prefill。 |
| prefill / decode | 推理的两个阶段：prefill 一次算完整段 prompt，decode 逐 token 生成；decode 是访存瓶颈。 |
| vLLM / vllm-ascend | 高吞吐推理服务；vllm-ascend 是其在昇腾上的适配版，rollout 常用它起服务。 |

## 下一步

下一课「数据工程」把「覆盖」拆开：偏好数据从哪来、怎么清洗去重、标注一致性怎么保证、长度偏置怎么在数据侧先削一半。再往后，「推理模型与 test-time compute」把 GRPO 与 RLVR 接成长 CoT 的故事。

## 来源

- Direct Preference Optimization: Your Language Model is Secretly a Reward Model（DPO，arXiv:2305.18290） — https://arxiv.org/abs/2305.18290
- Training language models to follow instructions with human feedback（InstructGPT，arXiv:2203.02155） — https://arxiv.org/abs/2203.02155
- Proximal Policy Optimization Algorithms（PPO，arXiv:1707.06347） — https://arxiv.org/abs/1707.06347
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models（GRPO 出处，arXiv:2402.03300） — https://arxiv.org/abs/2402.03300
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning（RLVR，arXiv:2501.12948） — https://arxiv.org/abs/2501.12948
- Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study（似然位移，arXiv:2404.10719） — https://arxiv.org/abs/2404.10719
- SimPO: Simple Preference Optimization with a Reference-Free Reward（长度偏置，arXiv:2405.14734） — https://arxiv.org/abs/2405.14734
- Understanding R1-Zero-Like Training: A Critical Perspective（GRPO 的偏置分析，arXiv:2503.20783） — https://arxiv.org/abs/2503.20783
- Hugging Face TRL 官方文档：DPOTrainer（参考模型、β 参数） — https://huggingface.co/docs/trl/dpo_trainer
- Hugging Face TRL 官方文档：GRPOTrainer（GRPO 训练实现） — https://huggingface.co/docs/trl/grpo_trainer
- OpenRLHF 官方仓库（GRPO 训练示例） — https://github.com/OpenRLHF/OpenRLHF
- vLLM 官方文档：并行与扩容（rollout 阶段的张量并行） — https://docs.vllm.ai/en/latest/serving/parallelism_scaling/
