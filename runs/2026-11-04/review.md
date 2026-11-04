# 审校记录 · 2026-11-04 · pretrain-basics · 第 15 课 · 预训练在学什么：从下一个 token 到 scaling law

## verdict

revise

## summary

主体扎实：交叉熵/困惑度、移位标签、scaling law 的算例我逐项复核基本正确（0.805→PPL 2.24；1.69+0.354+0.536=2.580→13.2；44N；1.2e20 FLOPs→26h→833 元；e^-0.1=9.5%），Llama 3 配比（50/25/17/8）与 15T/8B 也对得上原文。但有 3 处硬伤必须改：第 7 步拿 1B 的训练费除 7B 的单请求价、z-loss 被安到 attention logits 上、把 20 token/参数 当成与算力无关的常数（文中自己引的拟合在 Chinchilla 算力下给出约 90）。另有因果 mask 缺失、E=1.69 普适化、自造定量结论、批量解码吞吐说法过强、自测题不完整等问题。

## issues

### 1

- **severity**: high
- **claim**: 「z-loss 则在目标里加 1e-4·(logsumexp(logits))²……PaLM 之后成为标配」写在「attention 的 logits = QKᵀ/√d，一旦数值变大，softmax 饱和成 one-hot」这一段里
- **问题**: PaLM 的 z-loss 惩罚的是输出层（lm_head）对词表 softmax 的 log-partition，即 1e-4·log(Z)²，作用是锚住 logits 的共同偏移、防输出 logits 数值漂移；它不作用在 attention logits 上，也治不了 attention softmax 饱和（那是 qk-norm 的职责）。文中把两者写成同一因果链，又并列「attention 熵崩塌」，初学者会直接记成「z-loss 管 attention」。（router z-loss 作用在 MoE 路由 logits，是另一件事，不能拿来支撑这里的说法。）
- **建议**: 拆成并列两条：qk-norm 归一化 Q、K 以钉住 attention logits 尺度；z-loss 惩罚输出 logits 的 logsumexp（先解释 logsumexp 即 log 配分函数），防的是 lm_head 漂移。删掉「z-loss 治 attention 饱和」的因果，并把「成为标配」降级为「PaLM 之后常用」。

### 2

- **severity**: high
- **claim**: 「833 ÷ 0.00193 ≈ 43 万次请求才回本训练」
- **问题**: 833 元是 N=1e9、D=2e10 的 1B 模型训练成本，0.00193 元却是第 14 课 7B 模型的单请求成本。用 7B 的推理单价去除 1B 的训练费，回本请求数被高估约 7 倍（同 token 数、memory-bound 粗算 1B 约 0.00028 元/请求，即约 300 万次）。紧接着「放大 10 倍」又跳到 10B，参照系与 7B 混在一起，读者无法复核这个结论。
- **建议**: 统一模型口径：或把训练成本换成 7B/140B token（C=6·7e9·1.4e11≈5.9e22 FLOPs，按同法约 4 万元）重算，或把 1B 的推理单价按 N 同比例折算（≈0.00028 元）再除。把「放大 10 倍」改成对同一模型做 N→2N 扫描，并注明只在同 token 数的 memory-bound 请求下成立。

### 3

- **severity**: high
- **claim**: 「Chinchilla 的结论是 N 与 D 等比例放大（各约 ∝ C^0.5）……工程口诀就是 D ≈ 20N」；「44 比 20 大是拟合常数的不确定性（20–50 浮动）」
- **问题**: 用文中自己给的 E/A/B/α/β 解 αA N^−α = βB D^−β 与 6ND=C：在 N=1e9、D=2e10 处最优确是 D≈44N，但在 Chinchilla 自己的算力 C≈5.9e23 处解得 N*≈32B、D*≈3.0T，D/N≈93；且 D/N ∝ C^((α−β)/(α+β)) ≈ C^0.097，随算力缓慢上升，不是常数。所以 20 并非这组常数推出的结论，「20–50 浮动」也不成立（该拟合给出的是 44→93→约 130）；「各约 ∝ C^0.5」同样与 α≠β 推出的 N∝C^0.45、D∝C^0.55 不符。真正成立的是：按同一拟合，70B/1.4T 与 32B/3T 的 loss 只差约 0.005，loss 面在 N–D 方向极平，所以 20/44/90 的实际差别很小。
- **建议**: 把 20 明确为「Chinchilla 实际训练点与 Approach 1–2 的经验口诀」，而非从 E/A/B/α/β 推出的常数；补一句 D/N 随算力缓升（本课算例约 44，Chinchilla 算力约 90+），并点明 loss 面平坦才是 20 好用的原因。把「令两个 ΔL 相等解得 44N」的一步代数写出来（否则无法跟算），删掉「20–50」。

### 4

- **severity**: medium
- **claim**: 「永远喂真实前缀……一次前向并行算出所有位置的预测」「位置 2 的输入永远是真实的『我』」
- **问题**: 并行的前提是 causal mask（下三角）：位置 t 的 attention 只能看 ≤t，所以一次前向里位置 2 看不到位置 3 的「学」。全文没有出现因果 mask，只有 loss 曲线一节提「mask 写错（padding 位参与 loss）」，初学者容易把 causal mask 与 padding mask 混为一谈，也想不通「并行」与「不泄漏未来」为何能同时成立。
- **建议**: 在移位标签后补三句：attention 加因果 mask，位置 t 只见 ≤t；teacher forcing 解决「喂什么前缀」，causal mask 解决「并行时不看未来」；并区分 causal mask（防泄漏）与 padding mask（防 pad 参与 attention/loss）。

### 5

- **severity**: medium
- **claim**: 「E 是不可约损失——真实文本自带随机性……的熵底」「约 1.69」「loss 低于 E 就是背下来了」
- **问题**: 1.69 是 Chinchilla 在其数据配比与分词器上、以 nats/token 拟合出的截距（外推到 N,D→∞），不是「文本」的普适熵底；换数据分布（代码/多语占比）或分词器，E 会明显不同。文中还把「低于 E」同时解释成「背下来」和「几乎总是 bug」，没有区分训练集、分布内验证集、分布外验证集，读者无法判断该信哪条。
- **建议**: 给 E 加口径限定（Chinchilla 数据+分词器、nats/token），说明它是拟合截距、随数据分布变化；判断记忆改用 train/val gap、去重后评测或同分布对照，而不是「loss < 1.69」这一条绝对阈值。

### 6

- **severity**: medium
- **claim**: 「把 10% 的顶级数据刷 20 遍，不如把 40% 的中等数据各过 2 遍」「这条也是定量依据」「低质数据……等价于抬高了 B/D^β」
- **问题**: Muennighoff 等的实测结论是「固定算力、数据受限时，≤4 epoch 的重复与全新数据 loss 差异可忽略，再多则收益衰减」，并没有做「10%×20 vs 40%×2」这组对比；这个数字结论与「等价于抬高 B/D^β」都是文中自造的推断，却被写成「定量依据/等价」，属于把推测写成事实。
- **建议**: 改成定性表述并标注为推断（如「按收益递减外推，极端重复多半不如扩数据，但具体倍数未经验证」），或换成论文真实做过的重复比例对比；「等价于抬高 B/D^β」降级为「可粗略理解为有效 D 变小」的类比。

### 7

- **severity**: medium
- **claim**: 「每吐一个 token 都要把约 2N 字节权重读一遍」「越不过『N 定死了每秒能搬多少字节』这个物理事实」「max-num-seqs 和量化只能救一点点」
- **问题**: 这只在 batch=1、memory-bound 时成立。批量解码时权重每步只读一次、摊到 B 个 token 上，吞吐转为 compute-bound，上限约 峰值FLOPS×MFU/(2N)：7B、3 TB/s 下单流约 214 token/s，而 160 TFLOPS 有效算力下可到约 1.1 万 token/s。因此 max-num-seqs 不是「救一点点」，而是能把吞吐提一个量级；把「N 定死每秒字节数」说成物理事实，会让人误判容量与成本。
- **建议**: 区分两个 regime：单请求/低 batch 是 memory-bound（成本 ∝ N），高 batch 是 compute-bound（吞吐 ∝ 有效FLOPS/N）；说明 per-token 成本随 batch 下降，N 决定的是两者交界。再连回第 14 课的「有效 token 数 × 利用率」。

### 8

- **severity**: medium
- **claim**: 「课前自测」4 题；全篇无课后自测题
- **问题**: 课前自测没有答案，弱基础读者做完无法自检（第 1、3 题都要算 PPL）；模块六件套里的「自测题」缺失，只有「一句话总结」和「下一步」，起不到检验作用。
- **建议**: 给课前自测补折叠答案与一句解析（0.693 / 2.23；1.71 与 20；7.39→6.69，约 9.5%；先查该 step 的 batch 与数据）；再补 5–6 道课后自测（手算 PPL、scaling law 代入、loss 曲线判读、bf16 vs fp16、20 token/参数 的含义），附答案。
