---
date: "2026-11-26"
title: "第 23 课 · MoE 训练：专家并行怎么排、路由为什么抖、dropless 付什么账"
domain: "训练"
module_id: "moe-training"
module: "MoE 训练"
status: "已学"
lesson_no: 23
goals: [
  "画出一次 MoE 前向/反向里 all-to-all 的完整通信模式——哪一步发、发给谁、每 token 发多少元素、反向为何要再反向发一遍，并说清 EP 与 DP/TP 的组怎么切、通信量各随什么增长。",
  "从 loss 曲线、专家负载分布、gate 概率幅值三个信号认出路由不稳定（坍缩、震荡、饱和冻结），并说清 aux loss / 噪声 / z-loss / bias 均衡在训练动态上各自改变了什么、代价是什么。",
  "解释 dropless 在训练侧的实现与代价（token 重排、按最忙专家定每步缓冲、缓冲区与显存预留、负载不均导致卡空转），并说清训练与推理对 MoE 的约束为何不同（训练要梯度与稳定、推理要延迟与显存）。"
]
tags: ["MoE", "专家并行", "all-to-all", "路由稳定性", "dropless", "DeepSeek-V3"]
---

# 第 23 课 · MoE 训练：专家并行怎么排、路由为什么抖、dropless 付什么账

> 学完今天，你能：
> 1. 画出一次 MoE 前向/反向里 all-to-all 的完整通信模式——哪一步发、发给谁、每 token 发多少元素、反向为何要再反向发一遍，并说清 EP 与 DP/TP 的组怎么切、通信量各随什么增长。
> 2. 从 loss 曲线、专家负载分布、gate 概率幅值三个信号认出路由不稳定（坍缩、震荡、饱和冻结），并说清 aux loss / 噪声 / z-loss / bias 均衡在训练动态上各自改变了什么、代价是什么。
> 3. 解释 dropless 在训练侧的实现与代价（token 重排、按最忙专家定每步缓冲、缓冲区与显存预留、负载不均导致卡空转），并说清训练与推理对 MoE 的约束为何不同（训练要梯度与稳定、推理要延迟与显存）。

## 课前自测

1. 一层 256 个 routed 专家、每 token 选 8，64 卡各放 4 个：每卡驻留几个专家？均衡时这一步这张卡要算多少个专家-token 对？
2. all-reduce 求和、all-to-all 搬运——哪个有归约？各自搬的是什么？
3. dispatch 和它后面的 grouped GEMM 能换顺序吗？为什么？
4. decode 每步只来 1 个 token，EP 的固定通信开销摊给几个 token？
5. capacity factor 设 1.0 时一定有 token 被 drop，那 dropless 是不是就等于把 C 设得很大？

## 为什么需要它

上一课把 MoE 的账算到了 router（路由器，给每个专家打分的那个小矩阵）的 top-k 上，并埋了两条线：all-to-all 成了 EP 的新瓶颈、decode 又把省下的算力还回去。今天收这两条线。

一句话立论：**MoE 训练的瓶颈常不在矩阵乘而在 all-to-all，事故也常不是 NaN，而是路由悄悄坏掉**——前者是带宽账，后者是仪表盘账。

## 核心概念

### 一、EP 切的是专家集合

**定义**：EP（专家并行）把一层 N 个 routed 专家（由 router 打分、每 token 只选 k 个的专家）切成 N/EP 份分到多卡，每卡只驻留并算自己那几份，token 靠 all-to-all（按目的地多发多收、不做归约）送过去；shared 专家（共享专家）永远激活、不参与 top-k（V3 每个 MoE 层 1 个）。

**直觉**：TP（张量并行）是拆机床、四人接力同一零件，段间 all-reduce 对齐；EP 是四条完整产线，零件自己走过去。

**机制**：DP（数据并行）复制模型（纯 DP 下每卡都有全部专家，叠 EP 时复制的是 EP 分片后的整份模型）、梯度 all-reduce（各卡求和/取平均后广播）；TP 切单个专家矩阵、段间 all-reduce；EP 切专家集合、通信从归约变搬运。一张卡同时在 DP、TP、EP 三个组里，组员与通信原语都不同。V3 是 16-way PP（流水线并行）× 64-way EP（跨 8 节点）× ZeRO-1 DP（只分片优化器状态），256 个 routed 专家摊到 64 卡，每卡 4 个。

**例子与坑**：1024 卡排 EP=32×TP=4×DP=8；每卡在 EP 组换 token、TP 组做 all-reduce、DP 组同步梯度。EP=48 时 256/48=5.33，除不尽要 padding 或冗余专家；vLLM 开 EPLB 时还要求（routed＋冗余）能被 EP size 整除。全局均衡≠卡间均衡——某卡 4 个专家都热门，它就是慢卡。

### 二、训练期的四跳通信

**定义**：前向两跳（dispatch、combine）＋反向两跳（combine 的转置、dispatch 的转置）＝**每层每步 4 次 all-to-all**；重排与 grouped GEMM 不是 all-to-all。前向把激活搬过去，反向就要把对应梯度搬回来，所以每次前向 all-to-all 在反向都有一次转置。

**直觉**：跨城快递——按目的地码垛发出，收件地分堆，再按寄件地址装车退回，最后插回原序乘 gate（router 给的权重）。

**机制**：①重排：按目标专家/卡排序、拼连续块并计数。②dispatch：多对多 all-to-all，每 token 最多 k 份 d 维向量（同卡可合并）；**不到齐，本卡 GEMM 不能开始**，所以卡在关键路径。③grouped GEMM：各专家权重不同、长度不等，拼不成普通 GEMM，只能打包成一次 kernel；单卡耗时≈本卡各专家工作量之和，卡间 straggler 看最忙的卡（4 个专家总负载最大），只有按 max 段 padding 的实现才由最忙专家定。④combine：发回发方、还原顺序、按 gate 加权；同卡多专家结果先本地合并，体积从 k·d 压到“目标卡数·d”。⑤反向：combine^T 退回梯度 → 专家 GEMM 反向 → dispatch^T 退回输入梯度；router 梯度走 all-reduce，不占 all-to-all。每次 all-reduce 约 d 个元素、每跳 all-to-all 约 k·d，二者都随 token 数线性涨。

**例子**：d=4096、bf16、k=4 时，每 token 每层四跳 4×k×d=65536 个元素，最坏 128 KiB；V3 的 k=8、d=7168 是 448 KiB，约 3.5 倍。

### 三、节点受限路由

**定义**：每个 token 最多送到 M 个节点。V3 取 M=4：节点分＝该节点上最高的 K_r/M=2 个亲和分数（affinity score，router 给 token-专家对的打分，V3 用 sigmoid）之和；取前 4 个节点，再在其中取 top-8。

**直觉**：跨城贵、同城便宜；先圈 4 个城市，再在城里挑 8 家店。

**机制**：不限制时，8 个专家最多落在 8 个不同节点（上界 min(k, 节点数)=8）；限制后必落在 ≤4 个节点内，目的地址数被常数封顶。收益不是单次更快，而是**流量有上界、可切片、可重叠**——V3 把 all-to-all 藏进计算靠 DualPipe 流水重叠，节点受限路由是前提之一。代价：M 越小越好通信，越难均衡。

**例子**：每节点 32 个专家、每 token 选 8，8 个专家被压进 4 个节点；不限制时最坏跨 8 个。

### 四、路由不稳的三张脸

**直觉**：分诊台三种坏法——所有病人都挤一个科（坍缩）、病人来回换科（震荡）、分诊台只认一个科（冻结）。

| 表现 | 症状 | 可观测信号 |
| --- | --- | --- |
| 坍缩 | 少数专家吃掉大部分 token、其余梯度≈0，退化成更小的 dense（稠密）模型 | fᵢ（分到专家 i 的 token 占比）的 max/mean 持续远离 1（经验上长期 >2–3 值得查，阈值需按模型标定），或归一化熵持续下降 |
| 震荡 | 相邻 step/检查点的 top-k 集合频繁翻转，loss 出尖刺 | 路由翻转率：同一批固定 probe（探针）样本上，相邻检查点 top-k 集合的 Jaccard 距离 1−\|A∩B\|/\|A∪B\| |
| 饱和冻结 | logits 幅值膨胀、gate 接近 one-hot（只一个位置为 1）、top-k 冻住（看着稳，其实死了） | 逐层 log-sum-exp(logits) 均值、max gate 概率 |

**为什么只抖不报错**：前向数值合法、反向仍有梯度，坏在有效容量变小、梯度方差变大、被丢或闲置那份像乘性噪声，结果是 loss 抖动、收敛变慢，而不是直接报错。最明显的硬故障是数值溢出/NaN；它常常不抛异常，只让 loss 变 NaN。

**先查数据与初始化**：数据重复或模板单一可能让表示聚团、router 更难区分；LR 过大或 warmup 过短可能让 router 早期锁死；初始化尺度过大可能让 logits 一上来就很大。这三样不改，加正则只是压症状。

### 五、稳定手段的靶点与代价

| 手段 | 靶点 | 代价 |
| --- | --- | --- |
| aux loss（α·N·ΣfᵢPᵢ） | 把 fᵢ 与 gate 概率拉均匀 | 与语言建模目标拉扯；均≠对；α 要调（Switch 常用 0.01） |
| router 噪声 | 给 logits 加噪声，早期探索，防第一步锁死 | 引入随机性，推理必须关掉 |
| z-loss | 惩罚 logits 的 log-sum-exp 平方，防饱和、稳数值 | 不直接优化均衡，但会改 logits 尺度、间接改分配；系数大压制 router（ST-MoE 用 1e-3） |
| bias 调整（V3 无辅助损失均衡） | 均衡，但不进主目标 | 多一组调步长的超参：过载减 γ、欠载加 γ，前 14.3T token 取 0.001、后 500B 置 0；bias 只参与“选谁”，gate 仍用原始分数 |
| 梯度裁剪＋小 LR warmup | 降低早期把 router 推到饱和的风险 | 拖慢早期收敛；阈值敏感（V3 取 1.0） |

V3 并非完全不用 aux loss：仍留一个极小的序列级均衡损失（α=1e-4），只防单条序列内的极端不均。

### 六、dropless 与两端分道

**定义**：dropless 不设固定 capacity（每专家容量 ⌈(T·k/N)·C⌉，T 是一步 token 数、C 为容量系数），用“重排＋按实际收到数动态定缓冲”代替按 C 截断。

**直觉**：固定容量是给每间诊室发固定号源、发完退号；dropless 是不发号，但候诊区得按最挤的诊室留够座位。

**机制与代价**：①先交换各目的地计数（小 all-to-all 或 all-gather）；计数回主机再分配缓冲会同步、打断 CUDA graph（把 kernel 调用录成图重放）。②缓冲按最坏单专家负载预留（或按块稀疏分块，MegaBlocks 即此路），均不均都占着。③静态放置下卡间不能偷活。④好处是不因 drop 引入梯度偏差：被截断 token 少算一项梯度；被丢的 token 与路由分布、实现如何处理超容有关，不能一概说成“热门方向的高频模式”，但偏差方向确定。

**两端为何分道**：训练要无偏梯度＋可预测吞吐，所以 EP 尽量大、留超节点、M 封顶，变长可接受。推理要确定延迟：decode 吃带宽、每步 token 极少、固定开销摊不开，变长还破坏图捕获；vLLM 默认不做训练期的容量截断，其 low-latency 后端用 masked layout（按预算预留固定形状、无效位置 mask 掉）；并行是 EP 叠 DP（EP_SIZE = TP_SIZE × DP_SIZE），再叠 EPLB（专家并行负载均衡器，重排专家映射并加冗余专家）。训练侧也能重排，但要在 checkpoint 边界搬权重与优化器状态、打断流水线，代价高，一般不做或低频做；推理权重稳定，可频繁做。

**昇腾落点**：`--enable-expert-parallel` 开 EP，all-to-all 走 HCCL（华为集合通信库）；先单节点再跨节点。变长切分的 all-to-all 支持随 CANN（昇腾软件栈）版本变化，存疑，以对应版本文档为准。

## 一步一步理解

教学假设：d_model=4096、bf16（8 KiB/向量）、k=4、N=128、EP=32（4 专家/卡）、1024 token/卡/micro-batch、61 层。**只为算通信账，不追求可部署**：4 专家×61 层×176M×2 B≈86 GB/卡，已超单卡 80 GB。

**第 1 步 每 token 每层**：最坏情况（k 个专家不同卡），dispatch 32 KiB＋combine 32 KiB＋反向两跳各 32 KiB＝128 KiB；同卡合并会略低。

**第 2 步 每卡每层**：按最坏情况且负载均衡时，四跳各发 32 MiB（dispatch 发本地 token，combine 发替别人算的结果，反向两跳对应梯度），共 128 MiB；全组 32 卡一层 4 GiB。

**第 3 步 整步**：4 GiB×61≈244 GiB≈262 GB。

**第 4 步 换时间**：单卡单层 0.134 GB；超节点（高带宽互联的一组机器）内按每卡单向 400 GB/s→0.34 ms；跨节点按每卡 400 Gb/s IB（InfiniBand，50 GB/s）→2.7 ms。

**第 5 步 整步通信**：0.34×61≈21 ms；2.7×61≈164 ms。

**第 6 步 和算力比**：每 token 激活 4×176M=705M 参数，FLOPs=2×705M≈1.41 GFLOP。均衡时每卡算 4096 个专家-token 对——本卡 1024 token 只有 1024×4×4/128=128 对落在本地，其余由别卡送来；4096×2×176M≈1.44 TFLOP，按假设有效 400 TFLOPS→3.6 ms/层。

**第 7 步 结论**：在这笔账里，EP 组的物理边界比 EP size 更重要。超节点内 0.34÷3.6≈9%，跨节点 2.7÷3.6≈75%。9% 要靠另一个 micro-batch 或 PP 阶段的计算叠才藏得住，单 micro-batch 串行仍占关键路径；75% 已接近算力时间，只有把通信几乎完全叠进计算才不拖慢，这靠 M 封顶＋DualPipe 式重叠。同样 32 卡，一个超节点 9%，摊到 32 个节点（每节点 1 卡）75%。

在昇腾上复现：先 8 卡单节点跑 EP=8 量 all-to-all 占比，再扩两节点 16 卡，看跨节点流量是否按 400/50≈8 倍的带宽比上升；涨幅接近“全部按跨节点带宽”的预测，说明没重叠上。

## 常见误区

**一、「EP 越大越好，反正显存能摊开。」** 错。EP 8→64，单卡专家 32→4，显存摊开了；但 k 份激活多跳目的地、固定开销涨、每卡 token 少让 GEMM 更碎，跨出超节点带宽又掉一个量级。最优点是刚好装下模型、又不跨超节点。

**二、「drop 掉 1% 的 token 无所谓，稀疏模型本来就随机。」** 错。被截断 token 在超容专家少算一项梯度，偏差方向确定；但“被丢的一定是高频模式”是过度推断，取决于路由分布与实现如何截断。dropless 也不等于 C 很大：C 大仍是固定上限、只是少丢；dropless 取消上限，代价是预留、不能偷活、打断图捕获。

**三、「路由抖就是 router 没设计好，先加 aux loss。」** 错。常见怀疑对象是数据重复/模板单一、LR 大或 warmup 短、初始化尺度大；不改这些，aux loss 只压症状，还可能压掉能力——它逼的是“均匀”，不是“正确”。

## 一句话总结

MoE 训练＝按路由把 token 搬到专家卡（前向两跳＋反向两跳＝4 次 all-to-all，卡在关键路径）＋用超节点和 M 封顶把通信关进预算＋盯住路由三病（坍缩看 fᵢ、震荡看翻转率、冻结看 log-sum-exp）＋dropless 换“不 drop”的梯度、代价是预留与空转；训练要无偏与吞吐，推理要确定延迟，同一份权重因此两套并行。

## 自测题

### 1. 四跳怎么数？

写出前向与反向各有哪些 all-to-all，并说明为什么是 4 次而不是 2 次或 6 次。

<details>
<summary>查看答案</summary>

前向只有 dispatch 与 combine 两次；重排是卡内排序/散射，grouped GEMM 是计算，都不是 all-to-all。反向有 combine 的转置（梯度从发方回专家卡）与 dispatch 的转置（输入梯度从专家卡回发方）两次。合计 4。router 参数的梯度走 all-reduce，不占 all-to-all。

</details>

### 2. 容量与 dropless

capacity 公式怎么写？C=1.0 为什么一定有人被 drop？dropless 是不是把 C 设得很大？

<details>
<summary>查看答案</summary>

capacity=⌈(T·k/N)·C⌉，T 是一步 token 数、N 专家数、k 是每 token 选几个。C=1.0 时容量等于平均负载，而真实负载有波动，必有专家超容。dropless 不是调大 C：它取消固定上限，先交换各目的地计数，再按实际收到数动态分配缓冲；代价是多一次计数同步、按最坏单专家负载预留显存、静态放置下卡间不能偷活，变长还会打断 CUDA graph。

</details>

### 3. 节点受限路由

M=4 时节点怎么选？不限制时一个 token 的专家最多跨几个节点？收益与代价各是什么？

<details>
<summary>查看答案</summary>

节点分＝该节点上最高的 K_r/M 个（V3 是 8/4=2 个）亲和分数之和；取前 4 个节点，再在其中取 top-8。不限制时最多跨 min(k, 节点数)=8 个节点；限制后 ≤4 个。收益是跨节点目的地址数被封顶、流量可预测，能切片并与计算重叠；代价是牺牲路由自由度，M 越小越难均衡。

</details>

### 4. 路由三病各看什么？

坍缩、震荡、饱和冻结各给一个可观测信号，并说出加 aux loss 之前应先查什么。

<details>
<summary>查看答案</summary>

坍缩看 fᵢ 的 max/mean 是否持续远离 1、归一化熵是否持续下降；震荡看固定 probe 样本上相邻检查点 top-k 集合的 Jaccard 距离（1−|A∩B|/|A∪B|）；冻结看逐层 log-sum-exp(logits) 均值与 max gate 概率是否持续上升、top-k 是否长期不变。加 aux loss 前先查数据（重复、模板单一）、优化（LR 过大、warmup 过短）、初始化（logits 一开始就很大）；这些不改，aux loss 只是压症状。

</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| 专家并行（EP） | 一层 N 个 routed 专家切成 N/EP 份分到多卡，token 靠 all-to-all 送过去算。 |
| 数据并行（DP） | 复制整份模型/专家分片，算不同 batch，梯度 all-reduce 同步。 |
| 张量并行（TP） | 把单个专家的矩阵切到多卡，段间 all-reduce 汇总。 |
| all-reduce | 各卡把同形状张量求和/取平均后广播回所有卡；有归约，常用于 DP 梯度与 TP 段间。 |
| all-to-all | 每卡按目的地发自己的一块、同时收别人的；多对多、不归约。 |
| dispatch / combine | 前向第一跳把 token 送去专家卡，第二跳把结果送回发方。 |
| routed / shared 专家 | routed 由 router 选 top-k；shared 永远激活、不参与选择（V3 每个 MoE 层 1 个）。 |
| 亲和分数 | router 对 token-专家对的打分；节点受限路由按节点上最高的 K_r/M 个之和选节点。 |
| grouped GEMM | 把各专家权重不同、长度不等的多个小矩阵乘打包成一次 kernel 调用。 |
| 超节点 | 由高带宽低延迟互联连成的一组机器，内部带宽比跨节点网络高一个量级。 |
| 节点受限路由 | 每个 token 只送到至多 M 个节点上的专家，给跨节点流量封顶。 |
| fᵢ | 被路由到专家 i 的 token 占比。 |
| 路由翻转率 | 固定 probe 样本上相邻检查点 top-k 集合的 Jaccard 距离。 |
| log-sum-exp | log Σe^{z_i}，logits 幅度的平滑度量；持续上涨说明 softmax 在饱和。 |
| aux loss | α·N·ΣfᵢPᵢ，逼负载与 gate 概率均匀，与语言建模目标拉扯。 |
| z-loss | 惩罚 logits 的 log-sum-exp 平方；稳数值、防饱和，会间接改变分配。 |
| 无辅助损失均衡 | 给每个专家一个偏置 bᵢ 只参与“选谁”，按负载方向以步长 γ 调整。 |
| capacity factor | 容量系数 C，每专家容量＝⌈(T·k/N)·C⌉。 |
| dropless | 不丢 token：重排加按实际负载动态定尺寸的缓冲，代替固定容量。 |
| EPLB | vLLM 的专家并行负载均衡器：重排专家映射、加冗余专家抗倾斜；要求 (routed＋冗余) 能被 EP size 整除。 |
| masked layout | decode 低延迟后端按固定 token 预算预留形状、用 mask 标有效行，换取图捕获。 |

## 下一步

下一课「评估与红队」把仪表盘从训练过程抬到模型输出：评测集怎么设计、benchmark 怎么读、红队怎么找失效模式。今天看 fᵢ、翻转率这类过程指标，下一课看能力与安全的结果指标；MoE 还要记录激活参数与路由分布，否则平均分好看、长尾却塌了也发现不了。

## 来源

- DeepSeek-V3 Technical Report（256 routed + 1 shared、top-8、M=4、γ 前 14.3T 取 0.001 后 500B 置 0、序列级 aux loss α=1e-4、梯度裁剪 1.0，arXiv:2412.19437） — https://arxiv.org/abs/2412.19437
- DeepSeek-V2（device-limited routing 前身，arXiv:2405.04434） — https://arxiv.org/abs/2405.04434
- DeepSeekMoE（细粒度专家与共享专家，arXiv:2401.06066） — https://arxiv.org/abs/2401.06066
- Switch Transformers（aux loss 与容量因子，arXiv:2101.03961） — https://arxiv.org/abs/2101.03961
- GShard（capacity / drop 的工程源头，arXiv:2006.16668） — https://arxiv.org/abs/2006.16668
- ST-MoE（router z-loss，arXiv:2202.08906） — https://arxiv.org/abs/2202.08906
- MegaBlocks（dropless 的块稀疏实现，arXiv:2211.15841） — https://arxiv.org/abs/2211.15841
- Megatron-LM（张量并行的经典做法，arXiv:1909.08053） — https://arxiv.org/abs/1909.08053
- vLLM 文档：Expert Parallel Deployment（EP_SIZE = TP_SIZE × DP_SIZE、deepep_low_latency 的 masked layout、EPLB 整除约束） — https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- vLLM 文档：Fused MoE Kernel Features — https://docs.vllm.ai/en/latest/design/moe_kernel_features/
- PyTorch 文档：torch.distributed（all_to_all_single 的变长 split sizes） — https://pytorch.org/docs/stable/distributed.html
- 昇腾社区 CANN / HCCL 文档（AlltoAll、AlltoAllV） — https://www.hiascend.com/document
