# 第 13 课 · Prefill/Decode 分离与长上下文

> 学完今天，你能：
> 1. 解释 prefill 与 decode 为什么瓶颈不同
> 2. 说出 PD 分离的收益与代价
> 3. 估算长上下文请求的 KV 显存占用

## 课前自测

1. 一个请求「输入 8000 token、输出 500 token」，你愿意让它和 31 个「输入短、输出长」的请求挤在同一张卡上吗？
2. 同一张卡、同一个模型，上下文从 8K 涨到 128K，能同时服务的请求数大约变成原来的几分之几？
3. 两张卡之间搬 1 GiB 数据，链路 25 GB/s，要多久？这段算进首 token 延迟还是每 token 延迟？

第 2、3 题在「一步一步理解」里会亲手算出来。

## 为什么需要它

先回顾《serving-basics》：**Prefill（预填充）** 把整段 prompt 一次性喂进模型，并行算出所有位置的 Key/Value 塞进 **KV cache（键值缓存）**；**Decode（解码）** 每次只算一个新 token，直接复用缓存。省了算力，代价是「上下文」变成必须一直占着显存的实体。

两段的工作性质相反，却挤在同一张卡上：

- Prefill 是**算力密集型**：8K token 一次前向，矩阵单元几乎跑满，只产出 1 个有用的 token。
- Decode 是**显存带宽密集型**：每个请求每步只算 1 个 token，但一张卡会把几十个请求拼成一批；权重只读一次摊给整批，吞吐随批增大而升，直到带宽打满，代价是每步变慢。

混跑时，任何新请求的长 prefill 都会把正在解码的一批请求整体卡住几百毫秒到一秒，用户体感是「打字突然停住」。

两个指标先立在这：**TTFT**（Time To First Token，请求到首 token 的时间，由排队、prefill、KV 传输决定）、**TPOT**（Time Per Output Token，首 token 之后平均每 token 的时间，由 decode 决定）。

PD 分离（Prefill/Decode Disaggregation）就是把这两人分到不同屋子干活，中间只递半成品。它买到的不是「更快」，而是**两段资源独立扩缩容、两段延迟独立控制**。

## 核心概念

### PD 分离

定义：prefill 与 decode 部署成两套独立实例（P / D 节点），请求在 P 上做完 prefill 得到 KV cache，再搬到 D 继续 decode。

直觉：餐厅前厅与后厨。混在一起时一桌大单就堵死后厨；拆开后大单只堵前厅，后厨照常出菜。

机制：代理先选 P；P 只跑到首 token 就停（vLLM 用 `max_tokens=1`），KV 暂不释放，元信息回给代理；代理再选 D，D 把 KV 拉来或等它推来，decode 完通知 P 释放。P、D 可各配并行策略（张量并行 TP、数据并行 DP、专家并行 EP——EP 指按专家切分，面向 MoE），这是收益来源之一。

代价：多一次网络传输、多一跳代理、要处理传输失败与超时。显存上多的**不是「权重翻倍」**：每张卡本来就各存一份权重，总卡数固定时集群权重总量不变，翻倍只在最小 1P+1D 部署；额外峰值是传输期间同一份 KV 在 P、D 各驻留一份，加上 P 侧延迟释放。

例子：白天客服短 prompt + 长回复（D 吃紧），夜间批量摘要反过来（P 吃紧）；分离后 P 池白天缩、夜间扩，D 池不动。

### KV 传输

定义：把 P 的 KV cache 按层、按 block（PagedAttention 的固定大小 KV 分页单元）搬进 D 的显存。

直觉：搬家。东西越多、路越窄越慢，而这段时间直接算在用户等首 token 里。

机制：三条路——① 同机跨卡走片间互联（昇腾上是 HCCS，经 HCCL 通信库）；② 跨机走 RDMA 网卡，如 RoCE（RDMA over Converged Ethernet）；③ 落到 KV 池（CPU 内存 / SSD）。公式只有一条：`传输时间 ≈ KV 大小 ÷ 有效带宽`。胜负手是**重叠**，且在 P 侧：vllm-ascend 的 MooncakeLayerwiseConnector 里，P 算完一层就推一层，与 P 自己下一层的 prefill 计算重叠；D 预分配 KV 后等待，全部层到齐才被通知 decode——decode 每步要读全部层的 KV，D 不可能「收到前几层就先算」。

例子：8K 请求的 1 GiB KV，200 Gbps 上 43 ms，25 Gbps 上 344 ms——同一次分离，链路差 8 倍，TTFT 差 300 ms。

### 资源配比

定义：P 与 D 的数量比（含各自 TP/DP 配置），记作 P:D。

直觉：停车场出入口的车道数。入口堵就加 P，出口堵就加 D。

机制：P 的工作量 = 到达率 × 单请求 prefill 时间；D 的工作量 = 输出 token 速率 ÷ 单卡 decode 吞吐；各除以单卡能力就是卡数。P:D 由**输入/输出长度分布**决定，必须实测再配，并留 burst 余量。

例子：同为 10 请求/秒，输入 8K 时 P:D ≈ 2.6:1，输入 512 时 ≈ 1:1。

### 长上下文

定义：单请求上下文远超常见规模，如 32K、128K 甚至 1M token。

直觉：一张书桌。书越多摊得越满（KV 占显存），能同时摊开的书越少（并发下降），翻书找答案也越慢（每步读的 KV 更多）。

机制：四个随长度 S 增长的量——① KV 显存 ∝ S，挤掉并发；② prefill 的注意力计算 ∝ S²（线性层只 ∝ S），长 prompt 变成注意力主导；③ decode 每步要读的 KV ∝ S，TPOT 持续变差；④ KV 传输量 ∝ S。

**②+③ 才是长上下文最值得分离的原因**：prefill 阻塞的绝对时长随 S² 放大，decode 又随 S 变慢、更经不起打断；① 说明显存压力。④ 是**代价**，必须靠高速互联和重叠压回几十毫秒，否则收益被 TTFT 吃掉。

例子：128K 请求单条吃 16 GiB，64 GB 卡只剩 2 个并发位；其 prefill 里注意力项约是线性项的 4.3 倍；这条 KV 搬到另一台机器，200 Gbps 要 690 ms，25 Gbps 要 5.5 s。

### SLO

定义：SLO（Service Level Objective）是可量化的服务承诺，如「TTFT 的 P90 ≤ 2 s，TPOT 的 P90 ≤ 50 ms」。P90 指 90% 请求满足该值，P99 是最慢 1% 请求的分位；用分位而非均值，是为了不让少数慢请求被平均掉。

直觉：外卖承诺 30 分钟送达。你考核超时率，不是平均时长。

机制：PD 分离把两个 SLO 解耦——TTFT 归 P 管（排队 + prefill + KV 传输），TPOT 归 D 管（批大小与显存带宽），可单独加 P 卡或 D 卡。整体效率看 **goodput（有效吞吐）**：单位时间内**同时满足所有 SLO** 的请求数，Mooncake 的调度目标正是它。

例子：每天 100 万请求、5% 超时，goodput 95 万/天；把 P90 从 2.4 s 压到 1.8 s、超时率降到 1%，goodput 99 万/天——这个差值才该拿去要预算。

## 一步一步理解

统一假设（都可手算）：8B 级模型、32 层、32 个 Q 头、**8 个 KV 头**（GQA，Grouped-Query Attention：多个 Q 头共用一组 K/V 头，KV 头数远少于 Q 头数）、head_dim 128、BF16（2 字节/元素）；单卡 HBM（高带宽显存）64 GB、带宽 1.6 TB/s、prefill 有效算力 150 TFLOPS（约峰值 × 40% 利用率，即 MFU）。卡型规格请用 `npu-smi info` 与华为官方数据核对。下文把 1 GiB 近似当 1 GB（误差约 7%）。

**第 1 步：一个 token 的 KV 有多大？**
每层存 K、V 两份，每份 = KV 头数 × head_dim：

```
2（K,V）× 32 层 × 8 头 × 128 维 × 2 字节 = 131,072 字节 = 128 KiB
```

记住：**每 1000 token ≈ 128 MiB**。

**第 2 步：不同上下文的单请求 KV**

| 上下文 | 单请求 KV |
| --- | --- |
| 8K | 1 GiB |
| 32K | 4 GiB |
| 128K | 16 GiB |

**第 3 步：显存账 = 并发上限**
64 GB 卡，权重 8B × 2 字节 = 16 GB，留 8 GB 给激活、workspace 和框架，可用 KV 约 40 GB：

- 8K：40 ÷ 1 ≈ **40 个并发**
- 128K：40 ÷ 16 ≈ **2 个并发**（还剩 8 GB 空着，浪费）

上下文涨 16 倍，并发掉约 20 倍。

**第 4 步：prefill 要多久？**
`2 × 参数量 × token 数` 只算**线性层**；长 prompt 还要加注意力项 `4 × 层数 × S² × d_model`（d_model = 32 头 × 128 = 4096；QKᵀ 与 AV 各 2·S²·d_model）：

```
线性层：2 × 8e9 × 8192 = 1.31e14 FLOP = 131 TFLOP
注意力：4 × 32 × 8192² × 4096 ≈ 3.5e13 FLOP = 35 TFLOP
合计 166 TFLOP ÷ 150 TFLOPS ≈ 1.1 s

512 token：线性层 8.2 TFLOP，注意力仅 0.14 TFLOP → ≈ 0.055 s
```

只算线性项会把 8K 低估约 21%，所以后面一律用 **1.1 s**。两项相等的位置在 S ≈ 30K：短 prompt 线性层主导，长 prompt 注意力主导——这就是「长上下文 prefill 特别贵」的来源。

**第 5 步：decode 一步要多久？**
每步把全部权重读一遍，再加上这一批所有请求的 KV：

```
权重 16 GB ÷ 1.6 TB/s ≈ 10 ms   ← 单请求的理论下限
批量 32 不是拍的：40 GB KV 预算 ÷ 1 GiB ≈ 40 条，留余量取 32
32 个 8K 请求：16 GB + 32 × 1 GiB = 48 GB ÷ 1.6 TB/s ≈ 30 ms/步
              → 吞吐 ≈ 32 ÷ 0.030 s ≈ 1067 token/s
```

这 30 ms 里算力只花约 3 ms，其余都在等数据：加算力不加速 decode，只有加带宽或减小 KV 才有效。

**第 6 步：混跑时发生什么（这就是要分离的理由）**
这 32 个请求正以 30 ms/步出字，突然来了个 8K 新请求要 prefill 1.1 s。若它不可打断地插进来，TPOT 就从 30 ms 跳到约 1.1 s，P99（最慢 1% 请求的延迟）直接爆掉。分离后这次 prefill 发生在 P 节点，D 的节奏不受影响。

**第 7 步：算 P:D 配比**
设到达率 10 请求/秒、输入 8K、输出 500：

```
P 侧：10 × 1.1 s = 11.1 卡 → 留约 15% 余量 → 13 张
D 侧：需要 10 × 500 = 5000 token/s
      单卡（批量 32）1067 token/s → 5000 ÷ 1067 ≈ 4.7 → 5 张
P:D ≈ 13:5 ≈ 2.6:1
```

把输入换成 512 token、输出仍是 500：

```
P 侧：10 × 0.055 = 0.55 卡 → 1 张
D 侧：按平均上下文 512 + 500 ≈ 1000 token 估，每请求约 126 MiB
      批量取 128：按 TPOT SLO 定的 max_num_seqs，不是显存上限
      （40 GB 预算其实放得下 300 条）
      16 GB + 128 × 126 MiB ≈ 32 GB ÷ 1.6 TB/s ≈ 20 ms/步
      → 128 ÷ 0.020 s ≈ 6400 token/s
      5000 ÷ 6400 ≈ 0.8 卡 → 1 张
P:D ≈ 1:1
```

同样 10 请求/秒，P:D 从 2.6:1 变成 1:1。D 侧还极敏感：max_num_seqs 放到 300 时一步约 34 ms，吞吐约 8900 token/s，D 侧从 0.8 卡降到 0.6 卡。**抄别人的配比是错的，必须用你自己的长度分布实测。**

**第 8 步：KV 传输要多久？**
拿第 7 步第一个例子的 1 GiB：

```
200 Gbps RoCE ≈ 25 GB/s → 1.07 GB ÷ 25 GB/s ≈ 43 ms  （TTFT 约 1.15 s，占 4%，可接受）
25 Gbps       ≈ 3.1 GB/s → ≈ 344 ms                    （TTFT 变 1.45 s，多出三成，不可接受）
```

PD 分离对网络有硬门槛，长上下文门槛更高：16 GiB 的 128K 请求，200 Gbps 要 690 ms，25 Gbps 要 5.5 s——后者等于判死刑。

**第 9 步：落到昇腾 + vLLM 上**
vllm-ascend 支持两种连接器：MooncakeConnector（D 向 P 拉 KV）与 MooncakeLayerwiseConnector（P 按层推，叠在 P 的下一层计算上）；Atlas A2 / A3 均可用，GQA 与 MLA（多头潜在注意力：把 KV 压成低秩潜向量，比 GQA 更小）模型都支持。用 `--kv-transfer-config` 指定 `kv_connector` 与 `kv_role`（P 侧 producer、D 侧 consumer，字段名以你装的版本为准），用 `ASCEND_RT_VISIBLE_DEVICES` 给 P/D 进程分卡。三条硬约束：不支持异构 P/D 混搭（A2 与 A3 不能混跑）；TP 不对称时只支持 P 的 TP 大于 D 的 TP 且为其整数倍；P 节点靠 `max_tokens=1` 只做 prefill。

## 常见误区

**误区 1：PD 分离能提升总算力，所以吞吐一定更高。**
它不增加峰值 FLOPS，也不降低单 token 计算量，原始 token 吞吐甚至可能更低。但在满足 TTFT/TPOT 的前提下 **goodput 通常显著提升**：混跑要把 P 的峰值和 D 的峰值叠在一起备容量，分离后两段各自按需配（不浪费卡），这是实打实的容量收益，也是 Mooncake 报告的 SLO 内有效吞吐提升。若负载同质（清一色短 prompt 长输出）或链路太慢，goodput 可能反而下降——分离不是默认正确。

**误区 2：decode 慢是算力不够，该给 decode 更多算力。**
decode 的瓶颈是显存带宽：每步必须完整读一遍权重和 KV，矩阵单元大部分时间在等数据。换更强算力几乎不提升 decode，有效的是更高带宽、更小的 KV（GQA/MLA、KV 量化）、更大且有效的批。反过来认为「prefill 是带宽瓶颈」同样错，prefill 是实打实的算力瓶颈。

**误区 3：KV 传输只是「顺手拷一下」，可以忽略。**
1 GiB 在 25 GB/s 上是 43 ms，在 25 Gbps 上是 344 ms，直接计入 TTFT，还与在线业务抢带宽。不做按层重叠、压缩或局部性调度，省下的 TPOT 抖动会以 TTFT 的形式还回去。

## 一句话总结

prefill 吃算力、decode 吃带宽，混跑时长 prefill 会卡住 decode；PD 分离用一次 KV 传输把两段解耦，换来独立的配比与延迟控制，代价是网络、显存驻留和运维复杂度——值不值，得用你自己的长度分布和链路带宽算过才知道。

## 自测题

### 1. 为什么给 decode 加算力几乎没用，加带宽却能接近线性提速？
<details><summary>查看答案</summary>因为 decode 每步的计算量与数据量差一个数量级。批量 32 时每步只算 32 个 token，`2 × 8e9 × 32 ≈ 0.5 TFLOP`，按 150 TFLOPS 只要约 3 ms；而每步必须把 16 GB 权重加 32 GiB KV 共 48 GB 读一遍，`48 ÷ 1.6 TB/s = 30 ms`。27 ms 在等数据：算力翻倍只把 3 ms 变 1.5 ms（约 5%），带宽翻倍则压到约 18 ms。所以救 TPOT 的顺序是：更高带宽 → 更小的 KV（GQA/MLA、KV 量化、更短上下文）→ 批大小是否打满 → 最后才考虑更强算力的卡。</details>

### 2. 128K 相比 8K，并发、prefill 与 KV 传输各变化多少？
<details><summary>查看答案</summary>KV：128 KiB/token × 131072 ≈ 16 GiB，8K 只要 1 GiB。并发：40 GB 预算下从 40 条掉到 2 条（约 20 倍），而这 2 条只占 32 GB，显存碎得厉害。Prefill：注意力项 `4 × 32 × S² × 4096` 在 128K 处约 9000 TFLOP，是线性项（约 2100 TFLOP）的 4.3 倍；8K 时注意力只占 21%。传输：16 GiB 在 200 Gbps 上 690 ms，25 Gbps 上 5.5 s——所以长上下文既最值得分离（prefill 阻塞随 S² 放大、decode 随 S 变慢更怕打断），也最难分离（传输 ∝ S，必须上高速互联加逐层重叠，否则省下的 TPOT 抖动会以 TTFT 还回去）。</details>

### 3. 方案 A 平均 TTFT 800 ms、P99 3 s；方案 B 平均 900 ms、P99 1.5 s。SLO 是「TTFT P90 ≤ 1.2 s 且 P99 ≤ 2 s」，选哪个？为什么不能看平均值？
<details><summary>查看答案</summary>选 B。goodput 只数**同时满足所有 SLO** 的请求：A 均值更漂亮，但 P99 = 3 s 越过 2 s 红线，最慢那 1% 不计入有效吞吐，而它们通常正是「长 prefill 撞上 decode」的那批；B 平均慢 100 ms，但尾巴收在 1.5 s 内，几乎全部计入 goodput。这就是 PD 分离的调度目标——不是把均值做低，而是在 SLO 内让尽量多的请求过关。实操两点：验收用 P90/P99 而非均值；P99 由尾部一两个长请求决定，比 P90 更难达标，正对应 P 侧的排队策略。</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| Prefill（预填充） | 对整段 prompt 做一次前向，算出所有位置的 KV，主要产出是首 token |
| Decode（解码） | 每次前向只处理最新一个 token，逐字生成，受显存带宽限制 |
| KV cache | 缓存历史 token 的 Key/Value，避免 decode 重算，随上下文线性增长 |
| PD 分离 | prefill 与 decode 部署到不同实例，中间传 KV cache，两段独立扩缩容 |
| KV 传输 | 把 P 的 KV 搬到 D；时间 ≈ KV 大小 ÷ 有效带宽，需与 P 侧计算重叠 |
| 资源配比（P:D） | P 与 D 的节点数量比及各自并行策略，由输入/输出长度分布决定 |
| 长上下文 | 单请求上下文远超常规规模（32K/128K+），并发、TTFT、TPOT 同时恶化 |
| TTFT | Time To First Token，请求到首 token 的延迟，由排队、prefill、KV 传输决定 |
| TPOT | Time Per Output Token，首 token 之后平均每 token 延迟，由 decode 决定 |
| SLO | Service Level Objective，可量化承诺，如「TTFT 的 P90 ≤ 2 s」 |
| goodput | 有效吞吐：单位时间内同时满足所有 SLO 的请求数 |
| GQA | Grouped-Query Attention，多 Q 头共用一组 K/V 头，KV 头数远少于 Q 头 |
| chunked prefill | 把长 prefill 切块与 decode 交替，压住 TPOT 抖动，PD 分离的替代方案 |
| HCCL / HCCS | 昇腾的集合通信库 / 片间高速互联链路，同机 P 到 D 的 KV 传输常走它 |
| RoCE | RDMA over Converged Ethernet，跨机低延迟高带宽传输，跨机 KV 传输常用 |

## 下一步

下一课《推理成本与调度》把这一课的容量账换成钱：给定 P:D 配比和 SLO，怎么算每百万 token 的成本、怎么在延迟与成本间取舍、批处理何时比在线服务更划算。

## 来源

- Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving — https://arxiv.org/abs/2407.00079 （论文）
- vLLM 文档：Disaggregated Prefilling (experimental) — https://docs.vllm.ai/en/latest/features/disagg_prefill/ （官方文档）
- vLLM Ascend 文档：Disaggregated-prefill — https://docs.vllm.ai/projects/ascend/en/v0.11.0/developer_guide/feature_guide/disaggregated_prefill.html （官方文档）
- vLLM Ascend 教程：PD disaggregation with Mooncake — https://vllm-ascend.readthedocs.io/en/latest/tutorials/pd_disaggregation_mooncake_multi_node.html （教程）
- vLLM 文档：Optimization — https://docs.vllm.ai/en/latest/configuration/optimization/ （官方文档）
- Efficient Memory Management for LLM Serving with PagedAttention — https://arxiv.org/abs/2309.06180 （论文）
