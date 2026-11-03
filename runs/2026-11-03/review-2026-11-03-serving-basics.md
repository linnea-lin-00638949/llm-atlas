# 审校记录 · 2026-11-03-serving-basics

- **verdict**: revise
- **summary**: 重算核验全部通过：2×32×128×2=16 KB/token·层 → ×32 层 = 512 KB/token → ×300 token ≈ 150 MB；TPOT = (4300−300)/199 ≈ 20.1 ms、总时延 4.3 s；8/0.03 ≈ 267 token/s，均正确。注意力公式、Llama-2-7B 配置（32 层/32 头/128 维）、PagedAttention 论文「60%–80% → <4%」、指标名 `vllm:time_to_first_token_seconds`/`vllm:time_per_output_token_seconds`、`--enable-prefix-caching`、`--max-num-seqs`、KV Cache CPU Offload 均属实。教学结构八要素齐全，四个模块目标全部覆盖，术语大多有即席解释。问题：1 处概念错误（--gpu-memory-utilization 被定义为「划给 KV 的显存比例」，实为引擎可占用的显存总比例）；3 处中等问题（TTFT 决定因素的「×」写法、decode「只算 Q」、"几十个并发吃满 HBM"与本课数字差一个数量级）；4 处初学者友好性可改进项。

## issues

### 1. high

- **claim**: `--gpu-memory-utilization`（划给 KV 的显存比例）
- **problem**: 概念错误。vLLM 的 --gpu-memory-utilization 是引擎允许占用的显存总比例，模型权重、激活与 KV cache 都从这份配额里出，不是「划给 KV 的比例」；KV cache 实际能用的显存是总配额减去权重与激活后的剩余部分。文中两处（「为什么需要它」和「一步一步理解」）都按错误定义解释，会直接误导昇腾工程师调参。
- **fix**: 改为「引擎允许占用的显存总比例；KV cache 实际分到的是扣除模型权重与激活后的剩余部分，调大它间接扩大了 KV 池」。

### 2. medium

- **claim**: TTFT 主要由 prefill 时长（prompt 长度 × 排队）决定
- **problem**: 「×」会被初学者读成乘法，而 prompt 长度与排队时间不是相乘关系；且 TTFT = 排队时间 + prefill 时间，负载高时排队可能占大头，「主要由 prefill 时长决定」过于绝对。
- **fix**: 改为「TTFT ≈ 排队时间 + prefill 时间：prefill 时间随 prompt 长度与可用算力变化，排队时间随系统负载变化」。

### 3. medium

- **claim**: 每步只计算新 token 的 Q（查询）向量，与 KV cache 里的历史 K/V 做注意力
- **problem**: 不准确：decode 每步也要为新 token 计算 K、V 并追加进 KV cache（否则后续步骤无从复用），只有历史 token 的 K/V 是查缓存复用的。基础弱的读者会误以为 decode 阶段完全不产生 K/V。
- **fix**: 改为「每步只为新 token 计算 Q、K、V（新 K/V 写入缓存），注意力中历史 K/V 全部直接查 KV cache，真正新算的只有一个 token 的量」。

### 4. medium

- **claim**: 一个请求 150 MB，几十个并发就吃满 HBM
- **problem**: 与本课自己的数字矛盾：几十个（如 50 个）并发 × 150 MB ≈ 7.5 GB，远小于昇腾 910B 的 64 GB HBM；按 150 MB/请求要约 400 个并发才吃满，差了一个数量级。
- **fix**: 改为「几百个并发就能吃满 HBM」，或限定为「在 gpu-memory-utilization 划出的余量内，几十个并发已占掉相当比例」。

### 5. low

- **claim**: prefill 定义「把整个 prompt……一次性送入模型并行计算」，后文又说「vLLM 还支持 chunked prefill（把长 prompt 切块）」
- **problem**: 前后看起来矛盾：初学者刚记住「一次性」，马上又看到「切块」，会困惑哪个才对。
- **fix**: 在 prefill 定义处补一句「（未启用 chunked prefill 时是整个 prompt 一次算完；chunked prefill 是后文介绍的工程优化）」。

### 6. low

- **claim**: 吞吐 ≈ 同批请求数 ÷ 单步时长
- **problem**: 该式只在同批请求全部处于 decode 时成立；continuous batching 恰恰是 prefill/decode 混批，混入 prefill 后单步时长与每步产出 token 数都会变化，公式只是粗近似，缺限定会误导手算。
- **fix**: 加限定：「（同批都在 decode 时成立；混入 prefill 后只能当数量级估计）」。

### 7. low

- **claim**: 假设 prefill 100 token 耗时 300 ms，decode 每步 20 ms：TTFT = 300 ms
- **problem**: 概念段刚说 TTFT 含排队时间，算例直接把 TTFT 等于 prefill 时长且未声明「假设无排队」，前后不一致，初学者会追问差在哪。
- **fix**: 写成「TTFT ≈ 300 ms（假设排队时间为 0）」。

### 8. low

- **claim**: 「并行采样、束搜索时还能共享物理块」；「在『迭代』粒度上批处理」
- **problem**: 「束搜索」「并行采样」「迭代」均未解释即使用，基础弱的读者会卡住。
- **fix**: 首次出现时加注：束搜索＝一次尝试多条候选路径取概率最高者；并行采样＝同一 prompt 生成多个结果；迭代＝把一批 token 提交给 NPU 计算的一轮。
