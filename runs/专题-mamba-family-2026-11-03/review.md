# 审校记录 · 2026-11-03 · 专题 · mamba-family · Mamba 家族与混合架构

## verdict

revise

## summary

三篇结构清楚、来源意识和「厂商数字单独标注」做得较好（KDA 的 75%/6×、Qwen3-Next 的 580 tok/s 都已限定），但仍有 10 处需改：两处数学表述过松（块间串行次数、B̄≈ΔB 的条件与示例数字），Ascend GDN 的默认算子路径、前缀缓存状态、vLLM V1 版本时间线在两篇之间互相矛盾或缺乏可核对来源（含 v0.30.0 这个可疑版本号），第三篇 chunk 算术强度无法用文中公式复算、且「不是带宽」的结论与第二篇的带宽收益论述冲突。均可在不改结构的前提下修正；另建议三篇对「块间传状态」互相引用，避免同一概念三种说法。

## issues

### 1 · model-evolution · medium

- **severity**: medium
- **part_id**: model-evolution
- **claim**: chunked scan：「块内用矩阵乘一次算完（带因果掩码），块间只传一个状态 h，串行 L 次」；第三步又写「总计算约 O(n·L + (n/L)·N)」
- **问题**: L 是块长，块间串行次数应是块数 n/L；写成「串行 L 次」会让读者把块长当成串行步数。同一步的复杂度还漏了通道维 D，正确量级是 O(n(L+N)D)，这样才能和「主体是 matmul」自洽。
- **建议**: 改为「块间串行 n/L 次」，复杂度写成 O(n(L+N)D)，并说明 L 与 D 各自的作用。

### 2 · model-evolution · medium

- **severity**: medium
- **part_id**: model-evolution
- **claim**: 「B̄ = (ΔA)⁻¹(exp(ΔA) − I)·ΔB，A 为标量时 B̄ ≈ ΔB」，随后用 a=−1、Δ=0.1 算出 h₂=0.1905、h*=1.05
- **问题**: 标量精确式是 B̄ = ((e^{Δa}−1)/a)·B，只有 |Δa|≪1 才能近似成 ΔB。这里 Δa=−0.1，精确 B̄=0.0952，h₂=0.181、h*=1.00，文中数字偏高约 5%。「≈」没写成立条件，示例又直接照抄近似值，基础弱的读者会把它当恒等式。
- **建议**: 写出标量精确式并注明近似条件；示例要么用近似并标明误差量级，要么用精确值重算。

### 3 · model-evolution · medium

- **severity**: medium
- **part_id**: model-evolution
- **claim**: 「attention 也是 y = M·x，只是 M = softmax(QKᵀ)（下三角、稠密）」
- **问题**: 这里的 x 必须是 V；M 本身由输入经 Q、K 算出，attention 对输入不是线性映射。照字面读会得出「attention 与 SSM 是同一个线性 y=Mx」，而 SSD 的对偶是「给定 Q,K 后对 V 线性」，两者不是一回事——这恰是本专题最容易被误读的一步。
- **建议**: 改成「给定 Q,K 时 attention 是 y=A·V，A=softmax(QKᵀ)⊙mask；SSD 的对偶指两者都能写成同一种下三角矩阵乘 V」，并点明 Q,K 依赖输入。

### 4 · model-evolution · medium

- **severity**: medium
- **part_id**: model-evolution
- **claim**: 「昇腾侧（vllm-ascend）prefill 走 Triton 版 chunk_gated_delta_rule，decode 走 torch_npu.npu_recurrent_gated_delta_rule 自定义 AscendC 算子」
- **问题**: 与后两篇冲突：第二篇写「优先调 torch_npu.npu_chunk_gated_delta_rule 融合 prefill……算子不存在或 CANN 不支持就回落 Triton FLA」，第三篇写 PR #12607 已用 aclnn/AscendC 的 ChunkGatedDeltaRule。三篇对同一条默认路径给出互斥描述，约束也不一致（第二篇「Dk=Dv=128」vs 第三篇「Dk/Dv ≤ 128、Nk/Nv ≤ 64」），读者无法判断默认跑什么、何时回落。
- **建议**: 固定一个 vllm-ascend 版本，按源码统一成一张表：prefill/decode 默认算子、回落条件、维度与精度约束；第一篇那句改成「aclnn 优先、Triton 回落」并注明版本。

### 5 · ascend · medium

- **severity**: medium
- **part_id**: ascend
- **claim**: 「Qwen3-Next 那一行的 chunked prefill、前缀缓存、投机解码、PD 分离全是空白——不是「不行」，是「没验证过」」；表格「前缀缓存/PD 分离在支持矩阵里是空白」
- **问题**: 与第一篇「all 模式的 GDN 前缀缓存已实现，但该模式下投机解码会回退到 align」、第二篇「PR #8829 做了 GDN 前缀缓存」直接冲突；第二篇还把 mamba_cache_mode 写成只有 none/align，漏掉第一篇提到的 all。支持矩阵是 v0.18.0 口径，PR #8829 在其后，两篇都没标版本边界，读者会得到相反结论。
- **建议**: 在第三篇表格加脚注「v0.18.0 口径，此后 PR #8829 增加 align/all 模式」；第一、二篇把「已实现」限定到具体版本与模式（align/all、是否含投机回退）。

### 6 · vllm · medium

- **severity**: medium
- **part_id**: vllm
- **claim**: 「v0.10.0 把 Mamba2 接进 V1，v0.10.1 加 Mamba1/Jamba，v0.10.2 才设默认并去掉 enforce_eager，v0.11.0 删掉 V0」及 #21194/#22594/#23035/#23716/#22541/#24593 等 PR 号
- **问题**: 这些版本与 PR 号以既成事实给出，但第三篇明说「Mamba/Jamba 进 V1 的版本时间线未核实」；第一篇又把「去掉 enforce_eager 的文档改动」记在 PR #21233 上，与第二篇「v0.10.1 文档仍要求 Mamba-1 enforce_eager、v0.10.2 才移除（#22594/#23035）」对不上，且没有一处给出可核对的 diff。
- **建议**: 以 pinned tag 的文档/PR 做一张时间线表（版本—PR—改了什么—引文），删掉无法核对的 PR 号；说明 #21233 与 #22594/#23035 的关系，并与第三篇的「未核实」口径保持一致。

### 7 · vllm · medium

- **severity**: medium
- **part_id**: vllm
- **claim**: 「Qwen3.5-27B……vllm-ascend v0.17.0rc1 起支持；Qwen3-Next 从 v0.30.0 起不再维护」
- **问题**: 同一段里 Qwen3.5 的支持起点是 v0.17.0rc1，第三篇引用的是 v0.18.0 文档；v0.30.0 晚于这两者又无对应发布说明，像笔误（可能想写 v0.13.0 一类），却被写成既成事实，读者会误判 Qwen3-Next 还能用多久。
- **建议**: 核对 vllm-ascend release notes，改成正确版本号并给链接；若只是计划移除，写明时间与来源。

### 8 · ascend · medium

- **severity**: medium
- **part_id**: ascend
- **claim**: 「L=128 约 100、L=256 约 190、L=512 约 366 FLOP/Byte（按 4L²D + 4LD² 计算，手算非实测）」
- **问题**: 按文中公式、D=128、搬 q/k/v/o 四个张量（8LD 字节）复算，算术强度是 (L+D)/2 = 128/192/320，只有 L=256 接近；把状态读写也算进去也对不上。字节口径没写出来，「手算」不可复现，而「L=256 仍偏带宽、512 才越脊点」的结论正依赖这组数（脊点本身还是存疑的 200–250）。
- **建议**: 写明字节口径（读哪些张量、是否含状态与输出写回）并给出可代入的算式；结论改成随脊点变化的区间判断，不要用单一阈值。

### 9 · ascend · medium

- **severity**: medium
- **part_id**: ascend
- **claim**: 「『Mamba 更省，所以在 NPU 上也一定更快』：省的是显存容量和随长度增长的 FLOP，不是带宽」「换更好的 kernel 也救不回来」
- **问题**: 与第二篇「decode 每步不再读 512 MiB KV，只读 8.3 MiB 状态……这是长上下文 TPOT 改善的来源」直接冲突；按字面等于否认 SSM 的带宽收益。实际是两件事：SSM 把「每步要搬的字节」大幅降低（长上下文尤甚），代价是这些字节上的算术强度只有 1.5 FLOP/Byte。「救不回来」也与本篇「少搬数据、融合、状态留片上」的优化主张自相矛盾。
- **建议**: 区分「总搬运量」（SSM 显著更少，是带宽收益）与「单位字节算术强度」（低，故仍带宽受限）；把「救不回来」改成「只能减少搬运，不能把 AI 提到算力受限区」。

### 10 · vllm · low

- **severity**: low
- **part_id**: vllm
- **claim**: 「但要求 HMA，两端 block size 必须一致」「Mamba 组上有 EAGLE/MTP」「n_groups % tp_size == 0」；第三篇「退化成 (α, β) 二元组，复合为 (α₂α₁, α₂β₁+β₂)」
- **问题**: HMA、MTP、EAGLE、block table、n_groups 都是判断「能不能用」的关键条件，却既没在首现处解释也没进术语卡；第一篇的 GSA、meta token、d_inner、associative recall，第三篇的 BSND、FLA、UB、A3-752T 同样缺解释。第三篇还把 β 一符两用：既是 delta rule 的学习率，又是仿射映射的常数项，读者会以为是同一个量。另外三篇各讲一遍「块间传状态」（chunked scan / chunked prefill / chunk_state）却互不引用，第一、二篇又用不同配置各算一遍 KV/状态账（4 GiB→512 MiB vs 8.3 MiB），容易被读成数字打架。
- **建议**: 补术语卡与首现解释；仿射常数项统一写成 b_t（与第一篇的 (a,b) 一致）；三篇在「块间传状态」处互相链接，并给一张标注了各模型配置的统一记账表。
