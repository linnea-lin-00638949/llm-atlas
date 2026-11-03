---
date: "2026-11-03"
title: "第 7 课 · 昇腾 NPU 架构与 CANN 生态（上）"
domain: "硬件"
module_id: "ascend-arch"
module: "昇腾 NPU 架构与 CANN 生态"
status: "已学"
lesson_no: 7
goals: ["解释 NPU 与 GPU 在设计哲学上的异同，并判断一类负载更适合哪种芯片", "说清达芬奇架构 Cube/Vector/Scalar 的分工，以及 910B 与初代 910 核组织方式的区别", "分清 Atlas 各产品线定位，并会估算一个 GEMM 层落在计算受限还是带宽受限"]
tags: ["昇腾", "NPU", "达芬奇架构", "CANN", "HBM", "Atlas"]
---

# 第 7 课 · 昇腾 NPU 架构与 CANN 生态（上）

> 学完今天，你能：
> 1. 解释 NPU 与 GPU 在设计哲学上的异同，并判断一类负载更适合哪种芯片
> 2. 说清达芬奇架构 Cube/Vector/Scalar 的分工，以及 910B 与初代 910 核组织方式的区别
> 3. 分清 Atlas 各产品线定位，并会估算一个 GEMM 层落在计算受限还是带宽受限

## 课前自测

先凭直觉作答，答错没关系，正文会逐一纠正：

1. GPU 和 NPU，哪个更「样样通」，哪个更「一招鲜」？
2. Cube 一个周期算一个 16×16×16 矩阵乘加块，一共多少次乘法？
3. 昇腾 910B 的显存是 HBM 还是 GDDR？单卡容量约多少？
4. Atlas 300I Duo 和 Atlas 800I A2 分别是什么产品形态（插卡 / 整机）？
5. vLLM 推理时，把 batch 从 1 增大到 1024，单卡吞吐会怎样变化？为什么？

## 为什么需要它

你的工作里到处是 NPU、AI Core、HBM：vLLM（你正在用的开源大模型推理引擎）的日志、容量规划、故障排查都会遇到。它们不是装饰：芯片设计决定算子（深度学习的基本计算单元，如矩阵乘、激活函数）怎么写、显存怎么规划、吞吐天花板在哪。GPU 有 CUDA 生态，昇腾有 CANN（Compute Architecture for Neural Networks，昇腾整套软件栈，下集展开）。本集先回答三个硬件问题：芯片为什么快、矩阵怎么算、产品怎么选。

## 核心概念

### NPU 与 GPU 的区别

**定义**：NPU（Neural-network Processing Unit，神经网络处理器）为矩阵/张量运算深度定制；GPU（Graphics Processing Unit，图形处理器）从图形渲染演化而来，靠大量小核心并行做通用计算。

**直觉**：GPU 像一支「百人厨师团」，人人会做各种菜（图形、科学计算、AI），靠人多取胜，后来为 AI 加装了一个「专用灶台」（Tensor Core，张量核心，专做矩阵乘）；NPU 是「只为矩阵乘法盖的流水线车间」，一道菜做到极致。

**机制**：GPU 由大量通用核心加少量张量核心组成，兼顾各类负载；NPU 以达芬奇架构的 AI Core 为主，晶体管几乎全投向矩阵与向量运算。设计哲学差别：GPU「通用优先，AI 是后加的加速器」；NPU「AI 优先，通用控制只是辅助」。代价也很清楚：跑分支密集、访存不规则的负载时 NPU 不占优，有时要靠 CPU 兜底。这也解释了为什么昇腾上跑 vLLM 必须依赖 CANN 深度优化的算子，不能把 CUDA 的 kernel（运行在 GPU 上的计算函数）直接搬过来。

### 达芬奇架构

**定义**：华为自研的 AI 处理器微架构（DaVinci Architecture），2018 年随昇腾 310/910 发布（910 于 2019 年商用）。

**直觉**：它是 AI Core 内部的「车间平面图」：哪台做矩阵乘、哪台做杂活、哪台当指挥。

**机制**：三大计算单元分工——
- Cube（矩阵/立方体计算单元）：一个时钟周期完成一个 16×16×16 矩阵乘加 = 4096 次 MAC（Multiply-Accumulate，一次乘加 = 一次乘法加一次加法）；支持 BF16（16 位脑浮点数）输入、FP32 累加。这是 AI 算力的源头。
- Vector（向量单元）：逐元素运算，如激活函数、归一化。
- Scalar（标量单元）：循环控制与分支。

配套存储：Unified Buffer（UB，每核约 192 KB 片上缓冲）、L0A/L0B/L0C（Cube 的操作数与累加缓冲）、MTE（搬运引擎）。UB/L0 是算子经 MTE 显式管理的 SRAM 缓冲，不是自动换入换出的 cache。大模型 90% 以上浮点运算集中在矩阵乘，Cube 的规格几乎决定整卡算力。

### AI Core

**定义**：昇腾处理器上执行算子的最小计算单元；一颗芯片集成几十个。核的组织随代际变化：初代 910 每个 AI Core 集成 Cube+Vector+Scalar 三件套；910B 把向量部分独立为 AIV（AI Vector），与带 Cube 的 AIC（AI Core）并列。

**直觉**：初代 910 的 AI Core 是「完整小工厂」——矩阵灶台、杂活间、指挥室一栋楼；910B 把杂活间拆出去另盖，工厂更专一。

**机制**：910B（B2 版本）约含 24 个带 Cube 的 AIC 与 48 个 AIV，官方产品页标称 FP16 峰值约 376 TFLOPS；初代 910 为 32 核、256 TFLOPS。核数属半公开信息（来自 CANN 工具配置与公开分析，存疑），卡片级官方规格更可靠。910C 则把两片 die（裸片，硅晶圆切出的芯片）封装在一起，公开评测约 800 TFLOPS FP16，官方规格未公开，存疑。

AI Core 干活流水线：Scalar 指挥 → MTE 把矩阵块从 HBM 搬进 UB/L0 → Cube 计算 → 结果写回 UB → 搬回 HBM；计算与搬运重叠，把访存延迟藏起来。

### HBM 显存

**定义**：HBM（High Bandwidth Memory，高带宽内存）把多层 DRAM 垂直堆叠，与处理器同基板封装，数千条短连线连接。

**直觉**：GDDR（图形用双速率同步显存）像「离工厂很远的公路」，车道少；HBM 像「贴着工厂墙的超宽码头」。堆叠 = 盖楼，省面积，但离芯片太近散热难、成本高。

**机制**：两个关键数字：容量（放得下多少权重与 KV cache——推理时缓存的键/值张量，避免对已生成 token 重复计算）和带宽（每秒能搬多少字节）。910B 单卡配 64 GB HBM，8 卡 Atlas 800I A2 合计 512 GB；带宽约 1.2 TB/s（官方未列出，来自 CANN 配置与公开评测，存疑）。AI 芯片选 HBM 的原因：矩阵乘太快、搬运跟不上，带宽才是瓶颈。

### Atlas 产品线（910B/910C）

**定义**：Atlas 是昇腾硬件产品品牌，覆盖模组、加速卡、服务器、集群；910B/910C 是芯片代号，即 910 系列的第二、三代。

**直觉**：芯片像发动机，产品线像整车目录：910B 发动机 → 双发插卡 → 八发整机 → 集群车队。

**机制**：定位速记——
- Atlas 300I Duo：双 910B 的 PCIe（高速串行扩展总线，连接 CPU 与加速卡）插卡，128 GB HBM、约 752 TFLOPS FP16，适合单机加卡与试点；
- Atlas 800I A2：8×910B 的推理整机，512 GB HBM，生产推理主力；
- Atlas 800T A2：8×910B 的训练整机；
- Atlas 900 SuperCluster / CloudMatrix 384：集群与超节点；公开报道称 CloudMatrix 384 由 384 颗 910C 构成（官方规格未公开，存疑）；
- 边缘场景另有 Atlas 200I A2 模组（310B 芯片）等。

选型口诀：生产推理优先 800I A2 整机，试点或单机扩容用 300I Duo，超大规模用 910C 超节点。

## 一步一步理解

**场景**：910B 单卡跑 vLLM，模型某线性层权重 W 是 4096×4096 的 BF16 矩阵（每元素 2 字节，共 32 MiB）。这类「矩阵乘矩阵」运算叫 GEMM（General Matrix Multiply）。问：decode 阶段（逐 token 生成：每生成一个 token 就做一次本轮矩阵乘）batch=1 时，这一层要多久？瓶颈是什么？注：约 1.2 TB/s 带宽属存疑数值，换值只按比例缩放搬运时间，结论不变。

**第 1 步，数运算量**。输入 x 是 1×4096 向量，输出 y = x·W 是 1×4096；y 每个元素 = 4096 次乘加，总 MAC = 1×4096×4096 = 16,777,216 ≈ 33.5 MFLOP（百万次浮点运算）。

**第 2 步，算计算时间**。910B 峰值 376 TFLOPS = 376×10¹² 次浮点运算/秒，折合 188×10¹² 次乘加/秒。t = 16,777,216 ÷ 188×10¹² ≈ 89 ns。小到可以忽略。

**第 3 步，算搬运时间**。算之前必须把 W 完整读进 AI Core：32 MiB = 33,554,432 字节，按 1.2×10¹² 字节/秒 ≈ 28 µs。

**第 4 步，对比**。28 µs ÷ 89 ns ≈ 300 倍：算得快、等得久，这一层是带宽受限（bandwidth-bound，耗时由搬运决定），decode 算力大量闲置。

**第 5 步，推论**。把 batch 提到 1024：运算量乘 1024，约 91 µs；而 W 只搬一次（总搬运约 35 µs < 91 µs），变成计算受限（compute-bound，耗时由算力决定）。单层理想吞吐从约 3.6 万 token/s 升到约 1100 万 token/s——但这仅是「这一层」的理想上界，不是整卡真实吞吐：一个 token 要依次过全部几十层，还要算注意力等非 GEMM 算子。量级对照：若模型有 80 个同规模层，batch=1 时每 token 约 80×28 µs ≈ 2.2 ms（约 450 token/s），才接近真实整卡量级。这就是 vLLM 增大 batch 能提吞吐、decode 阶段要精打细算 KV cache 的硬件根源；prefill（预填充，一次性算完整个 prompt）天然 M 大，也是计算受限。

## 常见误区

1. **「NPU 全面强于 GPU」**。错。NPU 只在矩阵密集负载上极致；分支多、访存乱、科学计算类任务效率不高。强不强看负载形状，不看芯片名头。
2. **「TFLOPS 越高，推理越快」**。只对了一半。decode 是带宽受限，峰值算力根本摸不到；选卡要同时看算力与带宽的配比。
3. **「Cube 一个周期算完一个大矩阵」**。错。任何大矩阵都要切成 16×16×16 的小块反复迭代；一个 4096³ 的 GEMM 是约 1677 万块、块块排队。同理，batch=1 时算力「闲着」是带宽受限的正常现象，不是故障。

## 一句话总结

昇腾 NPU = 达芬奇架构 AI Core（Cube 每周期做 16×16×16 矩阵乘加）+ HBM 高带宽显存；910B/910C 撑起 Atlas 从插卡到超节点的产品线（910B 上 Vector 已独立为 AIV）。推理快不快，取决于负载落在计算受限还是带宽受限。

## 自测题

### 1. GPU 和 NPU，哪个更「样样通」，哪个更「一招鲜」？

<details><summary>查看答案</summary>

GPU 更「样样通」：从图形渲染演化而来，靠大量通用小核心并行，图形、科学计算、AI 都能跑，AI 靠后加的 Tensor Core 加速。NPU 更「一招鲜」：晶体管几乎全投向矩阵/向量运算，矩阵乘做到极致，但分支密集、访存不规则的负载不占优，有时要 CPU 兜底。

</details>

### 2. Cube 一个周期算的 16×16×16 矩阵乘加块，一共多少次乘法？

<details><summary>查看答案</summary>

16×16×16 = 4096 次乘法，同时有 4096 次加法，即 4096 次 MAC = 8192 次浮点运算（BF16 输入、FP32 累加）。大矩阵都要切成 16×16×16 的小块反复迭代：4096³ 的 GEMM 约需 4096³ ÷ 16³ ≈ 1677 万块排队。

</details>

### 3. 昇腾 910B 的显存是 HBM 还是 GDDR？单卡容量约多少？

<details><summary>查看答案</summary>

是 HBM：多层 DRAM 垂直堆叠、与芯片同封装，带宽远高于 GDDR。910B 单卡 64 GB HBM，8 卡 Atlas 800I A2 合计 512 GB；带宽约 1.2 TB/s（官方未列出，存疑）。选 HBM 的原因：矩阵乘太快、搬运跟不上，带宽才是推理瓶颈。

</details>

### 4. Atlas 300I Duo 和 Atlas 800I A2 分别是什么产品形态？

<details><summary>查看答案</summary>

Atlas 300I Duo 是双 910B 的 PCIe 插卡（128 GB HBM，约 752 TFLOPS FP16），适合单机加卡与试点；Atlas 800I A2 是 8×910B 的推理整机（512 GB HBM），生产推理主力。往上还有训练整机 Atlas 800T A2、384 颗 910C 的 CloudMatrix 384 超节点（官方规格未公开，存疑），边缘有 Atlas 200I A2 模组（310B）。

</details>

### 5. vLLM 推理把并发 batch 从 1 增大到 1024，单卡吞吐会怎样变化？为什么？

<details><summary>查看答案</summary>

显著提升，瓶颈从带宽受限翻转为计算受限。batch=1：W 算一次只要约 89 ns，但 32 MiB 的 W 从 HBM 搬进 AI Core 要约 28 µs（约 300 倍等待），算力闲置。batch=1024：运算量乘 1024 约 91 µs，W 只搬一次（约 35 µs），变成计算受限。注意口径：3.6 万 → 约 1100 万 token/s 只是「单层 GEMM」的理想上界；一个 token 还要过全部几十层、算注意力等算子，80 层同规模模型 batch=1 时每 token 约 2.2 ms（约 450 token/s）才接近真实整卡量级。

</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| NPU | 神经网络处理器，为矩阵运算深度定制的 AI 芯片 |
| GPU | 图形处理器，大量通用核心并行，AI 靠后加的 Tensor Core 加速 |
| 达芬奇架构 | 华为自研 AI 处理器微架构，规定 AI Core 内单元分工；2018 年随昇腾 310/910 发布 |
| AI Core | 昇腾上执行算子的最小计算单元；初代 910 形态内含 Cube/Vector/Scalar，910B 上 Vector 独立为 AIV |
| AIC / AIV | 910B 上带 Cube 的 AI 计算核 / 向量核，二者并列；初代 910 的 Vector 集成在 AI Core 内 |
| Cube 单元 | 矩阵计算单元，一周期做 16×16×16 矩阵乘加 = 4096 次 MAC |
| MTE | 搬运引擎，在 HBM 与片上缓冲（UB/L0）间搬数据，与计算重叠隐藏访存延迟 |
| MAC | 乘加运算；1 MAC = 1 次乘法 + 1 次加法 = 2 次浮点运算 |
| TFLOPS | 每秒万亿次浮点运算，衡量峰值算力 |
| HBM | 高带宽内存：DRAM 堆叠并与芯片同封装，带宽远高于 GDDR |
| 带宽 | 每秒可搬运的字节数（B/s），AI 推理的关键瓶颈之一 |
| KV cache | 推理时缓存的键/值张量，避免对已生成 token 重复计算 |
| GEMM | General Matrix Multiply，矩阵乘矩阵，大模型计算的主体 |
| decode / prefill | 逐 token 生成（M 小、带宽受限）/ 一次性算完整个 prompt 的预填充（M 大、计算受限） |
| CANN | Compute Architecture for Neural Networks，昇腾全栈软件，下集展开 |
| AscendCL | Ascend Computing Language，CANN 的主机侧编程接口，下集展开 |
| Atlas | 昇腾硬件产品品牌：模组 / 加速卡 / 服务器 / 集群 |
| 910B / 910C | 昇腾芯片代号；910B 约 376 TFLOPS FP16 + 64 GB HBM，910C 双 die、官方未公开完整规格（存疑） |
| 计算受限 / 带宽受限 | compute-bound / bandwidth-bound，耗时由算力或由数据搬运决定 |

## 下一步

明天（本模块下集）讲软件栈：CANN/驱动/固件层次关系（并画出与框架的层次关系图）、AscendCL、torch-npu 与性能工具（msprof）；之后第 8 课进入昇腾上的 vLLM：vllm-ascend 部署与调优。实践：在昇腾环境跑 `npu-smi info` 看 HBM 占用，再跑 vLLM 逐步加大 batch，观察吞吐是否先陡升后趋平。

## 来源

- [昇腾社区官网（hiascend.com）](https://www.hiascend.com/) — 官方文档
- [CANN 社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/index/index.html) — 官方文档
- [Atlas 800I A2 推理服务器（华为企业业务）](https://e.huawei.com/cn/products/computing/ascend/atlas-800i-a2) — 官方文档
- [Atlas 300I Duo 推理卡（华为企业业务）](https://e.huawei.com/cn/products/computing/ascend/atlas-300i-duo) — 官方文档
- [Atlas 800T A2 训练服务器（华为企业业务）](https://e.huawei.com/cn/products/computing/ascend/atlas-800t-a2) — 官方文档
- [Atlas 900 SuperCluster（华为企业业务）](https://e.huawei.com/cn/products/computing/ascend/atlas-900-supercluster) — 官方文档
- [Atlas 200I A2 边缘模组（华为企业业务）](https://e.huawei.com/cn/products/computing/ascend/atlas-200i-a2) — 官方文档
- [昇腾硬件：集群与超节点（昇腾社区）](https://www.hiascend.com/hardware/cluster) — 官方文档
- [昇腾硬件架构与 SHMEM 约束（CANN 社区 shmem 项目参考）](https://atomgit.com/cann/shmem/blob/master/.agents/skills/shmem-ops-design/references/hardware-architecture.md) — 教程
