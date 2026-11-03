---
date: "2026-11-03"
title: "第 8 课 · 昇腾 NPU 架构与 CANN 生态（下）"
domain: "硬件"
module_id: "ascend-arch"
module: "昇腾 NPU 架构与 CANN 生态"
status: "已学"
lesson_no: 8
goals: ["能画出 CANN/驱动/固件与 PyTorch、MindSpore 的层次关系，说清「谁调用谁」", "能说清 AscendCL、torch-npu、MindSpore 各自的位置与接入路径", "能手算 decode 吞吐上限并判读带宽/算力瓶颈，知道 profiler 看什么"]
tags: ["昇腾 NPU", "CANN", "torch-npu", "AscendCL", "性能剖析", "vLLM"]
---

# 第 8 课 · 昇腾 NPU 架构与 CANN 生态（下）

> 学完今天，你能：
> 1. 画出 CANN/驱动/固件与 PyTorch、MindSpore 的层次关系，说清「谁调用谁」
> 2. 说清 AscendCL、torch-npu、MindSpore 各自的位置与接入路径
> 3. 手算 decode 吞吐上限并判读带宽/算力瓶颈，知道 profiler 看什么

## 课前自测

1. CANN 是什么的缩写？它和 PyTorch 是什么关系？
2. torch-npu 是独立深度学习框架吗？只装 torch-npu、不装 CANN，能跑起来吗？
3. 推理时每生成 1 个 token，为什么要把全部模型权重读一遍？decode 阶段是算力受限还是显存带宽受限？
4. 用什么工具能看到每个算子在 NPU 上各花了多少时间？
5. MindSpore 2.x 默认是动态图还是静态图？它接入昇腾走的是 AscendCL 吗？

## 为什么需要它

你的工作是「昇腾 NPU + vLLM（大模型推理引擎，基于 PyTorch）做推理」。回顾上集：NPU 与 GPU 设计哲学差异是——GPU 靠大量通用小核兼顾图形与并行，NPU 专为矩阵乘等神经网络负载定制；昇腾核心是达芬奇架构 AI Core，含 Cube（矩阵乘）、Vector（向量）、Scalar（标量）三类单元；Atlas 产品线按训练、推理、边缘定位划分。日常三个高频问题：部署报 `CANN not found`、算子不支持；吞吐上不去；不知瓶颈看什么指标。答案都在昇腾软件栈的层次里。本集讲清「谁调用谁、谁编译谁、谁搬数据」，为下一课 vLLM 调优铺路。

## 核心概念

### CANN 与软件栈层次

**定义**：CANN（Compute Architecture for Neural Networks）是昇腾软件底座：驱动、固件、运行时、编译器、算子库、编程接口与性能工具都属其中。

**直觉**：昇腾 NPU 像一台裸机：驱动/固件像主板固件与内核，AscendCL 像系统调用接口，torch-npu 像把 PyTorch 翻译成这些调用的适配器，vLLM 是应用。

**机制**（自顶向下，两条框架路径）：

```text
vLLM ─ PyTorch ─ torch-npu ─→ AscendCL（torch-npu 使用的编程接口）
MindSpore ─→ GE + MindRT（图引擎 / 运行时）
                 │
       算子库 / 图编译 / 算子编译 ─ Runtime（执行环境）
                 │
       驱动 Driver / 固件 Firmware
                 │
    达芬奇硬件：AI Core / HBM / 片上总线
```

两条路径平级，都落在 CANN 算子库与 Runtime 上。

**例子**：一次 `torch.mm` 的旅程：torch-npu → AscendCL 矩阵乘调用 → 算子库取（或现编译）NPU 二进制 → Runtime 下发 AI Core。框架不认识 NPU，CANN 才是 NPU 的「操作系统」。

### AscendCL 与 torch-npu

**定义**：AscendCL（Ascend Computing Language，ACL）是 CANN 的 C/C++ 编程接口，管设备初始化、显存分配、数据搬运与算子执行。torch-npu 是华为维护的 PyTorch 插件（设备名 `"npu"`），把每个算子映射到 CANN。

**直觉**：AscendCL 之于昇腾 ≈ CUDA Runtime 之于 NVIDIA；torch-npu 是「接线板」：上接熟悉的 PyTorch，下接昇腾。

**机制**：`import torch_npu` 注册 npu 设备后，`torch.tensor(..., device="npu")` 建 HBM 张量，`torch.mm` 被分派到 CANN 算子。它不重写 PyTorch，是插件而非新框架。

**例子**：排查自底向上：驱动 → CANN → torch-npu——能 import 但一跑就报错，说明下面某层缺失或版本不配套。

### MindSpore 的位置

**定义**：MindSpore 是华为开源的深度学习框架，原生支持昇腾；2.x 起默认 PyNative 动态图，静态图需显式 `set_context(GRAPH_MODE)` 或 `@jit`。

**直觉**：昇腾上有两条路：PyTorch + torch-npu（vLLM 走这条）与 MindSpore 原生路线，平级而非上下级。

**机制**：MindSpore 不经 AscendCL：昇腾后端由 CANN 图引擎 GE 负责图编译与执行、自有运行时 MindRT 负责调度。它以静态图整图下沉的优化见长（算子融合、内存复用、自动并行），生产训练常切 GRAPH_MODE。

**例子**：做 vLLM 不必学它，但要清楚它不替代 CANN、也不走 AscendCL。

### 编译与算子调度

**定义**：算子（operator）是 AI Core 上执行的计算内核，如矩阵乘、LayerNorm。图编译把整张图优化（算子融合、常量折叠）后生成执行序列；算子调度是运行时按依赖把算子排队派发 AI Core。

**直觉**：编译像把菜谱翻译成后厨工序：图编译=合并工序（切菜腌肉合一步，少洗一次砧板）；算子调度=决定哪道工序进哪口锅。

**机制**：PyTorch 默认动态图（eager）：算子逐个立即下发执行，不做整图编译，灵活但优化少；`torch.compile`（PyTorch 图编译 API，可做整图融合优化）与 MindSpore 静态图走整图编译。CANN 还有自动调优，为同一算子挑最快实现。

**例子**：算子融合对 vLLM 尤其重要：融合掉的中间结果不用写回 HBM，省带宽；decode 阶段省带宽就是提吞吐。

### 用 profiler 看瓶颈

**定义**：profiler（性能剖析工具）给程序打点计时。昇腾可用 msprof 命令行 + Ascend Insight 网页，或 PyTorch 里 `torch_npu.profiler`。

**直觉**：给每个算子装秒表、给整卡装监控：谁耗时最多、AI Core 多忙、显存拷贝花多久，一目了然。

**机制**：关键指标——AI Core 利用率（ai_core_utilization）、算子耗时排名（op_statistic）、显存拷贝耗时、算子间空泡。利用率高=算力瓶颈；利用率低但拷贝多/有空泡=带宽或调度瓶颈。

**例子**：下一课调 vLLM 的第一步永远是先跑 profiler，再决定加 batch 还是换算子。

## 一步一步理解

**目标**：手算 7B 模型在昇腾上 decode（逐 token 生成）阶段的吞吐上限并判断瓶颈，只用乘除法。

**已知（假设）**：7B 模型（70 亿参数），FP16（每参数 2 字节）→ 权重 14 GB。HBM 带宽取 1.6 TB/s（昇腾 910 系列量级流传值，官方未公布精确数字，存疑；换实测带宽重算即可）。NPU 算力取约 300 TFLOPS（量级值，存疑）。

**第 1 步：decode 1 个 token 读多少数据？** 每个权重在矩阵-向量乘里恰好被用一次（从 HBM 读进 Cube 单元），故约读 14 GB。另有 KV cache（键值缓存）等读取：短上下文时远小于权重，先忽略，不影响带宽受限结论；长上下文时会变大。

**第 2 步：读多久？** 14×10⁹ B ÷ 1.6×10¹² B/s = 8.75 ms。

**第 3 步：单请求吞吐。** 1 ÷ 0.00875 ≈ 114 tokens/s。

**第 4 步：batch=8 呢？** 权重只读一遍、服务 8 个 token：8 ÷ 0.00875 ≈ 914 tokens/s。总吞吐涨约 8 倍，单请求延迟几乎不变。

**第 5 步：为什么带宽受限？** 每权重读 2 字节、做 1 次乘加 ≈ 2 FLOP → 算术强度 1 FLOP/字节；卡能支撑 ≈ 300×10¹² ÷ 1.6×10¹² ≈ 188 FLOP/字节。1 ≪ 188：算力在等数据，瓶颈是显存带宽（memory-bound，带宽受限）。

**落到 vLLM**：decode 带宽受限 → 吞吐靠加大 batch；batch 上限由显存容量（KV cache 占大头）和算力封顶。这正是 continuous batching（动态合批）的价值，即下一课调优主线。

## 常见误区

**误区 1：「torch-npu 是昇腾版 PyTorch，装了它就不用装 CANN。」** 事实：torch-npu 只是适配插件，每个算子都要落到 CANN，驱动、固件、CANN 缺一不可，且版本配套。症状：能 import，一跑报 CANN 找不到/算子不支持。排查自底向上：驱动 → CANN → torch-npu。

**误区 2：「decode 慢就是 NPU 算力不够。」** 事实：单请求 decode 带宽受限（见手算），算力翻倍而带宽不变几乎不提升单请求吞吐。动作：看 AI Core 利用率——低利用率+长耗时=在等数据。提升靠加 batch 或量化（少读权重）。

**误区 3：「昇腾上必须用 MindSpore。」** 事实：vLLM 全线走 PyTorch + torch-npu，官方维护 vllm-ascend；MindSpore 是可选项。选型看生态，不看「纯血」。

## 一句话总结

昇腾软件栈自顶向下：框架（PyTorch/vLLM）→ 适配层（torch-npu）→ CANN（AscendCL+编译+算子库+Runtime）→ 驱动/固件 → 达芬奇硬件；MindSpore 走 GE+MindRT 旁路，同样落在 CANN 之上。decode 阶段显存带宽是主瓶颈：先用 profiler 定位（AI Core 利用率、算子耗时、拷贝），再靠 batch 提吞吐——这就是下一课调 vLLM 的地基。

## 自测题

### 1. CANN 与驱动、固件是什么关系？为什么「只装 torch-npu」跑不起来？
<details><summary>查看答案</summary>驱动与固件是 CANN 底座最底层，负责硬件初始化与任务下发，需与 CANN 版本配套；torch-npu 只是把算子映射到 CANN 的插件，缺了它们算子无处执行。症状：能 import，一跑报 CANN not found。排查自底向上：驱动 → CANN → torch-npu。</details>

### 2. MindSpore 接入昇腾走 AscendCL 吗？它默认是静态图吗？
<details><summary>查看答案</summary>不走 AscendCL：MindSpore 昇腾后端由 CANN 图引擎 GE 编译执行、自有运行时 MindRT 调度；AscendCL 是 torch-npu 与独立应用用的接口。2.x 起默认 PyNative 动态图，静态图需 GRAPH_MODE 或 @jit。</details>

### 3. decode 每生成 1 个 token 为什么要读一遍 14 GB 权重？瓶颈是算力还是带宽？
<details><summary>查看答案</summary>decode 是矩阵-向量乘，每个权重被读一次：14 GB ÷ 1.6 TB/s = 8.75 ms → 单请求约 114 tokens/s，batch=8 约 914 tokens/s。算术强度约 1 FLOP/字节，远低于卡能支撑的约 188 → 带宽受限。KV cache 短上下文可忽略、长上下文不可忽略。</details>

### 4. 用什么工具看每个算子在 NPU 上的耗时？拿到指标怎么判读？
<details><summary>查看答案</summary>msprof + Ascend Insight，或 torch_npu.profiler。看 ai_core_utilization、op_statistic、拷贝耗时与空泡：利用率高=算力瓶颈；利用率低但拷贝多/有空泡=带宽或调度瓶颈。</details>

### 5. batch 从 1 加到 8，为什么总吞吐涨约 8 倍而单请求延迟几乎不变？
<details><summary>查看答案</summary>权重读取的 8.75 ms 是固定成本，被 8 个请求分摊：单请求延迟几乎不变，总吞吐约 8 倍。batch 上限由显存（KV cache）与算力封顶——这正是 continuous batching 的价值。</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| CANN | 昇腾软件底座（Compute Architecture for Neural Networks），含驱动、固件、编译器、算子库、编程接口与性能工具 |
| AscendCL（ACL） | CANN 的 C/C++ 编程接口，管理设备、显存、搬运与算子执行，是 torch-npu 用的接口，角色类似 CUDA Runtime |
| torch-npu | 华为维护的 PyTorch 适配插件，把算子映射到 CANN，提供 device="npu" |
| MindSpore | 华为开源深度学习框架，2.x 起默认 PyNative 动态图；昇腾后端走 GE + MindRT，不经 AscendCL |
| vllm-ascend | vLLM 的昇腾后端插件，基于 torch-npu，昇腾上跑 vLLM 的官方路线 |
| 算子（operator） | AI Core 上执行的计算内核，如矩阵乘、LayerNorm |
| 图编译 | 对整张图做算子融合、常量折叠等优化后再执行；eager 则算子逐个立即下发、不整图编译 |
| AI Core | 昇腾 NPU 核心计算单元，含 Cube（矩阵乘）、Vector（向量）、标量三类单元 |
| HBM | NPU 高带宽显存，权重与 KV cache 存于此，带宽主要决定 decode 吞吐上限 |
| msprof / Ascend Insight | CANN 性能剖析工具链，输出算子耗时、AI Core 利用率等 |
| memory-bound（带宽受限） | 计算单元多数时间在等显存数据而非计算，decode 典型如此 |
| 驱动与固件 | 最底层设备软件，负责硬件初始化与任务下发，需与 CANN 版本配套 |

## 下一步

下一课《昇腾上的 vLLM：vllm-ascend 部署与调优》：部署 vllm-ascend、continuous batching 与 KV cache 如何影响吞吐、HCCL（昇腾集合通信库）多卡组网、拿到 profiler 结果后如何调参。

## 来源

1. 昇腾社区官方文档中心（CANN / AscendCL / Profiling） — https://www.hiascend.com/document
2. CANN 性能调优工具：op_statistic 算子耗时统计（历史版本示例，若失效请从昇腾文档中心搜索） — https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devaids/devtools/profiling/atlasprofiling_16_0068.html
3. torch_npu 官方仓库（Ascend/pytorch） — https://github.com/Ascend/pytorch
4. vllm-ascend 官方仓库 — https://github.com/vllm-project/vllm-ascend
5. vLLM 昇腾插件官方文档 — https://docs.vllm.ai/projects/ascend/zh-cn/v0.23.0rc1/installation.html
6. MindSpore 官网 — https://www.mindspore.cn
