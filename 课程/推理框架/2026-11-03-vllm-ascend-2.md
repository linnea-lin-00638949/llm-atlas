---
date: "2026-11-03"
title: "第 10 课 · 昇腾上的 vLLM：vllm-ascend 部署与调优（下）"
domain: "推理框架"
module_id: "vllm-ascend"
module: "昇腾上的 vLLM：vllm-ascend 部署与调优"
status: "已学"
lesson_no: 10
goals: ["在昇腾上拉起一个带 profiler 的 vLLM 服务，并说清 vllm-ascend 替换的是原生 vLLM 的哪一层", "把 decode 一步拆成权重读取、KV 读取、算力、通信四笔账，判断瓶颈在带宽、算力还是调度", "用 profiler 的证据识别算子回退、融合不足与通信开销，说出昇腾上最常见的三个性能坑"]
tags: ["vllm-ascend", "昇腾", "推理性能", "profiler", "量化", "性能建模"]
---

# 第 10 课 · 昇腾上的 vLLM：vllm-ascend 部署与调优（下）

> 学完今天，你能：
> 1. 在昇腾上拉起一个带 profiler 的 vLLM 服务，并说清 vllm-ascend 替换的是原生 vLLM 的哪一层
> 2. 把 decode 一步拆成权重读取、KV 读取、算力、通信四笔账，判断瓶颈在带宽、算力还是调度
> 3. 用 profiler 的证据识别算子回退、融合不足与通信开销，说出昇腾上最常见的三个性能坑

## 课前自测

1. 回顾 ascend-arch：decode 为什么是「带宽受限」而不是「算力受限」？
2. 回顾 vllm-core：V1 调度器每步按什么分配预算？`--max-num-batched-tokens` 调大后 TTFT 与 TPOT 怎么变？
3. 8B 模型用 BF16 存，权重多大？HBM（NPU 上的高带宽显存）带宽约 1.6 TB/s，读一遍最少要多久？
4. torch-npu 里张量放在哪个 device 字符串上？某算子在 NPU 上没有实现时，它会把张量悄悄搬到 CPU 算完，还是直接报错？
5. 线上 TPOT 多少毫秒？是算出来的还是测出来的？

（只给题不给答案；答不上来，先回看 vllm-core 与 ascend-arch。）

## 为什么需要它

上集讲了三层栈——vLLM（引擎）、vllm-ascend（昇腾硬件插件）、CANN（昇腾异构计算架构，含驱动、算子库与工具链）与 torch-npu（PyTorch 的昇腾后端）——以及怎么把服务拉起来。但「能跑」和「能上生产」之间隔着一道墙：同一个 8B 模型，有人单卡 70 token/s，有人只有 25；有人开量化省一半显存，有人显存没变、精度却掉了。原因几乎都归到三件事：算子落到昇腾上是什么形状（有没有融合、有没有回退）、精度怎么选（量化在昇腾上的真实路径）、以及你用什么证据判断瓶颈（profiler 加三步定位法）。

## 核心概念

### vllm-ascend（换一个视角）

定义：vLLM 官方社区维护的昇腾硬件插件（platform plugin），实现昇腾的平台类、worker（执行计算的进程）与自定义算子。

直觉：换发动机不改仪表盘——API、调度器、PagedAttention（按块表寻址的注意力）都原装，只换「跟硬件打交道的那层」。

机制：启动注册平台，日志出现 `Platform plugin ascend is activated` 即接管成功；插件另带补丁模块 `vllm_ascend/patch`，import 时替换 vLLM 中少数与硬件假设冲突的函数。

例子：某调度参数不按文档生效，可能出自补丁层而非 vLLM 本体。

### torch-npu

定义：PyTorch 的昇腾后端，注册设备类型 `npu`，把 aten 算子映射到 CANN 的 aclnn 算子库执行。

直觉：翻译官。翻不出来的词只能拆成几个词硬解释（慢），或直接报错（见 fallback）。

机制：管设备、流与事件、内存分配器（`PYTORCH_NPU_ALLOC_CONF`）与图捕获（`torch.npu.NPUGraph`）；vllm-ascend 的每行 Python 都落在这里，版本必须与 CANN 配套。

例子：版本错配常见表现是 import 就失败，或第一个 kernel 报符号找不到。

### 算子适配

定义：把模型需要的一项计算落到昇腾上「可执行且尽量快」的实现上的全过程。

直觉：乐高。官方件（aclnn 已有）→ 定制件（自研融合算子）→ 小积木硬拼（多个基础算子组合）→ 最后才是报错。

机制：`csrc/` 下分 `op_host`（主机侧）、`op_kernel`（内核侧）用 Ascend C（昇腾算子语言）写，编译成 aclnn、绑定 `torch.ops._C_ascend`，Python 侧 `enable_custom_op()` 打开。

例子：add + rms_norm 融成一个内核，省一次 HBM 往返与一次下发；decode 每步几百个内核，省的是「次数 × 固定开销」。

### fallback（算子回退）

定义：缺实现时的两种出路。① 组合重写：多个基础算子拼出等价计算，仍在 NPU 上跑，慢但透明；② 没有实现：直接抛 `NotImplementedError`，得改代码或换版本。

直觉：餐厅没这道菜，① 是拿现有食材拼一盘（味道对、出菜慢十倍），② 是直接说「本店不做」。

机制：回退不改结果，只改性能或直接失败；torch-npu 不会自动把张量搬上 CPU（要自己写 `.cpu()`，每步同步，线上不可用）。信号是 profiler 里大量小内核、算子前后有空洞。

例子：某新激活退化成 5 个逐元素小算子串联，8k prefill 慢 3 倍而输出完全正确。

### 量化（昇腾上的取舍）

定义：用更低位宽表示权重与激活，W8A8 即权重 8 bit、激活 8 bit。

直觉：精装书换平装，搬得快、翻得快，但字有损失。

机制：三条路。① ModelSlim（官方压缩工具）产 W8A8，启动加 `--quantization ascend`；② LLM-Compressor 产 compressed-tensors 格式，不加 flag；③ 直接用原生 FP8 checkpoint——只有块级（block-wise）缩放且硬件是 950PR/950DT 才走 FP8 矩阵乘；其它代际按块缩放把权重重算成 BF16 再算，近似但不等价，显存按 BF16 规划。

例子：同一个 FP8 checkpoint，950PR 上显存减半，别的卡显存没省——「开了量化显存没变」多出于此。

### profiler（性能剖析器）

定义：采集「谁在什么时候占用了多久」：时间戳、算子与内核耗时、调度与通信事件。

直觉：CT 与问诊，一个看器官（算子级），一个看流程（服务级）。

机制：Ascend PyTorch Profiler 内置、算子级：`--profiler-config '{"profiler":"torch","torch_profiler_dir":"./vllm_profile"}'` 加 `POST /start_profile`、`POST /stop_profile` 圈区间，产物 `*_ascend_pt` 经 `torch_npu.profiler.profiler.analyse()` 解析出 `kernel_details.csv`、`step_trace_time.csv`、`trace_view.json`；MS Service Profiler 是函数级，出 Chrome Tracing（浏览器时间轴）与 `request.csv`、`batch.csv`。

例子：查矩阵乘之间的空隙用前者，查 batch 组不起来、排队久用后者——查算子用前者，查调度用后者。

### 性能定位

定义：先算理论上限、再测实际、再把差值归因到算力/带宽/调度。

直觉：先算这段路最快几分钟，再实测，再查堵在哪。

机制：三步。(1) 上界：`T_mem = (权重字节 + KV 字节) ÷ 带宽`，`T_compute = 2N·B ÷ 有效算力`（2N 是每 token 乘加的常用近似）。(2) 测 TTFT（首 token 时延）、TPOT（每 token 时延）与吞吐。(3) 归因：矩阵乘占比低 → 融合不足或回退；内核间空洞大 → 下发/调度（试 `TASK_QUEUE_ENABLE=2`、`CPU_AFFINITY_CONF=1`）；all-reduce（多卡部分和相加后广播）占比高 → 看 TP 与 HCCL。

例子：实测 20 ms、下界 11.5 ms，那 42% 就是要去捞的东西；捞不到，说明下界漏了 KV 或通信。

## 一步一步理解

先回顾 ascend-arch：decode 逐 token 生成，每生成一个 token 都要把全部权重从 HBM 读一遍，所以天生带宽受限。下面算 8B 模型 decode 一步的账：N = 8×10⁹、BF16 每参数 2 B、HBM 带宽 1.6×10¹² B/s、FP16 峰值 376 TFLOPS（本课程统一取这组量级）。

第 1 步 权重：8×10⁹ × 2 B = 16 GB。

第 2 步 权重读取下界：16×10⁹ ÷ 1.6×10¹² = 10 ms。这是「带宽 100%」的理想地板；若带宽只跑到 70%（示例假设，实测常见 60%–80%），一步 14 ms，单请求上限约 70 token/s——「打七折」是假设，不是规律。

第 3 步 每 token 计算量 ≈ 2N = 16 GFLOP。

第 4 步 算力下界：MFU（算力利用率）20% 时有效算力 75.2 TFLOPS，每 token 16 ÷ 75.2 ≈ 0.213 ms。

第 5 步 求拐点：算术强度 = 每读 1 字节做多少次浮点运算。decode 每步读 2N 字节权重、做 2N·B FLOP，按权重算它 ≈ B。机器的算力/带宽比是 376 ÷ 1.6 = 235 FLOP/字节，乘算力侧 MFU 得 B* ≈ 235 × 0.2 ≈ 47——47 是这么来的。口径是「带宽 100%、算力 20%」：两效率同比例打折 B* 不变，但带宽若也跑到 70%，B* ≈ 235 × 0.2 ÷ 0.7 ≈ 67。所以记「47 上下、实际 47–70」，别背成常数。

第 6 步 补上漏掉的 KV：Qwen3-8B 每 token 每层 KV = 2 × 8 个 KV 头 × 128 维 × 2 B = 4 KB，36 层共 144 KB/token。8k 上下文单序列约 1.2 GB，B=32 时约 38 GB，是权重的两倍多，一步下界变成 (16+38) ÷ 1.6 = 34 ms，不是 10 ms；512 token 上下文时 KV 只 2.4 GB，一步 11.5 ms，10 ms 才近乎成立。更看边际成本：8k 时每多一个序列 KV 多 1.2 GB（+0.75 ms），算力只多 0.213 ms——带宽永远更贵。所以「B < 47 时加并发几乎不涨时间」只在 KV 极小时成立；8k 下加并发就是直接涨时间。

第 7 步 量化：权重换 INT8，8 GB，读取下界 5 ms；设 INT8 算力为 FP16 的 2 倍（150 TFLOPS），每 token 0.107 ms，B* = 5 ÷ 0.107 ≈ 47，与 BF16 几乎一样。收益看上下文：512 token 时 (8+2.4) ÷ 1.6 = 6.5 ms，对 11.5 ms 约 1.8×；8k 时 (8+38) ÷ 1.6 = 29 ms，对 34 ms 只有 1.2×——量化压的是权重，KV 一字节没省。这就是「decode 量化收益大」的边界。

第 8 步 profiler 验证：B=32、512 token 上下文，下界 11.5 ms、实测每步 20 ms，带宽利用率约 58%。打开 `kernel_details.csv` 按耗时排序：榜首是成堆的逐元素小算子而非矩阵乘 → 融合不足或回退；矩阵乘间空隙大 → 下发/CPU 瓶颈；all-reduce 占三成 → 通信。

### 张量并行的补充账

前提：hidden = 4096、B = 32、BF16 每元素 2 B、Qwen3-8B 共 36 层、TP = 4，先忽略 KV；卡间通信有效带宽 100 GB/s、单次延迟 30 μs（存疑，用实测的 HCCL 小消息延迟替换）。

① 每卡权重读取：4 GB ÷ 1.6 TB/s = 2.5 ms。
② 每卡每 token 计算：4 GFLOP ÷ 75.2 TFLOPS ≈ 0.053 ms；B=32 时约 1.7 ms。
③ 通信量：单次 all-reduce 消息 = 批内 token 数 × hidden × 每元素字节 = 32 × 4096 × 2 B = 256 KB；36 层 × 2 = 72 次共 18 MB ÷ 100 GB/s ≈ 0.18 ms，可忽略。
④ 通信延迟：72 × 30 μs ≈ 2.2 ms，不随 batch 缩小，也不因消息变小而消失。
⑤ 合计：max(2.5, 1.7) + 2.2 ≈ 4.7 ms，对单卡 10 ms 约 2.1×。

结论：TP 在 decode 有真实收益，但远不到线性，原因就在三笔账里：权重读取被摊薄（÷4），通信固定延迟被加上（2.2 ms，与 B 无关，batch 越小占比越高）。上下文变长时 KV 也按 KV 头切分（8 个头 ÷ 4 = 每卡 2 个），8k 时每卡 (4+9.5) ÷ 1.6 ≈ 8.4 ms，加通信约 10.6 ms，对单卡 34 ms 约 3.2×——长上下文更能摊薄通信。一句话：装不下必须 TP，prefill 收益接近线性，decode 有 2–3× 但不是 4×。

串成一条启动命令：

`export TASK_QUEUE_ENABLE=2 CPU_AFFINITY_CONF=1`
`vllm serve /models/Qwen3-8B-W8A8 --quantization ascend --tensor-parallel-size 4 --max-model-len 8192 --max-num-batched-tokens 16384 --profiler-config '{"profiler":"torch","torch_profiler_dir":"./prof"}'`

再用 `curl -X POST localhost:8000/start_profile`、发压、`curl -X POST localhost:8000/stop_profile` 收数据。

## 常见误区

1. 「用了 FP8 权重就一定省显存、一定更快」。错：非 950PR/950DT 上原生 FP8 checkpoint 被摊回 BF16 执行，显存按 BF16 规划（约 2 倍于 FP8 文件）；要真省得走 ModelSlim / LLM-Compressor 的 W8A8，并确认量化后端生效。

2. 「能跑通、输出正确，说明算子都在昇腾上、性能没问题」。错：组合重写只保正确不保速度，一个算子变五个，decode 每步都付；没实现的算子是直接报错，不是悄悄上 CPU。依据只能是 profiler 的内核构成。

3. 「TP 越大越快」。错：decode 有 2–3× 但远非线性，通信固定延迟不随 batch 变小；注意力头数（含 KV 头数）要能被 TP 整除，否则起不来。

## 一句话总结

昇腾上 vLLM 的性能，八成由三件事决定：算子有没有融合、精度路径对不对得上硬件、有没有用 profiler 把「实测减理论」的差值归因到带宽/算力/调度——而理论那一项里，KV 流量常常比权重更大。

## 自测题

### 1. 服务跑在 NPU 上、输出正确，能否断定算子都落在昇腾上？

<details><summary>查看答案</summary>不能。回退有两种：① 组合重写——多个基础算子拼出等价计算，仍在 NPU 上跑，正确但慢（一个变五个，decode 每步都付）；② 没实现——抛 `NotImplementedError`，不会自动搬 CPU 替你算完。「能跑通」只说明走通 ①，或没碰到缺的算子；算子在哪、融得好不好，只能看 profiler 的内核构成。</details>

### 2. 为什么 B* ≈ 47？带宽只跑到 70% 呢？8k 上下文还谈得上 B* 吗？

<details><summary>查看答案</summary>算术强度 = 每读 1 字节做多少 FLOP，decode 按权重算 ≈ B；算力/带宽比 376 ÷ 1.6 = 235，乘 MFU 0.2 得 ≈ 47（口径：带宽 100%、算力 20%）。两效率同比例打折 B* 不变；带宽若只 70%，B* ≈ 235 × 0.2 ÷ 0.7 ≈ 67。8k 时变形：每多一个序列 KV 多 1.2 GB（+0.75 ms），算力只多 0.213 ms，一步永远落在带宽侧，B* 失效——加并发直接涨时间。</details>

### 3. TP = 4、B = 32，为什么 decode 只有约 2.1×？

<details><summary>查看答案</summary>三笔账：每卡权重读取 4 GB ÷ 1.6 TB/s = 2.5 ms（摊薄 4 倍）；每卡计算 4 GFLOP ÷ 75.2 TFLOPS ≈ 0.053 ms/token、B=32 时 1.7 ms；通信每次 32×4096×2 B = 256 KB、72 次共 18 MB，传输仅 0.18 ms，但 72×30 μs ≈ 2.2 ms 的固定延迟摊不薄。合计 max(2.5,1.7)+2.2 ≈ 4.7 ms，对单卡 10 ms 约 2.1×。batch 越小，2.2 ms 占比越高。</details>

### 4. 「矩阵乘之间有大段空隙」，该用哪个 profiler、看什么、下一步动什么？

<details><summary>查看答案</summary>用算子级的 Ascend PyTorch Profiler（`--profiler-config` + `/start_profile`、`/stop_profile`），解析 `*_ascend_pt` 后看 `kernel_details.csv` 的时间戳间隙与 task 耗时，先分清是下发（host 侧）慢还是内核慢。若在下发/调度，先试 `TASK_QUEUE_ENABLE=2` 与 `CPU_AFFINITY_CONF=1`；若是 batch 组不起来、排队久，换 MS Service Profiler 看 `batch.csv`、`request.csv`。</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| vllm-ascend | vLLM 的昇腾硬件插件，替换与硬件交互的一层，带 `patch` 补丁模块 |
| torch-npu | PyTorch 的昇腾后端，把 aten 算子映射到 CANN 的 aclnn 算子库执行 |
| 算子适配 | 把模型计算落到昇腾可执行实现上的全过程 |
| fallback（算子回退） | 缺算子时组合重写（慢、仍在 NPU），或直接抛 `NotImplementedError` |
| 量化（W8A8） | 权重 8 bit、激活 8 bit，省显存与带宽，精度有损 |
| ModelSlim | 昇腾官方压缩工具，产 W8A8 权重，需 `--quantization ascend` |
| profiler | 采集算子/内核/调度耗时、定位瓶颈 |
| ascend_pt | Ascend PyTorch Profiler 的原始产物，解析后才能读 |
| 算术强度 | 每读 1 字节做多少次浮点运算；decode 按权重算 ≈ 批内 token 数 B |
| B*（拐点 batch） | 带宽与算力下界相等处的批大小 ≈ 峰值算力÷带宽 × MFU（本课约 47） |
| 性能定位 | 算理论下界、测实际，把差值归因到带宽/算力/调度 |
| HCCL | 昇腾的集合通信库，多卡 all-reduce 等通信的执行者 |
| 张量并行（TP） | 把权重切到多卡、每层做集合通信的并行方式 |

## 下一步

下一课「量化与编译基础」把本课点到为止的两处展开：量化的数学（scale/zero-point、per-tensor 与 per-channel 的差别、精度为什么掉、KV 能否量化）与编译（图捕获、算子融合、动态 shape 的代价）。

## 来源

1. vLLM Ascend 官方文档首页 — https://docs.vllm.ai/projects/ascend/en/latest/
2. 量化指南（vLLM Ascend） — https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html
3. 服务性能分析指南（vLLM Ascend） — https://docs.vllm.ai/projects/ascend/zh-cn/latest/developer_guide/performance_and_debug/service_profiling_guide.html
4. 优化与调优（vLLM Ascend） — https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/optimization_and_tuning.html
5. 添加自定义 aclnn 算子（vLLM Ascend） — https://docs.vllm.ai/projects/ascend/zh-cn/latest/developer_guide/Design_Documents/add_custom_aclnn_op.html
6. 图模式指南（vLLM Ascend） — https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/feature_guide/graph_mode.html
7. Williams, Waterman, Patterson. Roofline: An Insightful Visual Performance Model for Multicore Architectures. CACM 2009 — https://dl.acm.org/doi/10.1145/1498765.1498785
