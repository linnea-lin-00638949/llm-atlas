---
date: "2026-11-03"
title: "第 9 课 · 昇腾上的 vLLM：vllm-ascend 部署与调优（上）"
domain: "推理框架"
module_id: "vllm-ascend"
module: "昇腾上的 vLLM：vllm-ascend 部署与调优"
status: "已学"
lesson_no: 9
goals: ["按官方矩阵锁好版本，在昇腾机器上拉起 vLLM 服务并发出第一个请求", "用 CANN / torch-npu / vllm-ascend 三层栈解释它和原生 vLLM 的关系，判断故障落在哪一层", "用 KV 池与带宽手算并发上限和 TTFT/TPOT 量级，知道后续用 profiler 把瓶颈归到算力/带宽/调度"]
tags: ["vllm-ascend", "昇腾 NPU", "CANN", "torch-npu", "KV cache", "推理框架"]
---

# 第 9 课 · 昇腾上的 vLLM：vllm-ascend 部署与调优（上）

> 学完今天，你能：
> 1. 按官方矩阵锁好版本，在昇腾机器上拉起 vLLM 服务并发出第一个请求
> 2. 用 CANN / torch-npu / vllm-ascend 三层栈解释它和原生 vLLM 的关系，判断故障落在哪一层
> 3. 用 KV 池与带宽手算并发上限和 TTFT/TPOT 量级，知道后续用 profiler 把瓶颈归到算力/带宽/调度

## 课前自测

1. 同一个模型在 CUDA 卡上 `pip install vllm` 就能跑；换到昇腾 NPU，缺的那块东西是什么？它装在 vLLM 里面还是外面？
2. `torch==2.10.0` 配 `torch_npu==2.8.0`，你以为会「慢一点」，实际会发生什么？为什么？
3. 64 GB 的卡跑 Qwen3-0.6B、权重才 1.2 GB；`--max-model-len` 从 4096 改到 32768，并发会从 100+ 掉到十几条——为什么？

## 为什么需要它

vLLM 的主链路（vllm-core 那课）：API Server 收请求分词 → Engine Core 调度 → Worker 做前向 → 采样出下一个 token。Worker 需要一个真实设备后端来执行矩阵乘、注意力、通信。昇腾 NPU 的算力由 CANN 暴露（ascend-arch 那课）。

问题在接口处：vLLM 自带的平台后端里没有昇腾——它不认识 `npu`，也不知道怎么在昇腾上编译 kernel（在设备上跑的计算函数）。所以「原版 vLLM + 昇腾卡」不是「装不装插件差不多」，而是根本跑不起来——补上这一块的就是 vllm-ascend。

这一课打地基：三层栈谁管谁、版本为什么必须整行锁定、怎么装出可复现的环境、怎么发出第一个请求并确认「真的是 NPU 在算」。算子 fallback、量化、profiler、多卡 TP 留给后续课时。

## 核心概念

### 三层推理栈：CANN / torch-npu / vllm-ascend

定义：CANN（运行时 + 算子库）→ torch-npu（PyTorch 的昇腾后端）→ vllm-ascend（vLLM 的平台插件），三层压在昇腾驱动（HDK，含驱动和固件）与 NPU 硬件之上。

直觉：CANN 是「运行时 + 标准数学库」，torch-npu 是「这门语言的 SDK 绑定」，vllm-ascend 是「用这套 SDK 写出来的应用引擎」；上层只调下层，报错沿这条链往上冒。

机制：一次前向是 `vLLM(Python) → torch-npu → CANN 运行时 / ACLNN 算子库 → 驱动 → AI Core`。定位第一步就是判断错在哪一层：`npu-smi info` 看不到卡 → 驱动或容器设备映射；`torch.npu.is_available()` 为 False → CANN 或 torch-npu 版本；能 import 但日志没有 `Platform plugin ascend is activated` → 插件或 vLLM 版本对不上。

### CANN

定义：Compute Architecture for Neural Networks，昇腾的异构计算架构：驱动接口、运行时、算子库（ACLNN）、集合通信库（HCCL）、配套工具（msprof 等）。

直觉：相当于 CUDA Toolkit + cuBLAS + NCCL——但不含驱动，驱动与固件在 HDK 层。

机制：上层框架不直接写硬件指令，而是调用 CANN 的算子（operator，封装好的计算单元，如 MatMul、FlashAttention），由 CANN 调度到 AI Core（昇腾的矩阵计算单元）执行。一个算子有没有原生实现，决定它跑得快不快——这是后续「算子适配」的起点。

### torch-npu

定义：PyTorch 的昇腾后端，通过 PrivateUse1（PyTorch 给非 CUDA 设备预留的注册机制）注册出 `npu` 设备，让 `tensor.npu()`、`torch.npu.synchronize()`、`torch.npu.device_count()` 可用。

直觉：PyTorch 是通用遥控器，torch-npu 是昇腾这台机器的红外码表——没有码表，按键没反应。

机制：它把 aten 算子（`add`、`bmm`、`softmax` 等）逐个映射到 ACLNN；没被映射的走 fallback（回退）：轻则落到 CPU 执行（极慢），重则直接报错。`import torch_npu` 这个动作本身就是注册后端，所以只 `import torch` 时 `torch.npu` 并不存在。

### vllm-ascend

定义：vLLM 的昇腾平台插件（platform plugin），把 vLLM 的设备抽象层完整实现到 NPU 上。

直觉：vLLM 是「可以换发动机」的车，vllm-ascend 是昇腾发动机的安装支架加传动轴——它自己不是车。

机制：它通过 entry point（入口点，包安装时声明「我提供哪类扩展」的元数据）注册 `vllm_ascend:register`；vLLM 启动时枚举 `vllm.platform_plugins` 组下的插件，发现 ascend 就激活，日志打印 `Platform plugin ascend is activated`。此后 Platform（设备属性、显存查询）、Worker、ModelRunner（组 batch、管 KV 缓存）、Attention backend、量化方法都由它提供。它实现的是 vLLM 的内部接口，vLLM 一升级，接口就可能变——这是它脆弱的原因。

### 算子适配

定义：让模型里每个计算（MatMul、RMSNorm、RoPE 位置编码、注意力、MoE 路由……）在目标硬件上都有可用实现；没有就补实现，或退回低效路径。

直觉：把英文菜谱翻成本地厨房能执行的步骤；厨房没有烤箱就换蒸锅，做出来能吃，但口感不同（性能或精度有差异）。

机制：三条路——(1) CANN/ACLNN 已有该算子，直接调用（最快）；(2) 用 Triton Ascend 或 Ascend C（昇腾的自定义算子编程接口）写算子，要编译、要验精度；(3) fallback 成若干已有小算子拼起来，或搬回 CPU。vLLM 每次支持新模型结构，最先卡住的往往就是这一层。

### 版本矩阵

定义：官方维护的兼容表，把 vllm-ascend、vLLM、CANN、NNAL、torch/torch-npu、HDK、Python 钉成一行。

直觉：不是「装最新版最好」，而是「整行一起换」的套装；混搭等于自己造一个没人验证过的组合。

机制：vllm-ascend 自 vLLM 0.7.x 起与 vLLM 同号发版（PEP 440，Python 的版本号规范），v0.30.0 就对应 vLLM v0.30.0。写作时（2026-11）官方 v0.30.0 的验证组合是：vllm-ascend v0.30.0 / vLLM v0.30.0 / CANN Toolkit+Ops 9.2.1 / NNAL 9.2.1 / torch 2.10.0 + torch_npu 2.10.0.post7 / HDK 26.1.1 / Python >=3.10,<3.13。数字一定会过期：抄之前先看矩阵页当日内容、核对日期，再整行安装。

### 配套不上会怎样（自测第 2 题的答案）

torch 与 torch_npu 版本段不匹配，不是「慢一点」，典型是三种：`import torch_npu` 直接抛错（import 时校验 torch 版本与编译 ABI）；能 import，但首个 npu 算子报 not implemented；或能跑出结果但数值异常。机制是 torch_npu 的二进制针对特定 torch 版本编译，版本段或 ABI 对不上就找不到符号。vLLM 与 vllm-ascend 不配套则是在 register 或内部接口处崩。

## 一步一步理解

**第 1 步 先确认卡在**。`npu-smi info` 应列出设备、Health 为 OK；看不到卡先查驱动与容器 `--device /dev/davinci0` 映射，别急着装 Python 包。

**第 2 步 按矩阵选一行并写死**。把 vllm-ascend、vLLM、CANN、torch_npu 写进镜像 tag 或 requirements，不要 `pip install -U vllm vllm-ascend`。

**第 3 步 优先用官方镜像**。预构建镜像里 CANN、NNAL（CANN 中提供 ATB 算子库运行时的组件，版本随 CANN 对齐）、PyTorch、torch-npu、vLLM、vllm-ascend 已是一套验证过的组合，能省掉最耗时的版本对齐。进容器先做两行验证：

```python
import torch, torch_npu, vllm, vllm_ascend
print(torch.npu.is_available(), torch.npu.device_count())
print(torch.__version__, torch_npu.__version__)
```

**第 4 步 离线跑通十行代码**（官方 Quick Start 的 0.6B 例子）：

```python
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen3-0.6B')
out = llm.generate(['Hello, my name is'], SamplingParams(temperature=0.0, max_tokens=32))
print(out[0].outputs[0].text)
```

首次加载要几分钟（读权重、建 KV 池、图捕获）。看到 `Platform plugin ascend is activated` 才算插件接管；若报 CUDA 相关错误，是插件没加载。

**第 5 步 起服务、发第一个请求**：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
vllm serve Qwen/Qwen3-0.6B --max-model-len 4096 --gpu-memory-utilization 0.9
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"用一句话解释 KV cache"}],"max_tokens":64}'
```

两个坑：`--gpu-memory-utilization` 名字里带 gpu 是历史包袱，NPU 上含义不变（允许 vLLM 用掉的显存比例）；`--max-model-len` 不写就取模型 config 的最大上下文（Qwen3 常见 32768 甚至更长）。但 `--max-model-len` 并不直接决定启动成败：启动只校验「单条 max-model-len 的 KV 装不装得进 KV 池」，大模型或超长上下文（例如设到 20 万 token）才会启动报错，小模型上表现为并发骤降。

**手算：一张 64 GB 卡能装多少请求、一个 token 要多久**

设定：Qwen3-0.6B，BF16（每个数 2 字节），28 层，8 个 KV 头，每头 128 维；单卡 HBM 64 GB。

1. 权重：0.6×10⁹ 参数 × 2 B = 1.2 GB。
2. 可用显存：64×0.9 = 57.6 GB；扣掉 CANN 运行时、通信缓冲、图捕获约 4 GB（量级估计），约 53.6 GB 给 KV 池。
3. 每 token 的 KV：K、V 各一份 → 2×8×128×2 B = 4096 B/层；×28 层 = 114,688 B ≈ 112 KiB/token。
4. 一条 4096 token 请求：114,688 B × 4096 = 469.8 MB（448 MiB）；53.6 GB ÷ 469.8 MB ≈ 114 条。
5. 改成 32768：单条 114,688 B × 32768 ≈ 3.6 GB，只剩 53.6 ÷ 3.6 ≈ 14 条。同一张卡并发差 8 倍——这就是自测第 3 题。
6. 对账：块大小 16 token，每块 114,688×16 = 1,835,008 B ≈ 1.835 MB；53.6 GB ÷ 1.835 MB ≈ 29,200 块，约 46.7 万 token。启动日志的 KV cache 容量应落在这个量级，对不上说明参数或模型配置抄错了。（第 3 步的 112 KiB 是显示用近似值，连乘要用 114,688 B，别拿 112 去乘。）
7. decode 时延下界：batch = 1 时每 token 要把 1.2 GB 权重从 HBM 读一遍，按 1.6 TB/s 算约 0.75 ms；4096 上下文还要多读约 0.45 GB KV，合并约 1.0 ms/token，生成 32 个 token 至少 32 ms。实测常见是它的 2–4 倍（取决于图模式与算子实现，属经验值，不当基准）。

三条结论：单请求 decode 是带宽受限；TP 让每张卡每步只读 1/N 权重，能压低单请求 TPOT，但要付每层 all-reduce 通信；提吞吐仍靠加大 batch，让同一份权重服务更多请求、把带宽摊薄；并发上限由 `max-model-len` 直接决定，调参前先算账。TTFT、TPOT、并发上限正是后续用 profiler 要量的三个数。

作业：用启动日志的 KV cache 行对上第 6 步的账，再把 `--max-model-len` 从 4096 改到 8192，观察并发上限与 TTFT。

## 常见误区

1. 「vllm-ascend 是可选的加速包，不装也能跑」。错：原生 vLLM 没有昇腾后端，插件缺失时连设备都分配不出来；它也不是独立引擎，必须挂同版本的 vLLM。
2. 「pip 能装上就能跑，版本差点没关系」。错：三层是锁死的兼容集；torch 与 torch_npu 不配套是 import 报错或算子找不到（见「配套不上会怎样」），vLLM 与 vllm-ascend 不配套是在 register 或内部接口处崩。报错位置常离真正原因很远，先 `pip list` 对矩阵。
3. 「`--gpu-memory-utilization` 拉到 1.0 能多塞请求」。错：CANN 运行时、通信缓冲、图捕获都要预留，打满通常换来启动失败或运行中 OOM；0.8–0.9 是稳妥起点。
4. 「跑通就等于性能达标」。错：昇腾与 CUDA 的算子实现不同，某个算子走 fallback 就可能慢几倍；同一模型换个 vllm-ascend 版本性能也会变。性能必须实测，不能从「CUDA 上很快」外推。

## 一句话总结

vllm-ascend 是把 vLLM 的设备抽象层实现到昇腾 NPU 的插件，站在 CANN 与 torch-npu 之上；跑通第一个请求只需三件事——按官方矩阵整行锁版本、看到 `Platform plugin ascend is activated`、给 KV 池留够显存并把 `max-model-len` 设成真正需要的长度。

## 自测题

### 1. 昇腾上缺的那块是什么？装在 vLLM 里面还是外面？
<details><summary>查看答案</summary>缺的是设备后端：vLLM 自带的平台后端里没有昇腾，不会为昇腾编译 kernel，也不知道怎么在 NPU 上分配显存、跑注意力。补它的是 vllm-ascend，一个装在 vLLM 外面的独立 Python 包，通过 entry point `vllm_ascend:register` 注册进 `vllm.platform_plugins`，激活后接管 Platform/Worker/ModelRunner/Attention backend。它必须与 vLLM 同版本，因为实现的是 vLLM 内部接口。</details>

### 2. torch 2.10.0 + torch_npu 2.8.0 会发生什么？
<details><summary>查看答案</summary>不是变慢，而是大概率直接失败。torch_npu 的二进制针对特定 torch 版本编译，`import torch_npu` 时校验 torch 版本与 C++ ABI；版本段不匹配通常在 import 或首个 npu 算子调用处抛错（找不到符号 / not implemented），也可能能跑但数值异常。修法不是调参，而是回到版本矩阵换整行：torch 2.10.0 配 torch_npu 2.10.0.post7。</details>

### 3. 64 GB 卡跑 0.6B，max-model-len 从 4096 到 32768，为什么并发掉 8 倍？
<details><summary>查看答案</summary>权重只占 1.2 GB，KV 池约 53.6 GB，而单条请求的 KV 与上下文长度成正比：每 token 约 114,688 B，4096 token ≈ 470 MB，能装约 114 条；32768 token ≈ 3.6 GB，只能装约 14 条。所以不是「显存不够启动」，而是每条请求吃的 KV 多了 8 倍、并发被压到 1/8。启动只在单条 max-model-len 的 KV 装不进池子时报错。</details>

### 4. 怎么确认「真的是 NPU 在算」？
<details><summary>查看答案</summary>三处证据：① 启动日志有 `Platform plugin ascend is activated`，说明插件接管；② `torch.npu.is_available()` 为 True、`device_count()` 等于预期卡数；③ 请求期间 `npu-smi info` 能看到对应进程占用显存与 AI Core。日志若出现 CUDA 相关报错，说明插件没加载，回第 3 步查版本与 entry point。</details>

## 术语卡

| 术语 | 解释 |
| --- | --- |
| vllm-ascend | vLLM 的昇腾平台插件，实现 Platform/Worker/ModelRunner 等接口，让 vLLM 跑在 NPU 上 |
| CANN | 昇腾异构计算架构，提供运行时、ACLNN 算子库、HCCL 通信库与工具链 |
| torch-npu | PyTorch 的昇腾后端，注册 `npu` 设备并把 aten 算子映射到 ACLNN |
| ACLNN | CANN 的神经网络算子库，torch-npu 与 vllm-ascend 最终调用的算子集合 |
| NNAL | CANN 中提供 ATB 算子库运行时的组件，版本随 CANN 对齐 |
| HDK | 昇腾驱动与固件包，CANN 运行的前置条件 |
| PrivateUse1 | PyTorch 为 CUDA 之外的设备预留的注册机制，torch-npu 用它注册出 `npu` 设备 |
| 算子（operator） | 封装好的计算单元，如 MatMul、FlashAttention、RMSNorm |
| 算子适配 | 为模型中每个计算在目标硬件上准备可用实现，缺失则自定义补齐或 fallback |
| fallback | 算子无原生实现时的回退路径，常拆成小算子组合或落到 CPU，性能显著下降 |
| 版本矩阵 | 官方锁定的兼容表，vllm-ascend / vLLM / CANN / NNAL / torch-npu / HDK 按整行配套 |
| 平台插件 | vLLM 的扩展机制，通过 entry point 注册后端，使 vLLM 识别新设备 |
| TTFT | Time To First Token，首 token 时延，主要由 prefill 与排队决定 |
| TPOT | Time Per Output Token，平均每 token 生成时延，decode 阶段受带宽限制 |
| 量化 | 用低比特表示权重/激活以省显存、省带宽，用精度换性能（后续课时展开） |
| profiler | 性能采样工具（msprof、torch_npu.profiler），把耗时拆到算子/通信/调度上 |
| 性能定位 | 用 profile 数据判断瓶颈属于算力、带宽还是调度，再定优化方向 |

## 下一步

下一课《量化与编译基础》：把权重与激活压到低比特（W8A8 等）、用图编译替换逐算子下发，继续在昇腾上把吞吐与时延做下来。

## 来源

1. vLLM Ascend Quick Start — https://docs.vllm.ai/projects/ascend/en/latest/getting_started/quick_start.html
2. vLLM Ascend Installation Guide — https://docs.vllm.ai/projects/ascend/en/latest/getting_started/installation.html
3. vLLM Ascend Versioning Policy — https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html
4. vLLM Plugin System — https://docs.vllm.ai/en/latest/design/plugin_system.html
5. 昇腾社区 CANN 文档 — https://www.hiascend.com/document
6. vLLM Ascend 项目仓库 — https://github.com/vllm-project/vllm-ascend
7. Efficient Memory Management for Large Language Model Serving with PagedAttention — https://arxiv.org/abs/2309.06180
