# 审校记录 · 2026-11-03-vllm-ascend

- 讲义：`课程/推理框架/2026-11-03-vllm-ascend.md`
- 日期：2026-11-03

## verdict

revise

## summary

结构八项齐全（自测/核心概念/手算/误区/一句话总结/术语卡/下一步/来源），三层栈、entry point（vllm_ascend:register，与 setup.py 一致）、插件激活日志、同号发版与整行锁定这些主干都讲对了，goal 1、2 有落实；goal 3（profiler 定位瓶颈）与 goal 4（常见性能坑）本课只在术语卡与「下一步」出现，必须由下集兑现。主要问题：课前自测第 3 题的前提被正文自己的手算推翻、TP 那句结论与张量并行原理相反、版本示例行的 post 号与 CANN 小版本和官方 v0.30.0 行不符、手算有一组数字因四舍五入连乘偏高 3–5%。改完这几处即可通过。

## issues

### 1. severity: high

- claim: 课前自测第 3 题「64 GB 卡跑 0.6B、权重才 1.2 GB，为什么第一次 vllm serve 常常直接报显存不够」，以及第 5 步「--max-model-len 不写…KV 池按它算，很容易一启动就失败」
- problem: 与正文手算直接冲突：KV 池 53.6 GB，而 32768 token 的单条只占 3.6 GB（约 7%），vLLM 启动时只校验「单条 max-model-len 的 KV 能否装进 KV 池」，这个校验轻松通过，0.6B 在 64 GB 卡上不会因此 OOM。正文第 5 步自己算出的是并发从约 114 掉到约 14，即「并发骤降」而非「启动失败」。把推测写成「常常」，学员照做复现不出来只会怀疑环境坏了。
- fix: 把自测第 3 题改成「一条 32768 token 的请求会把并发从 100+ 压到十几条，为什么」；正文把「很容易一启动就失败」改成「只有单条 max-model-len 的 KV 装不进 KV 池（大模型或超长上下文）时才启动报错，小模型表现为并发骤降」。若坚持保留「启动失败」说法，须给出触发条件（如 max-model-len 设到 20 万以上）。

### 2. severity: high

- claim: 「单请求 decode 是带宽受限，多卡切开权重并不会让每步读得更少」
- problem: 与张量并行原理相反。TP 把每层权重切到 N 张卡，每卡每步只读 1/N 的权重，单请求 decode 的访存时间近似降到 1/N（代价是每层 all-reduce 通信）——这正是 TP 能压低 TPOT 的原因。原句会让学员在下集得出「TP 只省显存、不提速」的错误结论，也与「下一步」要讲的 TP 自相矛盾。
- fix: 改为「TP 让每张卡每步读的权重变成 1/N，所以能压低单请求 TPOT，但要付 all-reduce 通信；TP 提速单请求，提吞吐仍靠加大 batch」。

### 3. severity: medium

- claim: 「写作时稳定版的一整行是：vllm-ascend v0.30.0 / vLLM v0.30.0 / CANN 9.2.0 / torch 2.10.0 + torch_npu 2.10.0.post4 / Python 3.10–3.12」，以及「v0.23.0 行对应 CANN 9.1.0」
- problem: 官方 v0.30.0 验证组合是 CANN Toolkit+Ops 9.2.1、NNAL 9.2.1、PyTorch 2.10.0、TorchNPU 2.10.0.post7、HDK 26.1.1、Python >=3.10,<3.13；讲义写的 CANN 9.2.0 与 post4 都对不上（见 installation 页的 validated stack 表）。这一课的核心口号是「整行锁定、精确到 post 号」，示例行却错了两位，学员照抄 `torch_npu==2.10.0.post4` 会直接踩版本/ABI 校验。v0.23.0 对应 CANN 9.1.0 的跨课引用也无法核实。
- fix: 按官方 v0.30.0 行改成 CANN 9.2.1 / torch 2.10.0 + torch_npu 2.10.0.post7（Python >=3.10,<3.13，镜像 3.12），并补一句「以矩阵页当日内容为准 + 核对日期」；无法核实的 v0.23.0 那句删掉或标为待核。

### 4. severity: medium

- claim: 手算第 4、6 步的数字：458 MB、≈117 条并发、≈30,600 块、≈49 万 token
- problem: 第 3 步先把 114,688 B 四舍五入成 112 KB，后面又拿 112 KB 连乘，并混用 KB/MB/MiB，结果系统性偏高约 4%。重算：114,688 B × 4096 = 469.8 MB（448 MiB）→ 53.6 GB ÷ 469.8 MB ≈ 114 条；块大小 16 时每块 1,835,008 B ≈ 1.835 MB → 53.6 GB ÷ 1.835 MB ≈ 29,200 块 → ≈46.7 万 token。结论（8 倍差、数量级）不变，但「数量级对得上，说明没算错」的对账失去意义，学员按 112 KB 手算会和日志差出几万 token。
- fix: 统一用字节或统一用 GiB 重算，写成 112 KiB/token → 470 MB/条、≈114 条、≈2.92 万块、≈46.7 万 token；并注明「第 3 步的 112 KB 是显示用近似值，连乘时用 114,688 B」。

### 5. severity: medium

- claim: 课前自测第 2 题「torch==2.10.0 配 torch_npu==2.8.0…实际会发生什么？为什么？」，开头说「答案散在本课，先猜再看」
- problem: 正文没有回答这个组合。误区 2 只到「import 失败或算子找不到」，既没说清机制（torch_npu 在 import 时校验 torch 版本/编译 ABI，版本段不匹配通常在 `import torch_npu` 或首个 npu 算子调用处直接抛错，而不是变慢），也没解释「为什么不是慢一点」。自测题成了悬空题，学员对不上答案。
- fix: 在「版本矩阵」或误区 2 补一段「torch 与 torch_npu 不配套的三种典型表现：import 报错 / 算子 not implemented / 能跑但结果异常」，并明确回指自测第 2 题；或把自测答案单独列一小节。

### 6. severity: low

- claim: 第 7 步 decode 时延下界：0.75 ms/token、32 个 token 至少 24 ms
- problem: 只算了读 1.2 GB 权重，没算 KV 读取：4096 上下文时每步还要读约 448 MiB KV（约等于权重的 37%），并入后真实下界约 1.0–1.1 ms。虽然标了「下界」，但紧接着说「实测常见是 2–4 倍」，初学者容易把 0.75 ms 当成 TPOT 基准，误判机器性能。
- fix: 补一句「这里略去 KV 读取与激活访存，只给权重带宽的最低界；4096 上下文时每步还要多读约 0.45 GB KV」，或直接给出「权重 + KV」的合并估算 ≈1.0 ms。

### 7. severity: low

- claim: CANN 的直觉类比「相当于操作系统 + 显卡驱动 + 标准数学库」
- problem: 与本课自己的三层栈定义冲突：驱动与固件属于 HDK，CANN 只提供驱动接口和运行时。把 CANN 说成含「显卡驱动」，学员定位问题时会把 `npu-smi` 看不到卡归到 CANN 层，而正文其实归给了驱动/容器设备映射。
- fix: 改成「CANN ≈ CUDA Toolkit + cuBLAS + NCCL（不含驱动，驱动在 HDK 层）」，与前面「HDK 含驱动和固件」保持一致。

### 8. severity: low

- claim: 第 3 步「预构建镜像里 CANN、NNAL、PyTorch、torch-npu、vLLM、vllm-ascend 已是一套验证过的组合」，以及「进容器先做三行验证」
- problem: NNAL 首次出现未做任何解释（它是 CANN 里提供 libatb.so / ATB 运行时的组件，官方 v0.30.0 行里 NNAL 也有独立版本号 9.2.1），也不在术语卡；PrivateUse1、Ascend C 同样只提名字。基础弱的读者会被这几个词卡住，也不知道 NNAL 版本同样要跟 CANN 对齐。另外「三行验证」实际只给了两行代码。
- fix: 给 NNAL 一句解释并进术语卡（「CANN 中提供 ATB 算子库运行时的组件，版本随 CANN」），PrivateUse1 / Ascend C 各补半句；把「三行验证」改成「两行」或补第三行版本打印（如 `print(torch.__version__, torch_npu.__version__)`）。
