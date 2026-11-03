# 审校记录 · 2026-11-03-ascend-arch-2

- date: 2026-11-03
- file: 课程/硬件/2026-11-03-ascend-arch-2.md
- title: 第 8 课 · 昇腾 NPU 架构与 CANN 生态（下）
- module_id: ascend-arch
- lesson_no: 8

## verdict

revise

## summary

数学全部重算无误：14×10⁹×2=14 GB、14e9/1.6e12=8.75 ms、1/0.00875≈114 tokens/s、8/0.00875≈914 tokens/s、300e12/1.6e12≈188 FLOP/字节、算术强度 2 FLOP/2 B=1 FLOP/字节，均正确；对带宽与算力的『存疑/量级』标注（910B 流传值、300 TFLOPS 量级）处理得当，没有把推测写成事实。教学结构完整（课前自测→动机→核心概念→手算例子→误区→一句话总结→术语卡→下一步→来源），缩写基本都有解释，手算可跟。两处必须改的事实错误都在软件栈层次：①层次图把 MindSpore 画到 AscendCL 之下（实际 MindSpore 走 CANN 的 GE 图引擎 + 自有运行时 MindRT，不用 AscendCL），而『画出层次关系』正是模块目标的核心；②『MindSpore 默认静态图』已过时（官方文档明确 2.0 起默认 PyNative 动态图）。另有 eager『即时编译』用词、KV cache 读取未交代等限定性问题。模块 4 个目标中本课直接覆盖『CANN/驱动/固件与框架层次关系』，其余 3 个（NPU vs GPU 设计哲学、达芬奇矩阵单元、Atlas 产品线）依赖上集，建议加一行跨课回顾。

## issues

### 1. 层次图把 MindSpore 画到 AscendCL 之下（high）

- severity: high
- claim: 层次图中『MindSpore（原生）→ AscendCL（统一编程接口）』，且正文称 AscendCL 是『统一编程接口』、两条路线『都建在 CANN 之上』
- problem: MindSpore 并不使用 AscendCL：其昇腾后端由 CANN 的图引擎 GE 负责图编译/执行、由 MindSpore 自有的运行时 MindRT 负责调度；AscendCL 是 torch-npu 和独立应用开发用的 C/C++ 接口。把 MindSpore 画到 AscendCL 之下会让学生对『谁调用谁』形成错误印象，而『能画出 CANN/驱动/固件与框架之间的层次关系』正是本模块目标之一，画错的正是核心图。
- fix: 改为两条并列路径：PyTorch→torch-npu→AscendCL；MindSpore→GE（图编译）+ MindRT；二者都落在 CANN 的算子库与 Runtime 之上。正文把『统一编程接口』改为『torch-npu 使用的编程接口』，并在 MindSpore 小节说明它与 torch-npu 接入 CANN 的路径不同。

### 2. MindSpore 默认静态图为过时事实（high）

- severity: high
- claim: 『MindSpore 默认静态图：先整图编译……再执行』（术语卡同：『默认静态图编译』）
- problem: 与现行产品不符：MindSpore 官方教程明确 2.0 起『默认计算图模式为 PyNative 模式』（动态图），静态图需显式 set_context(GRAPH_MODE) 或加 @jit。这是过时的事实错误，初学者装新版 MindSpore 验证时会对不上。
- fix: 改为『MindSpore 以静态图/整图下沉的图编译优化见长（2.x 起默认动态图 PyNative，生产训练用 GRAPH_MODE 或 jit 切静态图）』，术语卡同步修改。

### 3. eager 被描述为「即时编译下发」（medium）

- severity: medium
- claim: 『PyTorch 默认动态图（eager，算子逐个即时编译下发）』
- problem: eager 模式并不做编译，而是直接调用已就绪的算子内核逐个立即下发执行；『即时编译』一词会把 eager 与 torch.compile/JIT 的图编译混为一谈，恰好模糊了本节要教的动态图 vs 图编译的区别。
- fix: 改为『算子逐个立即下发执行（不做整图编译）』；如想保留编译概念，可另加一句『torch.compile（PyTorch 的图编译 API）与 MindSpore 走整图编译』。

### 4. 手算只计权重、未交代 KV cache 读取（medium）

- severity: medium
- claim: 『第 1 步：decode 1 个 token 读多少数据？……故约读 14 GB』，术语卡 HBM 行『其带宽决定 decode 吞吐』
- problem: 14 GB 只算了权重；decode 实际还要读 KV cache（随上下文长度增长，短上下文时远小于 14 GB，数万 token 长上下文时可达数 GB，不再可忽略）和少量激活。初学者容易把 14 GB 当成精确总量，之后实测吞吐低于手算值时无从解释。结论（带宽受限）不受影响，但限定缺失。
- fix: 第 1 步补一句：『另有 KV cache 等读取，短上下文时量级远小于权重，这里先忽略，不影响带宽受限的结论；长上下文时它会变大』。术语卡『决定』改为『主要决定 decode 吞吐上限』。

### 5. 「910B 是旗舰 NPU 芯片」为时效性表述（low）

- severity: low
- claim: 『910B 是昇腾旗舰 NPU 芯片』
- problem: 『旗舰』是时效性表述，写死成无时限的事实容易过时（如后续 910C/950 等型号发布后），对初学者可能造成型号认知偏差。
- fix: 改为『910B 是昇腾 910 系列的高端芯片（截至讲义编写时）』，或直接说『取昇腾 910 系列量级的带宽』。

### 6. torch.compile 未加解释（low）

- severity: low
- claim: 『torch.compile 与 MindSpore 走整图编译』
- problem: torch.compile 对基础弱的读者是未解释的术语，而它恰是本节『图编译』概念在 PyTorch 侧的对照物，直接抛出会丢知识点。
- fix: 加括号说明：『torch.compile（PyTorch 的图编译 API，可对整图做融合优化）』。

### 7. 模块目标覆盖不闭环（low）

- severity: low
- claim: 本课（下）只直接覆盖模块目标中的『CANN/驱动/固件与框架的层次关系』，其余 3 个目标（NPU vs GPU 设计哲学、达芬奇矩阵单元、Atlas 产品线）全部依赖上集
- problem: 本课仅用『回顾上集』一句带过这些前置内容；若学生跳课或遗忘，本节课中 CUDA 类比、Cube 单元、910B/Atlas 等引用会失去支点，模块目标整体覆盖不闭环。
- fix: 在『为什么需要它』里加一行跨课回顾（NPU 与 GPU 设计哲学一句话对比、Cube/Vector/Scalar 一句话、Atlas 产品线一句话），或在课前自测加 1 道上集复习题。

### 8. 来源 2 链接钉死在旧 alpha 版本（low）

- severity: low
- claim: 来源 2 指向 CANN Community Edition 81RC1alpha001（alpha 版）的 op_statistic 文档页
- problem: 链接钉死在旧 alpha 版本上，文档站改版或版本下架后易失效，且 alpha 版内容可能与学员实际安装的 CANN 正式版不一致。
- fix: 换成当前正式版 CANN 的 Profiling 章节入口（如 hiascend.com/document 下最新 Community Edition 的调优工具文档），或附『若失效请从昇腾文档中心搜 op_statistic』的提示。
