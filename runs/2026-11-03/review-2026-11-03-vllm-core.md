# 审校记录 · 2026-11-03-vllm-core

## verdict

revise

## summary

概念与数学总体扎实：worked example 全部 4 步重算无误（16 KB/层、512 KB/token、2048 token≈1 GB、50 个请求、(64−14)÷1=50、2·N·P=1.4×10¹³、99×1.4×10¹³≈1.4×10¹⁵、100 TFLOPS≈14 秒）；版本事实经官方文档与仓库核对基本准确（V1 alpha 2025 年发布、V0 已完全废弃、vllm-ascend v0.23.0 对应 CANN 9.1.0、SHA-256 自 v0.10.2 起为默认哈希、API Server 负责分词均属实）；教学结构 8 要素齐全，4 条模块目标全覆盖。需改的地方：主链路把「采样」归给 Engine Core 有误（V1 中采样在 Worker 的 model runner 设备端完成，且这是模块目标 1 的主链路），另有多处未解释术语、个别估算缺关键限定、几处低风险细节可打磨。

## issues

### Issue 1

- severity: high
- claim: 主链路写「…→ NPU Worker 前向计算 → Engine Core 采样出 1 个 token → API Server 流式返回」，且核心概念里把 Engine Core 定义为「协调设备 worker 执行模型计算」、把前向计算归给 Worker
- problem: V1 中采样（sampler）是在 Worker 进程的 model runner 里随前向计算在设备端完成的（vLLM V1 明确把 sampler 移入 model runner，官方文档只给 Worker 分配「执行前向计算」），Engine Core 接收采样结果并更新调度状态，自己不执行采样。主链路是模块目标之一，把采样归给 Engine Core 会让学生画错进程边界并固化成错误认知。
- fix: 把主链路改为「HTTP 请求 → API Server 分词 → ZMQ → Engine Core 调度 → NPU Worker 前向计算并采样出 1 个 token → Engine Core 汇总结果、更新调度状态 → API Server 流式返回」，其余表述不变。

### Issue 2

- severity: medium
- claim: 多处直接使用未解释的术语/缩写：采样、哈希（哈希缓存/哈希=父块哈希+token ID）、算子、SOSP、CANN、TorchNPU。
- problem: 目标读者是基础较弱的工程师；「采样」出现在主链路、「哈希」是 prefix caching 机制的核心，不解释就跟不上；SOSP、CANN、TorchNPU 在昇腾落地段落裸用（CANN 下一课才展开）。
- fix: 首次出现时各加一句括注：采样=按概率分布从词表挑出下一个 token；哈希=把任意内容压成固定长度指纹（便于快速比对）；算子=矩阵乘等底层计算单元；SOSP=操作系统领域顶会；CANN=昇腾异构计算架构（Compute Architecture for Neural Networks），下一课展开；TorchNPU=PyTorch 的昇腾后端。

### Issue 3

- severity: medium
- claim: 第 3 步「64 GB 显存，减掉权重约 14 GB，最多同时装约 (64−14)÷1 = 50 个这样的请求」。
- problem: 只扣了权重就除以单请求 KV，未扣激活值、工作区等临时显存。初学者会把这个数当成真实并发容量，实际能同时承载的请求数显著低于 50，「最多」二字挡不住误读。
- fix: 在结果后补一句：「这是忽略激活值等临时显存占用的粗算上界，实际并发数会更低」，让学生把 50 理解为量级而非容量。

### Issue 4

- severity: low
- claim: 课前自测只有 3 道题，没有答案。
- problem: 自学者无法核对；Q2 的答案（99 次）藏在手算第 4 步，Q1、Q3 的答案分散在正文各处，自学闭环不完整。
- fix: 文末加「自测答案」小节：Q1 页式按需分配、碎片接近零、支持大并发；Q2 约 99 次（第一次照算，其余命中缓存）；Q3 不一定——超出 KV 容量会频繁抢占、重复 prefill，吞吐与延迟要一起看。

### Issue 5

- severity: low
- claim: 误区 3：「V1 有独立的 prefill 队列和 decode 队列。错：那是 V0 的设计」。
- problem: V0 的队列是 waiting/running/swapped，并非字面上的 prefill/decode 两个队列；官方对比口径是 V1 的调度「不再严格分离 prefill 与 decode 阶段」。把「队列」说成 V0 事实会顺带教错一个 V0 细节。
- fix: 改为：「V0 把 prefill 和 decode 分阶段分批调度、互不混批；V1 统一 token 预算，长 prompt 切成块（chunked prefill）与 decode 同批执行」。

### Issue 6

- severity: low
- claim: 启动参数逐项读里写「--enable-prefix-caching 开前缀缓存」。
- problem: 前缀缓存自 vLLM 0.11 起已默认开启（讲义自己锚定的 v0.23.0 同样默认开启），只字不提默认值，学生可能误以为不加这个参数就没有前缀缓存。
- fix: 补一句：「新版默认已开启，显式传入无妨；用 --no-enable-prefix-caching 可关闭」。

### Issue 7

- severity: low
- claim: prefix caching 机制写「默认 SHA-256 防碰撞」。
- problem: 默认 SHA-256 属实（自 v0.10.2 起），但「防碰撞」略有绝对化——官方文档注明哈希结构并非 100% 无碰撞，SHA-256 只是把碰撞概率压到可忽略。
- fix: 改为「默认 SHA-256（自 v0.10.2 起），碰撞概率可忽略」，不需要展开碰撞校验细节。
