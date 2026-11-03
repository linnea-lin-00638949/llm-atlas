---
aliases: [推理框架]
tags: [domain/inference-frameworks]
---

# ⚙️ 推理框架

> 一个 token 从请求到返回的系统旅程；主栈 vLLM + 昇腾。

**进度**：7/10 个模块 · 最近学习 2026-11-03 · 累计标签 33 个

## 课程模块

| 模块 | 状态 | 首次学习 | 笔记 |
| --- | --- | --- | --- |
| 推理服务基础：prefill/decode 与 KV cache | ✅ 已学 | 2026-11-03 | [[2026-11-03-serving-basics|2026-11-03]] |
| vLLM 引擎架构 | ✅ 已学 | 2026-11-03 | [[2026-11-03-vllm-core|2026-11-03]] |
| 昇腾上的 vLLM：vllm-ascend 部署与调优 | ✅ 已学 | 2026-11-03 | [[2026-11-03-vllm-ascend-2|2026-11-03]]、[[2026-11-03-vllm-ascend|2026-11-03]] |
| 量化与编译基础 | ✅ 已学 | 2026-11-03 | [[2026-11-03-quant-compile|2026-11-03]] |
| 投机解码 | ✅ 已学 | 2026-11-03 | [[2026-11-03-spec-decoding|2026-11-03]] |
| Prefill/Decode 分离与长上下文 | ✅ 已学 | 2026-11-03 | [[2026-11-03-pd-disagg|2026-11-03]] |
| 推理成本与调度 | ✅ 已学 | 2026-11-03 | [[2026-11-03-serving-econ|2026-11-03]] |
| SGLang 与 RadixAttention | ⬜ 待学 | — | — |
| TensorRT-LLM | ⬜ 待学 | — | — |
| llama.cpp 与本地推理 | ⬜ 待学 | — | — |

## 学习记录

| 日期 | 课题 | 模块 |
| --- | --- | --- |
| [[2026-11-03-pd-disagg|2026-11-03]] | 第 13 课 · Prefill/Decode 分离与长上下文 | Prefill/Decode 分离与长上下文 |
| [[2026-11-03-quant-compile|2026-11-03]] | 第 11 课 · 量化与编译基础 | 量化与编译基础 |
| [[2026-11-03-serving-basics|2026-11-03]] | 第 5 课 · 推理服务基础：prefill/decode 与 KV cache | 推理服务基础：prefill/decode 与 KV cache |
| [[2026-11-03-serving-econ|2026-11-03]] | 第 14 课 · 推理成本与调度 | 推理成本与调度 |
| [[2026-11-03-spec-decoding|2026-11-03]] | 第 12 课 · 投机解码 | 投机解码 |
| [[2026-11-03-vllm-ascend-2|2026-11-03]] | 第 10 课 · 昇腾上的 vLLM：vllm-ascend 部署与调优（下） | 昇腾上的 vLLM：vllm-ascend 部署与调优 |
| [[2026-11-03-vllm-ascend|2026-11-03]] | 第 9 课 · 昇腾上的 vLLM：vllm-ascend 部署与调优（上） | 昇腾上的 vLLM：vllm-ascend 部署与调优 |
| [[2026-11-03-vllm-core|2026-11-03]] | 第 6 课 · vLLM 引擎架构 | vLLM 引擎架构 |

## 关联

回到 [[首页]] · [[课程地图]] · [[术语库]]
其他领域：[[foundation|🧱 基础]] · [[model-evolution|🧬 模型演进]] · [[hardware|🔩 硬件]] · [[training|🔥 训练]]

