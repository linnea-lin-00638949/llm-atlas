---
date: "2026-11-03"
title: "Mamba 家族与混合架构（专题）"
type: "topic-index"
topic_id: "mamba-family"
tags: ["专题", "Mamba", "SSM", "混合架构", "长上下文"]
---

# Mamba 家族与混合架构（专题）

> 定长状态替掉越写越长的 KV cache：数学、服务与硬件。

## 为什么学
长上下文的大头开销是 KV cache。Mamba 把它压成 O(1) 状态，混合架构让 attention 只留少数层；昇腾 + vLLM 上已能遇到这类模型。

## 三篇导读
- 模型演进：S4 离散化 → Mamba 选择性扫描 → Mamba-2 的 SSD → GDN / KDA 与混合架构的分工。
- 推理框架：vLLM 如何分页管理 conv_state / ssm_state，前缀缓存与投机解码要改什么。
- 硬件：scan 为何喂不饱达芬奇 Cube，CANN 算子与 vllm-ascend 的适配。

## 建议阅读顺序
1 → 2 → 3：先记号，再服务，最后硅片。

## 与课程的关系
第 1 课的前向传播就是 h_t = Ā h_{t−1} + B̄ x_t；第 3 课的 attention 是 SSD 的另一端；KV cache 那课是第 2 篇的对照组；第 7 课的达芬奇架构与 roofline 课决定第 3 篇的结论。

## 三篇笔记

[[专题/Mamba 家族/model-evolution|模型演进：从 S4 到 Mamba-2，再到 GDN / KDA 等混合模型]]
[[专题/Mamba 家族/vllm|推理框架：vLLM 怎么处理 Mamba 与混合模型]]
[[专题/Mamba 家族/ascend|硬件：昇腾 NPU 上跑 Mamba 有什么特殊点]]
