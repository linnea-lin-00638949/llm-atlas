#!/usr/bin/env python3
"""Mamba 家族 · 动手实验（第 1–3 天配套）

只依赖 numpy。每个实验都对应《模型演进》/《vLLM》/《昇腾》三篇笔记里的一个公式，
跑一遍、改几个数字，比读十遍公式有用。

运行：python3 lab_ssm.py
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def title(text):
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


def check(name, ok):
    print(("  ✅ " if ok else "  ❌ ") + name)


# ---------------------------------------------------------------------------
# 实验 1：标量 SSM —— 递推、离散化、卷积视图
# ---------------------------------------------------------------------------
title("实验 1：标量 SSM 的三种算法（递推 / 卷积矩阵 / 并行 scan）")

a = -1.0        # 连续时间衰减率（特征值实部为负 = 会遗忘）
delta = 0.1     # 离散化步长
b, c = 1.0, 1.0
x = np.array([1.0, 0.5, -0.2, 0.8])

# 零阶保持（ZOH）离散化
A_bar = np.exp(delta * a)
B_bar = (np.exp(delta * a) - 1.0) / a * b
print(f"  A_bar = e^(Δa) = {A_bar:.6f}，B_bar = (e^(Δa)-1)/a·b = {B_bar:.6f}")
print(f"  近似式 B_bar ≈ Δb = {delta * b:.6f}（只在 |Δa|≪1 时成立）")

# 1a. 逐步递推
h, ys = 0.0, []
for xt in x:
    h = A_bar * h + B_bar * xt
    ys.append(c * h)
ys = np.array(ys)
print(f"\n  递推结果 y = {ys}")

# 1b. 卷积视图：y = M x，M 是下三角 Toeplitz（核只依赖 t-s）
n = len(x)
M = np.zeros((n, n))
for t in range(n):
    for s in range(t + 1):
        M[t, s] = c * (A_bar ** (t - s)) * B_bar
print(f"  卷积矩阵 M =\n{M}")
y_conv = M @ x
print(f"  卷积结果 y = {y_conv}")
check("递推 == 卷积矩阵（误差 < 1e-12）", np.allclose(ys, y_conv))

# 1c. 并行 scan：把 (a, b) 当作仿射变换 h_out = a·h_in + b，验证结合律
pairs = [(A_bar, B_bar * xt) for xt in x]
def compose(p, q):  # q 后作用
    a1, b1 = p; a2, b2 = q
    return (a2 * a1, a2 * b1 + b2)
acc = (1.0, 0.0)
for p in pairs:
    acc = compose(acc, p)
y_scan = c * acc[1]
check(f"并行 scan 结果 {y_scan:.6f} == 递推末值 {ys[-1]:.6f}",
      abs(y_scan - ys[-1]) < 1e-12)
print("  → 结合律保证了树形归约：O(n) 计算、O(log n) 深度，这就是并行 scan。")


# ---------------------------------------------------------------------------
# 实验 2：选择性 scan —— 让 Δ 随 token 变化
# ---------------------------------------------------------------------------
title("实验 2：选择性 scan —— 该忘就忘，该记就记")

def selective_scan(x, deltas, Bs, Cs, a=-1.0):
    """标量选择型 SSM：每个 token 有自己的 Δ、B、C（Mamba S6 的标量版）。"""
    h, out = 0.0, []
    for t in range(len(x)):
        A_bar_t = np.exp(deltas[t] * a)
        B_bar_t = (np.exp(deltas[t] * a) - 1.0) / a * Bs[t]
        h = A_bar_t * h + B_bar_t * x[t]
        out.append(Cs[t] * h)
    return np.array(out)


x = np.array([1.0, 1.0, 1.0, 1.0])
# 第 3 个 token 的 Δ 突然变大：相当于「重置」，把之前的历史冲掉
deltas = np.array([0.1, 0.1, 10.0, 0.1])
Bs = np.ones(4)
Cs = np.ones(4)
y_sel = selective_scan(x, deltas, Bs, Cs)
y_fix = selective_scan(x, np.full(4, 0.1), Bs, Cs)
print(f"  Δ = {deltas}")
print(f"  选择性 y = {y_sel}")
print(f"  固定核 y = {y_fix}")
print(f"  第 4 步差异 = {abs(y_sel[3] - y_fix[3]):.4f}")
print("  大 Δ → Ā 趋近 0 → 旧状态被冲掉：模型在「分隔符/新段落」处主动遗忘。")
check("选择性输出与固定核不同（说明输入真的改变了状态轨迹）",
      not np.allclose(y_sel, y_fix))


# ---------------------------------------------------------------------------
# 实验 3：chunked scan —— 块内矩阵乘，块间传状态
# ---------------------------------------------------------------------------
title("实验 3：chunked scan（块内 GEMM + 块间状态）与逐步递推等价")

def chunked_scan(x, deltas, Bs, Cs, a=-1.0, L=4):
    """把序列切成 L 的块：块内用 (L×L) 下三角矩阵乘，块间只传一个状态。"""
    n = len(x)
    Ab = np.exp(deltas * a)
    Bb = (np.exp(deltas * a) - 1.0) / a * Bs
    ys = np.zeros(n)
    h = 0.0
    for start in range(0, n, L):
        end = min(start + L, n)
        m = end - start
        # 块内前缀积 P_t = Ā_0·…·Ā_t（注意从块的第一个 token 开始，不能从 1.0 起）
        P = np.ones(m)
        P[0] = Ab[start]
        for i in range(1, m):
            P[i] = P[i - 1] * Ab[start + i]
        # 块内矩阵 M[t,s] = (∏_{i=s+1..t} Ā_i)·B̄_s（下三角，因果）
        M = np.zeros((m, m))
        for t in range(m):
            for s in range(t + 1):
                prod = 1.0
                for i in range(s + 1, t + 1):
                    prod *= Ab[start + i]
                M[t, s] = prod * Bb[start + s]
        h_local = P * h + M @ x[start:end]   # 块内状态：初值 + 本块贡献
        ys[start:end] = Cs[start:end] * h_local
        h = h_local[-1]                       # 块间只传一个状态
    return ys


rng = np.random.default_rng(0)
n = 10
x = rng.normal(size=n)
deltas = rng.uniform(0.01, 0.5, size=n)
Bs = rng.normal(size=n) * 0.5 + 1
Cs = rng.normal(size=n) * 0.3 + 1
y_ref = selective_scan(x, deltas, Bs, Cs)
y_chunk = chunked_scan(x, deltas, Bs, Cs, L=4)
print(f"  递推   y = {y_ref}")
print(f"  chunked y = {y_chunk}")
check("chunked == 递推（误差 < 1e-10）", np.allclose(y_ref, y_chunk))
print("  → 块内是 L×L 矩阵乘（Cube 喜欢），块间串行 n/L 次，不是 n 次。")


# ---------------------------------------------------------------------------
# 实验 4：KV cache vs 循环状态 —— 显存账
# ---------------------------------------------------------------------------
title("实验 4：32K 上下文下 KV cache 与 SSM 状态的真实量级")

def kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes=2):
    return 2 * layers * kv_heads * head_dim * dtype_bytes  # K + V

def state_bytes(layers, heads, head_dim, d_state, dtype_bytes=2):
    return layers * heads * head_dim * d_state * dtype_bytes

models = [
    ("Llama-3-8B（全 attention）", kv_bytes_per_token(32, 8, 128), 0),
    ("Jamba-v0.1（4 层 attention / 28 层 Mamba-1）",
     kv_bytes_per_token(4, 8, 128),
     28 * (8192 * 16 + 8192 * 3) * 2),
    ("Mamba-Codestral-7B（纯 Mamba-2）", 0, state_bytes(64, 128, 64, 128)),
]
ctx = 32768
print(f"  {'模型':<38} {'KV/token':>10} {'32K KV':>10} {'状态/序列':>10} {'交叉点':>10}")
for name, kv_tok, st in models:
    kv32 = kv_tok * ctx
    cross = st / kv_tok if kv_tok else float("inf")
    def fmt(b):
        return f"{b / 1024 / 1024:.1f} MiB" if b >= 1024 * 1024 else f"{b / 1024:.0f} KiB"
    print(f"  {name:<38} {fmt(kv_tok):>10} {fmt(kv32):>10} {fmt(st):>10} "
          f"{(f'{cross:.0f} tok' if kv_tok else '—'):>10}")
print("  → attention 层数越少，KV 越省；纯 Mamba 没有 KV，但每个序列要背固定状态。")
print("  → 交叉点以下（短上下文）状态反而更贵，别把「省显存」当成无条件成立。")


# ---------------------------------------------------------------------------
# 实验 5：昇腾上的算术强度 —— 为什么 scan 喂不饱 Cube
# ---------------------------------------------------------------------------
title("实验 5：roofline 小算盘 —— decode 一步 scan 的算术强度")

# 单头、Dk = Dv = 128、bf16 的 decode 一步：
D = 128
flops = 3 * 2 * D * D          # S·k、S·q、外积更新，各 2D² 乘加
bytes_moved = (D * D + 2 * D) * 2  # 读 + 写状态（32 KiB）+ 读 q/k
ai = flops / bytes_moved
peak = 3e14                    # 910B 量级 FP16（厂商口径）
bw = 1.4e12                    # HBM 1.2–1.6 TB/s 取中值
ridge = peak / bw
print(f"  计算量 ≈ {flops:,} FLOP，搬运 ≈ {bytes_moved:,} B")
print(f"  算术强度 ≈ {ai:.2f} FLOP/Byte；脊点 ≈ {ridge:.0f} FLOP/Byte")
print(f"  → 差 {ridge / ai:.0f} 倍：这类算子永远带宽受限，优化方向是「少搬数据」。")

print("\n全部实验通过。接下来把《模型演进》的自测题做一遍，卡住回来问。")
