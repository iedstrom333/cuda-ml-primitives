"""PyTorch baseline benchmarks — run on Colab T4 to fill README table."""
import torch
import torch.nn.functional as F

assert torch.cuda.is_available(), "No GPU found — switch to T4 runtime"

results = []

def bench_bw(fn, name, n_floats, reps=100):
    fn()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / reps
    bw = 2 * n_floats * 4 / ms / 1e6  # read + write, float32
    results.append((name, f"{bw:.2f} GB/s"))

def bench_tflops(fn, name, flops, reps=100):
    fn()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / reps
    tflops = flops / ms / 1e9
    results.append((name, f"{tflops:.2f} TFLOPS"))

# Vector add
N = 16_777_216
a = torch.randn(N, device="cuda")
b = torch.randn(N, device="cuda")
bench_bw(lambda: a + b, "Vector Add", N)

# MatMul (torch.mm)
M = 2048
A = torch.randn(M, M, device="cuda")
B = torch.randn(M, M, device="cuda")
bench_tflops(lambda: torch.mm(A, B), "Naive MatMul (torch.mm)", 2 * M * M * M)

# Activations
x = torch.randn(N, device="cuda")
bench_bw(lambda: torch.relu(x),  "ReLU", N)
bench_bw(lambda: F.gelu(x),      "GELU", N)
bench_bw(lambda: F.silu(x),      "SiLU", N)

# Softmax
rows, cols = 4096, 1024
s = torch.randn(rows, cols, device="cuda")
bench_bw(lambda: torch.softmax(s, dim=-1), "Softmax (torch)", rows * cols)

# Linear FP32
B_, I, O = 512, 1024, 2048
xf32 = torch.randn(B_, I, device="cuda")
wf32 = torch.randn(O, I, device="cuda")
bench_tflops(lambda: F.linear(xf32, wf32), "Linear FP32", 2 * B_ * I * O)

# Linear FP16
xf16 = xf32.half()
wf16 = wf32.half()
bench_tflops(lambda: F.linear(xf16, wf16), "Linear FP16", 2 * B_ * I * O)

# Attention
seq, d = 512, 64
q = torch.randn(1, 1, seq, d, device="cuda")
k = torch.randn(1, 1, seq, d, device="cuda")
v = torch.randn(1, 1, seq, d, device="cuda")
attn_flops = 2 * seq * seq * d + seq * seq + 2 * seq * seq * d
bench_tflops(
    lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
    "Attention (causal)", attn_flops
)

print()
print("── PyTorch baselines (T4) " + "─" * 40)
for name, val in results:
    print(f"  {name:<30} {val}")
print("─" * 66)
