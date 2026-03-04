"""Run all standalone CUDA kernels and print a summary table."""
import subprocess, re, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def compile_run(src, binary, *extra_flags):
    flags = " ".join(extra_flags)
    cmd = f"nvcc -arch=sm_75 -O2 {flags} -o {binary} {src} && {binary}"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT)
    return r.stdout + r.stderr

results = []

# matmul
out = compile_run("kernels/matmul.cu", "/tmp/bm_matmul")
m = re.search(r"([\d.]+) TFLOPS", out)
results.append(("matmul (tiled 32×32, 4096³)", f"{m.group(1)} TFLOPS" if m else "err"))

# activations
out = compile_run("kernels/activations.cu", "/tmp/bm_act", "-lm")
for label in [("ReLU", "ReLU  "), ("GELU", "GELU  "), ("SiLU", "SiLU  ")]:
    m = re.search(rf"{label[1].strip()}\s+([\d.]+) GB/s", out)
    results.append((f"activation {label[0]}", f"{m.group(1)} GB/s" if m else "err"))

# softmax
out = compile_run("kernels/softmax.cu", "/tmp/bm_smx", "-lm")
for label in [("Naive", "Naive"), ("Stable", "Stable")]:
    m = re.search(rf"{label[1]}\s+([\d.]+) GB/s", out)
    results.append((f"softmax {label[0].lower()}", f"{m.group(1)} GB/s" if m else "err"))

# attention
out = compile_run("kernels/attention.cu", "/tmp/bm_attn", "-lm")
m = re.search(r"([\d.]+) TFLOPS", out)
results.append(("attention (seq=512, d=64)", f"{m.group(1)} TFLOPS" if m else "err"))

print()
print("── cuda-ml-primitives benchmark (T4) " + "─" * 28)
for name, val in results:
    print(f"  {name:<38} {val}")
print("─" * 66)
