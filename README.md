# cuda-ml-primitives

CUDA C/C++ implementations of core deep learning primitives — benchmarked against PyTorch on an NVIDIA T4 GPU (Google Colab).

Built as a portfolio project targeting NVIDIA DevTech (AI) roles. Demonstrates GPU memory hierarchy optimization, parallel reduction patterns, and transformer-layer kernels from scratch.

---

## Table of Contents

| # | Notebook | Topic | Key Concepts |
|---|----------|-------|--------------|
| 01 | [Vector Addition](notebooks/01_vector_addition.ipynb) | CUDA basics | Threads/blocks/grids, `cudaMalloc`, `cudaMemcpy` |
| 02 | [Matrix Multiply](notebooks/02_matrix_multiply.ipynb) | Tiled GEMM | Shared memory, `__syncthreads`, TFLOPS benchmark |
| 03 | [Activations](notebooks/03_activations.ipynb) | ReLU / GELU / SiLU | Fast math intrinsics, elementwise parallelism |
| 04 | [Softmax](notebooks/04_softmax.ipynb) | Numerically stable softmax | Parallel reduction, `__shfl_down_sync` |
| 05 | [Linear Layer](notebooks/05_linear_layer.ipynb) | Batched linear + FP16 | cuBLAS comparison, memory bandwidth |
| 06 | [Attention](notebooks/06_attention.ipynb) | Scaled dot-product attention | Causal mask, full transformer primitive |

---

## Benchmark Results (NVIDIA T4, Google Colab)

> Run `benchmarks/benchmark.py` after connecting to a T4 runtime to reproduce.

| Kernel | Size | CUDA (GB/s or TFLOPS) | PyTorch (GB/s or TFLOPS) |
|--------|------|-----------------------|--------------------------|
| Vector Add | 16M floats | — | — |
| Naive MatMul | 2048³ | — | — |
| Tiled MatMul (32×32) | 4096³ | **0.94 TFLOPS** | — |
| ReLU | 16M floats | **245.39 GB/s** | — |
| GELU | 16M floats | **250.02 GB/s** | — |
| SiLU | 16M floats | **247.51 GB/s** | — |
| Softmax naive | 4096×1024 | **54.10 GB/s** | — |
| Softmax stable (warp) | 4096×1024 | **192.35 GB/s** | — |
| Linear FP32 | 512×1024→2048 | — | — |
| Linear FP16 | 512×1024→2048 | — | — |
| Attention (causal) | seq=512, d=64 | **0.17 TFLOPS** | — |

*Measured on NVIDIA T4 (Google Colab), CUDA 13.0, driver 580.82.07. Run `python benchmarks/benchmark.py` to reproduce.*

---

## How to Run

All notebooks are self-contained for **Google Colab with a T4 GPU**.

1. Open any notebook in Colab: `File > Open notebook > GitHub > paste this repo URL`
2. Runtime > Change runtime type > **T4 GPU**
3. Runtime > **Run all**

Each notebook:
- Writes a `.cu` file with `%%writefile`
- Compiles with `!nvcc -arch=sm_75 ...`
- Runs the binary for correctness check
- Benchmarks against equivalent PyTorch

---

## Repository Structure

```
cuda-ml-primitives/
├── notebooks/          # Colab-ready .ipynb files (one per primitive)
├── kernels/            # Standalone .cu source files extracted from notebooks
├── benchmarks/         # benchmark.py — runs all kernels and prints a summary table
└── docs/
    └── diagrams.md     # Mermaid architecture/sequence diagrams
```

---

## NVIDIA DevTech Alignment

| Job Requirement | Covered In |
|---|---|
| CUDA C/C++ programming | Notebooks 01–06, all `kernels/` |
| Deep learning optimization | Notebooks 04 (softmax), 05 (linear), 06 (attention) |
| GPU memory hierarchy | Notebook 02 (shared memory tiling), `docs/diagrams.md` |
| NLP primitives (transformer) | Notebook 06 (attention) |
| CV/ML algorithms | Notebooks 03 (activations), 05 (linear layer) |
| Benchmarking + performance analysis | Notebook 05, `benchmarks/benchmark.py`, README table |

---

## Requirements

See [`requirements.txt`](requirements.txt) — Python deps for the benchmark script.
CUDA kernels compile with `nvcc` (provided by Colab's CUDA toolkit, no local install needed).
