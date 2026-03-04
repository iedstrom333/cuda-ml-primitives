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

| Kernel | Size | CUDA (GB/s or TFLOPS) | PyTorch (GB/s or TFLOPS) | Speedup |
|--------|------|-----------------------|--------------------------|---------|
| Vector Add | 16M elements | — | — | — |
| Tiled MatMul | 1024×1024 | — | — | — |
| ReLU | 16M elements | — | — | — |
| GELU | 16M elements | — | — | — |
| Softmax | 8192×8192 | — | — | — |
| Linear Layer (FP16) | 2048×2048 | — | — | — |
| Attention | seq=512, d=64 | — | — | — |

*Table will be filled with real numbers after Colab run.*

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
