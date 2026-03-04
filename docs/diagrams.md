# CUDA ML Primitives — Architecture Diagrams

All diagrams render natively on GitHub via Mermaid.

---

## How to Read These Diagrams

Read them in this order — each one builds on the last.

### Start with the mental model (Diagrams 2 → 3 → 9)

**Diagram 2 (Architecture)** answers: *where does data live and how fast can we get to it?*

This is the most important diagram in the whole project. Every optimization you'll see in the notebooks is about moving data *up* this hierarchy — away from slow HBM (300 GB/s) and into shared memory or registers. If a kernel is slow, it's almost always because it's hitting HBM too much.

**Diagram 3 (Execution Model)** answers: *how does code actually run on the GPU?*

Your PyTorch mental model is "tensors transform into tensors." The CUDA mental model is "thousands of threads run the same function simultaneously." The hierarchy is:
- **Grid** = the whole problem
- **Block** = a group of threads that share memory and can synchronize
- **Warp** = 32 threads that execute *literally the same instruction at the same time* (SIMT)
- **Thread** = one worker, doing one element

The key insight: threads in the same *block* can communicate through shared memory. Threads in different blocks cannot.

**Diagram 9 (Memory Activity)** is the decision tree you'll use when writing every kernel: *should I use shared memory? Are my accesses coalesced?*

### Then understand the optimization journey (Diagram 10)

**Diagram 10 (State: Optimization Progression)** is the narrative arc of the whole project. Every notebook follows this path — you'll implement the naive version first, then apply each optimization and watch the numbers improve. The benchmark table in the README is the scorecard.

### Then read the sequences top-to-bottom as recipes (Diagrams 5 → 6 → 7 → 8)

Each sequence diagram is a recipe for one kernel. Read them like a protocol: what happens on the CPU, what gets sent to the GPU, what the GPU does internally, what comes back.

**Diagram 5 (Vector Add)** is the "hello world" — just the boilerplate lifecycle every CUDA program follows. Every other kernel does the same `malloc → memcpy → launch → sync → memcpy → free` pattern, just with more interesting work in the middle.

**Diagram 6 (Tiled MatMul)** is where shared memory first appears. The loop is the key: instead of each thread reading from slow HBM for every multiply, a whole block loads a *tile* into fast shared memory, does all the math, then loads the next tile. This is the core optimization pattern used everywhere in deep learning.

**Diagram 7 (Softmax)** introduces *parallel reduction* — how you compute a global aggregate (max, sum) across thousands of threads. This pattern (`__shfl_down_sync`, tree reduction) comes up in layer norm, batch norm, cross-entropy, and attention.

**Diagram 8 (Attention)** is the capstone. It's just diagrams 6 and 7 composed: QK^T is a matmul, the softmax over scores is diagram 7, then score@V is another matmul.

### Skip for now (Diagrams 1, 4)

**Diagram 1 (Use Case)** is for recruiters, not for learning. Skim it.

**Diagram 4 (Class Diagram)** is a reference. Don't read it front-to-back — use it later to look up a kernel's expected inputs and launch configuration when you're in a notebook and forget the shape.

### One-sentence summary

The entire project is diagram 9 applied repeatedly: *every kernel we write is asking "how do I move data from HBM into shared memory and registers as efficiently as possible?"* The sequence diagrams show the pattern for each primitive, and diagram 10 shows how we improve each one step by step.

---

## 1. Use Case Diagram

What this project demonstrates and who it's for.

```mermaid
flowchart LR
    U(["👤 Developer\n(Portfolio Viewer)"])

    subgraph Learn["Learn"]
        L1[CUDA C/C++ Programming]
        L2[GPU Memory Hierarchy]
        L3[Parallel Reduction Patterns]
    end

    subgraph Build["Build"]
        B1[Vector Addition]
        B2[Matrix Multiply — Tiled GEMM]
        B3[Activations — ReLU / GELU / SiLU]
        B4[Numerically Stable Softmax]
        B5[Batched Linear Layer FP16]
        B6[Scaled Dot-Product Attention]
    end

    subgraph Measure["Measure"]
        M1[Benchmark vs. PyTorch]
        M2[GB/s and TFLOPS Analysis]
        M3[Roofline Model Intuition]
    end

    U --> Learn
    U --> Build
    U --> Measure
    B2 --> M1
    B3 --> M1
    B4 --> M1
    B5 --> M1
    B6 --> M1
    M1 --> M2
    M2 --> M3
```

---

## 2. Architecture Diagram — Host ↔ Device Memory Hierarchy

How CPU (host) and GPU (device) interact, and the GPU memory hierarchy from slowest to fastest.

```mermaid
flowchart TD
    subgraph HOST["🖥️  CPU Host"]
        RAM["System RAM\n~50 GB/s DDR5"]
        DRIVER["CUDA Driver / Runtime"]
    end

    PCIe["⚡ PCIe 4.0 x16\n~32 GB/s"]

    subgraph DEVICE["🟢  GPU Device — NVIDIA T4"]
        HBM["HBM / GDDR6\nGlobal Memory\n300 GB/s  ·  16 GB"]

        subgraph SM["Streaming Multiprocessor (SM)"]
            L1["L1 Cache / Shared Memory\n~100 TB/s  ·  48 KB per SM"]
            RF["Register File\n~20 PB/s  ·  256 KB per SM"]
            CORE["CUDA Cores\nTensor Cores"]
        end

        L2["L2 Cache\n~1.5 TB/s  ·  4 MB"]
    end

    RAM -->|cudaMemcpy H2D| PCIe
    PCIe --> HBM
    HBM --> L2
    L2 --> L1
    L1 --> RF
    RF --> CORE
    CORE -->|write results| HBM
    HBM -->|cudaMemcpy D2H| PCIe
    PCIe --> RAM
```

---

## 3. CUDA Execution Model — Grid / Block / Warp / Thread

The four-level hierarchy launched by every kernel call.

```mermaid
flowchart TD
    KL["kernel<<<grid, block>>>()"]

    subgraph GRID["Grid  (maps to full problem)"]
        subgraph BLK0["Block (0,0)"]
            subgraph W0["Warp 0  — 32 threads"]
                T0["T0"] --- T1["T1"] --- T2["..."] --- T31["T31"]
            end
            subgraph W1["Warp 1  — 32 threads"]
                T32["T32"] --- T33["T33"] --- T34["..."] --- T63["T63"]
            end
        end
        subgraph BLK1["Block (0,1)"]
            W2["Warp 0 … Warp N"]
        end
        subgraph BLKN["Block (M,N)"]
            W3["Warp 0 … Warp N"]
        end
    end

    KL --> GRID
    BLK0 -->|"Scheduled on one SM\nShared memory visible\nto all threads in block"| SM_NOTE[" "]
    W0 -->|"Executed in lockstep\nSIMT — same instruction"| WARP_NOTE[" "]

    style SM_NOTE fill:none,stroke:none
    style WARP_NOTE fill:none,stroke:none
```

---

## 4. Class Diagram — Kernel Interface Summary

Each CUDA kernel modeled as a class showing inputs, outputs, and launch parameters.

```mermaid
classDiagram
    class VectorAdd {
        +float* A  [device, input]
        +float* B  [device, input]
        +float* C  [device, output]
        +int N
        ---
        +gridDim  = (N+255)/256
        +blockDim = 256
        +launch() void
    }

    class TiledMatMul {
        +float* A  [M x K, device]
        +float* B  [K x N, device]
        +float* C  [M x N, device]
        +int M, K, N
        +int TILE_SIZE = 16
        ---
        +gridDim  = (N/TILE, M/TILE)
        +blockDim = (TILE, TILE)
        +__shared__ float As, Bs
        +launch() void
    }

    class Activation {
        +float* input  [device]
        +float* output [device]
        +int N
        +ActivationType type
        ---
        +gridDim  = (N+255)/256
        +blockDim = 256
        +launch() void
    }

    class Softmax {
        +float* input  [B x N, device]
        +float* output [B x N, device]
        +int B, N
        ---
        +gridDim  = B
        +blockDim = 256
        +__shared__ float sdata
        +__shfl_down_sync() float
        +launch() void
    }

    class LinearLayer {
        +half* X   [B x in, device]
        +half* W   [out x in, device]
        +half* b   [out, device]
        +half* Y   [B x out, device]
        +int B, in_features, out_features
        ---
        +uses cuBLAS hgemm
        +launch() void
    }

    class Attention {
        +float* Q  [B x S x D, device]
        +float* K  [B x S x D, device]
        +float* V  [B x S x D, device]
        +float* O  [B x S x D, device]
        +int B, S, D
        +bool causal_mask
        ---
        +scale = 1/sqrt(D)
        +gridDim  = (B, S)
        +blockDim = 256
        +launch() void
    }

    Activation <|-- VectorAdd : elementwise pattern
    Softmax --> Attention : used internally
    TiledMatMul --> Attention : QK^T and score@V
    LinearLayer --> Attention : projection
```

---

## 5. Sequence Diagram — Vector Addition

The full host/device lifecycle for the simplest CUDA program.

```mermaid
sequenceDiagram
    participant CPU as CPU (Host)
    participant GPU as GPU (Device)

    CPU->>CPU: Allocate host arrays A[], B[], C[]
    CPU->>CPU: Fill A and B with data

    CPU->>GPU: cudaMalloc(&d_A, N*sizeof(float))
    CPU->>GPU: cudaMalloc(&d_B, N*sizeof(float))
    CPU->>GPU: cudaMalloc(&d_C, N*sizeof(float))

    CPU->>GPU: cudaMemcpy(d_A, A, H2D)
    CPU->>GPU: cudaMemcpy(d_B, B, H2D)

    CPU->>GPU: vecAdd<<<grid, block>>>(d_A, d_B, d_C, N)
    activate GPU
    GPU->>GPU: Each thread computes C[i] = A[i] + B[i]
    deactivate GPU

    CPU->>GPU: cudaDeviceSynchronize()
    GPU-->>CPU: done

    CPU->>GPU: cudaMemcpy(C, d_C, D2H)
    CPU->>CPU: Verify C[i] == A[i] + B[i]

    CPU->>GPU: cudaFree(d_A, d_B, d_C)
```

---

## 6. Sequence Diagram — Tiled Matrix Multiply

One output tile's computation showing the shared-memory tile loop.

```mermaid
sequenceDiagram
    participant CPU as CPU (Host)
    participant SM as SM — Shared Memory
    participant HBM as HBM — Global Memory

    CPU->>HBM: Launch tiledMatMul<<<grid, block>>>

    loop for t = 0 to K/TILE_SIZE
        HBM->>SM: Load tile of A [ row, t*TILE : (t+1)*TILE ]
        HBM->>SM: Load tile of B [ t*TILE : (t+1)*TILE, col ]
        SM->>SM: __syncthreads()  — all threads see full tile

        SM->>SM: Accumulate: sum += As[row][k] * Bs[k][col]
        SM->>SM: __syncthreads()  — safe to overwrite tiles
    end

    SM->>HBM: Write C[row][col] = sum
    HBM-->>CPU: cudaDeviceSynchronize()
```

---

## 7. Sequence Diagram — Numerically Stable Softmax

Three-pass parallel reduction: max → exp/sum → normalize.

```mermaid
sequenceDiagram
    participant T as Threads (one row)
    participant SH as Shared Memory
    participant R as Registers

    Note over T: Pass 1 — Find row maximum (avoid overflow)
    T->>R: Each thread loads chunk of row into registers
    T->>SH: Partial max → shared memory reduction
    SH->>SH: Tree reduction: stride /= 2 until stride == 0
    SH->>T: Broadcast row_max to all threads

    Note over T: Pass 2 — Compute exp(x - row_max) and partial sums
    T->>R: exp_val = __expf(x[i] - row_max)
    T->>SH: Partial sum → shared memory reduction
    SH->>SH: Tree reduction → row_sum

    Note over T: Pass 3 — Normalize
    T->>R: out[i] = exp_val / row_sum
    R->>T: Write output to global memory
```

---

## 8. Sequence Diagram — Scaled Dot-Product Attention

Full attention primitive: QK^T → scale → causal mask → softmax → score@V.

```mermaid
sequenceDiagram
    participant CPU as CPU
    participant K1 as QK^T Kernel
    participant K2 as Scale + Mask Kernel
    participant K3 as Softmax Kernel
    participant K4 as Score @ V Kernel

    CPU->>K1: Launch with Q [B,S,D], K [B,S,D]
    K1->>K1: scores[b,i,j] = dot(Q[b,i,:], K[b,j,:])
    K1-->>CPU: scores [B, S, S]

    CPU->>K2: Launch with scores, scale = 1/√D
    K2->>K2: scores[b,i,j] *= scale
    K2->>K2: if causal and j > i: scores[b,i,j] = -∞
    K2-->>CPU: masked_scores [B, S, S]

    CPU->>K3: Launch softmax over last dim (S)
    K3->>K3: attn_weights = softmax(masked_scores, dim=-1)
    K3-->>CPU: attn_weights [B, S, S]

    CPU->>K4: Launch with attn_weights [B,S,S], V [B,S,D]
    K4->>K4: out[b,i,:] = sum_j( attn_weights[b,i,j] * V[b,j,:] )
    K4-->>CPU: output [B, S, D]
```

---

## 9. Activity Diagram — GPU Memory Hierarchy Access Pattern

Decision flow a kernel author takes to minimize memory bottlenecks.

```mermaid
flowchart TD
    START([Kernel needs data]) --> Q1{Data reused\nacross threads?}

    Q1 -->|No — read once| COAL[Ensure coalesced\nglobal memory access\nHBM  ~300 GB/s]
    Q1 -->|Yes — reused| SHMEM[Load into\nShared Memory\n~100 TB/s  ·  48 KB/SM]

    COAL --> Q2{Access pattern\naligned to 128B?}
    Q2 -->|Yes| GOOD1[✅ Full cache line utilized\nL2 cache hit likely]
    Q2 -->|No — strided/random| BAD1[⚠️ Bandwidth wasted\nConsider padding or transpose]

    SHMEM --> Q3{Bank conflicts?}
    Q3 -->|No| GOOD2[✅ Full shared memory BW\nProceed to compute]
    Q3 -->|Yes — same bank| PAD[Add padding column\nto shared array]

    GOOD2 --> REG[Keep hot scalars in\nRegisters — no latency]
    REG --> COMPUTE([Compute in CUDA cores\nor Tensor Cores])

    GOOD1 --> COMPUTE
    PAD --> GOOD2
    BAD1 --> COAL
```

---

## 10. State Diagram — Kernel Optimization Progression

The iterative optimization journey from naive to high-performance.

```mermaid
stateDiagram-v2
    [*] --> Naive

    Naive : Naive Kernel
    Naive : One thread per element
    Naive : Uncoalesced global loads
    Naive : ~10% peak bandwidth

    Coalesced : Coalesced Access
    Coalesced : Threads access adjacent addresses
    Coalesced : Full 128-byte cache lines used
    Coalesced : ~40% peak bandwidth

    SharedMem : Shared Memory Tiling
    SharedMem : Tile data into __shared__
    SharedMem : Reuse loaded data K times
    SharedMem : Drastically reduces HBM traffic

    Float4 : Vectorized Loads
    Float4 : float4 / half2 load instructions
    Float4 : 4× elements per load instruction
    Float4 : Hides memory latency better

    Unrolled : Loop Unrolling + Tuned Config
    Unrolled : #pragma unroll on inner loops
    Unrolled : Tuned block size and tile size
    Unrolled : Near-roofline performance

    Naive --> Coalesced : Fix access pattern
    Coalesced --> SharedMem : Add __shared__ tile buffers
    SharedMem --> Float4 : Use float4 / half2 loads
    Float4 --> Unrolled : Unroll + tune occupancy
    Unrolled --> [*]
```
