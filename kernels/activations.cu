#include <stdio.h>
#include <cuda_runtime.h>

__global__ void relu(float *x, float *y, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < N) y[i] = fmaxf(x[i], 0.f);
}

__global__ void gelu(float *x, float *y, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < N) {
        float v = x[i];
        y[i] = 0.5f*v*(1.f+tanhf(0.7978845608f*(v + 0.044715f*v*v*v)));
    }
}

__global__ void silu(float *x, float *y, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < N) y[i] = x[i] / (1.f + expf(-x[i]));
}

#define BENCH(kernel, label) do { \
    kernel<<<(N+255)/256,256>>>(dx,dy,N); cudaDeviceSynchronize(); \
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); \
    int reps=100; cudaEventRecord(s); \
    for(int i=0;i<reps;i++) kernel<<<(N+255)/256,256>>>(dx,dy,N); \
    cudaEventRecord(e); cudaEventSynchronize(e); float ms; \
    cudaEventElapsedTime(&ms,s,e); \
    printf("%-8s%.2f GB/s\n", label, 2.0*N*4*reps/(ms*1e6)); \
    cudaEventDestroy(s); cudaEventDestroy(e); \
} while(0)

int main() {
    int N = 1<<24; float *dx, *dy;
    cudaMalloc(&dx, N*4); cudaMalloc(&dy, N*4);
    BENCH(relu, "ReLU  ");
    BENCH(gelu, "GELU  ");
    BENCH(silu, "SiLU  ");
    cudaFree(dx); cudaFree(dy);
}
