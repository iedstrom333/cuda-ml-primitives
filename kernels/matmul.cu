#include <stdio.h>
#include <cuda_runtime.h>

#define TILE 32

__global__ void matmul(float *A, float *B, float *C, int M, int K, int N) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y*TILE + threadIdx.y;
    int col = blockIdx.x*TILE + threadIdx.x;
    float sum = 0.f;
    for (int t = 0; t < (K+TILE-1)/TILE; t++) {
        As[threadIdx.y][threadIdx.x] = (row < M && t*TILE+threadIdx.x < K)
            ? A[row*K + t*TILE+threadIdx.x] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (t*TILE+threadIdx.y < K && col < N)
            ? B[(t*TILE+threadIdx.y)*N+col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++) sum += As[threadIdx.y][k]*Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N+col] = sum;
}

int main() {
    int M = 4096, K = 4096, N = 4096;
    float *A, *B, *C;
    cudaMalloc(&A, M*K*4); cudaMalloc(&B, K*N*4); cudaMalloc(&C, M*N*4);
    dim3 blk(TILE, TILE), grd((N+TILE-1)/TILE, (M+TILE-1)/TILE);
    matmul<<<grd, blk>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    int reps = 10; cudaEventRecord(s);
    for (int i = 0; i < reps; i++) matmul<<<grd, blk>>>(A, B, C, M, K, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    printf("matmul %dx%d  %.2f TFLOPS  %.2f ms\n",
           M, N, 2.0*M*K*N*reps/(ms*1e9), ms/reps);
    cudaFree(A); cudaFree(B); cudaFree(C);
}
