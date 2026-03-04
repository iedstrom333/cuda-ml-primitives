#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void qkCausalMask(float *Q, float *K, float *S, int seq, int d) {
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if (row >= seq || col >= seq) return;
    float acc = 0.f;
    for (int i = 0; i < d; i++) acc += Q[row*d+i]*K[col*d+i];
    S[row*seq+col] = (col > row) ? -1e9f : acc/sqrtf((float)d);
}

__global__ void rowSoftmax(float *S, int seq) {
    __shared__ float smem[32];
    int row = blockIdx.x, tid = threadIdx.x, lane = tid&31, wid = tid>>5;
    float *si = S + row*seq;
    float mx = -1e30f;
    for (int j = tid; j < seq; j += blockDim.x) mx = fmaxf(mx, si[j]);
    for (int off = 16; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
    if (lane == 0) smem[wid] = mx;
    __syncthreads();
    mx = (tid < (blockDim.x+31)/32) ? smem[tid] : -1e30f;
    for (int off = 16; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
    mx = __shfl_sync(0xffffffff, mx, 0);
    float sum = 0.f;
    for (int j = tid; j < seq; j += blockDim.x) sum += expf(si[j] - mx);
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) smem[wid] = sum;
    __syncthreads();
    sum = (tid < (blockDim.x+31)/32) ? smem[tid] : 0.f;
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    sum = __shfl_sync(0xffffffff, sum, 0);
    for (int j = tid; j < seq; j += blockDim.x) si[j] = expf(si[j]-mx)/sum;
}

__global__ void attnOut(float *P, float *V, float *O, int seq, int d) {
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if (row >= seq || col >= d) return;
    float acc = 0.f;
    for (int k = 0; k < seq; k++) acc += P[row*seq+k]*V[k*d+col];
    O[row*d+col] = acc;
}

int main() {
    int seq = 512, d = 64;
    float *Q, *K, *V, *S, *O;
    cudaMalloc(&Q,seq*d*4); cudaMalloc(&K,seq*d*4); cudaMalloc(&V,seq*d*4);
    cudaMalloc(&S,seq*seq*4); cudaMalloc(&O,seq*d*4);
    dim3 blk(16,16), grd_ss((seq+15)/16,(seq+15)/16), grd_sd((d+15)/16,(seq+15)/16);
    qkCausalMask<<<grd_ss,blk>>>(Q,K,S,seq,d);
    rowSoftmax<<<seq,256>>>(S,seq);
    attnOut<<<grd_sd,blk>>>(S,V,O,seq,d);
    cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    int reps = 100; cudaEventRecord(s);
    for (int i = 0; i < reps; i++) {
        qkCausalMask<<<grd_ss,blk>>>(Q,K,S,seq,d);
        rowSoftmax<<<seq,256>>>(S,seq);
        attnOut<<<grd_sd,blk>>>(S,V,O,seq,d);
    }
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e);
    double flops = reps * (2.0*seq*seq*d + seq*seq + 2.0*seq*seq*d);
    printf("Attention seq=%d d=%d  %.2f TFLOPS  %.2f ms\n",
           seq, d, flops/(ms*1e9), ms/reps);
    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(S); cudaFree(O);
}
