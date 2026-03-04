#include <stdio.h>
#include <cuda_runtime.h>

__global__ void naiveSoftmax(float *x, float *y, int rows, int cols) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float *xi = x+row*cols, *yi = y+row*cols;
    float sum = 0.f;
    for (int j = 0; j < cols; j++) sum += expf(xi[j]);
    for (int j = 0; j < cols; j++) yi[j] = expf(xi[j])/sum;
}

__global__ void stableSoftmax(float *x, float *y, int rows, int cols) {
    __shared__ float smem[32];
    int row = blockIdx.x, tid = threadIdx.x, lane = tid&31, wid = tid>>5;
    float *xi = x+row*cols, *yi = y+row*cols;
    float mx = -1e30f;
    for (int j = tid; j < cols; j += blockDim.x) mx = fmaxf(mx, xi[j]);
    for (int off = 16; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
    if (lane == 0) smem[wid] = mx;
    __syncthreads();
    mx = (tid < (blockDim.x+31)/32) ? smem[tid] : -1e30f;
    for (int off = 16; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
    mx = __shfl_sync(0xffffffff, mx, 0);
    float sum = 0.f;
    for (int j = tid; j < cols; j += blockDim.x) sum += expf(xi[j] - mx);
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) smem[wid] = sum;
    __syncthreads();
    sum = (tid < (blockDim.x+31)/32) ? smem[tid] : 0.f;
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    sum = __shfl_sync(0xffffffff, sum, 0);
    for (int j = tid; j < cols; j += blockDim.x) yi[j] = expf(xi[j]-mx)/sum;
}

int main() {
    int rows = 4096, cols = 1024;
    float *dx, *dy;
    cudaMalloc(&dx, rows*cols*4); cudaMalloc(&dy, rows*cols*4);
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    int reps = 100;
    int blk = 128, grd = (rows+blk-1)/blk;
    naiveSoftmax<<<grd,blk>>>(dx,dy,rows,cols); cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < reps; i++) naiveSoftmax<<<grd,blk>>>(dx,dy,rows,cols);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e);
    printf("Naive  %.2f GB/s\n", 2.0*rows*cols*4*reps/(ms*1e6));
    stableSoftmax<<<rows,256>>>(dx,dy,rows,cols); cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < reps; i++) stableSoftmax<<<rows,256>>>(dx,dy,rows,cols);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms,s,e);
    printf("Stable %.2f GB/s\n", 2.0*rows*cols*4*reps/(ms*1e6));
    cudaFree(dx); cudaFree(dy);
}
