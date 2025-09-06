#include <cuda_runtime.h>
#include <algorithm>
#include "denoiser_cuda.h"

__global__ void simpleDenoiseKernel(
    const uint8_t* cur, const uint8_t* prev, const uint8_t* next,
    uint8_t* dst, int w, int h, int stride,
    int strength, float temporal)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * stride + x;
    float val = cur[idx];

    // ŽžŠÔ•ûŒü•½‹Ï
    if (prev && next) {
        val = (1.0f - temporal) * val
            + temporal * 0.5f * (prev[idx] + next[idx]);
    }
    else if (prev) {
        val = (1.0f - temporal) * val + temporal * prev[idx];
    }
    else if (next) {
        val = (1.0f - temporal) * val + temporal * next[idx];
    }

    // ‹óŠÔ•ûŒü 3x3 ‚Ú‚©‚µ
    float sum = 0.0f;
    int count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int xx = min(max(x + dx, 0), w - 1);
            int yy = min(max(y + dy, 0), h - 1);
            sum += cur[yy * stride + xx];
            count++;
        }
    }
    float spatial = sum / count;

    // ŽžŠÔ‚Æ‹óŠÔ‚ÌƒuƒŒƒ“ƒh
    float blended = (1.0f - strength / 100.0f) * val
        + (strength / 100.0f) * spatial;

    dst[idx] = (uint8_t)min(max(int(blended + 0.5f), 0), 255);
}

void run_simple_denoise(
    const uint8_t* cur,
    const uint8_t* prev,
    const uint8_t* next,
    uint8_t* dst,
    int w, int h, int stride,
    int strength, float temporal)
{
    size_t frame_size = stride * h;
    uint8_t* d_cur, * d_prev = nullptr, * d_next = nullptr, * d_dst;

    cudaMalloc(&d_cur, frame_size);
    cudaMalloc(&d_dst, frame_size);
    if (prev) cudaMalloc(&d_prev, frame_size);
    if (next) cudaMalloc(&d_next, frame_size);

    cudaMemcpy(d_cur, cur, frame_size, cudaMemcpyHostToDevice);
    if (prev) cudaMemcpy(d_prev, prev, frame_size, cudaMemcpyHostToDevice);
    if (next) cudaMemcpy(d_next, next, frame_size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((w + 15) / 16, (h + 15) / 16);
    simpleDenoiseKernel << <blocks, threads >> > (
        d_cur, d_prev, d_next,
        d_dst, w, h, stride, strength, temporal);

    cudaMemcpy(dst, d_dst, frame_size, cudaMemcpyDeviceToHost);

    cudaFree(d_cur);
    cudaFree(d_dst);
    if (prev) cudaFree(d_prev);
    if (next) cudaFree(d_next);
}
