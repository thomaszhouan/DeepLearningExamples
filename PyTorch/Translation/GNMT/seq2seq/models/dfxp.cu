#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAContext.h>

#include <iostream>
#include <ctime>

#include <curand_kernel.h>

#include "dfxp.cuh"

namespace dfxp {

const int BACKWARD_BLOCK_SIZE = 1024;
const int BACKWARD_GRID_SIZE = 64;
unsigned long long curandOffset = 0;

template <int nt>
__global__ void cuda_quantize_kernel(
    float *out,
    float *in,
    float qmin,
    float qmax,
    float step,
    int64_t n) {

    int i = blockIdx.x * nt + threadIdx.x;
    if (i < n) {
        auto v = in[i];
        v /= step;
        v = fmaxf(v, qmin);
        v = fminf(v, qmax);
        v = nearbyintf(v);
        v *= step;
        out[i] = v;
    }
}

void cuda_quantize(
    float *out,
    float *in,
    float qmin,
    float qmax,
    float step,
    int64_t n) {

    const int BLOCK_SIZE = 1024;
    int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cuda_quantize_kernel<BLOCK_SIZE>
        <<<GRID_SIZE, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
            out, in, qmin, qmax, step, n);
}

template <int nb, int nt>
__global__ void cuda_grad_quantize_kernel(
    float *out,
    float *in,
    float qmin,
    float qmax,
    float step,
    int64_t n,
    unsigned long long offset) {

    int index = blockIdx.x * nt + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(
        /*seed=*/1024,
        /*subsequence=*/index,
        /*offset=*/offset,
        /*state=*/&state);

    for (int j = index; j < n; j += 4 * nt * nb) {
        auto noise = curand_uniform4(&state);

        int i = j;
        auto v = in[i];
        v /= step;
        v += noise.x;
        v = fmaxf(v, qmin);
        v = fminf(v, qmax);
        v = floorf(v);
        v *= step;
        out[i] = v;

        i += nt * nb;
        if (i >= n) break;
        v = in[i];
        v /= step;
        v += noise.y;
        v = fmaxf(v, qmin);
        v = fminf(v, qmax);
        v = floorf(v);
        v *= step;
        out[i] = v;

        i += nt * nb;
        if (i >= n) break;
        v = in[i];
        v /= step;
        v += noise.z;
        v = fmaxf(v, qmin);
        v = fminf(v, qmax);
        v = floorf(v);
        v *= step;
        out[i] = v;

        i += nt * nb;
        if (i >= n) break;
        v = in[i];
        v /= step;
        v += noise.w;
        v = fmaxf(v, qmin);
        v = fminf(v, qmax);
        v = floorf(v);
        v *= step;
        out[i] = v;
    }
}

void cuda_grad_quantize(
    float *out,
    float *in,
    float qmin,
    float qmax,
    float step,
    int64_t n) {

    const int BLOCK_SIZE = BACKWARD_BLOCK_SIZE;
    const int GRID_SIZE = BACKWARD_GRID_SIZE;
    cuda_grad_quantize_kernel<GRID_SIZE, BLOCK_SIZE>
        <<<GRID_SIZE, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
            out, in, qmin, qmax, step, n, curandOffset);

    // increment offset by max # of curand4 call per thread
    curandOffset += (n + 4 * BLOCK_SIZE * GRID_SIZE - 1) / (4 * BLOCK_SIZE * GRID_SIZE);
}

} // namespace dfxp
