#pragma once

#define checkCUDA(expr)                                           \
    {                                                             \
        cudaError_t status = (expr);                              \
        if (status != cudaSuccess) {                              \
            std::cerr << "Error on line " << __LINE__ << ": "     \
                      << cudaGetErrorString(status) << std::endl; \
            std::exit(EXIT_FAILURE);                              \
        }                                                         \
    }

namespace dfxp {

void cuda_quantize(float *out, float *in, float qmin, float qmax, float step, int64_t n);
void cuda_grad_quantize(float *out, float *in, float qmin, float qmax, float step, int64_t n);

} // namespace dfxp
