#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <cassert>

#include "dfxp.cuh"

void dfxp_update_step(
    at::Tensor input,
    at::Tensor qmin_,
    at::Tensor qmax_,
    at::Tensor step_) {

    float mx = input.max().item<float>();
    float mn = input.min().item<float>();

    float qmin = qmin_.item<float>();
    float qmax = qmax_.item<float>();
    float step = step_.item<float>();

    if (mx > qmax * step || mn < qmin * step) {
        step_.mul_(2);
    } else if (mx <= qmax * step / 2 && mn >= qmin * step / 2) {
        step_.div_(2);
    }
}

at::Tensor dfxp_quantize_forward(
    at::Tensor input,
    at::Tensor qmin_,
    at::Tensor qmax_,
    at::Tensor step_,
    bool update_step) {

    if (update_step)
        dfxp_update_step(input, qmin_, qmax_, step_);

    float qmin = qmin_.item<float>();
    float qmax = qmax_.item<float>();
    float step = step_.item<float>();

    auto x = at::empty_like(input);
    dfxp::cuda_quantize(x.data<float>(), input.data<float>(), qmin, qmax, step, x.numel());
    return x;
}

at::Tensor dfxp_grad_quantize_backward(
    at::Tensor grad,
    at::Tensor qmin_,
    at::Tensor qmax_,
    at::Tensor step_,
    bool update_step) {

    if (update_step)
        dfxp_update_step(grad, qmin_, qmax_, step_);

    float qmin = qmin_.item<float>();
    float qmax = qmax_.item<float>();
    float step = step_.item<float>();

    dfxp::cuda_grad_quantize(grad.data<float>(), grad.data<float>(), qmin, qmax, step, grad.numel());
    return grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dfxp_quantize_forward", &dfxp_quantize_forward, "DFXP quantize forward");
    m.def("dfxp_grad_quantize_backward", &dfxp_grad_quantize_backward, "DFXP grad quantize backward");
}
