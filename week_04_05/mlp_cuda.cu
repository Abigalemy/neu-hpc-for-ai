#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void fused_swiglu_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float g = gate[idx];
        float u = up[idx];
        out[idx] = silu(g) * u;
    }
}

torch::Tensor fused_swiglu_cuda(torch::Tensor gate, torch::Tensor up) {
    auto size = gate.sizes();
    auto out = torch::empty(size, gate.options());

    int total_elements = gate.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_swiglu_kernel<<<blocks, threads>>>(
        gate.data_ptr<float>(),
        up.data_ptr<float>(),
        out.data_ptr<float>(),
        total_elements
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_swiglu", &fused_swiglu_cuda, "Fused SwiGLU (CUDA)");
}