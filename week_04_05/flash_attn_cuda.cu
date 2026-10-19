#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void tiled_attention_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N, int D,
    float scale
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z;

    int tid = threadIdx.x;

    const float* q_ptr = Q + ((b * H + h) * N + i) * D;
    float* o_ptr = O + ((b * H + h) * N + i) * D;

    float acc_o = 0.0f;
    float m_prev = -1e20f;
    float l_prev = 0.0f;

    for (int j = 0; j < N; ++j) {
        const float* k_ptr = K + ((b * H + h) * N + j) * D;
        const float* v_ptr = V + ((b * H + h) * N + j) * D;

        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += q_ptr[d] * k_ptr[d];
        }
        score *= scale;

        float m_curr = fmaxf(m_prev, score);
        float p = expf(score - m_curr);
        float factor = expf(m_prev - m_curr);

        l_prev = l_prev * factor + p;

        if (tid < D) {
            acc_o = acc_o * factor + p * v_ptr[tid];
        }
        m_prev = m_curr;
    }

    if (tid < D) {
        o_ptr[tid] = acc_o / l_prev;
    }
}

torch::Tensor tiled_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    int B = q.size(0);
    int H = q.size(1);
    int N = q.size(2);
    int D = q.size(3);

    auto o = torch::empty_like(q);
    float scale = 1.0f / sqrtf((float)D);

    dim3 grid(B, H, N);
    dim3 block(D);

    tiled_attention_forward_kernel<<<grid, block>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        o.data_ptr<float>(),
        B, H, N, D,
        scale
    );

    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiled_attention", &tiled_attention_cuda, "Tiled Fused Attention Forward (CUDA)");
}