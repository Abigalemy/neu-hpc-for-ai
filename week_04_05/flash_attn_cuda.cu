#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <int HEAD_DIM>
__global__ void tiled_attention_opt_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N,
    float scale
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z;
    int tid = threadIdx.x;

    int offset_base = (b * H + h) * N;
    const float* q_ptr = Q + (offset_base + i) * HEAD_DIM;
    float* o_ptr = O + (offset_base + i) * HEAD_DIM;

    float q_val = (tid < HEAD_DIM) ? q_ptr[tid] : 0.0f;

    __shared__ float s_k[HEAD_DIM];
    __shared__ float s_v[HEAD_DIM];
    __shared__ float s_red[HEAD_DIM];

    float acc_o = 0.0f;
    float m_prev = -1e20f;
    float l_prev = 0.0f;

    for (int j = 0; j < N; ++j) {
        const float* k_ptr = K + (offset_base + j) * HEAD_DIM;
        const float* v_ptr = V + (offset_base + j) * HEAD_DIM;

        if (tid < HEAD_DIM) {
            s_k[tid] = k_ptr[tid];
            s_v[tid] = v_ptr[tid];
            s_red[tid] = q_val * s_k[tid];
        }
        __syncthreads();

        for (int stride = HEAD_DIM / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                s_red[tid] += s_red[tid + stride];
            }
            __syncthreads();
        }

        float score = s_red[0] * scale;
        float m_curr = fmaxf(m_prev, score);
        float p = expf(score - m_curr);
        float factor = expf(m_prev - m_curr);

        l_prev = l_prev * factor + p;
        if (tid < HEAD_DIM) {
            acc_o = acc_o * factor + p * s_v[tid];
        }
        m_prev = m_curr;
        __syncthreads();
    }

    if (tid < HEAD_DIM) {
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

    if (D == 64) {
        tiled_attention_opt_kernel<64><<<grid, block>>>(
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            o.data_ptr<float>(),
            B, H, N,
            scale
        );
    } else {
        TORCH_CHECK(false, "Unsupported HEAD_DIM: only 64 is supported");
    }

    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiled_attention", &tiled_attention_cuda, "Tiled Fused Attention Forward (CUDA)");
}