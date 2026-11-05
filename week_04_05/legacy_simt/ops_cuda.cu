#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int total_rows,
    int hidden_size,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= total_rows) return;

    const float* in_ptr = input + row * hidden_size;
    float* out_ptr = output + row * hidden_size;

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = in_ptr[i];
        sum_sq += val * val;
    }

    __shared__ float s_sum[256];
    s_sum[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    float mean_sq = s_sum[0] / hidden_size;
    float inv_rms = rsqrtf(mean_sq + eps);

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        out_ptr[i] = in_ptr[i] * inv_rms * weight[i];
    }
}

torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor weight, float eps) {
    int total_rows = x.numel() / x.size(-1);
    int hidden_size = x.size(-1);
    auto out = torch::empty_like(x);

    int threads = (hidden_size > 256) ? 256 : hidden_size;
    rms_norm_kernel<<<total_rows, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
        total_rows,
        hidden_size,
        eps
    );
    return out;
}

__global__ void rope_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int B, int H_q, int H_k, int N, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_half_d = (D / 2);
    int total_q = B * H_q * N * total_half_d;

    if (idx < total_q) {
        int half_d_idx = idx % total_half_d;
        int rem = idx / total_half_d;
        int pos = rem % N;
        int h = (rem / N) % H_q;
        int b = rem / (N * H_q);

        int q_base = ((b * H_q + h) * N + pos) * D;
        float cos_val = cos_cache[pos * total_half_d + half_d_idx];
        float sin_val = sin_cache[pos * total_half_d + half_d_idx];

        float q0 = q[q_base + half_d_idx];
        float q1 = q[q_base + half_d_idx + total_half_d];

        q[q_base + half_d_idx] = q0 * cos_val - q1 * sin_val;
        q[q_base + half_d_idx + total_half_d] = q1 * cos_val + q0 * sin_val;
    }
}

void apply_rope_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor cos_cache, torch::Tensor sin_cache) {
    int B = q.size(0);
    int H_q = q.size(1);
    int H_k = k.size(1);
    int N = q.size(2);
    int D = q.size(3);

    int total_q = B * H_q * N * (D / 2);
    int threads = 256;
    int blocks = (total_q + threads - 1) / threads;

    rope_kernel<<<blocks, threads>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        cos_cache.data_ptr<float>(),
        sin_cache.data_ptr<float>(),
        B, H_q, H_k, N, D
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm_cuda, "CUDA RMSNorm");
    m.def("apply_rope", &apply_rope_cuda, "CUDA Apply RoPE");
}