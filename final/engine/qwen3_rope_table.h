// RoPE cos/sin tables on the host, following HF Transformers step by step so the bf16 entries match bit for bit:
//   default RoPE : inv_freq = 1 / theta^(2i/dim)
//   YaRN         : transformers/modeling_rope_utils.py::_compute_yarn_parameters (beta_fast=32, beta_slow=1,
//                  truncate=True); cos and sin are additionally multiplied by attention_factor = 0.1 ln(factor) + 1.
// HF computes in fp32 (torch), so everything below is float, except the correction range, which HF computes with
// Python floats (double).
// Output: cos then sin, each [n_pos, dim/2], bf16.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include <cuda_bf16.h>

struct RopeParams {
    int head_dim = 128;
    double theta = 1e6;
    bool yarn = false;
    double factor = 1.0;                // YaRN scaling factor (e.g. 4.0 for 32K -> 128K)
    int orig_max = 32768;               // original_max_position_embeddings
    double beta_fast = 32.0, beta_slow = 1.0;
    double attention_factor = -1.0;     // < 0: derive as 0.1 ln(factor) + 1 (HF default)
};

static std::vector<__nv_bfloat16> rope_tables_bf16(const RopeParams& p, int n_pos) {
    const int dim = p.head_dim, half = dim / 2;
    std::vector<float> inv(half);
    float attention = 1.0f;

    // pos_freqs = base ** (arange(0, dim, 2) / dim)   (torch: float32 pow)
    std::vector<float> pos_freqs(half);
    for (int i = 0; i < half; ++i) pos_freqs[i] = std::pow((float)p.theta, (float)(2 * i) / (float)dim);

    if (!p.yarn) {
        for (int i = 0; i < half; ++i) inv[i] = 1.0f / pos_freqs[i];
    } else {
        double att = p.attention_factor;
        if (att < 0) att = p.factor <= 1 ? 1.0 : 0.1 * std::log(p.factor) + 1.0;
        attention = (float)att;

        auto correction_dim = [&](double rotations) {
            return (dim * std::log(p.orig_max / (rotations * 2 * M_PI))) / (2 * std::log(p.theta));
        };
        double low = std::floor(correction_dim(p.beta_fast));    // truncate = True
        double high = std::ceil(correction_dim(p.beta_slow));
        low = std::max(low, 0.0);
        high = std::min(high, (double)(dim - 1));
        double span_hi = high;
        if (low == span_hi) span_hi += 0.001;                     // HF: prevent a singularity

        for (int i = 0; i < half; ++i) {
            float extrapolation = 1.0f / pos_freqs[i];
            float interpolation = 1.0f / ((float)p.factor * pos_freqs[i]);
            float ramp = ((float)i - (float)low) / ((float)span_hi - (float)low);
            ramp = std::min(std::max(ramp, 0.0f), 1.0f);
            float extrap_factor = 1.0f - ramp;
            inv[i] = interpolation * (1.0f - extrap_factor) + extrapolation * extrap_factor;
        }
    }

    std::vector<__nv_bfloat16> out(2 * (size_t)n_pos * half);
    for (int pos = 0; pos < n_pos; ++pos)
        for (int i = 0; i < half; ++i) {
            float f = (float)pos * inv[i];
            out[(size_t)pos * half + i] = __float2bfloat16(std::cos(f) * attention);
            out[((size_t)n_pos + pos) * half + i] = __float2bfloat16(std::sin(f) * attention);
        }
    return out;
}
