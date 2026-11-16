// Linear layer without bias, HF weight layout:  Y[n, out] = T( X[n, in] @ W[out, in]^T )   (fp32 accumulation).
// Used for lm_head (Qwen3-4B ties it to the embedding table). Same structure and tile sizes as the MLP kernels
// (see qwen3_mlp_tile.cuh for why the tiles are small).
#pragma once
#include <cstdio>
#include "cuda_tile.h"

namespace ct = cuda::tiles;

constexpr int LIN_BM = 16;
constexpr int LIN_BN = 32;
constexpr int LIN_BK = 32;

template <typename T>
__tile_global__ void linear_kernel(const T* __restrict__ _X, const T* __restrict__ _W, T* __restrict__ _Y,
                                   int n_tokens, int _in, int _out) {
    using namespace ct::literals;
    using acc_t = ct::tile<float, ct::shape<LIN_BM, LIN_BN>>;
    const T* X = ct::assume_aligned(_X, 16_ic);
    const T* W = ct::assume_aligned(_W, 16_ic);
    T* Y = ct::assume_aligned(_Y, 16_ic);
    auto in = ct::assume_divisible(_in, 16_ic);
    auto out = ct::assume_divisible(_out, 16_ic);

    auto pX = ct::partition_view{ct::tensor_span{X, ct::extents{n_tokens, in}}, ct::shape<LIN_BM, LIN_BK>{}};
    auto pW = ct::partition_view{ct::tensor_span{W, ct::extents{out, in}}, ct::shape<LIN_BN, LIN_BK>{}};
    auto pY = ct::partition_view{ct::tensor_span{Y, ct::extents{n_tokens, out}}, ct::shape<LIN_BM, LIN_BN>{}};

    auto [mb, nb, dummy] = ct::bid();

    auto acc = ct::zeros<acc_t>();
    int k_blocks = (in + LIN_BK - 1) / LIN_BK;
    for (auto kb : ct::irange(0, k_blocks)) {
        ct::tile<T, ct::shape<LIN_BM, LIN_BK>> x;
        ct::tile<T, ct::shape<LIN_BN, LIN_BK>> w;
        x = pX.load_masked(mb, kb);
        w = pW.load(nb, kb);
        acc = ct::mma(x, ct::transpose(w), acc);
    }
    pY.store_masked(ct::element_cast<T>(acc), mb, nb);
}

template <typename T>
cudaError_t launch_linear(const T* x, const T* w, T* y, int n_tokens, int in, int out, cudaStream_t stream = 0) {
    if (in % LIN_BK || out % LIN_BN) { std::fprintf(stderr, "in/out must be multiples of 32\n"); return cudaErrorInvalidValue; }
    int gm = (n_tokens + LIN_BM - 1) / LIN_BM;
    linear_kernel<T><<<dim3(gm, out / LIN_BN, 1), 1, 0, stream>>>(x, w, y, n_tokens, in, out);
    return cudaGetLastError();
}
