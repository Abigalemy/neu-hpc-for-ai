// Qwen3 MLP (SwiGLU) as CUDA Tile kernels:   y = down( silu(gate(x)) * up(x) )   (no biases)
//
// Structure follows the course's Week 4 notebook and NVIDIA's tileMatmul sample: runtime extents on the
// tensor_span, assume_divisible on the dimensions, ct::irange over the K blocks, mma with an fp32 accumulator.
//
// Weights keep the HF layout: gate/up are [INTER, HID], down is [HID, INTER], so every linear layer is x @ W^T
// (hence the transpose of the weight tile).
// Two steps:
//   1. gate_up_silu: one block per (token tile, INTER tile). Both matmuls accumulate in fp32 over K; the
//      activation and the product are fused so gate/up never go to memory.
//   2. down: an ordinary linear layer (launch_linear).
// Rounding matches HF in the activation dtype T (bf16 or fp32):
//   g = T(x@Wg^T), u = T(x@Wu^T), s = T(silu(g)), h = T(s*u), y = T(h@Wd^T)
#pragma once
#include <cstdio>
#include "cuda_tile.h"
#include "qwen3_linear_tile.cuh"

namespace ct = cuda::tiles;

template <typename T, int BM, int BN, int BK>
__tile_global__ void gate_up_silu_kernel(const T* __restrict__ _X, const T* __restrict__ _WG,
                                         const T* __restrict__ _WU, T* __restrict__ _H,
                                         int n_tokens, int _hid, int _inter) {
    using namespace ct::literals;
    using acc_t = ct::tile<float, ct::shape<BM, BN>>;
    const T* X = ct::assume_aligned(_X, 16_ic);
    const T* WG = ct::assume_aligned(_WG, 16_ic);
    const T* WU = ct::assume_aligned(_WU, 16_ic);
    T* H = ct::assume_aligned(_H, 16_ic);
    auto hid = ct::assume_divisible(_hid, 16_ic);
    auto inter = ct::assume_divisible(_inter, 16_ic);

    auto pX = ct::partition_view{ct::tensor_span{X, ct::extents{n_tokens, hid}}, ct::shape<BM, BK>{}};
    auto pWG = ct::partition_view{ct::tensor_span{WG, ct::extents{inter, hid}}, ct::shape<BN, BK>{}};
    auto pWU = ct::partition_view{ct::tensor_span{WU, ct::extents{inter, hid}}, ct::shape<BN, BK>{}};
    auto pH = ct::partition_view{ct::tensor_span{H, ct::extents{n_tokens, inter}}, ct::shape<BM, BN>{}};

    auto [mb, nb, dummy] = ct::bid();

    auto acc_g = ct::zeros<acc_t>();
    auto acc_u = ct::zeros<acc_t>();
    int k_blocks = (hid + BK - 1) / BK;
    for (auto kb : ct::irange(0, k_blocks)) {
        ct::tile<T, ct::shape<BM, BK>> x;
        ct::tile<T, ct::shape<BN, BK>> wg;
        ct::tile<T, ct::shape<BN, BK>> wu;
        x = pX.load_masked(mb, kb);      // token rows past n_tokens are padding; their outputs are not stored
        wg = pWG.load(nb, kb);
        wu = pWU.load(nb, kb);
        acc_g = ct::mma(x, ct::transpose(wg), acc_g);
        acc_u = ct::mma(x, ct::transpose(wu), acc_u);
    }

    auto g = ct::element_cast<float>(ct::element_cast<T>(acc_g));
    auto u = ct::element_cast<float>(ct::element_cast<T>(acc_u));
    auto one = ct::full<acc_t>(1.0f);
    auto silu = g / (one + ct::exp(-g));
    auto s = ct::element_cast<float>(ct::element_cast<T>(silu));
    pH.store_masked(ct::element_cast<T>(s * u), mb, nb);
}

// h_buf is scratch of size [n_tokens, inter] (type T).
template <typename T, int BM, int BN, int BK>
cudaError_t launch_mlp_gate_up_tiled(const T* x, const T* wg, const T* wu, T* h_buf, int n_tokens, int hid, int inter,
                                     cudaStream_t stream = 0) {
    if (hid % BK || inter % BN) {
        std::fprintf(stderr, "gate_up: hid/inter must be multiples of %d/%d\n", BK, BN);
        return cudaErrorInvalidValue;
    }
    int gm = (n_tokens + BM - 1) / BM;
    gate_up_silu_kernel<T, BM, BN, BK><<<dim3(gm, inter / BN, 1), 1, 0, stream>>>(x, wg, wu, h_buf, n_tokens, hid, inter);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        std::fprintf(stderr, "gate_up_silu launch (grid %d x %d) failed: %s\n", gm, inter / BN, cudaGetErrorString(e));
    return e;
}

// Tile choice: see tile_profile() in qwen3_linear_tile.cuh.
template <typename T>
cudaError_t launch_mlp_gate_up(const T* x, const T* wg, const T* wu, T* h_buf, int n_tokens, int hid, int inter,
                               cudaStream_t stream = 0) {
    // Measured on an H100 (tests/gemm_sweep_test.cu, fused kernel, 3072x2560 @ 2560x9728 x2): prefill 16x32x32 ->
    // 38 TFLOP/s, 128x64x64 -> 601 TFLOP/s (128x128 collapses to 95-166 TFLOP/s: two accumulators spill registers);
    // decode (1 token): 16x32x32 -> 59 us, 16x64x64 -> 45 us.
    if (tile_profile() == TileProfile::BIG && hid % 64 == 0 && inter % 64 == 0) {
        if (n_tokens > 64) return launch_mlp_gate_up_tiled<T, 128, 64, 64>(x, wg, wu, h_buf, n_tokens, hid, inter, stream);
        if (n_tokens > 16) return launch_mlp_gate_up_tiled<T, 64, 64, 64>(x, wg, wu, h_buf, n_tokens, hid, inter, stream);
        return launch_mlp_gate_up_tiled<T, 16, 64, 64>(x, wg, wu, h_buf, n_tokens, hid, inter, stream);
    }
    return launch_mlp_gate_up_tiled<T, 16, 32, 32>(x, wg, wu, h_buf, n_tokens, hid, inter, stream);
}

template <typename T>
cudaError_t launch_mlp_down(const T* h_buf, const T* wd, T* y, int n_tokens, int hid, int inter,
                            cudaStream_t stream = 0) {
    return launch_linear<T>(h_buf, wd, y, n_tokens, inter, hid, stream);
}

template <typename T>
cudaError_t launch_mlp(const T* x, const T* wg, const T* wu, const T* wd, T* h_buf, T* y, int n_tokens, int hid,
                       int inter, cudaStream_t stream = 0) {
    cudaError_t e = launch_mlp_gate_up<T>(x, wg, wu, h_buf, n_tokens, hid, inter, stream);
    if (e != cudaSuccess) return e;
    return launch_mlp_down<T>(h_buf, wd, y, n_tokens, hid, inter, stream);
}
