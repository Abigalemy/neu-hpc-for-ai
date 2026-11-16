// Qwen3 MLP (SwiGLU) as CUDA Tile kernels:   y = down( silu(gate(x)) * up(x) )   (no biases)
//
// Structure follows the course's Week 4 notebook and NVIDIA's tileMatmul sample: runtime extents on the
// tensor_span, assume_divisible on the dimensions, ct::irange over the K blocks, mma with an fp32 accumulator.
//
// Weights keep the HF layout: gate/up are [INTER, HID], down is [HID, INTER], so every linear layer is x @ W^T
// (hence the transpose of the weight tile).
// Two kernels:
//   1. gate_up_silu: one block per (token tile, INTER tile). Both matmuls accumulate in fp32 over K; the
//      activation and the product are fused so gate/up never go to memory.
//   2. down: one block per (token tile, HID tile).
// Rounding matches HF in the activation dtype T (bf16 or fp32):
//   g = T(x@Wg^T), u = T(x@Wu^T), s = T(silu(g)), h = T(s*u), y = T(h@Wd^T)
#pragma once
#include <cstdio>
#include "cuda_tile.h"

namespace ct = cuda::tiles;

constexpr int MLP_BM = 16;   // token tile
constexpr int MLP_BN = 32;   // output-feature tile
constexpr int MLP_BK = 32;   // reduction tile (small on purpose for now; tuned later with a profile)

template <typename T>
__tile_global__ void gate_up_silu_kernel(const T* __restrict__ _X, const T* __restrict__ _WG,
                                         const T* __restrict__ _WU, T* __restrict__ _H,
                                         int n_tokens, int _hid, int _inter) {
    using namespace ct::literals;
    using acc_t = ct::tile<float, ct::shape<MLP_BM, MLP_BN>>;
    const T* X = ct::assume_aligned(_X, 16_ic);
    const T* WG = ct::assume_aligned(_WG, 16_ic);
    const T* WU = ct::assume_aligned(_WU, 16_ic);
    T* H = ct::assume_aligned(_H, 16_ic);
    auto hid = ct::assume_divisible(_hid, 16_ic);
    auto inter = ct::assume_divisible(_inter, 16_ic);

    auto pX = ct::partition_view{ct::tensor_span{X, ct::extents{n_tokens, hid}}, ct::shape<MLP_BM, MLP_BK>{}};
    auto pWG = ct::partition_view{ct::tensor_span{WG, ct::extents{inter, hid}}, ct::shape<MLP_BN, MLP_BK>{}};
    auto pWU = ct::partition_view{ct::tensor_span{WU, ct::extents{inter, hid}}, ct::shape<MLP_BN, MLP_BK>{}};
    auto pH = ct::partition_view{ct::tensor_span{H, ct::extents{n_tokens, inter}}, ct::shape<MLP_BM, MLP_BN>{}};

    auto [mb, nb, dummy] = ct::bid();

    auto acc_g = ct::zeros<acc_t>();
    auto acc_u = ct::zeros<acc_t>();
    int k_blocks = (hid + MLP_BK - 1) / MLP_BK;
    for (auto kb : ct::irange(0, k_blocks)) {
        ct::tile<T, ct::shape<MLP_BM, MLP_BK>> x;
        ct::tile<T, ct::shape<MLP_BN, MLP_BK>> wg;
        ct::tile<T, ct::shape<MLP_BN, MLP_BK>> wu;
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

template <typename T>
__tile_global__ void down_kernel(const T* __restrict__ _H, const T* __restrict__ _WD, T* __restrict__ _Y,
                                 int n_tokens, int _hid, int _inter) {
    using namespace ct::literals;
    using acc_t = ct::tile<float, ct::shape<MLP_BM, MLP_BN>>;
    const T* H = ct::assume_aligned(_H, 16_ic);
    const T* WD = ct::assume_aligned(_WD, 16_ic);
    T* Y = ct::assume_aligned(_Y, 16_ic);
    auto hid = ct::assume_divisible(_hid, 16_ic);
    auto inter = ct::assume_divisible(_inter, 16_ic);

    auto pH = ct::partition_view{ct::tensor_span{H, ct::extents{n_tokens, inter}}, ct::shape<MLP_BM, MLP_BK>{}};
    auto pWD = ct::partition_view{ct::tensor_span{WD, ct::extents{hid, inter}}, ct::shape<MLP_BN, MLP_BK>{}};
    auto pY = ct::partition_view{ct::tensor_span{Y, ct::extents{n_tokens, hid}}, ct::shape<MLP_BM, MLP_BN>{}};

    auto [mb, nb, dummy] = ct::bid();

    auto acc = ct::zeros<acc_t>();
    int k_blocks = (inter + MLP_BK - 1) / MLP_BK;
    for (auto kb : ct::irange(0, k_blocks)) {
        ct::tile<T, ct::shape<MLP_BM, MLP_BK>> h;
        ct::tile<T, ct::shape<MLP_BN, MLP_BK>> wd;
        h = pH.load_masked(mb, kb);
        wd = pWD.load(nb, kb);
        acc = ct::mma(h, ct::transpose(wd), acc);
    }
    pY.store_masked(ct::element_cast<T>(acc), mb, nb);
}

// Host launcher. h_buf is scratch of size [n_tokens, inter] (type T). hid and inter must be multiples of 32.
template <typename T>
cudaError_t launch_mlp(const T* x, const T* wg, const T* wu, const T* wd, T* h_buf, T* y, int n_tokens, int hid,
                       int inter, cudaStream_t stream = 0) {
    if (hid % MLP_BK || inter % MLP_BK || hid % MLP_BN || inter % MLP_BN) {
        std::fprintf(stderr, "hid/inter must be multiples of %d\n", MLP_BK);
        return cudaErrorInvalidValue;
    }
    int gm = (n_tokens + MLP_BM - 1) / MLP_BM;
    gate_up_silu_kernel<T><<<dim3(gm, inter / MLP_BN, 1), 1, 0, stream>>>(x, wg, wu, h_buf, n_tokens, hid, inter);
    cudaError_t e1 = cudaGetLastError();
    if (e1 != cudaSuccess) {
        std::fprintf(stderr, "gate_up_silu launch (grid %d x %d) failed: %s\n", gm, inter / MLP_BN, cudaGetErrorString(e1));
        return e1;
    }
    down_kernel<T><<<dim3(gm, hid / MLP_BN, 1), 1, 0, stream>>>(h_buf, wd, y, n_tokens, hid, inter);
    cudaError_t e2 = cudaGetLastError();
    if (e2 != cudaSuccess)
        std::fprintf(stderr, "down launch (grid %d x %d) failed: %s\n", gm, hid / MLP_BN, cudaGetErrorString(e2));
    return e2;
}
