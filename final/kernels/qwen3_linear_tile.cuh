// Linear layer without bias, HF weight layout:  Y[n, out] = T( X[n, in] @ W[out, in]^T )   (fp32 accumulation).
// Used for the q/k/v/o projections, the MLP down projection and lm_head (Qwen3-4B ties it to the embedding table).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include "cuda_tile.h"

namespace ct = cuda::tiles;

// Tile sizes. The big tiles need a lot of shared memory / registers: they launch on Hopper (H100/H200) but the L4
// (99 KB of shared memory per block) rejects bf16 launches that big ("invalid argument"), so the tile choice depends on
// the device (see tile_profile()). Measured on an H100 for a 3072x2560 @ 2560x9728 bf16 GEMM
// (tests/gemm_sweep_test.cu): 16x32x32 -> 21 TFLOP/s, 128x128x64 -> 630 TFLOP/s (prefill); decode (1 token):
// 16x32x32 -> 0.98 TB/s, 64x128x64 -> 2.17 TB/s.
enum class TileProfile { SMALL, BIG };

// BIG on devices with >= 200 KB of opt-in shared memory per block (Hopper and newer), SMALL otherwise.
static TileProfile tile_profile() {
    static int cached = -1;
    if (cached < 0) {
        int dev = 0, smem = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        cached = smem >= 200 * 1024 ? 1 : 0;
        const char* force = std::getenv("QWEN3_TILES");   // "small" forces the verified small tiles (for A/B checks)
        if (force && std::string(force) == "small") cached = 0;
    }
    return cached ? TileProfile::BIG : TileProfile::SMALL;
}

template <typename T, int BM, int BN, int BK>
__tile_global__ void linear_kernel(const T* __restrict__ _X, const T* __restrict__ _W, T* __restrict__ _Y,
                                   int n_tokens, int _in, int _out) {
    using namespace ct::literals;
    using acc_t = ct::tile<float, ct::shape<BM, BN>>;
    const T* X = ct::assume_aligned(_X, 16_ic);
    const T* W = ct::assume_aligned(_W, 16_ic);
    T* Y = ct::assume_aligned(_Y, 16_ic);
    auto in = ct::assume_divisible(_in, 16_ic);
    auto out = ct::assume_divisible(_out, 16_ic);

    auto pX = ct::partition_view{ct::tensor_span{X, ct::extents{n_tokens, in}}, ct::shape<BM, BK>{}};
    auto pW = ct::partition_view{ct::tensor_span{W, ct::extents{out, in}}, ct::shape<BN, BK>{}};
    auto pY = ct::partition_view{ct::tensor_span{Y, ct::extents{n_tokens, out}}, ct::shape<BM, BN>{}};

    auto [mb, nb, dummy] = ct::bid();

    auto acc = ct::zeros<acc_t>();
    int k_blocks = (in + BK - 1) / BK;
    for (auto kb : ct::irange(0, k_blocks)) {
        ct::tile<T, ct::shape<BM, BK>> x;
        ct::tile<T, ct::shape<BN, BK>> w;
        x = pX.load_masked(mb, kb);
        w = pW.load(nb, kb);
        acc = ct::mma(x, ct::transpose(w), acc);
    }
    pY.store_masked(ct::element_cast<T>(acc), mb, nb);
}

template <typename T, int BM, int BN, int BK>
cudaError_t launch_linear_tiled(const T* x, const T* w, T* y, int n_tokens, int in, int out, cudaStream_t stream = 0) {
    if (in % BK || out % BN) { std::fprintf(stderr, "linear: in/out must be multiples of %d/%d\n", BK, BN); return cudaErrorInvalidValue; }
    int gm = (n_tokens + BM - 1) / BM;
    linear_kernel<T, BM, BN, BK><<<dim3(gm, out / BN, 1), 1, 0, stream>>>(x, w, y, n_tokens, in, out);
    return cudaGetLastError();
}

// y = x @ W^T. Picks the tile from the device and the number of tokens.
template <typename T>
cudaError_t launch_linear(const T* x, const T* w, T* y, int n_tokens, int in, int out, cudaStream_t stream = 0) {
    if (tile_profile() == TileProfile::BIG && in % 64 == 0 && out % 128 == 0) {
        if (n_tokens > 64) return launch_linear_tiled<T, 128, 128, 64>(x, w, y, n_tokens, in, out, stream);
        return launch_linear_tiled<T, 64, 128, 64>(x, w, y, n_tokens, in, out, stream);   // decode / short chunks
    }
    return launch_linear_tiled<T, 16, 32, 32>(x, w, y, n_tokens, in, out, stream);
}
