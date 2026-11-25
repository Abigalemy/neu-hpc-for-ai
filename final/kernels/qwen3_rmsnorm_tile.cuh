// Qwen3 RMSNorm as a CUDA Tile kernel.
//
// Matches HF Qwen3RMSNorm exactly, including where it rounds:
//   x32  = float(x)
//   xn   = T( x32 * rsqrt(mean(x32^2) + eps) )    <- rounded to T (bf16/fp32) here
//   y    = T( float(weight) * float(xn) )
//
// One block handles BLOCK_N rows. D is split into D / BLOCK_D chunks (tile shapes must be powers of two, and
// 2560 is not), so the row is read twice: once for the sum of squares, once to write the output.
#pragma once
#include "cuda_tile.h"

namespace ct = cuda::tiles;

template <typename T, int BLOCK_N, int BLOCK_D, int D, float EPS>
__tile_global__ void rmsnorm_kernel(const T* __restrict__ X, T* __restrict__ Y, const T* __restrict__ W,
                                    int n_rows) {
    static_assert(D % BLOCK_D == 0, "D must be a multiple of BLOCK_D");
    constexpr int NCHUNK = D / BLOCK_D;
    using f32_rows = ct::tile<float, ct::shape<BLOCK_N>>;

    X = ct::assume_aligned<16>(X);
    Y = ct::assume_aligned<16>(Y);
    W = ct::assume_aligned<16>(W);

    auto pX = ct::partition_view(ct::tensor_span{X, ct::extents{n_rows, ct::integral_constant<D>{}}},
                                 ct::shape<BLOCK_N, BLOCK_D>{});
    auto pY = ct::partition_view(ct::tensor_span{Y, ct::extents{n_rows, ct::integral_constant<D>{}}},
                                 ct::shape<BLOCK_N, BLOCK_D>{});
    auto pW = ct::partition_view(ct::tensor_span{W, ct::extents{ct::integral_constant<D>{}}},
                                 ct::shape<BLOCK_D>{});

    int row_block = ct::bid().x;

    // Pass 1: sum of squares over the chunks of each row.
    auto sumsq = ct::zeros<f32_rows>();
    for (auto c : ct::irange(0, NCHUNK)) {
        auto x = ct::element_cast<float>(pX.load_masked(row_block, c));
        sumsq = sumsq + ct::reshape<ct::shape<BLOCK_N>>(ct::sum<1>(x * x));
    }
    auto rstd = ct::rsqrt(sumsq * ct::full<f32_rows>(1.0f / static_cast<float>(D)) + ct::full<f32_rows>(EPS));
    auto rstd_col = ct::reshape<ct::shape<BLOCK_N, 1>>(rstd);

    // Pass 2: normalize, round to T, multiply by weight, round to T.
    for (auto c : ct::irange(0, NCHUNK)) {
        auto x = ct::element_cast<float>(pX.load_masked(row_block, c));
        auto xn = ct::element_cast<T>(x * rstd_col);
        auto w = ct::reshape<ct::shape<1, BLOCK_D>>(ct::element_cast<float>(pW.load(c)));
        auto y = ct::element_cast<float>(xn) * w;
        pY.store_masked(ct::element_cast<T>(y), row_block, c);
    }
}

// Host launcher: Y[n_rows, D] = rmsnorm(X[n_rows, D]) * W[D].
template <typename T, int BLOCK_D, int D>
cudaError_t launch_rmsnorm(const T* x, T* y, const T* w, int n_rows, cudaStream_t stream = 0) {
    constexpr int BLOCK_N = 4;
    int grid = (n_rows + BLOCK_N - 1) / BLOCK_N;
    rmsnorm_kernel<T, BLOCK_N, BLOCK_D, D, 1e-6f><<<grid, 1, 0, stream>>>(x, y, w, n_rows);
    return cudaGetLastError();
}
