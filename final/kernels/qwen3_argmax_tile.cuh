// Greedy sampling: ids[r] = argmax over the vocabulary of logits[r, :]  (first index wins ties, like torch.argmax).
//
// One block per row. The row is walked in chunks of AM_BV columns; each chunk gives a max value (ct::reduce_max) and
// the position of its first maximum (ct::max_element, documented to return the lowest index on ties). A running
// (best value, best index) pair is updated only on a strictly greater value, so earlier chunks win ties.
// Columns past the vocabulary size are masked with -inf so zero padding can never win.
#pragma once
#include <cmath>
#include <cstdio>
#include "cuda_tile.h"

namespace ct = cuda::tiles;

constexpr int AM_BV = 1024;

template <typename T>
__tile_global__ void argmax_kernel(const T* __restrict__ _L, int* __restrict__ _OUT, int n_rows, int vocab) {
    using namespace ct::literals;
    using f32_cols = ct::tile<float, ct::shape<1, AM_BV>>;
    using i32_cols = ct::tile<int, ct::shape<1, AM_BV>>;
    using f32_one = ct::tile<float, ct::shape<1, 1>>;
    using i32_one = ct::tile<int, ct::shape<1, 1>>;

    const T* L = ct::assume_aligned(_L, 16_ic);
    int* OUT = ct::assume_aligned(_OUT, 4_ic);
    auto pL = ct::partition_view{ct::tensor_span{L, ct::extents{n_rows, vocab}}, ct::shape<1, AM_BV>{}};
    auto pOut = ct::partition_view{ct::tensor_span{OUT, ct::extents{n_rows, 1_ic}}, ct::shape<1, 1>{}};

    int row = ct::bid().x;
    auto best_v = ct::full<f32_one>(-INFINITY);
    auto best_i = ct::zeros<i32_one>();
    int chunks = (vocab + AM_BV - 1) / AM_BV;
    for (auto c : ct::irange(0, chunks)) {
        auto x = ct::element_cast<float>(pL.load_masked(row, c));
        i32_cols col = c * AM_BV + ct::iota<i32_cols>();
        auto xm = ct::select(col < vocab, x, ct::full<f32_cols>(-INFINITY));
        auto cv = ct::reduce_max(xm, 1_ic);
        // max_element yields an unsigned index tile; convert to int before adding the chunk offset
        i32_one ci = ct::element_cast<int>(ct::max_element(xm, 1_ic)) + ct::full<i32_one>(c * AM_BV);
        auto better = cv > best_v;
        best_i = ct::select(better, ci, best_i);
        best_v = ct::select(better, cv, best_v);
    }
    pOut.store(best_i, row, 0);
}

template <typename T>
cudaError_t launch_argmax(const T* logits, int* out_ids, int n_rows, int vocab, cudaStream_t stream = 0) {
    argmax_kernel<T><<<n_rows, 1, 0, stream>>>(logits, out_ids, n_rows, vocab);
    return cudaGetLastError();
}
