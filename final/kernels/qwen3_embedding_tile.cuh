// Token embedding lookup: Y[t, :] = W[ids[t], :]   (W is [vocab, hid]).
// One block per token; the row is copied in chunks of EMB_BD columns. A pure copy, so the result is bit-exact.
#pragma once
#include <cstdio>
#include "cuda_tile.h"

namespace ct = cuda::tiles;

constexpr int EMB_BD = 128;  // hid must be a multiple of this

template <typename T>
__tile_global__ void embedding_kernel(const int* __restrict__ ids, const T* __restrict__ _W, T* __restrict__ _Y,
                                      int vocab, int n_tokens, int _hid) {
    using namespace ct::literals;
    const T* W = ct::assume_aligned(_W, 16_ic);
    T* Y = ct::assume_aligned(_Y, 16_ic);
    auto hid = ct::assume_divisible(_hid, 16_ic);

    auto pW = ct::partition_view{ct::tensor_span{W, ct::extents{vocab, hid}}, ct::shape<1, EMB_BD>{}};
    auto pY = ct::partition_view{ct::tensor_span{Y, ct::extents{n_tokens, hid}}, ct::shape<1, EMB_BD>{}};

    int token = ct::bid().x;
    int id = ids[token];
    int chunks = _hid / EMB_BD;
    for (auto c : ct::irange(0, chunks)) {
        ct::tile<T, ct::shape<1, EMB_BD>> row;
        row = pW.load(id, c);
        pY.store(row, token, c);
    }
}

template <typename T>
cudaError_t launch_embedding(const int* ids, const T* w, T* y, int vocab, int n_tokens, int hid,
                             cudaStream_t stream = 0) {
    if (hid % EMB_BD) { std::fprintf(stderr, "hid must be a multiple of %d\n", EMB_BD); return cudaErrorInvalidValue; }
    embedding_kernel<T><<<n_tokens, 1, 0, stream>>>(ids, w, y, vocab, n_tokens, hid);
    return cudaGetLastError();
}
