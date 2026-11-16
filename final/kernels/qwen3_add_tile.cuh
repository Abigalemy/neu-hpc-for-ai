// Residual add: Y = T(float(A) + float(B)), elementwise over n values (same rounding as HF's `residual + hidden`).
#pragma once
#include "cuda_tile.h"

namespace ct = cuda::tiles;

constexpr int ADD_B = 1024;

template <typename T>
__tile_global__ void add_kernel(const T* __restrict__ _A, const T* __restrict__ _B, T* __restrict__ _Y, int n) {
    using namespace ct::literals;
    const T* A = ct::assume_aligned(_A, 16_ic);
    const T* B = ct::assume_aligned(_B, 16_ic);
    T* Y = ct::assume_aligned(_Y, 16_ic);
    auto pA = ct::partition_view{ct::tensor_span{A, ct::extents{n}}, ct::shape<ADD_B>{}};
    auto pB = ct::partition_view{ct::tensor_span{B, ct::extents{n}}, ct::shape<ADD_B>{}};
    auto pY = ct::partition_view{ct::tensor_span{Y, ct::extents{n}}, ct::shape<ADD_B>{}};
    int i = ct::bid().x;
    auto a = ct::element_cast<float>(pA.load_masked(i));
    auto b = ct::element_cast<float>(pB.load_masked(i));
    pY.store_masked(ct::element_cast<T>(a + b), i);
}

template <typename T>
cudaError_t launch_add(const T* a, const T* b, T* y, int n, cudaStream_t stream = 0) {
    add_kernel<T><<<(n + ADD_B - 1) / ADD_B, 1, 0, stream>>>(a, b, y, n);
    return cudaGetLastError();
}
