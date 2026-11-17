// Write the K and V of B new tokens into the KV cache at arbitrary rows (continuous batching: every token belongs to a
// different sequence / cache slot). ROWS[i] = slot_i * max_len + pos_i. One block per token.
#pragma once
#include "cuda_tile.h"

namespace ct = cuda::tiles;

template <typename T, int HKV>
__tile_global__ void kv_scatter_kernel(const T* __restrict__ _KN, const T* __restrict__ _VN, T* __restrict__ _KC,
                                       T* __restrict__ _VC, const int* __restrict__ ROWS, int n_tok, int n_rows) {
    using namespace ct::literals;
    constexpr int D = 128;
    int i = static_cast<int>(ct::bid().x);
    int r = ROWS[i];
    auto pKN = ct::partition_view{ct::tensor_span{_KN, ct::extents{n_tok, ct::integral_constant<HKV>{}, ct::integral_constant<D>{}}}, ct::shape<1, HKV, D>{}};
    auto pVN = ct::partition_view{ct::tensor_span{_VN, ct::extents{n_tok, ct::integral_constant<HKV>{}, ct::integral_constant<D>{}}}, ct::shape<1, HKV, D>{}};
    auto pKC = ct::partition_view{ct::tensor_span{_KC, ct::extents{n_rows, ct::integral_constant<HKV>{}, ct::integral_constant<D>{}}}, ct::shape<1, HKV, D>{}};
    auto pVC = ct::partition_view{ct::tensor_span{_VC, ct::extents{n_rows, ct::integral_constant<HKV>{}, ct::integral_constant<D>{}}}, ct::shape<1, HKV, D>{}};
    pKC.store(pKN.load(i, 0, 0), r, 0, 0);
    pVC.store(pVN.load(i, 0, 0), r, 0, 0);
}

template <typename T, int HKV>
cudaError_t launch_kv_scatter(const T* k_new, const T* v_new, T* k_cache, T* v_cache, const int* d_rows, int n_tok,
                              int n_rows, cudaStream_t stream = 0) {
    kv_scatter_kernel<T, HKV><<<n_tok, 1, 0, stream>>>(k_new, v_new, k_cache, v_cache, d_rows, n_tok, n_rows);
    return cudaGetLastError();
}
