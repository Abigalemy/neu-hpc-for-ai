// Qwen3 RoPE (rotate-half / split-half convention) as a CUDA Tile kernel, applied in place.
//
// X is [n_tokens, NH, 128] (token-major). One block handles one token: it loads the first and second halves of
// every head as (NH, 64) tiles, rotates the pairs (x[i], x[i + 64]) and stores them back.
// cos/sin tables are [max_pos, 64] (the first half of HF's duplicated [max_pos, 128] table), positions[token]
// selects the row.
//
// Rounding matches HF apply_rotary_pos_emb, which runs every op in the activation dtype T:
//   out1 = T( T(x1*cos) - T(x2*sin) ),   out2 = T( T(x2*cos) + T(x1*sin) )
#pragma once
#include "cuda_tile.h"

namespace ct = cuda::tiles;

constexpr int ROPE_HEAD_DIM = 128;
constexpr int ROPE_HALF = ROPE_HEAD_DIM / 2;

template <typename T, int NH>
__tile_global__ void rope_kernel(T* __restrict__ X, const T* __restrict__ COS, const T* __restrict__ SIN,
                                 const int* __restrict__ POS, int n_tokens, int max_pos) {
    X = ct::assume_aligned<16>(X);
    COS = ct::assume_aligned<16>(COS);
    SIN = ct::assume_aligned<16>(SIN);

    int token = ct::bid().x;
    int pos = POS[token];

    auto pX = ct::partition_view(
        ct::tensor_span{X, ct::extents{n_tokens, ct::integral_constant<NH>{}, ct::integral_constant<ROPE_HEAD_DIM>{}}},
        ct::shape<1, NH, ROPE_HALF>{});
    auto pCos = ct::partition_view(ct::tensor_span{COS, ct::extents{max_pos, ct::integral_constant<ROPE_HALF>{}}},
                                   ct::shape<1, ROPE_HALF>{});
    auto pSin = ct::partition_view(ct::tensor_span{SIN, ct::extents{max_pos, ct::integral_constant<ROPE_HALF>{}}},
                                   ct::shape<1, ROPE_HALF>{});

    auto c = ct::broadcast<ct::shape<NH, ROPE_HALF>>(ct::element_cast<float>(pCos.load(pos, 0)));
    auto s = ct::broadcast<ct::shape<NH, ROPE_HALF>>(ct::element_cast<float>(pSin.load(pos, 0)));

    auto x1 = ct::element_cast<float>(ct::reshape<ct::shape<NH, ROPE_HALF>>(pX.load(token, 0, 0)));
    auto x2 = ct::element_cast<float>(ct::reshape<ct::shape<NH, ROPE_HALF>>(pX.load(token, 0, 1)));

    // round every product to T, like HF does
    auto x1c = ct::element_cast<float>(ct::element_cast<T>(x1 * c));
    auto x2s = ct::element_cast<float>(ct::element_cast<T>(x2 * s));
    auto x2c = ct::element_cast<float>(ct::element_cast<T>(x2 * c));
    auto x1s = ct::element_cast<float>(ct::element_cast<T>(x1 * s));

    auto o1 = ct::element_cast<T>(x1c - x2s);
    auto o2 = ct::element_cast<T>(x2c + x1s);

    pX.store(ct::reshape<ct::shape<1, NH, ROPE_HALF>>(o1), token, 0, 0);
    pX.store(ct::reshape<ct::shape<1, NH, ROPE_HALF>>(o2), token, 0, 1);
}

// Host launcher: rotates q [n_tokens, NQ, 128] and k [n_tokens, NK, 128] in place.
template <typename T, int NQ, int NK>
cudaError_t launch_rope(T* q, T* k, const T* cos_tab, const T* sin_tab, const int* positions, int n_tokens,
                        int max_pos, cudaStream_t stream = 0) {
    rope_kernel<T, NQ><<<n_tokens, 1, 0, stream>>>(q, cos_tab, sin_tab, positions, n_tokens, max_pos);
    rope_kernel<T, NK><<<n_tokens, 1, 0, stream>>>(k, cos_tab, sin_tab, positions, n_tokens, max_pos);
    return cudaGetLastError();
}
