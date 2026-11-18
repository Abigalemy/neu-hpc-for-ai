// Qwen3-MoE sparse block (Qwen3MoeSparseMoeBlock, the layer that replaces Qwen3MLP in Qwen3-30B-A3B) as CUDA Tile
// kernels. HF semantics (transformers 5.x, modeling_qwen3_moe.py):
//   logits = T(x @ Wr^T)                       [tokens, E]   router, bf16 like F.linear
//   p      = softmax(float(logits))            fp32
//   w, idx = topk(p, K); w /= sum(w); w = T(w) norm_topk_prob (Qwen3-30B-A3B: E = 128, K = 8)
//   y[t]   = sum_j w[t,j] * expert_{idx[t,j]}(x[t]),   expert_e(x) = Wd_e (silu(Wg_e x) * Wu_e x)
// HF adds the K weighted expert outputs into a bf16 tensor one by one; here the sum is in fp32 and rounded once.
//
// Everything stays on the GPU (no host round trip for the routing). K assignments per token, a = t*K + j:
//   1. moe_router      logits + softmax + top-K + renormalise, fused (one block per BR tokens)
//   2. moe_count       COUNT[e] = number of assignments to expert e (one block per expert)
//   3. moe_sort        stable counting sort of the assignments by expert: SORTED[OFF[e] + i] = the i-th assignment of
//                      expert e (token order), INV[a] = its position, OFF[e] = sum of the counts of experts < e (one
//                      block per expert; the position inside a tile of assignments is a prefix sum, done as a mma
//                      with a lower-triangular matrix of ones)
//   4. moe_gate_up     per (expert, row tile, INTER tile): H rows gathered from x with gather_scatter_view by token
//                      id (SORTED / K), fused SwiGLU like gate_up_silu_kernel, rows stored in sorted order
//   5. moe_down        per (expert, row tile, HID tile): expert output rows in sorted order (T, like HF's expert out)
//   6. moe_combine     y[t] = T(sum_j float(w[t,j]) * float(Ysorted[INV[t*K+j]])), rows gathered by INV
// Steps 4/5 launch a grid for the worst case (every token on every expert); tiles past an expert's count do no work.
// Expert weights are packed per layer: Wg, Wu [E, INTER, HID], Wd [E, HID, INTER] (HF layout per expert).
#pragma once
#include <cstdio>
#include "cuda_tile.h"
#include "qwen3_linear_tile.cuh"

namespace ct = cuda::tiles;

template <typename T>
struct MoeWorkspace {     // device scratch for up to max_tokens tokens (A = max_tokens * K assignments)
    int* idx = nullptr;       // [tokens, K] chosen experts
    T* w = nullptr;           // [tokens, K] their weights (T, as HF)
    int* count = nullptr;     // [E]
    int* off = nullptr;       // [E] first sorted position of each expert (written by moe_sort)
    int* sorted = nullptr;    // [A]
    int* inv = nullptr;       // [A]
    T* hbuf = nullptr;        // [A, INTER]
    T* ysorted = nullptr;     // [A, HID]
    int max_tokens = 0;
};

// ---- 1. router ---------------------------------------------------------------------------------------------------
template <typename T, int E, int K, bool NORM, int BR, int BK>
__tile_global__ void moe_router_kernel(const T* __restrict__ _X, const T* __restrict__ _WR, int* __restrict__ IDX,
                                       T* __restrict__ W, int n_tokens, int _hid) {
    using namespace ct::literals;
    using f32_te = ct::tile<float, ct::shape<BR, E>>;
    using f32_tk = ct::tile<float, ct::shape<BR, K>>;
    using i32_tk = ct::tile<int, ct::shape<BR, K>>;
    const T* X = ct::assume_aligned(_X, 16_ic);
    const T* WR = ct::assume_aligned(_WR, 16_ic);
    auto hid = ct::assume_divisible(_hid, 16_ic);
    auto pX = ct::partition_view{ct::tensor_span{X, ct::extents{n_tokens, hid}}, ct::shape<BR, BK>{}};
    auto pW = ct::partition_view{ct::tensor_span{WR, ct::extents{ct::integral_constant<E>{}, hid}}, ct::shape<E, BK>{}};
    auto pI = ct::partition_view{ct::tensor_span{IDX, ct::extents{n_tokens, ct::integral_constant<K>{}}}, ct::shape<BR, K>{}};
    auto pO = ct::partition_view{ct::tensor_span{W, ct::extents{n_tokens, ct::integral_constant<K>{}}}, ct::shape<BR, K>{}};
    int tb = static_cast<int>(ct::bid().x);

    auto acc = ct::zeros<f32_te>();
    int k_blocks = (hid + BK - 1) / BK;
    for (auto kb : ct::irange(0, k_blocks)) {
        ct::tile<T, ct::shape<BR, BK>> x = pX.load_masked(tb, kb);
        ct::tile<T, ct::shape<E, BK>> w = pW.load(0, kb);
        acc = ct::mma(x, ct::transpose(w), acc);
    }
    auto logits = ct::element_cast<float>(ct::element_cast<T>(acc));      // HF: bf16 logits, then fp32 softmax
    auto m = ct::reduce_max(logits, 1_ic);
    auto e = ct::exp(logits - m);
    auto p = e / ct::sum(e, 1_ic);

    // top-K by repeated arg-max (ties -> the lower expert id); p of a chosen expert is then set to -1
    auto col = ct::element_cast<float>(ct::broadcast<ct::shape<BR, E>>(ct::iota<ct::tile<int, ct::shape<1, E>>>()));
    auto kcol = ct::broadcast<ct::shape<BR, K>>(ct::iota<ct::tile<int, ct::shape<1, K>>>());
    auto wsel = ct::zeros<f32_tk>();
    auto isel = ct::zeros<i32_tk>();
    for (int j = 0; j < K; ++j) {
        auto mx = ct::reduce_max(p, 1_ic);                                          // [BR, 1]
        auto rev = ct::select(p == mx, (float)(E - 1) - col, ct::full<f32_te>(-1.0f));
        auto best = (float)(E - 1) - ct::reduce_max(rev, 1_ic);                     // lowest id among the maxima
        wsel = ct::select(kcol == j, ct::broadcast<ct::shape<BR, K>>(mx), wsel);
        isel = ct::select(kcol == j, ct::broadcast<ct::shape<BR, K>>(ct::element_cast<int>(best)), isel);
        p = ct::select(col == best, ct::full<f32_te>(-1.0f), p);
    }
    if constexpr (NORM) wsel = wsel / ct::sum(wsel, 1_ic);
    pI.store_masked(isel, tb, 0);
    pO.store_masked(ct::element_cast<T>(wsel), tb, 0);
}

// ---- 2. count / 3. stable sort by expert -------------------------------------------------------------------------
constexpr int MOE_BA = 128;   // assignments per tile in count / sort

template <int E>
__tile_global__ void moe_count_kernel(const int* __restrict__ IDX, int n_assign, int* __restrict__ COUNT) {
    using namespace ct::literals;
    using i32_a = ct::tile<int, ct::shape<MOE_BA>>;
    int e = static_cast<int>(ct::bid().x);
    auto pA = ct::partition_view{ct::tensor_span{IDX, ct::extents{n_assign}}, ct::shape<MOE_BA>{}};
    auto total = ct::zeros<ct::tile<int, ct::shape<1>>>();
    for (auto ab : ct::irange(0, (n_assign + MOE_BA - 1) / MOE_BA)) {
        auto pos = ab * MOE_BA + ct::iota<i32_a>();
        auto ids = ct::select(pos < n_assign, pA.load_masked(ab), ct::full<i32_a>(-1));
        total = total + ct::sum(ct::select(ids == e, ct::full<i32_a>(1), ct::zeros<i32_a>()), 0_ic);
    }
    auto pC = ct::partition_view{ct::tensor_span{COUNT, ct::extents{ct::integral_constant<E>{}}}, ct::shape<1>{}};
    pC.store(total, e);
}

template <typename T, int E>
__tile_global__ void moe_sort_kernel(const int* __restrict__ IDX, int n_assign, const int* __restrict__ COUNT,
                                     int* __restrict__ SORTED, int* __restrict__ INV, int* __restrict__ OFF) {
    using namespace ct::literals;
    using i32_a = ct::tile<int, ct::shape<MOE_BA>>;
    using i32_aa = ct::tile<int, ct::shape<MOE_BA, MOE_BA>>;
    using t_a16 = ct::tile<T, ct::shape<MOE_BA, 16>>;
    int e = static_cast<int>(ct::bid().x);
    // OFF[e] = sum of the counts of the experts before e
    auto pCnt = ct::partition_view{ct::tensor_span{COUNT, ct::extents{ct::integral_constant<E>{}}}, ct::shape<E>{}};
    auto cnt = pCnt.load(0);
    auto eid = ct::iota<ct::tile<int, ct::shape<E>>>();
    auto off = ct::sum(ct::select(eid < e, cnt, ct::zeros<ct::tile<int, ct::shape<E>>>()), 0_ic);   // [1]
    auto pOff = ct::partition_view{ct::tensor_span{OFF, ct::extents{ct::integral_constant<E>{}}}, ct::shape<1>{}};
    pOff.store(off, e);

    // lower-triangular ones (strict): pre[r] = number of matches at positions < r inside the tile
    auto rr = ct::broadcast<ct::shape<MOE_BA, MOE_BA>>(ct::reshape<ct::shape<MOE_BA, 1>>(ct::iota<i32_a>()));
    auto cc = ct::broadcast<ct::shape<MOE_BA, MOE_BA>>(ct::reshape<ct::shape<1, MOE_BA>>(ct::iota<i32_a>()));
    auto ltri = ct::element_cast<T>(ct::select(cc < rr, ct::full<i32_aa>(1), ct::zeros<i32_aa>()));

    auto pA = ct::partition_view{ct::tensor_span{IDX, ct::extents{n_assign}}, ct::shape<MOE_BA>{}};
    auto gS = ct::gather_scatter_view{ct::tensor_span{SORTED, ct::extents{n_assign}}, ct::shape<MOE_BA>{}, ct::integral_constant<0>{}};
    auto gI = ct::gather_scatter_view{ct::tensor_span{INV, ct::extents{n_assign}}, ct::shape<MOE_BA>{}, ct::integral_constant<0>{}};
    auto running = ct::broadcast<ct::shape<MOE_BA>>(off);      // next free position of this expert
    for (auto ab : ct::irange(0, (n_assign + MOE_BA - 1) / MOE_BA)) {
        auto pos = ab * MOE_BA + ct::iota<i32_a>();
        auto ids = ct::select(pos < n_assign, pA.load_masked(ab), ct::full<i32_a>(-1));
        auto hit = ct::select(ids == e, ct::full<i32_a>(1), ct::zeros<i32_a>());
        auto hit16 = ct::element_cast<T>(ct::broadcast<ct::shape<MOE_BA, 16>>(ct::reshape<ct::shape<MOE_BA, 1>>(hit)));
        auto pre = ct::mma(ltri, hit16, ct::zeros<ct::tile<float, ct::shape<MOE_BA, 16>>>());   // every column the same
        auto pre1 = ct::element_cast<int>(ct::reshape<ct::shape<MOE_BA>>(ct::sum(pre, 1_ic) / 16.0f));
        auto dst = ct::select(hit == 1, running + pre1, ct::full<i32_a>(n_assign));   // n_assign = masked out
        gS.store_masked(pos, dst);                                                    // SORTED[dst] = assignment id
        gI.store_masked(dst, ct::select(hit == 1, pos, ct::full<i32_a>(n_assign)));   // INV[assignment] = dst
        running = running + ct::broadcast<ct::shape<MOE_BA>>(ct::sum(hit, 0_ic));
    }
}

// ---- 4. grouped gate/up + SiLU, 5. grouped down --------------------------------------------------------------------
template <typename T, int K, int BM, int BN, int BK>
__tile_global__ void moe_gate_up_kernel(const T* __restrict__ _X, int n_tokens, const T* __restrict__ WG,
                                        const T* __restrict__ WU, const int* __restrict__ COUNT,
                                        const int* __restrict__ OFF, const int* __restrict__ SORTED,
                                        T* __restrict__ HBUF, int _hid, int _inter) {
    using namespace ct::literals;
    using acc_t = ct::tile<float, ct::shape<BM, BN>>;
    using i32_m = ct::tile<int, ct::shape<BM>>;
    auto hid = ct::assume_divisible(_hid, 16_ic);
    auto inter = ct::assume_divisible(_inter, 16_ic);
    auto [e_u, mb_u, nb_u] = ct::bid();
    int e = static_cast<int>(e_u), mb = static_cast<int>(mb_u), nb = static_cast<int>(nb_u);
    int off = OFF[e], cnt = COUNT[e];
    int k_all = (hid + BK - 1) / BK;
    int k_blocks = cnt - mb * BM > 0 ? k_all : 0;               // a tile past the expert's rows does no work

    auto pSort = ct::partition_view{ct::tensor_span{SORTED + off, ct::extents{cnt}}, ct::shape<BM>{}};
    auto r = mb * BM + ct::iota<i32_m>();
    auto tok = ct::select(r < cnt, pSort.load_masked(mb) / K, ct::full<i32_m>(n_tokens));   // token of each row
    auto gX = ct::gather_scatter_view{ct::tensor_span{_X, ct::extents{n_tokens, hid}}, ct::shape<BM, BK>{}, ct::integral_constant<0>{}};
    const size_t wofs = (size_t)e * _inter * _hid;
    auto pWG = ct::partition_view{ct::tensor_span{WG + wofs, ct::extents{inter, hid}}, ct::shape<BN, BK>{}};
    auto pWU = ct::partition_view{ct::tensor_span{WU + wofs, ct::extents{inter, hid}}, ct::shape<BN, BK>{}};

    auto acc_g = ct::zeros<acc_t>();
    auto acc_u = ct::zeros<acc_t>();
    for (auto kb : ct::irange(0, k_blocks)) {
        ct::tile<T, ct::shape<BM, BK>> x = gX.load_masked(tok, kb);
        ct::tile<T, ct::shape<BN, BK>> wg = pWG.load(nb, kb);
        ct::tile<T, ct::shape<BN, BK>> wu = pWU.load(nb, kb);
        acc_g = ct::mma(x, ct::transpose(wg), acc_g);
        acc_u = ct::mma(x, ct::transpose(wu), acc_u);
    }
    auto g = ct::element_cast<float>(ct::element_cast<T>(acc_g));    // same roundings as gate_up_silu_kernel
    auto u = ct::element_cast<float>(ct::element_cast<T>(acc_u));
    auto s = ct::element_cast<float>(ct::element_cast<T>(g / (ct::full<acc_t>(1.0f) + ct::exp(-g))));
    auto pH = ct::partition_view{ct::tensor_span{HBUF + (size_t)off * _inter, ct::extents{cnt, inter}}, ct::shape<BM, BN>{}};
    pH.store_masked(ct::element_cast<T>(s * u), mb, nb);
}

template <typename T, int BM, int BN, int BK>
__tile_global__ void moe_down_kernel(const T* __restrict__ HBUF, const T* __restrict__ WD, const int* __restrict__ COUNT,
                                     const int* __restrict__ OFF, T* __restrict__ YS, int _hid, int _inter) {
    using namespace ct::literals;
    using acc_t = ct::tile<float, ct::shape<BM, BN>>;
    auto hid = ct::assume_divisible(_hid, 16_ic);
    auto inter = ct::assume_divisible(_inter, 16_ic);
    auto [e_u, mb_u, nb_u] = ct::bid();
    int e = static_cast<int>(e_u), mb = static_cast<int>(mb_u), nb = static_cast<int>(nb_u);
    int off = OFF[e], cnt = COUNT[e];
    int k_all = (inter + BK - 1) / BK;
    int k_blocks = cnt - mb * BM > 0 ? k_all : 0;
    auto pH = ct::partition_view{ct::tensor_span{HBUF + (size_t)off * _inter, ct::extents{cnt, inter}}, ct::shape<BM, BK>{}};
    auto pW = ct::partition_view{ct::tensor_span{WD + (size_t)e * _hid * _inter, ct::extents{hid, inter}}, ct::shape<BN, BK>{}};
    auto acc = ct::zeros<acc_t>();
    for (auto kb : ct::irange(0, k_blocks)) {
        ct::tile<T, ct::shape<BM, BK>> h = pH.load_masked(mb, kb);
        ct::tile<T, ct::shape<BN, BK>> w = pW.load(nb, kb);
        acc = ct::mma(h, ct::transpose(w), acc);
    }
    auto pY = ct::partition_view{ct::tensor_span{YS + (size_t)off * _hid, ct::extents{cnt, hid}}, ct::shape<BM, BN>{}};
    pY.store_masked(ct::element_cast<T>(acc), mb, nb);
}

// ---- 6. weighted combine -------------------------------------------------------------------------------------------
template <typename T, int K, int BR, int BN>
__tile_global__ void moe_combine_kernel(const T* __restrict__ YS, int n_assign, const int* __restrict__ INV,
                                        const T* __restrict__ W, T* __restrict__ Y, int n_tokens, int _hid) {
    using namespace ct::literals;
    using acc_t = ct::tile<float, ct::shape<BR, BN>>;
    auto hid = ct::assume_divisible(_hid, 16_ic);
    auto [tb_u, nb_u, dummy] = ct::bid();
    int tb = static_cast<int>(tb_u), nb = static_cast<int>(nb_u);
    auto pInv = ct::partition_view{ct::tensor_span{INV, ct::extents{n_tokens, ct::integral_constant<K>{}}}, ct::shape<BR, 1>{}};
    auto pW = ct::partition_view{ct::tensor_span{W, ct::extents{n_tokens, ct::integral_constant<K>{}}}, ct::shape<BR, 1>{}};
    auto gY = ct::gather_scatter_view{ct::tensor_span{YS, ct::extents{n_assign, hid}}, ct::shape<BR, BN>{}, ct::integral_constant<0>{}};
    auto t = tb * BR + ct::iota<ct::tile<int, ct::shape<BR>>>();
    auto acc = ct::zeros<acc_t>();
    for (int j = 0; j < K; ++j) {
        auto row = ct::select(t < n_tokens, ct::reshape<ct::shape<BR>>(pInv.load_masked(tb, j)),
                              ct::full<ct::tile<int, ct::shape<BR>>>(n_assign));
        auto w = ct::element_cast<float>(pW.load_masked(tb, j));                    // [BR, 1]
        acc = acc + ct::element_cast<float>(gY.load_masked(row, nb)) * w;
    }
    auto pY = ct::partition_view{ct::tensor_span{Y, ct::extents{n_tokens, hid}}, ct::shape<BR, BN>{}};
    pY.store_masked(ct::element_cast<T>(acc), tb, nb);
}

// ---- host side ------------------------------------------------------------------------------------------------------
// Router only (also used by the unit test): IDX [n, K], W [n, K].
template <typename T, int E, int K>
cudaError_t launch_moe_router(const T* x, const T* wr, int* idx, T* w, int n, int hid, bool norm_topk, cudaStream_t s = 0) {
    constexpr int BR = 16, BK = 64;
    if (hid % BK) { std::fprintf(stderr, "moe router: hidden %d not a multiple of %d\n", hid, BK); return cudaErrorInvalidValue; }
    if (norm_topk) moe_router_kernel<T, E, K, true, BR, BK><<<(n + BR - 1) / BR, 1, 0, s>>>(x, wr, idx, w, n, hid);
    else moe_router_kernel<T, E, K, false, BR, BK><<<(n + BR - 1) / BR, 1, 0, s>>>(x, wr, idx, w, n, hid);
    return cudaGetLastError();
}

// The whole block: y [n, hid] = MoE(x). wg, wu [E, inter, hid], wd [E, hid, inter] packed per layer.
template <typename T, int E, int K>
cudaError_t launch_moe(const T* x, const T* wr, const T* wg, const T* wu, const T* wd, T* y, int n, int hid, int inter,
                       bool norm_topk, const MoeWorkspace<T>& ws, cudaStream_t s = 0) {
    if (n > ws.max_tokens) { std::fprintf(stderr, "moe: %d tokens > workspace %d\n", n, ws.max_tokens); return cudaErrorInvalidValue; }
    if (inter % 64 || hid % 128) { std::fprintf(stderr, "moe: inter %% 64 and hid %% 128 required\n"); return cudaErrorInvalidValue; }
    const int A = n * K;
    cudaError_t e = launch_moe_router<T, E, K>(x, wr, ws.idx, ws.w, n, hid, norm_topk, s);
    if (e != cudaSuccess) return e;
    moe_count_kernel<E><<<E, 1, 0, s>>>(ws.idx, A, ws.count);
    moe_sort_kernel<T, E><<<E, 1, 0, s>>>(ws.idx, A, ws.count, ws.sorted, ws.inv, ws.off);
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
    // rows per expert: at most n (a token picks an expert once). Small batches: small row tiles.
    auto grouped = [&](auto bm) {
        constexpr int BM = decltype(bm)::value;
        const int mt = (n + BM - 1) / BM;
        moe_gate_up_kernel<T, K, BM, 64, 64><<<dim3(E, mt, inter / 64), 1, 0, s>>>(x, n, wg, wu, ws.count, ws.off,
                                                                                   ws.sorted, ws.hbuf, hid, inter);
        moe_down_kernel<T, BM, 128, 64><<<dim3(E, mt, hid / 128), 1, 0, s>>>(ws.hbuf, wd, ws.count, ws.off, ws.ysorted,
                                                                             hid, inter);
    };
    if ((long long)A / E > 32) grouped(std::integral_constant<int, 64>{});
    else grouped(std::integral_constant<int, 16>{});
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
    constexpr int BR = 16, BN = 256;
    moe_combine_kernel<T, K, BR, BN><<<dim3((n + BR - 1) / BR, hid / BN, 1), 1, 0, s>>>(ws.ysorted, A, ws.inv, ws.w, y, n, hid);
    return cudaGetLastError();
}
