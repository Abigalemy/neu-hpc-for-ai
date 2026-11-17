// Causal GQA attention over a KV cache as one fused FlashAttention-2-style CUDA Tile kernel (online softmax).
//
//   O[i, h, :] = softmax_j( Q[i, h, :] . K[j, g(h), :] / sqrt(128) , j <= pos(i) ) @ V[:, g(h), :]
//   g(h) = floor(h / (Hq / Hkv))        pos(i) = n_kv - n_q + i      (the n_q queries are the LAST n_q positions)
//
// Q is [n_q, HQ, 128]; K and V are the cache [n_kv, HKV, 128] (token-major, as written by the KV-cache step);
// O is [n_q, HQ, 128]. Prefill is n_q == n_kv; decode is n_q == 1. One block per (query tile, query head).
// Notation follows the Week 3 notebook / FlashAttention-2: m = running row max, l = running row sum,
// acc = unnormalised output. Scores and the softmax statistics are fp32; P@V accumulates in fp32.
#pragma once
#include <cfloat>
#include <cmath>
#include <cstdio>
#include "cuda_tile.h"
#include "qwen3_linear_tile.cuh"   // tile_profile()

namespace ct = cuda::tiles;

constexpr int FA_D = 128;   // Qwen3 head_dim
// BR = query rows per block, BC = keys per tile. PV_MODE selects how P (= exp(s - m)) enters the second matmul:
//   0: fp32 P, fp32 mma                     (most accurate, very slow: no tensor-core fast path)
//   1: P cast to the activation dtype       (what FlashAttention-2 does; one bf16 rounding of P)
//   2: P split into two bf16 terms, P_hi + P_lo, two bf16 mmas accumulated into the same fp32 accumulator
//      (P keeps ~16 significant bits, so the P rounding error of mode 1 disappears; second matmul costs 2x)

template <typename T, int HQ, int HKV, int FA_BR, int FA_BC, int PV_MODE>
__tile_global__ void flash_attention_kernel(const T* __restrict__ _Q, const T* __restrict__ _K,
                                            const T* __restrict__ _V, T* __restrict__ _O,
                                            int n_q, int n_kv, float scale) {
    static_assert(HQ % HKV == 0, "HQ must be a multiple of HKV");
    using namespace ct::literals;
    using f32_scores = ct::tile<float, ct::shape<FA_BR, FA_BC>>;
    using f32_rows = ct::tile<float, ct::shape<FA_BR, 1>>;
    using f32_out = ct::tile<float, ct::shape<FA_BR, FA_D>>;
    using i32_rows = ct::tile<int, ct::shape<FA_BR, 1>>;
    using i32_keys = ct::tile<int, ct::shape<1, FA_BC>>;

    const T* Q = ct::assume_aligned(_Q, 16_ic);
    const T* K = ct::assume_aligned(_K, 16_ic);
    const T* V = ct::assume_aligned(_V, 16_ic);
    T* O = ct::assume_aligned(_O, 16_ic);

    auto pQ = ct::partition_view{ct::tensor_span{Q, ct::extents{n_q, ct::integral_constant<HQ>{}, ct::integral_constant<FA_D>{}}},
                                 ct::shape<FA_BR, 1, FA_D>{}};
    auto pK = ct::partition_view{ct::tensor_span{K, ct::extents{n_kv, ct::integral_constant<HKV>{}, ct::integral_constant<FA_D>{}}},
                                 ct::shape<FA_BC, 1, FA_D>{}};
    auto pV = ct::partition_view{ct::tensor_span{V, ct::extents{n_kv, ct::integral_constant<HKV>{}, ct::integral_constant<FA_D>{}}},
                                 ct::shape<FA_BC, 1, FA_D>{}};
    auto pO = ct::partition_view{ct::tensor_span{O, ct::extents{n_q, ct::integral_constant<HQ>{}, ct::integral_constant<FA_D>{}}},
                                 ct::shape<FA_BR, 1, FA_D>{}};

    auto [qb_u, h_u, dummy] = ct::bid();   // unsigned; the tile arithmetic below needs signed ints
    int qb = static_cast<int>(qb_u);
    int h = static_cast<int>(h_u);
    int kvh = h / (HQ / HKV);

    ct::tile<T, ct::shape<FA_BR, FA_D>> q = ct::reshape<ct::shape<FA_BR, FA_D>>(pQ.load_masked(qb, h, 0));

    // absolute position of each query row of this block
    i32_rows qpos = (n_kv - n_q) + qb * FA_BR + ct::iota<i32_rows>();

    auto m = ct::full<f32_rows>(-FLT_MAX);
    auto l = ct::zeros<f32_rows>();
    auto acc = ct::zeros<f32_out>();

    // keys beyond the last query position of this block are fully masked: skip those tiles
    int last_q = (n_kv - n_q) + (qb + 1) * FA_BR - 1;
    if (last_q > n_kv - 1) last_q = n_kv - 1;
    int n_tiles = last_q / FA_BC + 1;

    for (auto kb : ct::irange(0, n_tiles)) {
        ct::tile<T, ct::shape<FA_BC, FA_D>> k = ct::reshape<ct::shape<FA_BC, FA_D>>(pK.load_masked(kb, kvh, 0));
        ct::tile<T, ct::shape<FA_BC, FA_D>> v = ct::reshape<ct::shape<FA_BC, FA_D>>(pV.load_masked(kb, kvh, 0));

        auto s = ct::mma(q, ct::transpose(k), ct::zeros<f32_scores>()) * scale;

        // causal mask: key j is visible to query i iff j <= pos(i). (Every real query row sees key 0, so a row is
        // never fully masked in its first tile; later fully masked tiles contribute exp(-FLT_MAX - m) = 0.)
        i32_keys kidx = kb * FA_BC + ct::iota<i32_keys>();
        s = ct::select(kidx <= qpos, s, ct::full<f32_scores>(-FLT_MAX));

        auto block_m = ct::reduce_max(s, 1_ic);
        auto new_m = ct::max(m, block_m);
        auto alpha = ct::exp(m - new_m);
        auto p = ct::exp(s - new_m);
        l = l * alpha + ct::sum(p, 1_ic);
        if constexpr (PV_MODE == 1) {
            acc = acc * alpha + ct::mma(ct::element_cast<T>(p), v, ct::zeros<f32_out>());
        } else if constexpr (PV_MODE == 2) {
            auto p_hi = ct::element_cast<T>(p);
            auto p_lo = ct::element_cast<T>(p - ct::element_cast<float>(p_hi));
            acc = acc * alpha;
            acc = ct::mma(p_hi, v, acc);
            acc = ct::mma(p_lo, v, acc);
        } else {
            acc = acc * alpha + ct::mma(p, ct::element_cast<float>(v), ct::zeros<f32_out>());
        }
        m = new_m;
    }

    auto out = acc / l;
    pO.store_masked(ct::reshape<ct::shape<FA_BR, 1, FA_D>>(ct::element_cast<T>(out)), qb, h, 0);
}

// Host launchers.
template <typename T, int HQ, int HKV, int BR, int BC, int PV_MODE>
cudaError_t launch_flash_attention_tiled(const T* q, const T* k, const T* v, T* o, int n_q, int n_kv,
                                         cudaStream_t stream = 0) {
    float scale = 1.0f / std::sqrt(static_cast<float>(FA_D));
    int gq = (n_q + BR - 1) / BR;
    flash_attention_kernel<T, HQ, HKV, BR, BC, PV_MODE><<<dim3(gq, HQ, 1), 1, 0, stream>>>(q, k, v, o, n_q, n_kv, scale);
    return cudaGetLastError();
}

// Tile choice (measured on an H100, tests/attn_sweep_test.cu, Hq 32 / Hkv 8, 3072 keys):
//   16x16, fp32 P            : prefill 6.92 ms (11 TFLOP/s), decode 285 us per layer
//   64x128, bf16 P (mode 1)  : prefill 0.299 ms, decode 44.6 us, max diff to fp32-P 4.6e-3 of the largest output
//   64x128, split P (mode 2) : prefill 0.354 ms (219 TFLOP/s), decode 54.2 us, max diff to fp32-P 2.3e-3 / 1.1e-3
// Keeping P in fp32 for the second matmul is what makes everything slow (fp32 mma has no tensor-core fast path).
// Rounding P to bf16 (mode 1) lowered the end-to-end decode cosine vs HF below the 0.999 criterion on Qwen3-4B
// (0.998993 vs 0.999399 with fp32 P, same HF reference); the split (mode 2) keeps P's precision for ~20% more time.
// Devices without Hopper-sized shared memory keep the verified 16x16 / fp32-P configuration.
template <typename T, int HQ, int HKV>
cudaError_t launch_flash_attention(const T* q, const T* k, const T* v, T* o, int n_q, int n_kv,
                                   cudaStream_t stream = 0) {
    static const bool force_small = [] { const char* e = std::getenv("QWEN3_ATTN"); return e && std::string(e) == "small"; }();
    if (tile_profile() == TileProfile::BIG && !force_small)   // QWEN3_ATTN=small: A/B switch for the attention only
        return launch_flash_attention_tiled<T, HQ, HKV, 64, 128, 2>(q, k, v, o, n_q, n_kv, stream);
    return launch_flash_attention_tiled<T, HQ, HKV, 16, 16, 0>(q, k, v, o, n_q, n_kv, stream);
}

// ---- decode (n_q == 1): split the KV length across blocks, then combine (flash-decoding) ---------------------------
//
// The kernel above gives a single query only HQ blocks (32 on Qwen3-4B), each walking every key tile of the cache one
// after another: at 128K that is ~74 ms per decode step for 19 GB of cache (~261 GB/s). Here block (h, s) handles query
// head h and the key tiles [s*tps, (s+1)*tps); it writes its partial softmax state (m, l, unnormalised output) to an
// fp32 workspace, and a second kernel merges the splits of each head:
//   m* = max_s m_s,  w_s = exp(m_s - m*),  O = sum_s w_s acc_s / sum_s w_s l_s.
// blockIdx.x is the head, so the HQ/HKV query heads that share a KV head read the same split at the same time (L2).
// Only rows of a 16-row tile are used for the single query (mma needs 16 rows); rows 1..15 are padding.
constexpr int FD_BR = 16;      // query rows of the tile (1 used)
constexpr int FD_BC = 128;     // keys per tile
constexpr int FD_SMAX = 64;    // max number of splits (power of two: the combine loads them as one tile)

struct DecodeWorkspace {        // fp32 partial results; sizes: o = HQ*FD_SMAX*FD_BR*FA_D, m = l = HQ*FD_SMAX*FD_BR
    float* o = nullptr;
    float* m = nullptr;
    float* l = nullptr;
};

template <typename T, int HQ, int HKV, int PV_MODE>
__tile_global__ void flash_decode_partial_kernel(const T* __restrict__ _Q, const T* __restrict__ _K,
                                                 const T* __restrict__ _V, float* __restrict__ _WO,
                                                 float* __restrict__ _WM, float* __restrict__ _WL,
                                                 int n_kv, int tiles_per_split, int n_splits, float scale) {
    using namespace ct::literals;
    using f32_scores = ct::tile<float, ct::shape<FD_BR, FD_BC>>;
    using f32_rows = ct::tile<float, ct::shape<FD_BR, 1>>;
    using f32_out = ct::tile<float, ct::shape<FD_BR, FA_D>>;
    using i32_rows = ct::tile<int, ct::shape<FD_BR, 1>>;
    using i32_keys = ct::tile<int, ct::shape<1, FD_BC>>;

    const T* Q = ct::assume_aligned(_Q, 16_ic);
    const T* K = ct::assume_aligned(_K, 16_ic);
    const T* V = ct::assume_aligned(_V, 16_ic);

    auto pQ = ct::partition_view{ct::tensor_span{Q, ct::extents{1, ct::integral_constant<HQ>{}, ct::integral_constant<FA_D>{}}},
                                 ct::shape<FD_BR, 1, FA_D>{}};
    auto pK = ct::partition_view{ct::tensor_span{K, ct::extents{n_kv, ct::integral_constant<HKV>{}, ct::integral_constant<FA_D>{}}},
                                 ct::shape<FD_BC, 1, FA_D>{}};
    auto pV = ct::partition_view{ct::tensor_span{V, ct::extents{n_kv, ct::integral_constant<HKV>{}, ct::integral_constant<FA_D>{}}},
                                 ct::shape<FD_BC, 1, FA_D>{}};
    auto pWO = ct::partition_view{ct::tensor_span{_WO, ct::extents{ct::integral_constant<HQ>{}, n_splits, ct::integral_constant<FD_BR>{}, ct::integral_constant<FA_D>{}}},
                                  ct::shape<1, 1, FD_BR, FA_D>{}};
    auto pWM = ct::partition_view{ct::tensor_span{_WM, ct::extents{ct::integral_constant<HQ>{}, n_splits, ct::integral_constant<FD_BR>{}}},
                                  ct::shape<1, 1, FD_BR>{}};
    auto pWL = ct::partition_view{ct::tensor_span{_WL, ct::extents{ct::integral_constant<HQ>{}, n_splits, ct::integral_constant<FD_BR>{}}},
                                  ct::shape<1, 1, FD_BR>{}};

    auto [h_u, s_u, dummy] = ct::bid();
    int h = static_cast<int>(h_u);
    int split = static_cast<int>(s_u);
    int kvh = h / (HQ / HKV);

    ct::tile<T, ct::shape<FD_BR, FA_D>> q = ct::reshape<ct::shape<FD_BR, FA_D>>(pQ.load_masked(0, h, 0));
    i32_rows qpos = (n_kv - 1) + ct::iota<i32_rows>();   // row 0 is the real query (position n_kv - 1)

    auto m = ct::full<f32_rows>(-FLT_MAX);
    auto l = ct::zeros<f32_rows>();
    auto acc = ct::zeros<f32_out>();

    int n_tiles = (n_kv + FD_BC - 1) / FD_BC;
    int t0 = split * tiles_per_split;
    int t1 = t0 + tiles_per_split < n_tiles ? t0 + tiles_per_split : n_tiles;
    for (auto kb : ct::irange(t0, t1)) {
        ct::tile<T, ct::shape<FD_BC, FA_D>> k = ct::reshape<ct::shape<FD_BC, FA_D>>(pK.load_masked(kb, kvh, 0));
        ct::tile<T, ct::shape<FD_BC, FA_D>> v = ct::reshape<ct::shape<FD_BC, FA_D>>(pV.load_masked(kb, kvh, 0));
        auto sc = ct::mma(q, ct::transpose(k), ct::zeros<f32_scores>()) * scale;
        i32_keys kidx = kb * FD_BC + ct::iota<i32_keys>();
        // keys past the end of the cache (padding of the last tile) are masked; every split contains real keys
        sc = ct::select(kidx <= qpos, sc, ct::full<f32_scores>(-FLT_MAX));
        auto block_m = ct::reduce_max(sc, 1_ic);
        auto new_m = ct::max(m, block_m);
        auto alpha = ct::exp(m - new_m);
        auto p = ct::exp(sc - new_m);
        l = l * alpha + ct::sum(p, 1_ic);
        if constexpr (PV_MODE == 2) {
            auto p_hi = ct::element_cast<T>(p);
            auto p_lo = ct::element_cast<T>(p - ct::element_cast<float>(p_hi));
            acc = acc * alpha;
            acc = ct::mma(p_hi, v, acc);
            acc = ct::mma(p_lo, v, acc);
        } else if constexpr (PV_MODE == 1) {
            acc = acc * alpha + ct::mma(ct::element_cast<T>(p), v, ct::zeros<f32_out>());
        } else {
            acc = acc * alpha + ct::mma(p, ct::element_cast<float>(v), ct::zeros<f32_out>());
        }
        m = new_m;
    }
    pWO.store(ct::reshape<ct::shape<1, 1, FD_BR, FA_D>>(acc), h, split, 0, 0);
    pWM.store(ct::reshape<ct::shape<1, 1, FD_BR>>(m), h, split, 0);
    pWL.store(ct::reshape<ct::shape<1, 1, FD_BR>>(l), h, split, 0);
}

template <typename T, int HQ>
__tile_global__ void flash_decode_combine_kernel(const float* __restrict__ _WO, const float* __restrict__ _WM,
                                                 const float* __restrict__ _WL, T* __restrict__ _O, int n_splits) {
    using namespace ct::literals;
    using f32_s = ct::tile<float, ct::shape<FD_SMAX, 1>>;
    using f32_so = ct::tile<float, ct::shape<FD_SMAX, FA_D>>;
    using i32_s = ct::tile<int, ct::shape<FD_SMAX, 1>>;

    auto pWO = ct::partition_view{ct::tensor_span{_WO, ct::extents{ct::integral_constant<HQ>{}, n_splits, ct::integral_constant<FD_BR>{}, ct::integral_constant<FA_D>{}}},
                                  ct::shape<1, FD_SMAX, 1, FA_D>{}};
    auto pWM = ct::partition_view{ct::tensor_span{_WM, ct::extents{ct::integral_constant<HQ>{}, n_splits, ct::integral_constant<FD_BR>{}}},
                                  ct::shape<1, FD_SMAX, 1>{}};
    auto pWL = ct::partition_view{ct::tensor_span{_WL, ct::extents{ct::integral_constant<HQ>{}, n_splits, ct::integral_constant<FD_BR>{}}},
                                  ct::shape<1, FD_SMAX, 1>{}};
    auto pO = ct::partition_view{ct::tensor_span{_O, ct::extents{1, ct::integral_constant<HQ>{}, ct::integral_constant<FA_D>{}}},
                                 ct::shape<1, 1, FA_D>{}};

    int h = static_cast<int>(ct::bid().x);
    i32_s sidx = ct::iota<i32_s>();
    auto valid = sidx < n_splits;                                   // splits past n_splits are padding
    f32_so o_s = ct::reshape<ct::shape<FD_SMAX, FA_D>>(pWO.load_masked(h, 0, 0, 0));   // row 0 of every split
    f32_s m_s = ct::reshape<ct::shape<FD_SMAX, 1>>(pWM.load_masked(h, 0, 0));
    f32_s l_s = ct::reshape<ct::shape<FD_SMAX, 1>>(pWL.load_masked(h, 0, 0));
    o_s = ct::select(valid, o_s, ct::zeros<f32_so>());
    l_s = ct::select(valid, l_s, ct::zeros<f32_s>());
    m_s = ct::select(valid, m_s, ct::full<f32_s>(-FLT_MAX));
    auto mstar = ct::reduce_max(m_s, 0_ic);
    auto w = ct::select(valid, ct::exp(m_s - mstar), ct::zeros<f32_s>());
    auto L = ct::sum(w * l_s, 0_ic);
    auto o = ct::sum(o_s * w, 0_ic) / L;
    pO.store(ct::reshape<ct::shape<1, 1, FA_D>>(ct::element_cast<T>(o)), 0, h, 0);
}

template <typename T, int HQ, int HKV>
cudaError_t launch_flash_decode(const T* q, const T* k, const T* v, T* o, int n_kv, const DecodeWorkspace& ws,
                                cudaStream_t stream = 0) {
    float scale = 1.0f / std::sqrt(static_cast<float>(FA_D));
    int n_tiles = (n_kv + FD_BC - 1) / FD_BC;
    int want = (512 + HQ - 1) / HQ;                                  // aim at ~512 blocks
    int n_splits = n_tiles < want ? n_tiles : want;
    if (n_splits > FD_SMAX) n_splits = FD_SMAX;
    int tps = (n_tiles + n_splits - 1) / n_splits;
    n_splits = (n_tiles + tps - 1) / tps;                           // no empty split
    flash_decode_partial_kernel<T, HQ, HKV, 2><<<dim3(HQ, n_splits, 1), 1, 0, stream>>>(
        q, k, v, ws.o, ws.m, ws.l, n_kv, tps, n_splits, scale);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    flash_decode_combine_kernel<T, HQ><<<HQ, 1, 0, stream>>>(ws.o, ws.m, ws.l, o, n_splits);
    return cudaGetLastError();
}

// Entry point used by the engine: split-KV decode on Hopper when a workspace is given, the kernels above otherwise.
template <typename T, int HQ, int HKV>
cudaError_t launch_attention_auto(const T* q, const T* k, const T* v, T* o, int n_q, int n_kv, const DecodeWorkspace* ws,
                                  cudaStream_t stream = 0) {
    static const bool force_small = [] { const char* e = std::getenv("QWEN3_ATTN"); return e && std::string(e) == "small"; }();
    static const bool no_split = [] { const char* e = std::getenv("QWEN3_DECODE_SPLIT"); return e && std::string(e) == "0"; }();
    if (n_q == 1 && ws && tile_profile() == TileProfile::BIG && !force_small && !no_split)
        return launch_flash_decode<T, HQ, HKV>(q, k, v, o, n_kv, *ws, stream);
    return launch_flash_attention<T, HQ, HKV>(q, k, v, o, n_q, n_kv, stream);
}
