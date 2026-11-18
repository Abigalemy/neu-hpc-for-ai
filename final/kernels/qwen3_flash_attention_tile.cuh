// Causal GQA attention over a KV cache as one fused FlashAttention-2-style CUDA Tile kernel (online softmax).
//
//   O[i, h, :] = softmax_j( Q[i, h, :] . K[j, g(h), :] / sqrt(128) , j <= pos(i) ) @ V[:, g(h), :]
//   g(h) = floor(h / (Hq / Hkv))        pos(i) = n_kv - n_q + i      (the n_q queries are the LAST n_q positions)
//
// Q is [n_q, HQ, 128]; K and V are the cache [n_kv, HKV, 128] (token-major, as written by the KV-cache step);
// O is [n_q, HQ, 128]. Prefill is n_q == n_kv; decode is n_q == 1. One block per (query tile, query head).
// Notation follows the Week 3 notebook / FlashAttention-2: m = running row max, l = running row sum,
// acc = unnormalised output. Scores and the softmax statistics are fp32; P@V accumulates in fp32.
//
// PagedAttention (Week 7 lecture; Kwon et al., SOSP 2023): the KV cache is a pool of fixed-size blocks of KV_BLOCK
// tokens, [n_blocks * KV_BLOCK, HKV, 128] per layer, and a sequence owns a block table BT (logical block i -> physical
// block BT[i]). Token position p of the sequence lives in pool row BT[p / KV_BLOCK] * KV_BLOCK + p % KV_BLOCK. The paged
// kernels read the blocks in place through the table (no copy into a contiguous buffer). KV_BLOCK equals the key tile
// of the decode kernel, so one key tile is exactly one physical block; the prefill key tile divides it.
// Rows of a block past the sequence's length hold stale K/V of an earlier owner. Their scores are masked to -FLT_MAX as
// before, so P = exp(-FLT_MAX - m) = 0 exactly and they add 0 * (stale V) = 0, provided the stale values are finite:
// the engine zeroes the pool once at start, and afterwards every row was written by a real token (finite K/V; a
// non-finite K/V would already be a wrong result). Zeroing those V rows with a select in the loop instead cost 1.9x in
// decode and 2.5x in prefill (tests/paged_bench_test.cu, H100, 2026-11-18), so it is not done. Reading the block
// table itself costs nothing measurable, even with shuffled blocks.
#pragma once
#include <cfloat>
#include <cmath>
#include <cstdio>
#include "cuda_tile.h"
#include "qwen3_linear_tile.cuh"   // tile_profile()

namespace ct = cuda::tiles;

constexpr int FA_D = 128;       // Qwen3 head_dim
constexpr int KV_BLOCK = 128;   // tokens per KV-cache block (PagedAttention block size)
// BR = query rows per block, BC = keys per tile. PV_MODE selects how P (= exp(s - m)) enters the second matmul:
//   0: fp32 P, fp32 mma                     (most accurate, very slow: no tensor-core fast path)
//   1: P cast to the activation dtype       (what FlashAttention-2 does; one bf16 rounding of P)
//   2: P split into two bf16 terms, P_hi + P_lo, two bf16 mmas accumulated into the same fp32 accumulator
//      (P keeps ~16 significant bits, so the P rounding error of mode 1 disappears; second matmul costs 2x)

// PAGED = false: K and V are this sequence's own contiguous cache of k_rows == n_kv rows.
// PAGED = true:  K and V are the whole block pool (k_rows rows) and BT is this sequence's block table.
template <typename T, int HQ, int HKV, int FA_BR, int FA_BC, int PV_MODE, bool PAGED>
__tile_global__ void flash_attention_kernel(const T* __restrict__ _Q, const T* __restrict__ _K,
                                            const T* __restrict__ _V, T* __restrict__ _O,
                                            int n_q, int n_kv, float scale, const int* __restrict__ BT, int k_rows) {
    static_assert(HQ % HKV == 0, "HQ must be a multiple of HKV");
    static_assert(KV_BLOCK % FA_BC == 0, "a key tile must not cross a KV-cache block");
    constexpr int TILES_PER_BLOCK = KV_BLOCK / FA_BC;
    using namespace ct::literals;
    using f32_scores = ct::tile<float, ct::shape<FA_BR, FA_BC>>;
    using f32_rows = ct::tile<float, ct::shape<FA_BR, 1>>;
    using f32_out = ct::tile<float, ct::shape<FA_BR, FA_D>>;
    using i32_rows = ct::tile<int, ct::shape<FA_BR, 1>>;
    using i32_keys = ct::tile<int, ct::shape<1, FA_BC>>;
    using kv_tile = ct::tile<T, ct::shape<FA_BC, FA_D>>;

    const T* Q = ct::assume_aligned(_Q, 16_ic);
    const T* K = ct::assume_aligned(_K, 16_ic);
    const T* V = ct::assume_aligned(_V, 16_ic);
    T* O = ct::assume_aligned(_O, 16_ic);

    auto pQ = ct::partition_view{ct::tensor_span{Q, ct::extents{n_q, ct::integral_constant<HQ>{}, ct::integral_constant<FA_D>{}}},
                                 ct::shape<FA_BR, 1, FA_D>{}};
    auto pK = ct::partition_view{ct::tensor_span{K, ct::extents{k_rows, ct::integral_constant<HKV>{}, ct::integral_constant<FA_D>{}}},
                                 ct::shape<FA_BC, 1, FA_D>{}};
    auto pV = ct::partition_view{ct::tensor_span{V, ct::extents{k_rows, ct::integral_constant<HKV>{}, ct::integral_constant<FA_D>{}}},
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
        int tile = kb;                                       // index of this key tile in the K / V views
        if constexpr (PAGED) tile = BT[kb / TILES_PER_BLOCK] * TILES_PER_BLOCK + kb % TILES_PER_BLOCK;
        kv_tile k = ct::reshape<ct::shape<FA_BC, FA_D>>(pK.load_masked(tile, kvh, 0));
        kv_tile v = ct::reshape<ct::shape<FA_BC, FA_D>>(pV.load_masked(tile, kvh, 0));

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

// Host launchers. Contiguous: k, v hold this sequence's n_kv rows. Paged (block_table != nullptr): k, v are the block
// pool of pool_rows rows and block_table is this sequence's table (device ints).
template <typename T, int HQ, int HKV, int BR, int BC, int PV_MODE>
cudaError_t launch_flash_attention_tiled(const T* q, const T* k, const T* v, T* o, int n_q, int n_kv,
                                         cudaStream_t stream = 0, const int* block_table = nullptr, int pool_rows = 0) {
    float scale = 1.0f / std::sqrt(static_cast<float>(FA_D));
    int gq = (n_q + BR - 1) / BR;
    if (block_table)
        flash_attention_kernel<T, HQ, HKV, BR, BC, PV_MODE, true><<<dim3(gq, HQ, 1), 1, 0, stream>>>(
            q, k, v, o, n_q, n_kv, scale, block_table, pool_rows);
    else
        flash_attention_kernel<T, HQ, HKV, BR, BC, PV_MODE, false><<<dim3(gq, HQ, 1), 1, 0, stream>>>(
            q, k, v, o, n_q, n_kv, scale, nullptr, n_kv);
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
                                   cudaStream_t stream = 0, const int* block_table = nullptr, int pool_rows = 0) {
    static const bool force_small = [] { const char* e = std::getenv("QWEN3_ATTN"); return e && std::string(e) == "small"; }();
    if (tile_profile() == TileProfile::BIG && !force_small)   // QWEN3_ATTN=small: A/B switch for the attention only
        return launch_flash_attention_tiled<T, HQ, HKV, 64, 128, 2>(q, k, v, o, n_q, n_kv, stream, block_table, pool_rows);
    return launch_flash_attention_tiled<T, HQ, HKV, 16, 16, 0>(q, k, v, o, n_q, n_kv, stream, block_table, pool_rows);
}

// ---- decode: split the KV length across blocks, then combine (flash-decoding) -----------------------------------------
//
// The kernel above gives a single query only HQ blocks (32 on Qwen3-4B), each walking every key tile of the cache one
// after another: at 128K that was ~74 ms per decode step for 19 GB of cache (~261 GB/s), and its decode accuracy was
// below the end-to-end criterion. Here block (h, s, b) handles query head h of sequence b and the key tiles
// [s*tps, (s+1)*tps); it writes its partial softmax state (m, l, unnormalised output) to an fp32 workspace, and a second
// kernel merges the splits of each (sequence, head):
//   m* = max_s m_s,  w_s = exp(m_s - m*),  O = sum_s w_s acc_s / sum_s w_s l_s.
// blockIdx.x is the head, so the HQ/HKV query heads that share a KV head read the same split at the same time (L2).
constexpr int FD_BR = 16;      // query rows of the tile (1 used)
constexpr int FD_BC = 128;     // keys per tile
constexpr int FD_SMAX = 64;    // max number of splits (power of two: the combine loads them as one tile)

// ---- batched decode over the paged KV cache (continuous batching) -------------------------------------------------
//
// B sequences, one new token each. Sequence b uses slot SLOTS[b], whose block table is row SLOTS[b] of BT
// ([n_slots, max_blocks] ints), and its query is at position POS[b], so it attends to keys 0..POS[b]: key tile t is
// physical block BT[slot][t] of the pool (FD_BC == KV_BLOCK). Grid: (head, split, sequence). Same split-KV
// scheme as above. Each sequence splits its OWN key tiles: s_b = min(split_target, n_tiles_b) splits of
// tps_b = ceil(n_tiles_b / s_b) tiles, so how a sequence is summed depends only on its length and split_target, not on
// which other sequences share the batch (with a fixed split_target, batched and alone decoding are bit-identical).
// Splits past a sequence's last tile run zero iterations and write an empty state (m = -FLT_MAX, l = 0), which the
// combine weights by exp(-huge) = 0.
// The single query row is broadcast to the 16 rows the mma needs.
struct BatchDecodeWorkspace {   // fp32 partial results for up to `max_parts` (sequence, split) pairs per head
    float* o = nullptr;         // [max_parts * HQ * FD_BR * FA_D]
    float* m = nullptr;         // [max_parts * HQ * FD_BR]
    float* l = nullptr;
    int max_parts = 0;
};

// KMODE: 1 = paged (the engine's path); 0 = no table, key tile kb of slot s is block s * max_blocks + kb (slots laid
// out contiguously, as before paging): kept only as the timing reference of tests/paged_bench_test.cu.
template <typename T, int HQ, int HKV, int PV_MODE, int KMODE = 1>
__tile_global__ void flash_decode_batch_partial_kernel(const T* __restrict__ _Q, const T* __restrict__ _K,
                                                       const T* __restrict__ _V, float* __restrict__ _WO,
                                                       float* __restrict__ _WM, float* __restrict__ _WL,
                                                       const int* __restrict__ SLOTS, const int* __restrict__ POS,
                                                       const int* __restrict__ BT, int n_seqs, int max_blocks,
                                                       int pool_rows, int split_target, int n_splits, float scale) {
    static_assert(FD_BC == KV_BLOCK, "one decode key tile = one KV-cache block");
    using namespace ct::literals;
    using f32_scores = ct::tile<float, ct::shape<FD_BR, FD_BC>>;
    using f32_rows = ct::tile<float, ct::shape<FD_BR, 1>>;
    using f32_out = ct::tile<float, ct::shape<FD_BR, FA_D>>;
    using i32_rows = ct::tile<int, ct::shape<FD_BR, 1>>;
    using i32_keys = ct::tile<int, ct::shape<1, FD_BC>>;
    using kv_tile = ct::tile<T, ct::shape<FD_BC, FA_D>>;

    auto [h_u, s_u, b_u] = ct::bid();
    int h = static_cast<int>(h_u), split = static_cast<int>(s_u), b = static_cast<int>(b_u);
    int kvh = h / (HQ / HKV);
    const int* table = BT + static_cast<long long>(SLOTS[b]) * max_blocks;   // this sequence's block table
    int n_kv = POS[b] + 1;

    const T* Q = ct::assume_aligned(_Q, 16_ic);
    const T* K = ct::assume_aligned(_K, 16_ic);
    const T* V = ct::assume_aligned(_V, 16_ic);

    auto pQ = ct::partition_view{ct::tensor_span{Q, ct::extents{n_seqs, ct::integral_constant<HQ>{}, ct::integral_constant<FA_D>{}}},
                                 ct::shape<1, 1, FA_D>{}};
    auto pK = ct::partition_view{ct::tensor_span{K, ct::extents{pool_rows, ct::integral_constant<HKV>{}, ct::integral_constant<FA_D>{}}},
                                 ct::shape<FD_BC, 1, FA_D>{}};
    auto pV = ct::partition_view{ct::tensor_span{V, ct::extents{pool_rows, ct::integral_constant<HKV>{}, ct::integral_constant<FA_D>{}}},
                                 ct::shape<FD_BC, 1, FA_D>{}};
    auto pWO = ct::partition_view{ct::tensor_span{_WO, ct::extents{n_seqs, ct::integral_constant<HQ>{}, n_splits, ct::integral_constant<FD_BR>{}, ct::integral_constant<FA_D>{}}},
                                  ct::shape<1, 1, 1, FD_BR, FA_D>{}};
    auto pWM = ct::partition_view{ct::tensor_span{_WM, ct::extents{n_seqs, ct::integral_constant<HQ>{}, n_splits, ct::integral_constant<FD_BR>{}}},
                                  ct::shape<1, 1, 1, FD_BR>{}};
    auto pWL = ct::partition_view{ct::tensor_span{_WL, ct::extents{n_seqs, ct::integral_constant<HQ>{}, n_splits, ct::integral_constant<FD_BR>{}}},
                                  ct::shape<1, 1, 1, FD_BR>{}};

    ct::tile<T, ct::shape<FD_BR, FA_D>> q =
        ct::broadcast<ct::shape<FD_BR, FA_D>>(ct::reshape<ct::shape<1, FA_D>>(pQ.load(b, h, 0)));
    i32_rows qpos = ct::full<i32_rows>(n_kv - 1);

    auto m = ct::full<f32_rows>(-FLT_MAX);
    auto l = ct::zeros<f32_rows>();
    auto acc = ct::zeros<f32_out>();

    int n_tiles = (n_kv + FD_BC - 1) / FD_BC;
    int s_b = split_target < n_tiles ? split_target : n_tiles;
    int tps = (n_tiles + s_b - 1) / s_b;
    int t0 = split * tps;
    int t1 = t0 + tps < n_tiles ? t0 + tps : n_tiles;   // t1 <= t0: empty split
    for (auto kb : ct::irange(t0, t1)) {
        int blk;                                             // physical block of key tile kb
        if constexpr (KMODE == 0) blk = SLOTS[b] * max_blocks + kb;
        else blk = table[kb];
        kv_tile k = ct::reshape<ct::shape<FD_BC, FA_D>>(pK.load_masked(blk, kvh, 0));
        kv_tile v = ct::reshape<ct::shape<FD_BC, FA_D>>(pV.load_masked(blk, kvh, 0));
        auto sc = ct::mma(q, ct::transpose(k), ct::zeros<f32_scores>()) * scale;
        i32_keys kidx = kb * FD_BC + ct::iota<i32_keys>();
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
        } else {
            acc = acc * alpha + ct::mma(ct::element_cast<T>(p), v, ct::zeros<f32_out>());
        }
        m = new_m;
    }
    pWO.store(ct::reshape<ct::shape<1, 1, 1, FD_BR, FA_D>>(acc), b, h, split, 0, 0);
    pWM.store(ct::reshape<ct::shape<1, 1, 1, FD_BR>>(m), b, h, split, 0);
    pWL.store(ct::reshape<ct::shape<1, 1, 1, FD_BR>>(l), b, h, split, 0);
}

template <typename T, int HQ>
__tile_global__ void flash_decode_batch_combine_kernel(const float* __restrict__ _WO, const float* __restrict__ _WM,
                                                       const float* __restrict__ _WL, T* __restrict__ _O, int n_seqs,
                                                       int n_splits) {
    using namespace ct::literals;
    using f32_s = ct::tile<float, ct::shape<FD_SMAX, 1>>;
    using f32_so = ct::tile<float, ct::shape<FD_SMAX, FA_D>>;
    using i32_s = ct::tile<int, ct::shape<FD_SMAX, 1>>;

    auto pWO = ct::partition_view{ct::tensor_span{_WO, ct::extents{n_seqs, ct::integral_constant<HQ>{}, n_splits, ct::integral_constant<FD_BR>{}, ct::integral_constant<FA_D>{}}},
                                  ct::shape<1, 1, FD_SMAX, 1, FA_D>{}};
    auto pWM = ct::partition_view{ct::tensor_span{_WM, ct::extents{n_seqs, ct::integral_constant<HQ>{}, n_splits, ct::integral_constant<FD_BR>{}}},
                                  ct::shape<1, 1, FD_SMAX, 1>{}};
    auto pWL = ct::partition_view{ct::tensor_span{_WL, ct::extents{n_seqs, ct::integral_constant<HQ>{}, n_splits, ct::integral_constant<FD_BR>{}}},
                                  ct::shape<1, 1, FD_SMAX, 1>{}};
    auto pO = ct::partition_view{ct::tensor_span{_O, ct::extents{n_seqs, ct::integral_constant<HQ>{}, ct::integral_constant<FA_D>{}}},
                                 ct::shape<1, 1, FA_D>{}};

    auto [h_u, b_u, dummy] = ct::bid();
    int h = static_cast<int>(h_u), b = static_cast<int>(b_u);
    i32_s sidx = ct::iota<i32_s>();
    auto valid = sidx < n_splits;
    f32_so o_s = ct::reshape<ct::shape<FD_SMAX, FA_D>>(pWO.load_masked(b, h, 0, 0, 0));
    f32_s m_s = ct::reshape<ct::shape<FD_SMAX, 1>>(pWM.load_masked(b, h, 0, 0));
    f32_s l_s = ct::reshape<ct::shape<FD_SMAX, 1>>(pWL.load_masked(b, h, 0, 0));
    o_s = ct::select(valid, o_s, ct::zeros<f32_so>());
    l_s = ct::select(valid, l_s, ct::zeros<f32_s>());
    m_s = ct::select(valid, m_s, ct::full<f32_s>(-FLT_MAX));
    auto mstar = ct::reduce_max(m_s, 0_ic);
    auto w = ct::select(valid, ct::exp(m_s - mstar), ct::zeros<f32_s>());
    auto L = ct::sum(w * l_s, 0_ic);
    auto o = ct::sum(o_s * w, 0_ic) / L;
    pO.store(ct::reshape<ct::shape<1, 1, FA_D>>(ct::element_cast<T>(o)), b, h, 0);
}

// q: [B, HQ, 128]; k_cache / v_cache: the block pool [pool_rows, HKV, 128]; o: [B, HQ, 128]. d_slots / d_pos: device
// ints; d_bt: device block tables [n_slots, max_blocks]. max_kv: the longest attended length in the batch (host value;
// it only sets the grid's split dimension).
template <typename T, int HQ, int HKV, int KMODE = 1>
cudaError_t launch_flash_decode_batch(const T* q, const T* k_cache, const T* v_cache, T* o, int B, const int* d_slots,
                                      const int* d_pos, const int* d_bt, int max_blocks, int pool_rows, int max_kv,
                                      const BatchDecodeWorkspace& ws, cudaStream_t stream = 0) {
    float scale = 1.0f / std::sqrt(static_cast<float>(FA_D));
    int n_tiles = (max_kv + FD_BC - 1) / FD_BC;
    int want = (512 + HQ * B - 1) / (HQ * B);                       // ~512 blocks in total
    // QWEN3_DECODE_SPLITS=<n> fixes the split count, so that the same sequence is summed in the same groups whatever the
    // batch size (used to tell numerical regrouping apart from a bug when comparing batched and alone decoding).
    static const int forced = [] { const char* e = std::getenv("QWEN3_DECODE_SPLITS"); return e ? std::atoi(e) : 0; }();
    if (forced > 0) want = forced;
    int target = want < 1 ? 1 : want;
    if (target > FD_SMAX) target = FD_SMAX;
    int n_splits = target < n_tiles ? target : n_tiles;   // no sequence uses more splits than this
    if (B * n_splits > ws.max_parts) {
        std::fprintf(stderr, "decode workspace too small: %d sequences x %d splits > %d\n", B, n_splits, ws.max_parts);
        return cudaErrorInvalidValue;
    }
    flash_decode_batch_partial_kernel<T, HQ, HKV, 2, KMODE><<<dim3(HQ, n_splits, B), 1, 0, stream>>>(
        q, k_cache, v_cache, ws.o, ws.m, ws.l, d_slots, d_pos, d_bt, B, max_blocks, pool_rows, target, n_splits, scale);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    flash_decode_batch_combine_kernel<T, HQ><<<dim3(HQ, B, 1), 1, 0, stream>>>(ws.o, ws.m, ws.l, o, B, n_splits);
    return cudaGetLastError();
}
