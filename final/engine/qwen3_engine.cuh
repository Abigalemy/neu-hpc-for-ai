// Qwen3 dense inference engine on CUDA Tile kernels. No PyTorch.
//
// Everything that is per-model happens ONCE in the constructor: weights go to the GPU, the KV cache and the scratch
// buffers are allocated, the RoPE tables are built. forward() then only launches kernels (it never copies weights).
//
// One decoder layer (HF Qwen3DecoderLayer):
//   h = rmsnorm(x);  q,k,v = h@Wq^T, h@Wk^T, h@Wv^T
//   q = rmsnorm_head(q);  k = rmsnorm_head(k);  rope(q, k)         (per-head RMSNorm, then RoPE)
//   append k, v to this layer's KV cache;  a = flash_attention(q, K_cache, V_cache);  x = x + a@Wo^T
//   x = x + mlp(rmsnorm(x))
// Final: logits = rmsnorm(x_last) @ lm_head^T (tied to the embedding table in the small Qwen3 models).
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "qwen3_weights.h"
#include "qwen3_rope_table.h"
#include "../tests/common.h"
#include "../kernels/qwen3_add_tile.cuh"
#include "../kernels/qwen3_argmax_tile.cuh"
#include "../kernels/qwen3_embedding_tile.cuh"
#include "../kernels/qwen3_flash_attention_tile.cuh"
#include "../kernels/qwen3_kv_scatter_tile.cuh"
#include "../kernels/qwen3_linear_tile.cuh"
#include "../kernels/qwen3_mlp_tile.cuh"
#include "../kernels/qwen3_moe_tile.cuh"
#include "../kernels/qwen3_rmsnorm_tile.cuh"
#include "../kernels/qwen3_rope_tile.cuh"

using bf16 = __nv_bfloat16;

#define CKL(call) CK((call))

// ---- compile-time dispatch of the kernels that are specialised on model dimensions -----------------------------------
static cudaError_t rmsnorm_rows(const bf16* x, const bf16* w, bf16* y, int rows, int D) {
    switch (D) {
        case 128:  return launch_rmsnorm<bf16, 128, 128>(x, y, w, rows);
        case 1024: return launch_rmsnorm<bf16, 512, 1024>(x, y, w, rows);
        case 2048: return launch_rmsnorm<bf16, 512, 2048>(x, y, w, rows);
        case 2560: return launch_rmsnorm<bf16, 512, 2560>(x, y, w, rows);
    }
    std::fprintf(stderr, "rmsnorm: unsupported D=%d\n", D);
    return cudaErrorInvalidValue;
}

static cudaError_t rope_dispatch(bf16* q, bf16* k, const bf16* c, const bf16* s, const int* pos, int n, int max_pos,
                                 int hq, int hkv) {
    if (hq == 16 && hkv == 8) return launch_rope<bf16, 16, 8>(q, k, c, s, pos, n, max_pos);
    if (hq == 32 && hkv == 8) return launch_rope<bf16, 32, 8>(q, k, c, s, pos, n, max_pos);
    if (hq == 32 && hkv == 4) return launch_rope<bf16, 32, 4>(q, k, c, s, pos, n, max_pos);
    std::fprintf(stderr, "rope: unsupported heads %d/%d\n", hq, hkv);
    return cudaErrorInvalidValue;
}

// Prefill / chunk of ONE sequence: q [n_q, HQ, 128] against the block pool kc, vc (pool_rows rows) through that
// sequence's block table d_bt (device).
static cudaError_t attention_prefill(const bf16* q, const bf16* kc, const bf16* vc, bf16* o, int n_q, int n_kv,
                                     const int* d_bt, int pool_rows, int hq, int hkv) {
    if (hq == 16 && hkv == 8) return launch_flash_attention<bf16, 16, 8>(q, kc, vc, o, n_q, n_kv, 0, d_bt, pool_rows);
    if (hq == 32 && hkv == 8) return launch_flash_attention<bf16, 32, 8>(q, kc, vc, o, n_q, n_kv, 0, d_bt, pool_rows);
    if (hq == 32 && hkv == 4) return launch_flash_attention<bf16, 32, 4>(q, kc, vc, o, n_q, n_kv, 0, d_bt, pool_rows);
    std::fprintf(stderr, "attention: unsupported heads %d/%d\n", hq, hkv);
    return cudaErrorInvalidValue;
}

// One decode step of B sequences (split-KV kernel over the block pool; d_bt = all block tables [slots, max_blocks]).
// QWEN3_ATTN=small or a device without big tiles: one launch per sequence with the verified 16x16 / fp32-P kernel.
static cudaError_t attention_decode(const bf16* q, const bf16* kc, const bf16* vc, bf16* o, int B, const int* d_slots,
                                    const int* d_pos, const int* h_slots, const int* h_pos, const int* d_bt, int max_blocks,
                                    int pool_rows, int hq, int hkv, const BatchDecodeWorkspace& ws) {
    static const bool force_small = [] { const char* e = std::getenv("QWEN3_ATTN"); return e && std::string(e) == "small"; }();
    if (tile_profile() != TileProfile::BIG || force_small) {
        const size_t QD = (size_t)hq * FA_D;
        for (int b = 0; b < B; ++b) {
            cudaError_t e = attention_prefill(q + b * QD, kc, vc, o + b * QD, 1, h_pos[b] + 1,
                                              d_bt + (size_t)h_slots[b] * max_blocks, pool_rows, hq, hkv);
            if (e != cudaSuccess) return e;
        }
        return cudaSuccess;
    }
    int max_kv = 0;
    for (int b = 0; b < B; ++b) max_kv = std::max(max_kv, h_pos[b] + 1);
    if (hq == 16 && hkv == 8) return launch_flash_decode_batch<bf16, 16, 8>(q, kc, vc, o, B, d_slots, d_pos, d_bt, max_blocks, pool_rows, max_kv, ws);
    if (hq == 32 && hkv == 8) return launch_flash_decode_batch<bf16, 32, 8>(q, kc, vc, o, B, d_slots, d_pos, d_bt, max_blocks, pool_rows, max_kv, ws);
    if (hq == 32 && hkv == 4) return launch_flash_decode_batch<bf16, 32, 4>(q, kc, vc, o, B, d_slots, d_pos, d_bt, max_blocks, pool_rows, max_kv, ws);
    std::fprintf(stderr, "attention: unsupported heads %d/%d\n", hq, hkv);
    return cudaErrorInvalidValue;
}

static cudaError_t kv_scatter_dispatch(const bf16* k, const bf16* v, bf16* kc, bf16* vc, const int* d_rows, int n, int n_rows,
                                       int hkv) {
    if (hkv == 8) return launch_kv_scatter<bf16, 8>(k, v, kc, vc, d_rows, n, n_rows);
    if (hkv == 4) return launch_kv_scatter<bf16, 4>(k, v, kc, vc, d_rows, n, n_rows);
    std::fprintf(stderr, "kv scatter: unsupported kv heads %d\n", hkv);
    return cudaErrorInvalidValue;
}

struct LayerWeights {
    const bf16 *ln1, *ln2, *wq, *wk, *wv, *wo, *qn, *kn, *wg, *wu, *wd;   // wg/wu/wd: dense MLP (nullptr in an MoE layer)
    const bf16* router = nullptr;                                         // MoE layer: [E, H] router
    const bf16 *ewg = nullptr, *ewu = nullptr, *ewd = nullptr;            //   packed experts [E, I, H], [E, I, H], [E, H, I]
};

// Packed MoE expert weights: per layer one buffer each for gate, up [E, I, H] and down [E, H, I], filled while loading
// (load_weights_to_gpu's place callback copies every expert tensor straight into its slot: one copy on the GPU).
struct MoePack {
    int E = 0, I = 0, H = 0;
    int first_layer = 0;          // buffers hold layers first_layer .. first_layer + g.size() - 1 (a test loads one)
    std::vector<bf16*> g, u, d;
    long long placed = 0;
    void alloc(int layers, int E_, int I_, int H_) {
        E = E_; I = I_; H = H_;
        const size_t elems = (size_t)E * I * H;
        for (int l = 0; l < layers; ++l)
            for (auto* v : {&g, &u, &d}) {
                void* ptr;
                if (cudaMalloc(&ptr, elems * sizeof(bf16)) != cudaSuccess) { std::fprintf(stderr, "out of GPU memory for experts\n"); std::exit(2); }
                v->push_back((bf16*)ptr);
            }
    }
    // "model.layers.<l>.mlp.experts.<e>.<gate|up|down>_proj.weight" -> its slot; anything else -> DEFAULT
    Placement place(const std::string& name, const std::vector<int64_t>& shape, size_t bytes, void** dst) {
        int l = -1, e = -1;
        char which[8] = {0};
        if (std::sscanf(name.c_str(), "model.layers.%d.mlp.experts.%d.%7[a-z]_proj.weight", &l, &e, which) != 3) return Placement::DEFAULT;
        std::string w = which;
        const bool down = w == "down";
        l -= first_layer;
        if (l < 0 || l >= (int)g.size() || e < 0 || e >= E || !(down || w == "gate" || w == "up") || shape.size() != 2 ||
            shape[0] != (down ? H : I) || shape[1] != (down ? I : H) || bytes != (size_t)I * H * sizeof(bf16)) {
            std::fprintf(stderr, "unexpected expert tensor %s\n", name.c_str());
            std::exit(2);
        }
        bf16* base = down ? d[l] : w == "gate" ? g[l] : u[l];
        *dst = base + (size_t)e * I * H;
        ++placed;
        return Placement::PLACED;
    }
};

class Qwen3Engine {
public:
    Qwen3Config cfg;
    int max_len, max_chunk, max_logit_rows, max_seqs;
    int max_blocks;          // block-table width: blocks of one sequence of max_len tokens
    int n_blocks;            // KV-cache blocks in the pool (KV_BLOCK tokens each), shared by all sequences
    bool profile = false;    // when true, forward() times every stage with CUDA events (see prof_report)

    // yarn_factor > 0 turns YaRN on (original context 32768) even if the checkpoint's config has no rope_scaling, as
    // needed for Qwen3-4B at 128K (factor 4). If it is <= 0 the environment variable QWEN3_YARN is used the same way.
    // max_seqs: number of slots, i.e. sequences that can be decoded together (continuous batching). A slot is only a
    // row of the block table; the KV memory is the pool of kv_tokens tokens (rounded up to whole blocks) shared by all
    // slots (PagedAttention). kv_tokens <= 0: max_seqs * max_len, i.e. enough for every slot at full length.
    Qwen3Engine(const std::string& dir, int max_len_, int max_chunk_, int max_logit_rows_ = 1, double yarn_factor = 0,
                int max_seqs_ = 1, long long kv_tokens = 0)
        : max_len(max_len_), max_chunk(max_chunk_), max_logit_rows(std::max(max_logit_rows_, max_seqs_)), max_seqs(max_seqs_) {
        max_blocks = (max_len + KV_BLOCK - 1) / KV_BLOCK;
        long long want = kv_tokens > 0 ? (kv_tokens + KV_BLOCK - 1) / KV_BLOCK : (long long)max_seqs * max_blocks;
        if (want * KV_BLOCK > 0x7fffffffLL) { std::fprintf(stderr, "KV pool too large for 32-bit row indices\n"); std::exit(2); }
        n_blocks = (int)want;
        cfg = load_config(dir);
        if (cfg.head_dim != FA_D) { std::fprintf(stderr, "head_dim must be 128\n"); std::exit(2); }
        if (cfg.unsupported_rope_scaling) { std::fprintf(stderr, "this rope_scaling type is not supported (only default and YaRN)\n"); std::exit(2); }
        if (yarn_factor <= 0 && std::getenv("QWEN3_YARN")) yarn_factor = std::atof(std::getenv("QWEN3_YARN"));
        if (yarn_factor > 0) { cfg.yarn = true; cfg.yarn_factor = yarn_factor; }

        // 1) weights to the GPU, once (MoE experts straight into their packed per-layer buffers)
        if (cfg.experts > 0) {
            if (cfg.experts != 128 || cfg.top_k != 8 || cfg.sparse_step != 1 || cfg.has_mlp_only_layers || cfg.moe_inter % 64 ||
                cfg.hidden % 256) {
                std::fprintf(stderr, "MoE config not supported (compiled for 128 experts, top-8, every layer MoE)\n");
                std::exit(2);
            }
            moe_.alloc(cfg.layers, cfg.experts, cfg.moe_inter, cfg.hidden);
            weights_ = load_weights_to_gpu(dir, [&](const std::string& n, const std::vector<int64_t>& sh, size_t b, void** dst) {
                return moe_.place(n, sh, b, dst);
            });
            if (moe_.placed != (long long)cfg.layers * cfg.experts * 3) {
                std::fprintf(stderr, "found %lld expert tensors, expected %d\n", moe_.placed, cfg.layers * cfg.experts * 3);
                std::exit(2);
            }
        } else {
            weights_ = load_weights_to_gpu(dir);
        }
        embed_ = W("model.embed_tokens.weight");
        final_norm_ = W("model.norm.weight");
        lm_head_ = weights_.count("lm_head.weight") ? W("lm_head.weight") : embed_;
        for (int l = 0; l < cfg.layers; ++l) {
            std::string p = "model.layers." + std::to_string(l) + ".";
            const bool moe = cfg.is_moe_layer(l);
            LayerWeights L{W(p + "input_layernorm.weight"), W(p + "post_attention_layernorm.weight"),
                           W(p + "self_attn.q_proj.weight"), W(p + "self_attn.k_proj.weight"),
                           W(p + "self_attn.v_proj.weight"), W(p + "self_attn.o_proj.weight"),
                           W(p + "self_attn.q_norm.weight"), W(p + "self_attn.k_norm.weight"),
                           moe ? nullptr : W(p + "mlp.gate_proj.weight"), moe ? nullptr : W(p + "mlp.up_proj.weight"),
                           moe ? nullptr : W(p + "mlp.down_proj.weight")};
            if (moe) {
                L.router = W(p + "mlp.gate.weight");
                L.ewg = moe_.g[l]; L.ewu = moe_.u[l]; L.ewd = moe_.d[l];
            }
            layers_.push_back(L);
        }
        {
            size_t wb = 0;
            for (auto& kv : weights_) wb += kv.second.bytes;
            if (cfg.experts > 0) wb += (size_t)cfg.layers * 3 * cfg.experts * cfg.moe_inter * cfg.hidden * sizeof(bf16);
            std::printf("weights: %.2f GB on the GPU (%s)\n", wb / 1e9, cfg.experts > 0 ? "MoE, experts packed per layer" : "dense");
            std::fflush(stdout);
        }

        // 2) KV cache: per layer a block pool [n_blocks * KV_BLOCK, hkv, 128] for K and for V; block tables
        //    [max_seqs, max_blocks] (host copy = the truth, device copy uploaded when a table changes). Zeroed once, so
        //    a block that was never written holds finite values.
        size_t kv_elems = (size_t)n_blocks * KV_BLOCK * cfg.hkv * FA_D;
        for (int l = 0; l < cfg.layers; ++l) {
            kcache_.push_back(alloc<bf16>(kv_elems));
            vcache_.push_back(alloc<bf16>(kv_elems));
            CK(cudaMemset(kcache_.back(), 0, kv_elems * sizeof(bf16)));
            CK(cudaMemset(vcache_.back(), 0, kv_elems * sizeof(bf16)));
        }
        tables_.assign(max_seqs, {});
        table_dirty_.assign(max_seqs, false);
        for (int b = n_blocks - 1; b >= 0; --b) free_.push_back(b);   // pop_back hands out 0, 1, 2, ...
        d_bt_ = alloc<int>((size_t)max_seqs * max_blocks);
        CK(cudaMemset(d_bt_, 0, (size_t)max_seqs * max_blocks * sizeof(int)));
        std::printf("KV cache: %d blocks of %d tokens = %lld tokens, %.2f GB (%d layers)\n", n_blocks, KV_BLOCK,
                    (long long)n_blocks * KV_BLOCK, 2.0 * kv_elems * sizeof(bf16) * cfg.layers / 1e9, cfg.layers);
        std::fflush(stdout);

        // 3) scratch buffers sized for the largest chunk
        size_t N = std::max(max_chunk, max_seqs), H = cfg.hidden, QD = (size_t)cfg.hq * FA_D, KD = (size_t)cfg.hkv * FA_D;
        d_tokens_ = alloc<int>(N); d_pos_ = alloc<int>(N); d_slots_ = alloc<int>(N); d_rows_ = alloc<int>(N);
        d_argmax_ = alloc<int>(max_seqs);
        xa_ = alloc<bf16>(N * H); xb_ = alloc<bf16>(N * H); h_ = alloc<bf16>(N * H); o_ = alloc<bf16>(N * H);
        qraw_ = alloc<bf16>(N * QD); qn_ = alloc<bf16>(N * QD); attn_ = alloc<bf16>(N * QD);
        kraw_ = alloc<bf16>(N * KD); kn_ = alloc<bf16>(N * KD); v_ = alloc<bf16>(N * KD);
        if (cfg.experts > 0) {
            // one MoE workspace shared by all layers, for N tokens: A = N * K assignments
            const size_t A = N * (size_t)cfg.top_k;
            moe_ws_.max_tokens = (int)N;
            moe_ws_.idx = alloc<int>(A); moe_ws_.w = alloc<bf16>(A);
            moe_ws_.count = alloc<int>(cfg.experts); moe_ws_.off = alloc<int>(cfg.experts);
            moe_ws_.sorted = alloc<int>(A); moe_ws_.inv = alloc<int>(A);
            moe_ws_.hbuf = alloc<bf16>(A * cfg.moe_inter); moe_ws_.ysorted = alloc<bf16>(A * H);
            std::printf("MoE workspace: %zu tokens, %.2f GB\n", N, (A * (cfg.moe_inter + H) * sizeof(bf16) + A * 16) / 1e9);
            std::fflush(stdout);
        } else {
            mlp_h_ = alloc<bf16>(N * cfg.inter);
        }
        logits_ = alloc<bf16>((size_t)max_logit_rows * cfg.vocab);
        // split-KV decode workspace (fp32 partial softmax states). B sequences x n_splits <= ~512/HQ + B <= FD_SMAX + B.
        dec_ws_.max_parts = FD_SMAX + max_seqs;
        dec_ws_.o = alloc<float>((size_t)dec_ws_.max_parts * cfg.hq * FD_BR * FA_D);
        dec_ws_.m = alloc<float>((size_t)dec_ws_.max_parts * cfg.hq * FD_BR);
        dec_ws_.l = alloc<float>((size_t)dec_ws_.max_parts * cfg.hq * FD_BR);

        // 4) RoPE tables [max_len, 64] (cos, sin), built like HF: fp32 inverse frequencies, rounded to bf16
        build_rope_tables();
    }

    // ---- block manager (PagedAttention) ---------------------------------------------------------------------------
    // reserve(slot, n): make the slot's table cover positions 0..n-1, taking blocks from the free list on demand.
    // All or nothing: false (nothing taken) if the pool or the table width is too small. release(slot): return all of
    // the slot's blocks. The server decides who gets blocks (admission, preemption); forward() reserves the positions
    // it writes and exits if that fails, so callers that share the pool must reserve first.
    int free_blocks() const { return (int)free_.size(); }
    int blocks_of(int slot) const { return (int)tables_[slot].size(); }
    bool reserve(int slot, int n_tokens) {
        if (slot < 0 || slot >= max_seqs) return false;
        int need = (n_tokens + KV_BLOCK - 1) / KV_BLOCK;
        std::vector<int>& t = tables_[slot];
        if (need <= (int)t.size()) return true;
        if (need > max_blocks || need - (int)t.size() > (int)free_.size()) return false;
        while ((int)t.size() < need) { t.push_back(free_.back()); free_.pop_back(); }
        table_dirty_[slot] = true;
        return true;
    }
    void release(int slot) {
        if (slot < 0 || slot >= max_seqs) return;
        std::vector<int>& t = tables_[slot];
        for (int i = (int)t.size() - 1; i >= 0; --i) free_.push_back(t[i]);
        t.clear();
    }
    // pool row of position pos of the sequence in `slot` (the position must be reserved)
    int kv_row(int slot, int pos) const { return tables_[slot][pos / KV_BLOCK] * KV_BLOCK + pos % KV_BLOCK; }

    // Prefill (or one chunk) of ONE sequence: `n` new tokens at positions start_pos .. start_pos+n-1 of cache slot `slot`
    // (earlier positions must already be in that slot). Returns device logits: [n, vocab] if all_logits (needs
    // max_logit_rows >= n), else [1, vocab] for the last new token. n == 1 runs the decode path (batch of one).
    const bf16* forward(const int* h_tokens, int n, int start_pos, bool all_logits, int slot = 0) {
        if (n <= 0 || start_pos < 0 || n > max_len - start_pos) {
            std::fprintf(stderr, "forward: n=%d start=%d out of range (max_len %d)\n", n, start_pos, max_len);
            std::exit(2);
        }
        if (n == 1) return forward_decode(h_tokens, &slot, &start_pos, 1);
        if (n > max_chunk || slot < 0 || slot >= max_seqs || (all_logits && n > max_logit_rows)) {
            std::fprintf(stderr, "forward: n=%d start=%d slot=%d exceeds engine limits\n", n, start_pos, slot);
            std::exit(2);
        }
        std::vector<int> pos(n);
        std::iota(pos.begin(), pos.end(), start_pos);
        return run(h_tokens, n, pos.data(), nullptr, slot, all_logits);
    }

    // One decode step for B sequences: token b is at position h_pos[b] of cache slot h_slots[b] (all slots distinct).
    // Returns device logits [B, vocab].
    const bf16* forward_decode(const int* h_tokens, const int* h_slots, const int* h_pos, int B) {
        if (B < 1 || B > max_seqs) { std::fprintf(stderr, "forward_decode: B=%d exceeds max_seqs=%d\n", B, max_seqs); std::exit(2); }
        for (int b = 0; b < B; ++b)
            if (h_slots[b] < 0 || h_slots[b] >= max_seqs || h_pos[b] < 0 || h_pos[b] >= max_len) {
                std::fprintf(stderr, "forward_decode: slot %d / position %d out of range\n", h_slots[b], h_pos[b]);
                std::exit(2);
            }
        return run(h_tokens, B, h_pos, h_slots, -1, true);
    }

    // Greedy tokens of `rows` logits rows (device pointer).
    std::vector<int> argmax_rows(const bf16* logits, int rows) {
        CKL(launch_argmax<bf16>(logits, d_argmax_, rows, cfg.vocab));
        std::vector<int> ids(rows);
        CK(cudaMemcpy(ids.data(), d_argmax_, rows * sizeof(int), cudaMemcpyDeviceToHost));
        return ids;
    }

    // Greedy token for one logits row (device pointer).
    int argmax_row(const bf16* row) { return argmax_rows(row, 1)[0]; }

    // RoPE tables as the engine built them, for testing: cos then sin, each [max_len, 64] bf16.
    std::vector<bf16> rope_tables_host() const {
        std::vector<bf16> v(2 * (size_t)max_len * 64);
        CK(cudaMemcpy(v.data(), d_cos_, (size_t)max_len * 64 * sizeof(bf16), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(v.data() + (size_t)max_len * 64, d_sin_, (size_t)max_len * 64 * sizeof(bf16), cudaMemcpyDeviceToHost));
        return v;
    }

    // ---- profiling -----------------------------------------------------------------------------------------------
    void prof_reset() { prof_.clear(); }

    // Table of the stages accumulated since prof_reset(): time per call group, share of the total, achieved memory
    // bandwidth (weights / cache bytes read, a lower bound of real traffic) and achieved TFLOP/s (useful flops).
    // `divisor` turns totals into per-forward numbers (e.g. the number of decode steps that were accumulated).
    std::string prof_report(const char* title, int divisor) const {
        std::vector<std::pair<std::string, ProfAcc>> rows(prof_.begin(), prof_.end());
        std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.second.ms > b.second.ms; });
        double total = 0;
        for (auto& r : rows) total += r.second.ms;
        char buf[256];
        std::string out = std::string("PROFILE ") + title + " (per forward, summed over all layers; stages are timed with CUDA events)\n";
        std::snprintf(buf, sizeof buf, "  %-18s %6s %10s %7s %10s %10s\n", "stage", "calls", "ms", "share", "GB/s", "TFLOP/s");
        out += buf;
        for (auto& r : rows) {
            const ProfAcc& a = r.second;
            std::snprintf(buf, sizeof buf, "  %-18s %6d %10.3f %6.1f%% %10.1f %10.2f\n", r.first.c_str(), a.calls / divisor,
                          a.ms / divisor, 100.0 * a.ms / total, a.ms > 0 ? a.bytes / (a.ms * 1e-3) / 1e9 : 0.0,
                          a.ms > 0 ? a.flops / (a.ms * 1e-3) / 1e12 : 0.0);
            out += buf;
        }
        std::snprintf(buf, sizeof buf, "  %-18s %6s %10.3f\n", "TOTAL (stages)", "", total / divisor);
        return out + buf;
    }

private:
    struct ProfAcc { double ms = 0, bytes = 0, flops = 0; int calls = 0; };
    struct ProfEntry { cudaEvent_t a, b; const char* name; double bytes, flops; };
    std::map<std::string, ProfAcc> prof_;
    std::vector<ProfEntry> prof_log_;
    std::vector<cudaEvent_t> ev_pool_;

    // The forward pass. Prefill: h_slots == nullptr, all n tokens belong to `slot` at consecutive positions.
    // Decode: h_slots != nullptr, token i belongs to slot h_slots[i] at position h_pos[i].
    const bf16* run(const int* h_tokens, int n, const int* h_pos, const int* h_slots, int slot, bool all_logits) {
        const bool decode = h_slots != nullptr;
        const int H = cfg.hidden, HQ = cfg.hq, HKV = cfg.hkv, QD = HQ * FA_D, KD = HKV * FA_D;
        const int start_pos = decode ? 0 : h_pos[0];
        // blocks for the positions written now (a no-op when the caller reserved them), then the pool row of each token
        std::vector<int> kv_rows(n);
        for (int i = 0; i < n; ++i) {
            int s = decode ? h_slots[i] : slot;
            if (!reserve(s, h_pos[i] + 1)) {
                std::fprintf(stderr, "KV cache: no free block for slot %d position %d (%d free)\n", s, h_pos[i], free_blocks());
                std::exit(2);
            }
            kv_rows[i] = kv_row(s, h_pos[i]);
        }
        upload_tables();
        CK(cudaMemcpy(d_tokens_, h_tokens, n * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pos_, h_pos, n * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_rows_, kv_rows.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        if (decode) CK(cudaMemcpy(d_slots_, h_slots, n * sizeof(int), cudaMemcpyHostToDevice));
        const int pool_rows = n_blocks * KV_BLOCK;
        const int* seq_bt = decode ? nullptr : d_bt_ + (size_t)slot * max_blocks;   // prefill: this sequence's table
        const double nn = n;
        auto lin_bytes = [](double o, double i) { return o * i * 2.0; };                 // weight bytes of one linear layer
        auto lin_flops = [&](double o, double i) { return 2.0 * nn * o * i; };
        double pairs = 0, kv_read = 0;                                                    // for the profile only
        for (int i = 0; i < n; ++i) { pairs += decode ? h_pos[i] + 1 : start_pos + i + 1; kv_read += decode ? h_pos[i] + 1 : 0; }
        if (!decode) kv_read = start_pos + nn;
        bf16 *x = xa_, *other = xb_;
        stage("embedding", nn * H * 2.0, 0, [&] { CKL(launch_embedding<bf16>(d_tokens_, embed_, x, cfg.vocab, n, H)); });
        for (int l = 0; l < cfg.layers; ++l) {
            const LayerWeights& L = layers_[l];
            stage("rmsnorm", 2.0 * nn * H * 2, 0, [&] { CKL(rmsnorm_rows(x, L.ln1, h_, n, H)); });
            stage("q_proj", lin_bytes(QD, H), lin_flops(QD, H), [&] { CKL(launch_linear<bf16>(h_, L.wq, qraw_, n, H, QD)); });
            stage("k_proj", lin_bytes(KD, H), lin_flops(KD, H), [&] { CKL(launch_linear<bf16>(h_, L.wk, kraw_, n, H, KD)); });
            stage("v_proj", lin_bytes(KD, H), lin_flops(KD, H), [&] { CKL(launch_linear<bf16>(h_, L.wv, v_, n, H, KD)); });
            stage("qk_norm", 2.0 * nn * (QD + KD) * 2, 0, [&] {
                CKL(rmsnorm_rows(qraw_, L.qn, qn_, n * HQ, FA_D));    // per-head RMSNorm: rows = n*heads, D = 128
                CKL(rmsnorm_rows(kraw_, L.kn, kn_, n * HKV, FA_D));
            });
            stage("rope", 2.0 * nn * (QD + KD) * 2, 0, [&] { CKL(rope_dispatch(qn_, kn_, d_cos_, d_sin_, d_pos_, n, max_len, HQ, HKV)); });
            stage("kv_write", 2.0 * nn * KD * 2 * 2, 0, [&] {   // every token to its pool row (blocks are not contiguous)
                CKL(kv_scatter_dispatch(kn_, v_, kcache_[l], vcache_[l], d_rows_, n, pool_rows, HKV));
            });
            stage("attention", 2.0 * kv_read * KD * 2, 4.0 * HQ * FA_D * pairs, [&] {
                if (decode)
                    CKL(attention_decode(qn_, kcache_[l], vcache_[l], attn_, n, d_slots_, d_pos_, h_slots, h_pos, d_bt_,
                                         max_blocks, pool_rows, HQ, HKV, dec_ws_));
                else
                    CKL(attention_prefill(qn_, kcache_[l], vcache_[l], attn_, n, start_pos + n, seq_bt, pool_rows, HQ, HKV));
            });
            stage("o_proj", lin_bytes(H, QD), lin_flops(H, QD), [&] { CKL(launch_linear<bf16>(attn_, L.wo, o_, n, QD, H)); });
            stage("residual_add", 3.0 * nn * H * 2, 0, [&] { CKL(launch_add<bf16>(x, o_, other, n * H)); });
            std::swap(x, other);
            stage("rmsnorm", 2.0 * nn * H * 2, 0, [&] { CKL(rmsnorm_rows(x, L.ln2, h_, n, H)); });
            if (L.router) {   // MoE layer: K of E experts per token (bytes: the experts that can be touched, flops: K)
                const double ex_bytes = std::min<double>(cfg.experts, nn * cfg.top_k) * 3.0 * lin_bytes(cfg.moe_inter, H);
                stage("moe", ex_bytes, 3.0 * cfg.top_k * lin_flops(cfg.moe_inter, H), [&] {
                    CKL((launch_moe<bf16, 128, 8>(h_, L.router, L.ewg, L.ewu, L.ewd, o_, n, H, cfg.moe_inter, cfg.norm_topk, moe_ws_)));
                });
                if (moe_dump_) dump_routing(l, n, h_pos);
            } else {
                stage("mlp_gate_up_silu", 2.0 * lin_bytes(cfg.inter, H), 2.0 * lin_flops(cfg.inter, H),
                      [&] { CKL(launch_mlp_gate_up<bf16>(h_, L.wg, L.wu, mlp_h_, n, H, cfg.inter)); });
                stage("mlp_down", lin_bytes(H, cfg.inter), lin_flops(H, cfg.inter),
                      [&] { CKL(launch_mlp_down<bf16>(mlp_h_, L.wd, o_, n, H, cfg.inter)); });
            }
            stage("residual_add", 3.0 * nn * H * 2, 0, [&] { CKL(launch_add<bf16>(x, o_, other, n * H)); });
            std::swap(x, other);
        }
        int rows = all_logits ? n : 1;
        const bf16* xin = all_logits ? x : x + (size_t)(n - 1) * H;
        stage("rmsnorm", 0, 0, [&] { CKL(rmsnorm_rows(xin, final_norm_, h_, rows, H)); });
        stage("lm_head", lin_bytes(cfg.vocab, H), 2.0 * rows * cfg.vocab * H,
              [&] { CKL(launch_linear<bf16>(h_, lm_head_, logits_, rows, H, cfg.vocab)); });
        CK(cudaDeviceSynchronize());
        if (profile) prof_collect();
        return logits_;
    }

    // QWEN3_MOE_DUMP=<file>: append every MoE layer's chosen experts, record = int32 layer, n, positions[n],
    // experts[n * K] (debugging only: a device-to-host copy per layer)
    std::FILE* moe_dump_ = [] { const char* p = std::getenv("QWEN3_MOE_DUMP"); return p ? std::fopen(p, "wb") : nullptr; }();
    void dump_routing(int layer, int n, const int* h_pos) {
        std::vector<int> idx((size_t)n * cfg.top_k);
        CK(cudaMemcpy(idx.data(), moe_ws_.idx, idx.size() * sizeof(int), cudaMemcpyDeviceToHost));
        int head[2] = {layer, n};
        std::fwrite(head, sizeof(int), 2, moe_dump_);
        std::fwrite(h_pos, sizeof(int), n, moe_dump_);
        std::fwrite(idx.data(), sizeof(int), idx.size(), moe_dump_);
        std::fflush(moe_dump_);
    }

    void upload_tables() {   // device copy of the block tables that changed since the last forward
        for (int s = 0; s < max_seqs; ++s) {
            if (!table_dirty_[s]) continue;
            CK(cudaMemcpy(d_bt_ + (size_t)s * max_blocks, tables_[s].data(), tables_[s].size() * sizeof(int), cudaMemcpyHostToDevice));
            table_dirty_[s] = false;
        }
    }

    cudaEvent_t ev(size_t i) {
        while (ev_pool_.size() <= i) { cudaEvent_t e; CK(cudaEventCreate(&e)); ev_pool_.push_back(e); }
        return ev_pool_[i];
    }

    template <typename F>
    void stage(const char* name, double bytes, double flops, F&& f) {
        if (!profile) { f(); return; }
        cudaEvent_t a = ev(2 * prof_log_.size()), b = ev(2 * prof_log_.size() + 1);
        CK(cudaEventRecord(a));
        f();
        CK(cudaEventRecord(b));
        prof_log_.push_back({a, b, name, bytes, flops});
    }

    void prof_collect() {   // after the device is synchronised: turn the recorded event pairs into totals
        for (auto& e : prof_log_) {
            float ms;
            CK(cudaEventElapsedTime(&ms, e.a, e.b));
            ProfAcc& acc = prof_[e.name];
            acc.ms += ms; acc.bytes += e.bytes; acc.flops += e.flops; acc.calls += 1;
        }
        prof_log_.clear();
    }

    std::map<std::string, DevTensor> weights_;
    std::vector<LayerWeights> layers_;
    std::vector<bf16*> kcache_, vcache_;
    std::vector<std::vector<int>> tables_;   // block table of every slot (host)
    std::vector<bool> table_dirty_;
    std::vector<int> free_;                  // free physical blocks
    int* d_bt_ = nullptr;                    // block tables on the device [max_seqs, max_blocks]
    const bf16 *embed_ = nullptr, *final_norm_ = nullptr, *lm_head_ = nullptr;
    int *d_tokens_ = nullptr, *d_pos_ = nullptr, *d_slots_ = nullptr, *d_rows_ = nullptr, *d_argmax_ = nullptr;
    bf16 *xa_, *xb_, *h_, *o_, *qraw_, *qn_, *attn_, *kraw_, *kn_, *v_, *mlp_h_, *logits_;
    bf16 *d_cos_ = nullptr, *d_sin_ = nullptr;
    BatchDecodeWorkspace dec_ws_;
    MoePack moe_;
    MoeWorkspace<bf16> moe_ws_;

    template <typename X>
    X* alloc(size_t n) {
        void* p;
        if (cudaMalloc(&p, n * sizeof(X)) != cudaSuccess) { std::fprintf(stderr, "out of GPU memory\n"); std::exit(2); }
        return (X*)p;
    }

    const bf16* W(const std::string& name) {
        auto it = weights_.find(name);
        if (it == weights_.end()) { std::fprintf(stderr, "missing weight %s\n", name.c_str()); std::exit(2); }
        return (const bf16*)it->second.ptr;
    }

    void build_rope_tables() {
        RopeParams rp;
        rp.head_dim = FA_D;
        rp.theta = cfg.rope_theta;
        rp.yarn = cfg.yarn;
        rp.factor = cfg.yarn_factor;
        rp.orig_max = cfg.yarn_orig_max;
        rp.beta_fast = cfg.yarn_beta_fast;
        rp.beta_slow = cfg.yarn_beta_slow;
        rp.attention_factor = cfg.yarn_attention_factor;
        std::vector<bf16> t = rope_tables_bf16(rp, max_len);          // cos then sin, [max_len, 64] each
        const size_t n = (size_t)max_len * (FA_D / 2);
        d_cos_ = alloc<bf16>(n); d_sin_ = alloc<bf16>(n);
        CK(cudaMemcpy(d_cos_, t.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sin_, t.data() + n, n * sizeof(bf16), cudaMemcpyHostToDevice));
        if (cfg.yarn) { std::printf("RoPE: YaRN factor %g, original context %d\n", cfg.yarn_factor, cfg.yarn_orig_max); std::fflush(stdout); }
    }
};
