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
#include "../kernels/qwen3_linear_tile.cuh"
#include "../kernels/qwen3_mlp_tile.cuh"
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

static cudaError_t attention_dispatch(const bf16* q, const bf16* k, const bf16* v, bf16* o, int n_q, int n_kv, int hq,
                                      int hkv, const DecodeWorkspace* ws) {
    if (hq == 16 && hkv == 8) return launch_attention_auto<bf16, 16, 8>(q, k, v, o, n_q, n_kv, ws);
    if (hq == 32 && hkv == 8) return launch_attention_auto<bf16, 32, 8>(q, k, v, o, n_q, n_kv, ws);
    if (hq == 32 && hkv == 4) return launch_attention_auto<bf16, 32, 4>(q, k, v, o, n_q, n_kv, ws);
    std::fprintf(stderr, "attention: unsupported heads %d/%d\n", hq, hkv);
    return cudaErrorInvalidValue;
}

struct LayerWeights {
    const bf16 *ln1, *ln2, *wq, *wk, *wv, *wo, *qn, *kn, *wg, *wu, *wd;
};

class Qwen3Engine {
public:
    Qwen3Config cfg;
    int max_len, max_chunk, max_logit_rows;
    bool profile = false;   // when true, forward() times every stage with CUDA events (see prof_report)

    // yarn_factor > 0 turns YaRN on (original context 32768) even if the checkpoint's config has no rope_scaling, as
    // needed for Qwen3-4B at 128K (factor 4). If it is <= 0 the environment variable QWEN3_YARN is used the same way.
    Qwen3Engine(const std::string& dir, int max_len_, int max_chunk_, int max_logit_rows_ = 1, double yarn_factor = 0)
        : max_len(max_len_), max_chunk(max_chunk_), max_logit_rows(max_logit_rows_) {
        cfg = load_config(dir);
        if (cfg.head_dim != FA_D) { std::fprintf(stderr, "head_dim must be 128\n"); std::exit(2); }
        if (cfg.unsupported_rope_scaling) { std::fprintf(stderr, "this rope_scaling type is not supported (only default and YaRN)\n"); std::exit(2); }
        if (yarn_factor <= 0 && std::getenv("QWEN3_YARN")) yarn_factor = std::atof(std::getenv("QWEN3_YARN"));
        if (yarn_factor > 0) { cfg.yarn = true; cfg.yarn_factor = yarn_factor; }

        // 1) weights to the GPU, once
        weights_ = load_weights_to_gpu(dir);
        embed_ = W("model.embed_tokens.weight");
        final_norm_ = W("model.norm.weight");
        lm_head_ = weights_.count("lm_head.weight") ? W("lm_head.weight") : embed_;
        for (int l = 0; l < cfg.layers; ++l) {
            std::string p = "model.layers." + std::to_string(l) + ".";
            layers_.push_back({W(p + "input_layernorm.weight"), W(p + "post_attention_layernorm.weight"),
                               W(p + "self_attn.q_proj.weight"), W(p + "self_attn.k_proj.weight"),
                               W(p + "self_attn.v_proj.weight"), W(p + "self_attn.o_proj.weight"),
                               W(p + "self_attn.q_norm.weight"), W(p + "self_attn.k_norm.weight"),
                               W(p + "mlp.gate_proj.weight"), W(p + "mlp.up_proj.weight"), W(p + "mlp.down_proj.weight")});
        }

        // 2) KV cache: per layer [max_len, hkv, 128] for K and for V
        size_t kv_elems = (size_t)max_len * cfg.hkv * FA_D;
        for (int l = 0; l < cfg.layers; ++l) {
            kcache_.push_back(alloc<bf16>(kv_elems));
            vcache_.push_back(alloc<bf16>(kv_elems));
        }

        // 3) scratch buffers sized for the largest chunk
        size_t N = max_chunk, H = cfg.hidden, QD = (size_t)cfg.hq * FA_D, KD = (size_t)cfg.hkv * FA_D;
        d_tokens_ = alloc<int>(N); d_pos_ = alloc<int>(N); d_argmax_ = alloc<int>(1);
        xa_ = alloc<bf16>(N * H); xb_ = alloc<bf16>(N * H); h_ = alloc<bf16>(N * H); o_ = alloc<bf16>(N * H);
        qraw_ = alloc<bf16>(N * QD); qn_ = alloc<bf16>(N * QD); attn_ = alloc<bf16>(N * QD);
        kraw_ = alloc<bf16>(N * KD); kn_ = alloc<bf16>(N * KD); v_ = alloc<bf16>(N * KD);
        mlp_h_ = alloc<bf16>(N * cfg.inter);
        logits_ = alloc<bf16>((size_t)max_logit_rows * cfg.vocab);
        // split-KV decode workspace (fp32 partial softmax states)
        dec_ws_.o = alloc<float>((size_t)cfg.hq * FD_SMAX * FD_BR * FA_D);
        dec_ws_.m = alloc<float>((size_t)cfg.hq * FD_SMAX * FD_BR);
        dec_ws_.l = alloc<float>((size_t)cfg.hq * FD_SMAX * FD_BR);

        // 4) RoPE tables [max_len, 64] (cos, sin), built like HF: fp32 inverse frequencies, rounded to bf16
        build_rope_tables();
    }

    // Runs the model on `n` new tokens placed at positions start_pos .. start_pos+n-1 (earlier positions must already be in
    // the KV cache). Returns the device logits: [n, vocab] if all_logits (needs max_logit_rows >= n), else [1, vocab] for
    // the last new token.
    const bf16* forward(const int* h_tokens, int n, int start_pos, bool all_logits) {
        if (n > max_chunk || start_pos + n > max_len || (all_logits && n > max_logit_rows)) {
            std::fprintf(stderr, "forward: n=%d start=%d exceeds engine limits\n", n, start_pos);
            std::exit(2);
        }
        std::vector<int> pos(n);
        std::iota(pos.begin(), pos.end(), start_pos);
        CK(cudaMemcpy(d_tokens_, h_tokens, n * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pos_, pos.data(), n * sizeof(int), cudaMemcpyHostToDevice));

        const int H = cfg.hidden, HQ = cfg.hq, HKV = cfg.hkv, QD = HQ * FA_D, KD = HKV * FA_D;
        const double nn = n;
        auto lin_bytes = [](double o, double i) { return o * i * 2.0; };                 // weight bytes of one linear layer
        auto lin_flops = [&](double o, double i) { return 2.0 * nn * o * i; };
        const double pairs = nn * start_pos + nn * (nn + 1) / 2;                        // visible (query, key) pairs
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
            // append this chunk's K and V to the cache (token-major, so it is one contiguous block)
            stage("kv_write", 2.0 * nn * KD * 2 * 2, 0, [&] {
                CK(cudaMemcpyAsync(kcache_[l] + (size_t)start_pos * KD, kn_, (size_t)n * KD * sizeof(bf16), cudaMemcpyDeviceToDevice));
                CK(cudaMemcpyAsync(vcache_[l] + (size_t)start_pos * KD, v_, (size_t)n * KD * sizeof(bf16), cudaMemcpyDeviceToDevice));
            });
            stage("attention", 2.0 * (start_pos + nn) * KD * 2, 4.0 * HQ * FA_D * pairs,
                  [&] { CKL(attention_dispatch(qn_, kcache_[l], vcache_[l], attn_, n, start_pos + n, HQ, HKV, &dec_ws_)); });
            stage("o_proj", lin_bytes(H, QD), lin_flops(H, QD), [&] { CKL(launch_linear<bf16>(attn_, L.wo, o_, n, QD, H)); });
            stage("residual_add", 3.0 * nn * H * 2, 0, [&] { CKL(launch_add<bf16>(x, o_, other, n * H)); });
            std::swap(x, other);
            stage("rmsnorm", 2.0 * nn * H * 2, 0, [&] { CKL(rmsnorm_rows(x, L.ln2, h_, n, H)); });
            stage("mlp_gate_up_silu", 2.0 * lin_bytes(cfg.inter, H), 2.0 * lin_flops(cfg.inter, H),
                  [&] { CKL(launch_mlp_gate_up<bf16>(h_, L.wg, L.wu, mlp_h_, n, H, cfg.inter)); });
            stage("mlp_down", lin_bytes(H, cfg.inter), lin_flops(H, cfg.inter),
                  [&] { CKL(launch_mlp_down<bf16>(mlp_h_, L.wd, o_, n, H, cfg.inter)); });
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

    // Greedy token for one logits row (device pointer).
    int argmax_row(const bf16* row) {
        CKL(launch_argmax<bf16>(row, d_argmax_, 1, cfg.vocab));
        int id;
        CK(cudaMemcpy(&id, d_argmax_, sizeof(int), cudaMemcpyDeviceToHost));
        return id;
    }

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
    const bf16 *embed_ = nullptr, *final_norm_ = nullptr, *lm_head_ = nullptr;
    int *d_tokens_ = nullptr, *d_pos_ = nullptr, *d_argmax_ = nullptr;
    bf16 *xa_, *xb_, *h_, *o_, *qraw_, *qn_, *attn_, *kraw_, *kn_, *v_, *mlp_h_, *logits_;
    bf16 *d_cos_ = nullptr, *d_sin_ = nullptr;
    DecodeWorkspace dec_ws_;

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
        if (cfg.yarn) std::printf("RoPE: YaRN factor %g, original context %d\n", cfg.yarn_factor, cfg.yarn_orig_max);
    }
};
