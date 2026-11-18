// PagedAttention kernels vs the contiguous ones: the same keys and values, once in a contiguous cache and once scattered
// over a shuffled block pool whose other rows hold large finite garbage (stale K/V of earlier sequences). The paged
// kernels read the same tiles in the same order and give masked keys P = 0 exactly, so their outputs must be
// BIT-IDENTICAL to the contiguous ones (and finite).
//
//   paged_attention_test            (no arguments; random data, Hq 32 / Hkv 8 as Qwen3-4B; exit code 0 = pass)
//
// 1) prefill kernel (big 64x128 split-P tiles and small 16x16 fp32-P tiles): full prefill (n_q == n_kv) and a chunk
//    (the last n_q queries of n_kv keys), contiguous vs paged.
// 2) batched split-KV decode: B sequences of different lengths, reference pool = each sequence in consecutive blocks
//    with zero rows past its end (what the contiguous cache gave), test pool = shuffled blocks, garbage elsewhere.
#include <algorithm>
#include <cstring>
#include <random>
#include <cuda_bf16.h>
#include "common.h"
#include "../kernels/qwen3_flash_attention_tile.cuh"

using bf16 = __nv_bfloat16;
constexpr int HQ = 32, HKV = 8, KD = HKV * FA_D, QD = HQ * FA_D;

static std::mt19937 rng(1234);

static std::vector<bf16> random_bf16(size_t n) {
    std::normal_distribution<float> d(0.f, 1.f);
    std::vector<bf16> v(n);
    for (auto& x : v) x = __float2bfloat16(d(rng));
    return v;
}

// stale rows: finite but large (+-1000), so any leak of a masked key into the result would show

template <typename X>
static X* dev(const std::vector<X>& h) {
    X* p; CK(cudaMalloc(&p, h.size() * sizeof(X)));
    CK(cudaMemcpy(p, h.data(), h.size() * sizeof(X), cudaMemcpyHostToDevice));
    return p;
}

template <typename X>
static std::vector<X> host(const X* d, size_t n) {
    std::vector<X> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(X), cudaMemcpyDeviceToHost));
    return h;
}

static bool same_and_finite(const std::vector<bf16>& a, const std::vector<bf16>& b, const char* what) {
    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(bf16)) != 0) {
        size_t i = 0;
        while (std::memcmp(&a[i], &b[i], sizeof(bf16)) == 0) ++i;
        std::printf("FAIL %s: first difference at element %zu (%g vs %g)\n", what, i, __bfloat162float(a[i]), __bfloat162float(b[i]));
        return false;
    }
    for (auto x : a)
        if (!std::isfinite(__bfloat162float(x))) { std::printf("FAIL %s: non-finite output\n", what); return false; }
    std::printf("ok   %s\n", what);
    return true;
}

// a pool of n_blocks blocks filled with stale garbage; the sequence's rows go to the blocks of `table`
static std::vector<bf16> stale_pool(int n_blocks) {
    static std::vector<bf16> pat;        // a 64K random pattern, repeated (filling ~10^8 values one by one is slow)
    if (pat.empty()) {
        std::uniform_real_distribution<float> d(-1000.f, 1000.f);
        pat.resize(1 << 16);
        for (auto& x : pat) x = __float2bfloat16(d(rng));
    }
    std::vector<bf16> v((size_t)n_blocks * KV_BLOCK * KD);
    for (size_t i = 0; i < v.size(); i += pat.size())
        std::memcpy(&v[i], pat.data(), std::min(pat.size(), v.size() - i) * sizeof(bf16));
    return v;
}

static void put(std::vector<bf16>& pool, const std::vector<int>& table, const std::vector<bf16>& seq, int n_kv) {
    for (int p = 0; p < n_kv; ++p) {
        size_t row = (size_t)table[p / KV_BLOCK] * KV_BLOCK + p % KV_BLOCK;
        std::memcpy(&pool[row * KD], &seq[(size_t)p * KD], KD * sizeof(bf16));
    }
}

template <int BR, int BC, int PV>
static bool prefill_case(int n_q, int n_kv, const char* cfg) {
    auto q = random_bf16((size_t)n_q * QD), k = random_bf16((size_t)n_kv * KD), v = random_bf16((size_t)n_kv * KD);
    int nb = (n_kv + KV_BLOCK - 1) / KV_BLOCK, pool_blocks = nb + 5;
    std::vector<int> perm(pool_blocks);
    for (int i = 0; i < pool_blocks; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int> table(perm.begin(), perm.begin() + nb);
    auto kp = stale_pool(pool_blocks), vp = stale_pool(pool_blocks);
    put(kp, table, k, n_kv); put(vp, table, v, n_kv);

    bf16 *dq = dev(q), *dk = dev(k), *dv = dev(v), *dkp = dev(kp), *dvp = dev(vp), *o1, *o2;
    int* dt = dev(table);
    CK(cudaMalloc(&o1, (size_t)n_q * QD * sizeof(bf16))); CK(cudaMalloc(&o2, (size_t)n_q * QD * sizeof(bf16)));
    CK((launch_flash_attention_tiled<bf16, HQ, HKV, BR, BC, PV>(dq, dk, dv, o1, n_q, n_kv)));
    CK((launch_flash_attention_tiled<bf16, HQ, HKV, BR, BC, PV>(dq, dkp, dvp, o2, n_q, n_kv, 0, dt, pool_blocks * KV_BLOCK)));
    CK(cudaDeviceSynchronize());
    char what[128];
    std::snprintf(what, sizeof what, "prefill %s n_q=%d n_kv=%d", cfg, n_q, n_kv);
    bool ok = same_and_finite(host(o1, (size_t)n_q * QD), host(o2, (size_t)n_q * QD), what);
    for (void* p : {(void*)dq, (void*)dk, (void*)dv, (void*)dkp, (void*)dvp, (void*)o1, (void*)o2, (void*)dt}) CK(cudaFree(p));
    return ok;
}

static bool decode_case(const std::vector<int>& lens) {
    const int B = (int)lens.size();
    int max_blocks = 0, total = 0, max_kv = 0;
    for (int n : lens) { int nb = (n + KV_BLOCK - 1) / KV_BLOCK; max_blocks = std::max(max_blocks, nb); total += nb; max_kv = std::max(max_kv, n); }
    int pool_blocks = total + 7;
    auto q = random_bf16((size_t)B * QD);
    // reference: sequence b in consecutive blocks, zero rows past its end; test: shuffled blocks, garbage elsewhere
    std::vector<bf16> kr((size_t)pool_blocks * KV_BLOCK * KD, __float2bfloat16(0.f)), vr = kr;
    auto kt = stale_pool(pool_blocks), vt = stale_pool(pool_blocks);
    std::vector<int> perm(pool_blocks);
    for (int i = 0; i < pool_blocks; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int> bt_ref((size_t)B * max_blocks, 0), bt_test((size_t)B * max_blocks, 0);
    int next_ref = 0, next_test = 0;
    for (int b = 0; b < B; ++b) {
        int nb = (lens[b] + KV_BLOCK - 1) / KV_BLOCK;
        std::vector<int> tr(nb), tt(nb);
        for (int i = 0; i < nb; ++i) { tr[i] = next_ref++; tt[i] = perm[next_test++]; }
        std::copy(tr.begin(), tr.end(), bt_ref.begin() + (size_t)b * max_blocks);
        std::copy(tt.begin(), tt.end(), bt_test.begin() + (size_t)b * max_blocks);
        auto k = random_bf16((size_t)lens[b] * KD), v = random_bf16((size_t)lens[b] * KD);
        put(kr, tr, k, lens[b]); put(vr, tr, v, lens[b]);
        put(kt, tt, k, lens[b]); put(vt, tt, v, lens[b]);
    }
    std::vector<int> slots(B), pos(B);
    for (int b = 0; b < B; ++b) { slots[b] = b; pos[b] = lens[b] - 1; }

    BatchDecodeWorkspace ws;
    ws.max_parts = FD_SMAX + B;
    CK(cudaMalloc(&ws.o, (size_t)ws.max_parts * HQ * FD_BR * FA_D * sizeof(float)));
    CK(cudaMalloc(&ws.m, (size_t)ws.max_parts * HQ * FD_BR * sizeof(float)));
    CK(cudaMalloc(&ws.l, (size_t)ws.max_parts * HQ * FD_BR * sizeof(float)));
    bf16 *dq = dev(q), *dkr = dev(kr), *dvr = dev(vr), *dkt = dev(kt), *dvt = dev(vt), *o1, *o2;
    int *ds = dev(slots), *dp = dev(pos), *dbr = dev(bt_ref), *dbt = dev(bt_test);
    CK(cudaMalloc(&o1, (size_t)B * QD * sizeof(bf16))); CK(cudaMalloc(&o2, (size_t)B * QD * sizeof(bf16)));
    int rows = pool_blocks * KV_BLOCK;
    CK((launch_flash_decode_batch<bf16, HQ, HKV>(dq, dkr, dvr, o1, B, ds, dp, dbr, max_blocks, rows, max_kv, ws)));
    CK((launch_flash_decode_batch<bf16, HQ, HKV>(dq, dkt, dvt, o2, B, ds, dp, dbt, max_blocks, rows, max_kv, ws)));
    CK(cudaDeviceSynchronize());
    std::string what = "decode batch, lengths";
    for (int n : lens) what += " " + std::to_string(n);
    return same_and_finite(host(o1, (size_t)B * QD), host(o2, (size_t)B * QD), what.c_str());
}

int main() {
    bool ok = true;
    for (int n_kv : {1, 127, 128, 129, 1000, 3000}) {
        for (int n_q : {n_kv, std::min(n_kv, 37)}) {
            ok &= prefill_case<64, 128, 2>(n_q, n_kv, "64x128 split-P");
            ok &= prefill_case<16, 16, 0>(n_q, n_kv, "16x16 fp32-P");
        }
    }
    ok &= decode_case({1});
    ok &= decode_case({1000});
    ok &= decode_case({1, 128, 129, 300, 1000, 4100});
    std::vector<int> many;
    for (int i = 0; i < 100; ++i) many.push_back(1000 + 5 * i);   // a lecture-sized batch
    ok &= decode_case(many);
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
