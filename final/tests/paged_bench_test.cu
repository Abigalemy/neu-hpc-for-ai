// Timing of the attention kernels with and without the PagedAttention indirection. Hq 32 / Hkv 8 (Qwen3-4B), bf16, one
// layer. Written when the 128K run got slower after paging (TTFT 23.6 -> 46.0 s, ITL 25.7 -> 40.0 ms); the first run
// (H100, 2026-11-18) showed the cause was zeroing the stale V rows with a select in the inner loop (decode 1.9x, prefill
// 2.5x slower), not the block-table lookup (same time as before paging, also with shuffled blocks). The select was
// removed; this test keeps the comparison "before paging" vs "paged" (identity and shuffled tables).
//
//   paged_bench_test        (no arguments; random data; prints times only)
//
// decode (split-KV):  KMODE 0 = no table, slot laid out contiguously (the kernel before paging)
//                     KMODE 1 = block table (the engine's path), with the identity table and with shuffled blocks
// prefill (64x128 split-P): contiguous kernel vs paged (identity / shuffled table)
// KV write of a 130944-token prompt: one contiguous copy (before paging) vs the per-token scatter (now)
#include <algorithm>
#include <random>
#include <type_traits>
#include <cuda_bf16.h>
#include "common.h"
#include "../kernels/qwen3_flash_attention_tile.cuh"
#include "../kernels/qwen3_kv_scatter_tile.cuh"

using bf16 = __nv_bfloat16;
constexpr int HQ = 32, HKV = 8, KD = HKV * FA_D, QD = HQ * FA_D;

// random values for timing only: a 1M-element random pattern, repeated on the device (generating hundreds of millions
// of samples on the host would keep the paid GPU idle for tens of seconds)
static bf16* dev_random(size_t n) {
    static std::vector<bf16> pat;
    if (pat.empty()) {
        pat.resize(1 << 20);
        std::mt19937 r(7);
        std::normal_distribution<float> d(0.f, 1.f);
        for (auto& x : pat) x = __float2bfloat16(d(r));
    }
    bf16* p; CK(cudaMalloc(&p, n * sizeof(bf16)));
    size_t first = std::min(n, pat.size());
    CK(cudaMemcpy(p, pat.data(), first * sizeof(bf16), cudaMemcpyHostToDevice));
    for (size_t done = first; done < n; done *= 2)   // double the filled prefix until n elements
        CK(cudaMemcpy(p + done, p, std::min(done, n - done) * sizeof(bf16), cudaMemcpyDeviceToDevice));
    return p;
}

static int* dev_ints(const std::vector<int>& h) {
    int* p; CK(cudaMalloc(&p, h.size() * sizeof(int)));
    CK(cudaMemcpy(p, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
    return p;
}

template <typename F>
static float time_ms(F&& f, int reps) {
    f(); CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) f();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaGetLastError());
    return ms / reps;
}

// B sequences of n_kv keys each; slot s owns blocks s*mb .. (identity) or a random permutation of all blocks
static void decode_case(int B, int n_kv) {
    int mb = (n_kv + KV_BLOCK - 1) / KV_BLOCK, nblk = B * mb, rows = nblk * KV_BLOCK;
    bf16 *k = dev_random((size_t)rows * KD), *v = dev_random((size_t)rows * KD), *q = dev_random((size_t)B * QD), *o;
    CK(cudaMalloc(&o, (size_t)B * QD * sizeof(bf16)));
    std::vector<int> ident(nblk), shuf(nblk), slots(B), pos(B, n_kv - 1);
    for (int i = 0; i < nblk; ++i) ident[i] = shuf[i] = i;
    std::shuffle(shuf.begin(), shuf.end(), std::mt19937(3));
    for (int b = 0; b < B; ++b) slots[b] = b;
    int *dI = dev_ints(ident), *dS = dev_ints(shuf), *ds = dev_ints(slots), *dp = dev_ints(pos);
    BatchDecodeWorkspace ws;
    ws.max_parts = FD_SMAX + B;
    CK(cudaMalloc(&ws.o, (size_t)ws.max_parts * HQ * FD_BR * FA_D * sizeof(float)));
    CK(cudaMalloc(&ws.m, (size_t)ws.max_parts * HQ * FD_BR * sizeof(float)));
    CK(cudaMalloc(&ws.l, (size_t)ws.max_parts * HQ * FD_BR * sizeof(float)));
    auto run = [&](auto kmode, const int* bt) {
        constexpr int KM = decltype(kmode)::value;
        return time_ms([&] { CK((launch_flash_decode_batch<bf16, HQ, HKV, KM>(q, k, v, o, B, ds, dp, bt, mb, rows, n_kv, ws))); }, 50);
    };
    double gb = 2.0 * B * n_kv * KD * 2 / 1e9;   // K + V bytes read
    auto line = [&](const char* name, float ms) {
        std::printf("DECODE B=%-3d n_kv=%-6d %-34s %8.1f us  %6.0f GB/s\n", B, n_kv, name, ms * 1e3, gb / (ms * 1e-3));
    };
    line("KMODE 0 no table (before paging)", run(std::integral_constant<int, 0>{}, dI));
    line("KMODE 1 paged, identity table", run(std::integral_constant<int, 1>{}, dI));
    line("KMODE 1 paged, shuffled blocks", run(std::integral_constant<int, 1>{}, dS));
    std::fflush(stdout);
    for (void* p : {(void*)k, (void*)v, (void*)q, (void*)o, (void*)dI, (void*)dS, (void*)ds, (void*)dp, (void*)ws.o, (void*)ws.m, (void*)ws.l})
        CK(cudaFree(p));
}

static void prefill_case(int n) {
    int nb = (n + KV_BLOCK - 1) / KV_BLOCK, rows = nb * KV_BLOCK;
    bf16 *k = dev_random((size_t)rows * KD), *v = dev_random((size_t)rows * KD), *q = dev_random((size_t)n * QD), *o;
    CK(cudaMalloc(&o, (size_t)n * QD * sizeof(bf16)));
    std::vector<int> ident(nb), shuf(nb);
    for (int i = 0; i < nb; ++i) ident[i] = shuf[i] = i;
    std::shuffle(shuf.begin(), shuf.end(), std::mt19937(5));
    int *dI = dev_ints(ident), *dS = dev_ints(shuf);
    double tflop = 4.0 * HQ * FA_D * (double)n * (n + 1) / 2 / 1e12;
    auto line = [&](const char* name, float ms) {
        std::printf("PREFILL n=%-6d %-34s %9.2f ms  %6.1f TFLOP/s\n", n, name, ms, tflop / (ms * 1e-3));
    };
    line("contiguous (before paging)", time_ms([&] { CK((launch_flash_attention_tiled<bf16, HQ, HKV, 64, 128, 2>(q, k, v, o, n, n))); }, 3));
    line("paged, identity table", time_ms([&] { CK((launch_flash_attention_tiled<bf16, HQ, HKV, 64, 128, 2>(q, k, v, o, n, n, 0, dI, rows))); }, 3));
    line("paged, shuffled blocks", time_ms([&] { CK((launch_flash_attention_tiled<bf16, HQ, HKV, 64, 128, 2>(q, k, v, o, n, n, 0, dS, rows))); }, 3));
    std::fflush(stdout);
    for (void* p : {(void*)k, (void*)v, (void*)q, (void*)o, (void*)dI, (void*)dS}) CK(cudaFree(p));
}

static void kv_write_case(int n) {
    bf16 *kn = dev_random((size_t)n * KD), *vn = dev_random((size_t)n * KD), *kc, *vc;
    CK(cudaMalloc(&kc, (size_t)n * KD * sizeof(bf16))); CK(cudaMalloc(&vc, (size_t)n * KD * sizeof(bf16)));
    std::vector<int> rows(n);
    for (int i = 0; i < n; ++i) rows[i] = i;
    int* dr = dev_ints(rows);
    float copy = time_ms([&] {
        CK(cudaMemcpyAsync(kc, kn, (size_t)n * KD * sizeof(bf16), cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(vc, vn, (size_t)n * KD * sizeof(bf16), cudaMemcpyDeviceToDevice));
    }, 10);
    float scat = time_ms([&] { CK((launch_kv_scatter<bf16, HKV>(kn, vn, kc, vc, dr, n, n))); }, 10);
    std::printf("KV WRITE n=%d per layer: contiguous copy %.3f ms, per-token scatter %.3f ms (x36 layers: %.1f vs %.1f ms)\n",
                n, copy, scat, copy * 36, scat * 36);
    std::fflush(stdout);
    for (void* p : {(void*)kn, (void*)vn, (void*)kc, (void*)vc, (void*)dr}) CK(cudaFree(p));
}

int main() {
    decode_case(1, 131072);    // the 128K decode step
    decode_case(1, 3072);
    decode_case(100, 1500);    // the lecture workload's decode batch
    prefill_case(16384);       // a long prompt (the 128K prefill is ~64x this attention work)
    prefill_case(2048);
    kv_write_case(130944);
    std::printf("DONE\n");
    return 0;
}
