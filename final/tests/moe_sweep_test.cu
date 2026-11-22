// Part 4: tile sweep of the MoE expert GEMMs (gate/up + SiLU, down) at the real Qwen3-30B-A3B sizes (128 experts,
// hidden 2048, expert inter 768), with the work list, for the token counts of the suite (decode batches, prefill
// lengths). Uniform routing (8 distinct experts per token), random weights.
// Every configuration keeps BK = 64, so each output element is accumulated in the same order: its hbuf / ysorted must
// be bit-for-bit those of the configuration the engine uses today (checked); only the speed differs.
//   moe_sweep_test
#include <cuda_bf16.h>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include "common.h"
#include "../kernels/qwen3_moe_tile.cuh"

using bf16 = __nv_bfloat16;
constexpr int E = 128, HID = 2048, INTER = 768, K = 8;

template <typename V>
static V* dev(size_t n) { V* p; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(V))); return p; }

struct Bufs { bf16 *x, *wg, *wu, *wd; MoeWorkspace<bf16> ws; int* tiles2; };

// one configuration: time gate_up (BMG x BNG) and down (BMD x BND), each with its own work list
template <int BMG, int BNG, int BMD, int BND, int HG = 7, int HD = 7>
static void run(Bufs& b, int n, float& t_gu, float& t_dn, std::vector<bf16>& h, std::vector<bf16>& y, int reps) {
    const int A = n * K;
    const int capg = (int)moe_tile_cap(A, BMG, E), capd = (int)moe_tile_cap(A, BMD, E);
    moe_tile_list_kernel<E, BMG, 128><<<1, 1>>>(b.ws.count, 0, E, capg, b.ws.tiles);
    moe_tile_list_kernel<E, BMD, 128><<<1, 1>>>(b.ws.count, 0, E, capd, b.tiles2);
    CK(cudaGetLastError());
    auto gu = [&] {
        moe_gate_up_kernel<bf16, K, BMG, BNG, 64, HG><<<dim3(capg, 1, INTER / BNG), 1>>>(b.x, n, b.wg, b.wu, b.ws.count, b.ws.off,
                                                                                    b.ws.sorted, b.ws.hbuf, HID, INTER, 0, b.ws.tiles, capg);
    };
    auto dn = [&] {
        moe_down_kernel<bf16, BMD, BND, 64, HD><<<dim3(capd, 1, HID / BND), 1>>>(b.ws.hbuf, b.wd, b.ws.count, b.ws.off, b.ws.ysorted,
                                                                          HID, INTER, 0, b.tiles2, capd);
    };
    CK(cudaMemset(b.ws.hbuf, 0xFF, (size_t)A * INTER * 2));
    CK(cudaMemset(b.ws.ysorted, 0xFF, (size_t)A * HID * 2));
    gu(); dn();
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    h.resize((size_t)A * INTER); y.resize((size_t)A * HID);
    CK(cudaMemcpy(h.data(), b.ws.hbuf, h.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y.data(), b.ws.ysorted, y.size() * 2, cudaMemcpyDeviceToHost));
    cudaEvent_t e0, e1, e2;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2));
    float a = 0, c = 0;
    for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(e0)); gu(); CK(cudaEventRecord(e1)); dn(); CK(cudaEventRecord(e2));
        CK(cudaEventSynchronize(e2));
        float x1, x2;
        CK(cudaEventElapsedTime(&x1, e0, e1)); CK(cudaEventElapsedTime(&x2, e1, e2));
        a += x1; c += x2;
    }
    t_gu = a / reps; t_dn = c / reps;
}

int main() {
    const int MAXN = 3072, MAXA = MAXN * K;
    Bufs b;
    unsigned st = 1;
    auto fill = [&](bf16* d, size_t n, float scale) {
        std::vector<bf16> v(n);
        for (auto& x : v) { st = st * 1664525u + 1013904223u; x = __float2bfloat16(scale * (((st >> 9) & 0xFFFF) / 32768.0f - 1.0f)); }
        CK(cudaMemcpy(d, v.data(), n * 2, cudaMemcpyHostToDevice));
    };
    b.x = dev<bf16>((size_t)MAXN * HID); b.wg = dev<bf16>((size_t)E * INTER * HID); b.wu = dev<bf16>((size_t)E * INTER * HID);
    b.wd = dev<bf16>((size_t)E * HID * INTER);
    fill(b.x, (size_t)MAXN * HID, 1.0f); fill(b.wg, (size_t)E * INTER * HID, 0.05f); fill(b.wu, (size_t)E * INTER * HID, 0.05f);
    fill(b.wd, (size_t)E * HID * INTER, 0.05f);
    b.ws.idx = dev<int>(MAXA); b.ws.count = dev<int>(E); b.ws.off = dev<int>(E); b.ws.sorted = dev<int>(MAXA); b.ws.inv = dev<int>(MAXA);
    b.ws.hbuf = dev<bf16>((size_t)MAXA * INTER); b.ws.ysorted = dev<bf16>((size_t)MAXA * HID);
    b.ws.tiles = dev<int>(2 * moe_tile_cap(MAXA, 16, E)); b.tiles2 = dev<int>(2 * moe_tile_cap(MAXA, 16, E));
    b.ws.max_tokens = MAXN;
    std::mt19937 rng(3);
    int failed = 0;
    for (int n : {1, 8, 64, 256, 2048, 3072}) {
        std::vector<int> ids(n * K), perm(E);
        for (int t = 0; t < n; ++t) {
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int j = 0; j < K; ++j) ids[t * K + j] = perm[j];
        }
        CK(cudaMemcpy(b.ws.idx, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
        CK((moe_sort<bf16, E, K>(n, b.ws)));
        const double flops = 2.0 * n * K * HID * INTER;                 // per GEMM: gate_up is 2 of these, down 1
        const int reps = n >= 2048 ? 10 : 30;
        std::vector<bf16> h0, y0;
        float g0, d0;
        const bool big = (long long)n * K / E > 32;                     // the engine's row tile choice (moe_bm)
        if (big) run<64, 64, 64, 128>(b, n, g0, d0, h0, y0, reps); else run<16, 64, 16, 128>(b, n, g0, d0, h0, y0, reps);
        std::printf("tokens %d (rows per expert ~%d): with all hints (7), BM %d: gate_up %.3f ms (%.1f TFLOP/s), down %.3f ms (%.1f TFLOP/s)\n",
                    n, n * K / E, big ? 64 : 16, g0, 2 * flops / g0 / 1e9, d0, flops / d0 / 1e9);
        auto cmp = [&](const char* name, auto f) {
            std::vector<bf16> h, y;
            float g, d;
            f(h, y, g, d);
            bool same = h.size() == h0.size() && y.size() == y0.size() && !std::memcmp(h.data(), h0.data(), h.size() * 2) &&
                        !std::memcmp(y.data(), y0.data(), y.size() * 2);
            failed += !same;
            std::printf("  %-26s gate_up %8.3f ms %7.1f TFLOP/s | down %8.3f ms %7.1f TFLOP/s | %s\n", name, g, 2 * flops / g / 1e9,
                        d, flops / d / 1e9, same ? "bit-exact" : "DIFFERS");
        };
#define CFG(a, b_, c, d_, hg, hd) cmp("gate " #a "x" #b_ " h" #hg ", down " #c "x" #d_ " h" #hd, \
    [&](auto& h, auto& y, float& g, float& d) { run<a, b_, c, d_, hg, hd>(b, n, g, d, h, y, reps); })
        // h = alignment hints (1 input rows, 2 weights, 4 output rows; 0 = none = the engine before Part 4)
        if (big) {
            CFG(64, 64, 64, 128, 0, 0); CFG(64, 64, 64, 128, 2, 2); CFG(64, 64, 64, 128, 6, 3); CFG(64, 64, 64, 256, 1, 7);
            CFG(64, 128, 64, 256, 0, 7); CFG(64, 128, 64, 256, 2, 7); CFG(64, 128, 64, 256, 4, 7); CFG(128, 128, 64, 256, 0, 7);
        } else {
            CFG(16, 64, 16, 128, 0, 0); CFG(16, 64, 16, 128, 2, 2); CFG(16, 64, 16, 256, 6, 7); CFG(16, 128, 16, 256, 0, 7);
            CFG(16, 128, 16, 256, 2, 7);
        }
#undef CFG
    }
    std::printf(failed ? "MOE SWEEP: %d configuration(s) DIFFER\n" : "MOE SWEEP: every configuration bit-exact with the engine's\n", failed);
    return failed != 0;
}
