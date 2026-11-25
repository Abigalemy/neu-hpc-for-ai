// Attention tile sweep on a prefill-sized problem (n_q = n_kv = 3072) and a decode-sized one (n_q = 1, n_kv = 3072),
// Hq = 32, Hkv = 8 (Qwen3-4B), bf16.
// For each (BR, BC, P in bf16 or fp32): does it launch, how fast, and how far from the verified default (16x16, fp32 P).
// A configuration that fails to launch is reported, not fatal.
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include "common.h"
#include "../kernels/qwen3_flash_attention_tile.cuh"

using bf16 = __nv_bfloat16;
constexpr int HQ = 32, HKV = 8;

struct Bufs { bf16 *q, *k, *v, *o; int n; };

template <int BR, int BC, int PV>
static void run_cfg(const Bufs& b, const std::vector<bf16>& ref, double flops, const std::vector<bf16>& ref_dec) {
    char name[64];
    std::snprintf(name, sizeof name, "BR=%-3d BC=%-3d P=%s", BR, BC, PV == 0 ? "fp32" : PV == 1 ? "bf16" : "bf16x2");
    cudaError_t e = launch_flash_attention_tiled<bf16, HQ, HKV, BR, BC, PV>(b.q, b.k, b.v, b.o, b.n, b.n);
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { std::printf("ATTN %s FAIL %s\n", name, cudaGetErrorString(e)); cudaGetLastError(); return; }
    std::vector<bf16> got((size_t)b.n * HQ * FA_D);
    CK(cudaMemcpy(got.data(), b.o, got.size() * 2, cudaMemcpyDeviceToHost));
    double mx = 0, mr = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double g = __bfloat162float(got[i]), r = __bfloat162float(ref[i]);
        mx = std::max(mx, std::fabs(g - r)); mr = std::max(mr, std::fabs(r));
    }
    cudaEvent_t a, c; cudaEventCreate(&a); cudaEventCreate(&c);
    cudaEventRecord(a);
    for (int i = 0; i < 5; ++i) launch_flash_attention_tiled<bf16, HQ, HKV, BR, BC, PV>(b.q, b.k, b.v, b.o, b.n, b.n);
    cudaEventRecord(c); cudaEventSynchronize(c);
    float ms; cudaEventElapsedTime(&ms, a, c); ms /= 5;
    // decode shape: one query (the last position) against the 3072-key cache
    e = launch_flash_attention_tiled<bf16, HQ, HKV, BR, BC, PV>(b.q, b.k, b.v, b.o, 1, b.n);
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    double dmx = -1, dmr = 1;
    float dus = -1;
    if (e == cudaSuccess) {
        std::vector<bf16> dg((size_t)HQ * FA_D);
        CK(cudaMemcpy(dg.data(), b.o, dg.size() * 2, cudaMemcpyDeviceToHost));
        dmx = 0; dmr = 0;
        for (size_t i = 0; i < dg.size(); ++i) {
            double g = __bfloat162float(dg[i]), r = __bfloat162float(ref_dec[i]);
            dmx = std::max(dmx, std::fabs(g - r)); dmr = std::max(dmr, std::fabs(r));
        }
        cudaEventRecord(a);
        for (int i = 0; i < 100; ++i) launch_flash_attention_tiled<bf16, HQ, HKV, BR, BC, PV>(b.q, b.k, b.v, b.o, 1, b.n);
        cudaEventRecord(c); cudaEventSynchronize(c);
        float dms; cudaEventElapsedTime(&dms, a, c); dus = dms / 100 * 1e3f;
    } else { cudaGetLastError(); }
    std::printf("ATTN %-22s ok  prefill(3072) %8.3f ms %7.1f TFLOP/s diff %.2e | decode(1 x 3072) %8.1f us (cache %.0f GB/s) diff %.2e\n", name, ms,
                flops / (ms * 1e-3) / 1e12, mx / mr, dus, 2.0 * b.n * HKV * FA_D * 2 / (dus * 1e-6) / 1e9, dmx / dmr);
    std::fflush(stdout);
}

int main() {
    const int n = 3072;
    std::vector<bf16> hq((size_t)n * HQ * FA_D), hk((size_t)n * HKV * FA_D), hv(hk.size());
    unsigned s = 11;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return (((s >> 8) & 0xFFFF) / 65536.0f - 0.5f) * 3.4f; };
    for (auto& x : hq) x = __float2bfloat16(rnd());
    for (auto& x : hk) x = __float2bfloat16(rnd());
    for (auto& x : hv) x = __float2bfloat16(rnd());
    Bufs b{nullptr, nullptr, nullptr, nullptr, n};
    CK(cudaMalloc(&b.q, hq.size() * 2)); CK(cudaMalloc(&b.k, hk.size() * 2)); CK(cudaMalloc(&b.v, hv.size() * 2));
    CK(cudaMalloc(&b.o, hq.size() * 2));
    CK(cudaMemcpy(b.q, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.k, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.v, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice));
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    std::printf("device: %s\n", p.name);

    CK((launch_flash_attention_tiled<bf16, HQ, HKV, 16, 16, 0>(b.q, b.k, b.v, b.o, n, n)));
    CK(cudaDeviceSynchronize());
    std::vector<bf16> ref((size_t)n * HQ * FA_D);
    CK(cudaMemcpy(ref.data(), b.o, ref.size() * 2, cudaMemcpyDeviceToHost));
    const double flops = 4.0 * HQ * FA_D * ((double)n * (n + 1) / 2);   // useful flops of the causal prefill
    CK((launch_flash_attention_tiled<bf16, HQ, HKV, 16, 16, 0>(b.q, b.k, b.v, b.o, 1, n)));   // decode reference: verified config
    CK(cudaDeviceSynchronize());
    std::vector<bf16> ref_dec((size_t)HQ * FA_D);
    CK(cudaMemcpy(ref_dec.data(), b.o, ref_dec.size() * 2, cudaMemcpyDeviceToHost));

    run_cfg<16, 16, 0>(b, ref, flops, ref_dec);
    run_cfg<16, 16, 1>(b, ref, flops, ref_dec);
    run_cfg<16, 64, 0>(b, ref, flops, ref_dec);
    run_cfg<16, 64, 1>(b, ref, flops, ref_dec);
    run_cfg<64, 32, 0>(b, ref, flops, ref_dec);
    run_cfg<64, 32, 1>(b, ref, flops, ref_dec);
    run_cfg<64, 64, 0>(b, ref, flops, ref_dec);
    run_cfg<64, 64, 1>(b, ref, flops, ref_dec);
    run_cfg<128, 32, 0>(b, ref, flops, ref_dec);
    run_cfg<128, 32, 1>(b, ref, flops, ref_dec);
    run_cfg<128, 64, 0>(b, ref, flops, ref_dec);
    run_cfg<128, 64, 1>(b, ref, flops, ref_dec);
    run_cfg<64, 128, 1>(b, ref, flops, ref_dec);
    // P split in two bf16 terms: keeps P's precision, 2x the second matmul
    run_cfg<16, 64, 2>(b, ref, flops, ref_dec);
    run_cfg<64, 64, 2>(b, ref, flops, ref_dec);
    run_cfg<64, 128, 2>(b, ref, flops, ref_dec);
    run_cfg<128, 64, 2>(b, ref, flops, ref_dec);
    return 0;
}
