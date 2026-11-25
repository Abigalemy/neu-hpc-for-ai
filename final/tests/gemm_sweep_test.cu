// GEMM tile sweep: y[M,N] = x[M,K] @ w[N,K]^T in bf16 (fp32 accumulate), the shape of a Qwen3-4B MLP projection.
//
// For every (BM, BN, BK, occupancy-hint) configuration: does it launch, how fast is it for prefill (M = 3072) and for
// decode (M = 1), and does it agree with the already-verified baseline kernel (16x32x32, qwen3_linear_tile.cuh).
// A configuration that fails to launch is reported, not fatal.
//   gemm_sweep_test            -> plain GEMM sweep, then the fused gate/up + SiLU kernel sweep
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include "common.h"
#include "../kernels/qwen3_mlp_tile.cuh"

using bf16 = __nv_bfloat16;

#define SWEEP_BODY                                                                                              \
    using namespace ct::literals;                                                                               \
    using acc_t = ct::tile<float, ct::shape<BM, BN>>;                                                           \
    const bf16* X = ct::assume_aligned(_X, 16_ic);                                                              \
    const bf16* W = ct::assume_aligned(_W, 16_ic);                                                              \
    bf16* Y = ct::assume_aligned(_Y, 16_ic);                                                                    \
    auto in = ct::assume_divisible(_in, 16_ic);                                                                 \
    auto out = ct::assume_divisible(_out, 16_ic);                                                               \
    auto pX = ct::partition_view{ct::tensor_span{X, ct::extents{n_tokens, in}}, ct::shape<BM, BK>{}};           \
    auto pW = ct::partition_view{ct::tensor_span{W, ct::extents{out, in}}, ct::shape<BN, BK>{}};                \
    auto pY = ct::partition_view{ct::tensor_span{Y, ct::extents{n_tokens, out}}, ct::shape<BM, BN>{}};          \
    auto [mb, nb, dummy] = ct::bid();                                                                           \
    auto acc = ct::zeros<acc_t>();                                                                              \
    int k_blocks = (in + BK - 1) / BK;                                                                          \
    for (auto kb : ct::irange(0, k_blocks)) {                                                                   \
        ct::tile<bf16, ct::shape<BM, BK>> x;                                                                    \
        ct::tile<bf16, ct::shape<BN, BK>> w;                                                                    \
        x = pX.load_masked(mb, kb);                                                                             \
        w = pW.load(nb, kb);                                                                                    \
        acc = ct::mma(x, ct::transpose(w), acc);                                                                \
    }                                                                                                           \
    pY.store_masked(ct::element_cast<bf16>(acc), mb, nb);

// default occupancy (no hint)
template <int BM, int BN, int BK>
__tile_global__ void sweep_kernel(const bf16* __restrict__ _X, const bf16* __restrict__ _W, bf16* __restrict__ _Y,
                                  int n_tokens, int _in, int _out) {
    SWEEP_BODY
}

// with an occupancy hint
template <int BM, int BN, int BK, int OCC>
[[ cutile::hint(0, occupancy = OCC) ]]
__tile_global__ void sweep_kernel_occ(const bf16* __restrict__ _X, const bf16* __restrict__ _W, bf16* __restrict__ _Y,
                                      int n_tokens, int _in, int _out) {
    SWEEP_BODY
}

struct Buffers {
    bf16 *x, *w, *y, *ref;
    int K, N;
};

template <int BM, int BN, int BK, int OCC>
static cudaError_t launch(const Buffers& b, int M, bf16* y) {
    dim3 grid((M + BM - 1) / BM, b.N / BN, 1);
    if constexpr (OCC == 0) sweep_kernel<BM, BN, BK><<<grid, 1>>>(b.x, b.w, y, M, b.K, b.N);
    else sweep_kernel_occ<BM, BN, BK, OCC><<<grid, 1>>>(b.x, b.w, y, M, b.K, b.N);
    return cudaGetLastError();
}

template <int BM, int BN, int BK, int OCC>
static void run_cfg(const Buffers& b, const std::vector<bf16>& ref_pre) {
    const int M_PRE = 3072;
    char name[64];
    std::snprintf(name, sizeof name, "BM=%-3d BN=%-3d BK=%-3d occ=%-4s", BM, BN, BK, OCC ? std::to_string(OCC).c_str() : "def");
    cudaError_t e = launch<BM, BN, BK, OCC>(b, M_PRE, b.y);
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::printf("CFG %s FAIL %s\n", name, cudaGetErrorString(e));
        cudaGetLastError();  // clear
        return;
    }
    // correctness against the verified baseline
    std::vector<bf16> got((size_t)M_PRE * b.N);
    CK(cudaMemcpy(got.data(), b.y, got.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    double max_abs = 0, max_ref = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double g = __bfloat162float(got[i]), r = __bfloat162float(ref_pre[i]);
        max_abs = std::max(max_abs, std::fabs(g - r));
        max_ref = std::max(max_ref, std::fabs(r));
    }
    cudaEvent_t a, c; cudaEventCreate(&a); cudaEventCreate(&c);
    auto bench = [&](int M, int reps) {
        launch<BM, BN, BK, OCC>(b, M, b.y);
        cudaDeviceSynchronize();
        cudaEventRecord(a);
        for (int i = 0; i < reps; ++i) launch<BM, BN, BK, OCC>(b, M, b.y);
        cudaEventRecord(c);
        cudaEventSynchronize(c);
        float ms; cudaEventElapsedTime(&ms, a, c);
        return ms / reps;
    };
    float pre_ms = bench(M_PRE, 10), dec_ms = bench(1, 200);
    double tflops = 2.0 * M_PRE * b.N * b.K / (pre_ms * 1e-3) / 1e12;
    double gbps = (double)b.N * b.K * 2 / (dec_ms * 1e-3) / 1e9;
    std::printf("CFG %s ok  prefill %8.3f ms %7.1f TFLOP/s | decode %8.1f us %7.0f GB/s | max|diff|/max|ref| %.2e\n", name, pre_ms,
                tflops, dec_ms * 1e3, gbps, max_abs / max_ref);
    std::fflush(stdout);
}

int main() {
    const int M_PRE = 3072, K = 2560, N = 9728;   // Qwen3-4B gate/up projection
    std::vector<bf16> hx((size_t)M_PRE * K), hw((size_t)N * K);
    unsigned s = 7;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xFFFF) / 65536.0f - 0.5f; };
    for (auto& v : hx) v = __float2bfloat16(rnd());
    for (auto& v : hw) v = __float2bfloat16(0.1f * rnd());
    Buffers b{nullptr, nullptr, nullptr, nullptr, K, N};
    CK(cudaMalloc(&b.x, hx.size() * 2)); CK(cudaMalloc(&b.w, hw.size() * 2));
    CK(cudaMalloc(&b.y, (size_t)M_PRE * N * 2)); CK(cudaMalloc(&b.ref, (size_t)M_PRE * N * 2));
    CK(cudaMemcpy(b.x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));

    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    std::printf("device: %s, %d SMs, shared mem per block (opt-in) %zu KB, regs per block %d\n", p.name, p.multiProcessorCount,
                p.sharedMemPerBlockOptin / 1024, p.regsPerBlock);

    // baseline (verified) kernel -> reference output
    CK((launch_linear_tiled<bf16, 16, 32, 32>(b.x, b.w, b.ref, M_PRE, K, N)));
    CK(cudaDeviceSynchronize());
    std::vector<bf16> ref((size_t)M_PRE * N);
    CK(cudaMemcpy(ref.data(), b.ref, ref.size() * 2, cudaMemcpyDeviceToHost));

    // the current production tile first, for comparison
    run_cfg<16, 32, 32, 0>(b, ref);
#define SWEEP_ONE(BM, BN, BK)          \
    run_cfg<BM, BN, BK, 0>(b, ref);    \
    run_cfg<BM, BN, BK, 1>(b, ref);    \
    run_cfg<BM, BN, BK, 2>(b, ref);
    SWEEP_ONE(16, 64, 64)
    SWEEP_ONE(64, 64, 32)
    SWEEP_ONE(64, 64, 64)
    SWEEP_ONE(64, 128, 32)
    SWEEP_ONE(64, 128, 64)
    SWEEP_ONE(128, 64, 32)
    SWEEP_ONE(128, 64, 64)
    SWEEP_ONE(128, 128, 32)
    SWEEP_ONE(128, 128, 64)

    // ---- fused gate/up + SiLU (two accumulators), the production kernel, vs. its verified small-tile version ----------
    std::printf("\nfused gate_up_silu: out[3072, 9728] = silu(x@Wg^T) * (x@Wu^T), K=2560\n");
    bf16 *wu_d, *h_ref, *h_out;
    CK(cudaMalloc(&wu_d, hw.size() * 2));
    std::vector<bf16> hw2(hw.size());
    for (size_t i = 0; i < hw2.size(); ++i) hw2[i] = __float2bfloat16(__bfloat162float(hw[(i * 7 + 3) % hw.size()]));
    CK(cudaMemcpy(wu_d, hw2.data(), hw2.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&h_ref, (size_t)M_PRE * N * 2)); CK(cudaMalloc(&h_out, (size_t)M_PRE * N * 2));
    CK((launch_mlp_gate_up_tiled<bf16, 16, 32, 32>(b.x, b.w, wu_d, h_ref, M_PRE, K, N)));
    CK(cudaDeviceSynchronize());
    std::vector<bf16> hr((size_t)M_PRE * N);
    CK(cudaMemcpy(hr.data(), h_ref, hr.size() * 2, cudaMemcpyDeviceToHost));
    auto fused = [&](const char* name, auto launch_fn) {
        cudaError_t e = launch_fn(M_PRE);
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { std::printf("FUSED %s FAIL %s\n", name, cudaGetErrorString(e)); cudaGetLastError(); return; }
        std::vector<bf16> got((size_t)M_PRE * N);
        CK(cudaMemcpy(got.data(), h_out, got.size() * 2, cudaMemcpyDeviceToHost));
        double mx = 0, mr = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            double g = __bfloat162float(got[i]), r = __bfloat162float(hr[i]);
            mx = std::max(mx, std::fabs(g - r)); mr = std::max(mr, std::fabs(r));
        }
        cudaEvent_t a, c; cudaEventCreate(&a); cudaEventCreate(&c);
        launch_fn(M_PRE); cudaDeviceSynchronize();
        cudaEventRecord(a); for (int i = 0; i < 10; ++i) launch_fn(M_PRE); cudaEventRecord(c); cudaEventSynchronize(c);
        float pre; cudaEventElapsedTime(&pre, a, c); pre /= 10;
        launch_fn(1); cudaDeviceSynchronize();
        cudaEventRecord(a); for (int i = 0; i < 200; ++i) launch_fn(1); cudaEventRecord(c); cudaEventSynchronize(c);
        float dec; cudaEventElapsedTime(&dec, a, c); dec /= 200;
        std::printf("FUSED %-22s ok  prefill %8.3f ms %7.1f TFLOP/s | decode %8.1f us %7.0f GB/s | max|diff|/max|ref| %.2e\n", name,
                    pre, 4.0 * M_PRE * N * K / (pre * 1e-3) / 1e12, dec * 1e3, 2.0 * N * K * 2 / (dec * 1e-3) / 1e9, mx / mr);
        std::fflush(stdout);
    };
#define FUSED_CFG(BM, BN, BK) \
    fused("BM=" #BM " BN=" #BN " BK=" #BK, [&](int M) { return launch_mlp_gate_up_tiled<bf16, BM, BN, BK>(b.x, b.w, wu_d, h_out, M, K, N); });
    FUSED_CFG(16, 32, 32)
    FUSED_CFG(16, 64, 64)
    FUSED_CFG(64, 64, 32)
    FUSED_CFG(64, 64, 64)
    FUSED_CFG(64, 128, 64)
    FUSED_CFG(128, 64, 32)
    FUSED_CFG(128, 64, 64)
    FUSED_CFG(128, 128, 32)
    FUSED_CFG(128, 128, 64)
    return 0;
}
