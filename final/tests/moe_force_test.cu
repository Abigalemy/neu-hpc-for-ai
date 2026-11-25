// Part 1 check: the router with forced experts (QWEN3_MOE_FORCE) against the router choosing them itself.
//   1. forced = the router's own choice: indices and weights bit-identical (same probabilities, same order, same
//      renormalisation), so forcing changes nothing but the choice
//   2. forced = another set of K distinct experts per token: the indices are the given ones, and the weights are the
//      router's own probabilities of those experts, renormalised: checked against a host softmax of the logits
//      (computed on the host from the same bf16 inputs, rounded to bf16 as the kernel does), to 2e-2 relative (a logit one bf16 step apart moves its probability by up to ~1%, the bf16 weight ~0.4%)
// 128 experts, top-8, as Qwen3-30B-A3B; small hidden size; token counts across the 16-row tile edge.
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include "common.h"
#include "../kernels/qwen3_moe_tile.cuh"

using bf16 = __nv_bfloat16;
constexpr int E = 128, K = 8, HID = 256;

template <typename V>
static V* dev(size_t n) { V* p; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(V))); return p; }
template <typename V>
static std::vector<V> host(const V* d, size_t n) { std::vector<V> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(V), cudaMemcpyDeviceToHost)); return h; }

int main() {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    int failed = 0;
    for (int n : {1, 5, 16, 17, 300}) {
        std::vector<bf16> hx((size_t)n * HID), hw((size_t)E * HID);
        for (auto& v : hx) v = __float2bfloat16(nd(rng));
        for (auto& v : hw) v = __float2bfloat16(0.1f * nd(rng));
        bf16 *x = dev<bf16>(hx.size()), *wr = dev<bf16>(hw.size()), *w0 = dev<bf16>((size_t)n * K), *w1 = dev<bf16>((size_t)n * K);
        int *i0 = dev<int>((size_t)n * K), *i1 = dev<int>((size_t)n * K), *f = dev<int>((size_t)n * K);
        CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(wr, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
        CK((launch_moe_router<bf16, E, K>(x, wr, i0, w0, n, HID, true)));
        // 1. forced = own choice
        CK((launch_moe_router<bf16, E, K>(x, wr, i1, w1, n, HID, true, 0, i0)));
        CK(cudaDeviceSynchronize());
        auto a = host(i0, (size_t)n * K), b = host(i1, (size_t)n * K);
        auto wa = host(w0, (size_t)n * K), wb = host(w1, (size_t)n * K);
        const bool same = a == b && std::memcmp(wa.data(), wb.data(), wa.size() * 2) == 0;
        // 2. forced = random distinct experts
        std::vector<int> forced((size_t)n * K), perm(E);
        for (int t = 0; t < n; ++t) {
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), rng);
            std::copy_n(perm.begin(), K, &forced[(size_t)t * K]);
        }
        CK(cudaMemcpy(f, forced.data(), forced.size() * 4, cudaMemcpyHostToDevice));
        CK((launch_moe_router<bf16, E, K>(x, wr, i1, w1, n, HID, true, 0, f)));
        CK(cudaDeviceSynchronize());
        auto c = host(i1, (size_t)n * K);
        auto wc = host(w1, (size_t)n * K);
        double worst = 0;
        for (int t = 0; t < n; ++t) {
            std::vector<double> lg(E);
            for (int e = 0; e < E; ++e) {
                double s = 0;
                for (int h = 0; h < HID; ++h) s += (double)__bfloat162float(hx[(size_t)t * HID + h]) * __bfloat162float(hw[(size_t)e * HID + h]);
                lg[e] = __bfloat162float(__float2bfloat16((float)s));
            }
            const double mx = *std::max_element(lg.begin(), lg.end());
            double z = 0, zs = 0;
            for (double v : lg) z += std::exp(v - mx);
            for (int j = 0; j < K; ++j) zs += std::exp(lg[forced[(size_t)t * K + j]] - mx) / z;
            for (int j = 0; j < K; ++j) {
                const double want = std::exp(lg[forced[(size_t)t * K + j]] - mx) / z / zs;
                worst = std::max(worst, std::fabs(__bfloat162float(wc[(size_t)t * K + j]) - want) / want);
            }
        }
        const bool ok2 = c == forced && worst < 2e-2;
        std::printf("n %3d: forced = own choice %s; forced = random experts: indices %s, weights worst rel err %.2e %s\n", n,
                    same ? "bit-identical" : "DIFFERS", c == forced ? "as given" : "WRONG", worst, ok2 ? "ok" : "FAIL");
        failed += !same || !ok2;
    }
    std::printf(failed ? "MOE FORCE: %d case(s) FAILED\n" : "MOE FORCE: all cases pass\n", failed);
    return failed != 0;
}
