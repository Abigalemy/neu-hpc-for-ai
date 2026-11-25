// Part 3: one real MoE layer of Qwen3-30B-A3B with expert parallelism over the visible GPUs (one thread per GPU, as
// in part2/a2a_mock.cu; NCCL between them), against the single-GPU layer of Part 1 on the SAME tokens.
// Rank r loads only its experts [floor(r*128/W), floor((r+1)*128/W)) with the engine's packer; GPU 0 also loads all
// 128 experts for the reference. Every case gives each rank its own tokens (data parallelism) and checks that each
// rank's EP output is bit-for-bit the single-GPU output, and how many rows each rank received.
//   ep_layer_test <model_dir> <layer>
#include <barrier>
#include <tuple>
#include <random>
#include <thread>
#include "../engine/qwen3_engine.cuh"

struct Case { const char* name; std::vector<int> n; bool hot; };

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: ep_layer_test <model_dir> <layer>\n"); return 1; }
    const std::string dir = argv[1];
    const int layer = std::atoi(argv[2]);
    int W = 0;
    CK(cudaGetDeviceCount(&W));
    if (W < 2) { std::fprintf(stderr, "needs at least 2 GPUs, found %d\n", W); return 2; }
    Qwen3Config cfg = load_config(dir);
    const int E = cfg.experts, K = cfg.top_k, H = cfg.hidden, I = cfg.moe_inter, N = 1024;
    std::vector<int> first = EpGroup::table(E, W);
    // tokens per rank; the hot cases use a router that sends every token to experts 0..7 (all on rank 0)
    std::vector<Case> cases = {
        {"one_token_each", std::vector<int>(W, 1), false},
        {"only_rank0_33", [&] { std::vector<int> v(W, 0); v[0] = 33; return v; }(), false},
        {"only_last_33", [&] { std::vector<int> v(W, 0); v[W - 1] = 33; return v; }(), false},
        {"uneven_5_vs_1000", [&] { std::vector<int> v(W, 1000); v[0] = 5; return v; }(), false},
        {"full_1024_each", std::vector<int>(W, N), false},
        {"hot_1024_each", std::vector<int>(W, N), true},
    };
    ncclUniqueId id;
    NCK(ncclGetUniqueId(&id));
    std::barrier sync(W);
    // inputs per (case, rank) and the reference outputs, computed by rank 0 on GPU 0
    std::vector<std::vector<std::vector<bf16>>> xin(cases.size(), std::vector<std::vector<bf16>>(W)), ref = xin, got = xin;
    for (size_t c = 0; c < cases.size(); ++c)
        for (int r = 0; r < W; ++r) {
            std::mt19937 rng(100 * c + r);
            std::normal_distribution<float> nd(0.f, 1.f);
            xin[c][r].resize((size_t)cases[c].n[r] * H);
            for (auto& v : xin[c][r]) v = __float2bfloat16(cases[c].hot ? std::fabs(nd(rng)) : nd(rng));
        }
    std::vector<std::vector<int>> recv_rows(cases.size(), std::vector<int>(W));
    const std::string pre = "model.layers." + std::to_string(layer) + ".mlp.";

    auto worker = [&](int r) {
        CK(cudaSetDevice(r));
        EpGroup g;
        g.rank = r; g.world = W; g.first = first;
        NCK(ncclCommInitRank(&g.comm, W, id, r));
        MoePack mine, full;
        mine.first_layer = layer; mine.alloc(1, g.e_local(), I, H, g.e0(), E);
        if (r == 0) { full.first_layer = layer; full.alloc(1, E, I, H); }
        auto w = load_weights_to_gpu(dir, [&](const std::string& n, const std::vector<int64_t>& sh, size_t b, void** dst) {
            if (n.rfind(pre + "experts.", 0) == 0) {
                Placement p = mine.place(n, sh, b, dst);
                if (r == 0 && p == Placement::SKIP) return full.place(n, sh, b, dst);   // rank 0's reference: the rest
                return p;
            }
            return n == pre + "gate.weight" ? Placement::DEFAULT : Placement::SKIP;
        });
        if (mine.placed != 3LL * g.e_local() || !mine.complete()) { std::fprintf(stderr, "rank %d: %lld expert tensors\n", r, mine.placed); std::exit(2); }
        const bf16* router = (const bf16*)w[pre + "gate.weight"].ptr;
        bf16* hot_router;                 // experts 0..7 get positive, decreasing logits for positive inputs; others 0
        CK(cudaMalloc(&hot_router, (size_t)E * H * sizeof(bf16)));
        {
            std::vector<bf16> h((size_t)E * H, __float2bfloat16(0.f));
            for (int e = 0; e < K; ++e) for (int k = 0; k < H; ++k) h[(size_t)e * H + k] = __float2bfloat16((K - e) * 1e-3f);
            CK(cudaMemcpy(hot_router, h.data(), h.size() * sizeof(bf16), cudaMemcpyHostToDevice));
        }
        auto al = [](size_t bytes) { void* p; CK(cudaMalloc(&p, std::max<size_t>(bytes, 4))); return p; };
        MoeWorkspace<bf16> ws;
        const size_t A = (size_t)N * K;
        ws.max_tokens = N;
        ws.idx = (int*)al(A * 4); ws.w = (bf16*)al(A * 2); ws.count = (int*)al(E * 4); ws.off = (int*)al(E * 4);
        ws.sorted = (int*)al(A * 4); ws.inv = (int*)al(A * 4); ws.hbuf = (bf16*)al(A * I * 2); ws.ysorted = (bf16*)al(A * H * 2);
        EpWorkspace<bf16> ep;
        size_t eb = 0;
        alloc_ep_workspace(ep, N, K, E, H, I, W, &eb);
        if (r == 0) {   // rank 0's pack + its reference pack = all 128 experts, each once
            if (full.placed != 3LL * (E - g.e_local()) || mine.placed + full.placed != 3LL * E) { std::fprintf(stderr, "reference pack incomplete\n"); std::exit(2); }
        }
        bf16 *dx = (bf16*)al((size_t)N * H * 2), *dy = (bf16*)al((size_t)N * H * 2);
        sync.arrive_and_wait();
        if (r == 0) {   // reference: every rank's tokens through the single-GPU layer. GPU 0 needs ALL 128 experts in one
                        // packed buffer: copy its own slice and the reference pack's slice into a full buffer.
            MoePack all; all.alloc(1, E, I, H);
            const size_t per = (size_t)I * H;
            for (auto [dst, a, b] : {std::tuple{all.g[0], mine.g[0], full.g[0]}, std::tuple{all.u[0], mine.u[0], full.u[0]},
                                     std::tuple{all.d[0], mine.d[0], full.d[0]}}) {
                CK(cudaMemcpy(dst, a, g.e_local() * per * 2, cudaMemcpyDeviceToDevice));               // experts 0..e_local-1
                CK(cudaMemcpy(dst + g.e_local() * per, b + g.e_local() * per, (E - g.e_local()) * per * 2,
                              cudaMemcpyDeviceToDevice));                                             // the rest
            }
            for (size_t c = 0; c < cases.size(); ++c)
                for (int q = 0; q < W; ++q) {
                    const int n = cases[c].n[q];
                    ref[c][q].resize((size_t)n * H);
                    if (!n) continue;
                    CK(cudaMemcpy(dx, xin[c][q].data(), (size_t)n * H * 2, cudaMemcpyHostToDevice));
                    CK((launch_moe<bf16, 128, 8>(dx, cases[c].hot ? hot_router : router, all.g[0], all.u[0], all.d[0], dy, n,
                                                 H, I, cfg.norm_topk, ws)));
                    CK(cudaMemcpy(ref[c][q].data(), dy, (size_t)n * H * 2, cudaMemcpyDeviceToHost));
                }
        }
        sync.arrive_and_wait();
        for (size_t c = 0; c < cases.size(); ++c) {
            const int n = cases[c].n[r];
            if (n) CK(cudaMemcpy(dx, xin[c][r].data(), (size_t)n * H * 2, cudaMemcpyHostToDevice));
            moe_ep_layer<bf16, 128, 8>(n ? dx : nullptr, cases[c].hot ? hot_router : router, mine.g[0], mine.u[0], mine.d[0],
                                       n ? dy : nullptr, n, H, I, cfg.norm_topk, ws, ep, g, 0, ncclBfloat16);
            CK(cudaDeviceSynchronize());
            got[c][r].resize((size_t)n * H);
            if (n) CK(cudaMemcpy(got[c][r].data(), dy, (size_t)n * H * 2, cudaMemcpyDeviceToHost));
            long long rc = 0;
            for (int q = 0; q < W; ++q) for (int e = g.e0(); e < g.e0() + g.e_local(); ++e) rc += ep.h_counts[(size_t)q * E + e];
            recv_rows[c][r] = (int)rc;
        }
        NCK(ncclCommDestroy(g.comm));
    };
    std::vector<std::thread> th;
    for (int r = 0; r < W; ++r) th.emplace_back(worker, r);
    for (auto& t : th) t.join();

    int failed = 0;
    for (size_t c = 0; c < cases.size(); ++c) {
        long long bad = 0, total = 0;
        float maxd = 0;
        for (int r = 0; r < W; ++r)
            for (size_t i = 0; i < ref[c][r].size(); ++i) {
                float a = __bfloat162float(ref[c][r][i]), b = __bfloat162float(got[c][r][i]);
                bad += std::memcmp(&ref[c][r][i], &got[c][r][i], sizeof(bf16)) != 0;
                maxd = std::max(maxd, std::fabs(a - b));
                ++total;
            }
        failed += bad != 0;
        std::printf("%-18s tokens", cases[c].name);
        for (int r = 0; r < W; ++r) std::printf(" %d", cases[c].n[r]);
        std::printf(" | rows received per rank");
        for (int r = 0; r < W; ++r) std::printf(" %d", recv_rows[c][r]);
        std::printf(" | %s (%lld of %lld values differ, max |diff| %g)\n", bad ? "DIFFERS" : "bit-exact", bad, total, maxd);
    }
    if (failed) std::printf("EP LAYER: %d case(s) differ from the single-GPU layer\n", failed);
    else std::printf("EP LAYER: all %zu cases bit-exact vs the single-GPU layer on %d GPUs\n", cases.size(), W);
    return failed != 0;
}
