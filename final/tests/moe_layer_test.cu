// One real MoE layer of Qwen3-30B-A3B on the GPU: loads layer L's router and its 128 experts from the checkpoint with
// the engine's own packer (MoePack, the same code the engine uses), runs the MoE block on random inputs and writes
// everything tools/moe_layer_check.py needs to compare with numpy. Also times the block.
//   moe_layer_test <model_dir> <layer> <tokens> <out_dir>
#include <chrono>
#include <random>
#include "../engine/qwen3_engine.cuh"

int main(int argc, char** argv) {
    if (argc != 5) { std::fprintf(stderr, "usage: moe_layer_test <model_dir> <layer> <tokens> <out_dir>\n"); return 1; }
    std::string dir = argv[1], out = argv[4];
    const int layer = std::atoi(argv[2]), T = std::atoi(argv[3]);
    Qwen3Config cfg = load_config(dir);
    if (cfg.experts != 128 || cfg.top_k != 8) { std::fprintf(stderr, "not a 128-expert top-8 model\n"); return 2; }
    const int E = cfg.experts, K = cfg.top_k, H = cfg.hidden, I = cfg.moe_inter;
    MoePack pack;
    pack.first_layer = layer;
    pack.alloc(1, E, I, H);
    const std::string pre = "model.layers." + std::to_string(layer) + ".mlp.";
    auto t0 = std::chrono::steady_clock::now();
    auto w = load_weights_to_gpu(dir, [&](const std::string& n, const std::vector<int64_t>& sh, size_t b, void** dst) {
        if (n.rfind(pre + "experts.", 0) == 0) return pack.place(n, sh, b, dst);
        return n == pre + "gate.weight" ? Placement::DEFAULT : Placement::SKIP;
    });
    double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (pack.placed != E * 3 || !pack.complete() || !w.count(pre + "gate.weight")) { std::fprintf(stderr, "layer %d: %lld expert tensors\n", layer, pack.placed); return 2; }
    std::printf("layer %d: router + %lld expert tensors loaded in %.1f s\n", layer, pack.placed, load_s);

    std::mt19937 rng(1000 + T);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<bf16> x((size_t)T * H);
    for (auto& v : x) v = __float2bfloat16(nd(rng));
    bf16 *dx, *dy;
    CK(cudaMalloc(&dx, x.size() * 2)); CK(cudaMalloc(&dy, x.size() * 2));
    CK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    MoeWorkspace<bf16> ws;
    const size_t A = (size_t)T * K;
    ws.max_tokens = T;
    auto al = [](size_t bytes) { void* p; CK(cudaMalloc(&p, std::max<size_t>(bytes, 4))); return p; };
    ws.idx = (int*)al(A * 4); ws.w = (bf16*)al(A * 2); ws.count = (int*)al(E * 4); ws.off = (int*)al(E * 4);
    ws.sorted = (int*)al(A * 4); ws.inv = (int*)al(A * 4); ws.hbuf = (bf16*)al(A * I * 2); ws.ysorted = (bf16*)al(A * H * 2);
    const bf16* router = (const bf16*)w[pre + "gate.weight"].ptr;
    auto run = [&] { CK((launch_moe<bf16, 128, 8>(dx, router, pack.g[0], pack.u[0], pack.d[0], dy, T, H, I, cfg.norm_topk, ws))); };
    run();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int i = 0; i < 20; ++i) run();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    std::printf("MOE_LAYER layer=%d tokens=%d: %.3f ms per call (x48 layers = %.2f ms)\n", layer, T, ms / 20, ms / 20 * 48);
    auto save = [&](const char* name, const void* d, size_t bytes) {
        std::vector<char> h(bytes);
        CK(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost));
        write_file((out + "/" + name).c_str(), h.data(), bytes);
    };
    save("x.bin", dx, x.size() * 2); save("y.bin", dy, x.size() * 2);
    save("idx.bin", ws.idx, A * 4); save("w.bin", ws.w, A * 2);
    return 0;
}
