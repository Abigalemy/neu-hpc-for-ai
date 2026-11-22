// Part 4, C1: where the time of a Qwen3-30B-A3B forward pass goes, stage by stage (the engine's profile: CUDA events
// around every stage, the MoE split into router / sort / gate_up / down / combine, or with expert parallelism into its
// ep_* steps; plus the host clock, so time in no stage (host work, waits, the per-layer count sync) shows too).
//
//   moe_profile <model_dir> <prefill_lens, e.g. 256,2048,3072> <decode_batches, e.g. 1,8,32,64> <context> <steps>
//
// Prefill: one sequence of each length (warm-up pass first), TTFT's forward. Decode: B sequences, each prefilled with
// `context` tokens, then `steps` profiled decode steps of all B together (ITL's forward at batch B).
// Expert parallelism: QWEN3_WORLD=K QWEN3_RANK=r QWEN3_NCCL_ID=<hex> as engine_test; every rank runs the same plan, so
// the ranks run the same passes (each rank's report is its own view).
#include <chrono>
#include <sstream>
#include "../engine/qwen3_engine.cuh"

static std::vector<int> ints(const char* s) {
    std::vector<int> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');) if (!t.empty()) v.push_back(std::atoi(t.c_str()));
    return v;
}

int main(int argc, char** argv) {
    if (argc != 6) { std::fprintf(stderr, "usage: moe_profile <model_dir> <prefill_lens> <decode_batches> <context> <steps>\n"); return 1; }
    const std::string dir = argv[1];
    const std::vector<int> lens = ints(argv[2]), batches = ints(argv[3]);
    const int ctx = std::atoi(argv[4]), steps = std::atoi(argv[5]);
    const int world = std::getenv("QWEN3_WORLD") ? std::atoi(std::getenv("QWEN3_WORLD")) : 1;
    const int rank = std::getenv("QWEN3_RANK") ? std::atoi(std::getenv("QWEN3_RANK")) : 0;
    std::vector<unsigned char> nccl_id;
    if (world > 1) {
        const char* hex = std::getenv("QWEN3_NCCL_ID");
        for (size_t i = 0; hex && hex[i] && hex[i + 1]; i += 2) nccl_id.push_back((unsigned char)std::strtol(std::string(hex + i, 2).c_str(), nullptr, 16));
        if (nccl_id.size() != sizeof(ncclUniqueId)) { std::fprintf(stderr, "QWEN3_NCCL_ID: %zu bytes\n", nccl_id.size()); return 1; }
    }
    const int max_len = std::max(*std::max_element(lens.begin(), lens.end()), ctx + steps + 8) + 8;
    const int max_chunk = std::max(*std::max_element(lens.begin(), lens.end()), ctx);
    const int max_b = *std::max_element(batches.begin(), batches.end());
    auto t0 = std::chrono::steady_clock::now();
    Qwen3Engine eng(dir, max_len, max_chunk, 1, 0, max_b, 0, rank, world, world > 1 ? nccl_id.data() : nullptr);
    std::printf("rank %d of %d: loaded in %.1f s; max_len %d, max_chunk %d, %d slots\n", rank, world,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), max_len, max_chunk, max_b);
    std::fflush(stdout);

    std::vector<int> prompt(std::max(max_chunk, ctx));
    unsigned s = 12345 + 7919u * rank;                // each rank its own tokens (data parallelism)
    for (auto& t : prompt) { s = s * 1664525u + 1013904223u; t = 1000 + (s >> 8) % 100000; }
    char title[160];

    for (int L : lens) {
        eng.forward(prompt.data(), L, 0, false, 0);   // warm-up of this shape
        eng.release(0);
        eng.prof_reset();
        eng.profile = true;
        eng.forward(prompt.data(), L, 0, false, 0);
        eng.profile = false;
        eng.release(0);
        std::snprintf(title, sizeof title, "rank %d/%d prefill %d tokens", rank, world, L);
        std::printf("%s\n", eng.prof_report(title, 1).c_str());
        std::fflush(stdout);
    }
    for (int B : batches) {
        std::vector<int> slots(B), toks(B), pos(B);
        for (int b = 0; b < B; ++b) {
            slots[b] = b;
            toks[b] = eng.argmax_row(eng.forward(prompt.data(), ctx, 0, false, b));
            pos[b] = ctx;
        }
        auto step = [&] {
            std::vector<int> next = eng.argmax_rows(eng.forward_decode(toks.data(), slots.data(), pos.data(), B), B);
            toks = next;
            for (auto& p : pos) ++p;
        };
        for (int i = 0; i < 3; ++i) step();           // warm-up
        eng.prof_reset();
        eng.profile = true;
        for (int i = 0; i < steps; ++i) step();
        eng.profile = false;
        for (int b = 0; b < B; ++b) eng.release(b);
        std::snprintf(title, sizeof title, "rank %d/%d decode batch %d, context ~%d", rank, world, B, ctx);
        std::printf("%s\n", eng.prof_report(title, steps).c_str());
        std::fflush(stdout);
    }
    std::printf("rank %d: PROFILE DONE\n", rank);
    return 0;
}
