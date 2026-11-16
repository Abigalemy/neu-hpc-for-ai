// engine_bench <model_dir> <prompt_len> <gen_len> <repeats> <warmup_gen> [profile_decode_steps [logits_dump.bin]]
//
// Single-sequence timing of the engine, shaped like the vLLM/AIPerf baseline (concurrency 1, random prompt of exactly
// prompt_len tokens, exactly gen_len generated tokens, EOS ignored, greedy):
//   TTFT = time from "request start" until the first generated token is available (prefill + argmax)
//   ITL  = time between consecutive generated tokens (one decode step + argmax, host-synchronised)
// Engine only: no HTTP server, no tokenizer. Wall-clock timing around forward() + argmax, which synchronise the GPU.
#include <algorithm>
#include <chrono>
#include "../engine/qwen3_engine.cuh"

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

int main(int argc, char** argv) {
    if (argc < 6 || argc > 8) { std::fprintf(stderr, "bad usage\n"); return 1; }
    std::string dir = argv[1];
    int prompt_len = std::atoi(argv[2]), gen_len = std::atoi(argv[3]), repeats = std::atoi(argv[4]), warm = std::atoi(argv[5]);

    auto t0 = clk::now();
    int prof_steps = argc >= 7 ? std::atoi(argv[6]) : 0;
    Qwen3Engine eng(dir, prompt_len + std::max(gen_len, prof_steps) + 8, prompt_len, 1);   // room for the profiled decode steps too
    std::printf("config: hidden=%d inter=%d layers=%d heads=%d/%d | load %.1f s (weights copied to the GPU once)\n",
                eng.cfg.hidden, eng.cfg.inter, eng.cfg.layers, eng.cfg.hq, eng.cfg.hkv, ms_since(t0) / 1000.0);
    std::fflush(stdout);

    std::printf("tile profile: %s (QWEN3_TILES=%s)\n", tile_profile() == TileProfile::BIG ? "BIG" : "SMALL",
                std::getenv("QWEN3_TILES") ? std::getenv("QWEN3_TILES") : "unset");
    std::vector<int> prompt(prompt_len);
    unsigned s = 12345;
    for (auto& t : prompt) { s = s * 1664525u + 1013904223u; t = 1000 + (s >> 8) % 100000; }

    auto request = [&](int gen, std::vector<double>* steps) {
        auto ts = clk::now();
        const bf16* lg = eng.forward(prompt.data(), prompt_len, 0, false);
        int next = eng.argmax_row(lg);
        double ttft = ms_since(ts);
        for (int i = 1; i < gen; ++i) {
            auto t = clk::now();
            lg = eng.forward(&next, 1, prompt_len + i - 1, false);
            next = eng.argmax_row(lg);
            if (steps) steps->push_back(ms_since(t));
        }
        return ttft;
    };

    request(warm, nullptr);   // warm-up: first use of every kernel shape
    std::printf("warm-up done (%d tokens generated)\n", warm);
    std::fflush(stdout);

    double sum_ttft = 0, sum_itl = 0;
    for (int r = 0; r < repeats; ++r) {
        std::vector<double> steps;
        auto t = clk::now();
        double ttft = request(gen_len, &steps);
        double e2e = ms_since(t);
        std::vector<double> sorted = steps;
        std::sort(sorted.begin(), sorted.end());
        double mean = 0; for (double x : steps) mean += x; mean /= steps.size();
        std::printf("REQ %d: TTFT_MS %.2f  ITL_MEAN_MS %.3f  ITL_P50_MS %.3f  ITL_P99_MS %.3f  E2E_MS %.1f  (in %d, out %d)\n",
                    r, ttft, mean, sorted[sorted.size() / 2], sorted[(size_t)(sorted.size() * 0.99)], e2e, prompt_len, gen_len);
        std::fflush(stdout);
        sum_ttft += ttft; sum_itl += mean;
    }
    std::printf("AVERAGE over %d requests: TTFT_MS %.2f  ITL_MS %.3f\n", repeats, sum_ttft / repeats, sum_itl / repeats);

    // Optional per-stage profile: one prefill, then `steps` decode steps right after a prefill (context ~ prompt_len).
    int steps = argc >= 7 ? std::atoi(argv[6]) : 0;
    if (argc == 8) {   // last-position logits of the full prompt, for comparing tile configurations
        const bf16* lg = eng.forward(prompt.data(), prompt_len, 0, false);
        std::vector<bf16> row(eng.cfg.vocab);
        CK(cudaMemcpy(row.data(), lg, row.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
        write_file(argv[7], row.data(), row.size() * sizeof(bf16));
    }
    if (steps > 0) {
        eng.prof_reset();
        eng.profile = true;
        eng.forward(prompt.data(), prompt_len, 0, false);
        eng.profile = false;
        std::printf("\n%s\n", eng.prof_report("prefill, 1 request of the prompt length", 1).c_str());
        eng.forward(prompt.data(), prompt_len, 0, false);   // refill the cache, not profiled
        eng.prof_reset();
        eng.profile = true;
        int tok = prompt[0];
        for (int i = 0; i < steps; ++i) eng.forward(&tok, 1, prompt_len + i, false);
        eng.profile = false;
        std::printf("%s\n", eng.prof_report("decode step at context ~prompt_len (averaged)", steps).c_str());
    }
    return 0;
}
