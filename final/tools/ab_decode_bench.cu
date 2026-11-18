// Engine-only timing used to compare two versions of the engine in ONE container (tools/ab_engine_bench.py builds this
// file once against each tree with -I<tree>/final). Uses only the API both versions have.
//
//   ab_decode_bench <model_dir> <label>
//
// For each case (B sequences, prompt L): prefill every sequence (timed, summed), then time `steps` batched decode
// steps (wall clock, each step synchronised by the argmax copy), and one profiled step set (per-stage CUDA events).
#include <algorithm>
#include <chrono>
#include <numeric>
#include "engine/qwen3_engine.cuh"

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: ab_decode_bench <model_dir> <label>\n"); return 1; }
    std::string dir = argv[1], label = argv[2];
    struct Case { int B, L, steps; bool prof; };
    const Case cases[] = {{1, 256, 128, false}, {1, 3072, 128, false}, {32, 64, 128, false}, {64, 256, 128, true}, {100, 1000, 64, false}};
    const int max_B = 100, max_len = 3072 + 256, max_chunk = 3072;
    Qwen3Engine eng(dir, max_len, max_chunk, 1, 0, max_B);
    std::vector<int> vocab_ids(max_len);
    for (int i = 0; i < max_len; ++i) vocab_ids[i] = 1000 + (i * 7919) % 50000;   // fixed pseudo-random tokens
    for (const Case& c : cases) {
        std::vector<int> slots(c.B), pos(c.B), toks(c.B);
        auto t0 = clk::now();
        for (int b = 0; b < c.B; ++b) {
            const bf16* lg = eng.forward(vocab_ids.data() + b % 7, c.L, 0, false, b);
            toks[b] = eng.argmax_row(lg);
            slots[b] = b;
            pos[b] = c.L;
        }
        double prefill_ms = ms_since(t0);
        auto step = [&] {
            std::vector<int> out = eng.argmax_rows(eng.forward_decode(toks.data(), slots.data(), pos.data(), c.B), c.B);
            for (int b = 0; b < c.B; ++b) { toks[b] = out[b]; ++pos[b]; }
        };
        for (int i = 0; i < 4; ++i) step();   // warm-up
        t0 = clk::now();
        for (int i = 0; i < c.steps; ++i) step();
        double step_ms = ms_since(t0) / c.steps;
        std::printf("AB %-6s B=%-3d L=%-4d prefill %8.2f ms per sequence | decode step %7.3f ms\n", label.c_str(), c.B, c.L,
                    prefill_ms / c.B, step_ms);
        std::fflush(stdout);
        if (c.prof) {
            eng.profile = true;
            eng.prof_reset();
            for (int i = 0; i < 16; ++i) step();
            eng.profile = false;
            std::printf("%s", eng.prof_report((label + " decode B=" + std::to_string(c.B)).c_str(), 16).c_str());
            std::fflush(stdout);
        }
    }
    return 0;
}
