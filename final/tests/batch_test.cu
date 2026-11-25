// batch_test <model_dir> <steps> <len_0> <len_1> ...
//
// Batched decode must give the same logits as decoding each sequence alone. For every sequence (random tokens of the
// given length): (1) alone in slot 0: prefill, then `steps` greedy decode steps (batch of one), keeping each step's
// logits and tokens; (2) all sequences together, sequence i in slot i: prefill each, then `steps` decode steps with all
// of them in one batch, feeding the tokens from (1) (teacher forcing, so both runs see identical inputs). Each step's
// logits row of (2) is compared with (1): cosine, max |difference|, top-1.
// With QWEN3_DECODE_SPLITS=<n> both runs sum every sequence in the same groups, so the batched logits must be
// bit-identical to the alone ones: exit code 0 iff all rows are. Without it the split count depends on the batch size
// (different fp32 summation order), so the run is informational: it reports the drift and fails only on a top-1 change.
#include <algorithm>
#include <cmath>
#include "../engine/qwen3_engine.cuh"

static std::vector<float> to_float(const bf16* dev, int n) {
    std::vector<bf16> h(n);
    CK(cudaMemcpy(h.data(), dev, n * sizeof(bf16), cudaMemcpyDeviceToHost));
    std::vector<float> f(n);
    for (int i = 0; i < n; ++i) f[i] = __bfloat162float(h[i]);
    return f;
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s model_dir steps len...\n", argv[0]); return 1; }
    std::string dir = argv[1];
    int steps = std::atoi(argv[2]);
    std::vector<int> lens;
    for (int i = 3; i < argc; ++i) lens.push_back(std::atoi(argv[i]));
    const int B = (int)lens.size();
    int max_prompt = *std::max_element(lens.begin(), lens.end());

    Qwen3Engine eng(dir, max_prompt + steps + 8, max_prompt, 1, 0, B);
    const int V = eng.cfg.vocab;
    std::printf("model hidden=%d layers=%d heads=%d/%d | %d sequences, %d decode steps, tile profile %s\n", eng.cfg.hidden,
                eng.cfg.layers, eng.cfg.hq, eng.cfg.hkv, B, steps, tile_profile() == TileProfile::BIG ? "BIG" : "SMALL");

    std::vector<std::vector<int>> prompts(B);
    unsigned s = 99;
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < lens[b]; ++i) { s = s * 1664525u + 1013904223u; prompts[b].push_back(1000 + (s >> 8) % 100000); }

    // (1) each sequence alone, slot 0
    std::vector<std::vector<int>> toks(B);                         // token fed at each decode step
    std::vector<std::vector<std::vector<float>>> ref(B);           // logits after each decode step
    for (int b = 0; b < B; ++b) {
        const bf16* lg = eng.forward(prompts[b].data(), lens[b], 0, false, 0);
        int t = eng.argmax_row(lg);
        for (int k = 0; k < steps; ++k) {
            int pos = lens[b] + k, slot = 0;
            toks[b].push_back(t);
            lg = eng.forward_decode(&t, &slot, &pos, 1);
            ref[b].push_back(to_float(lg, V));
            t = eng.argmax_row(lg);
        }
    }

    // (2) all together, sequence b in slot b
    for (int b = 0; b < B; ++b) eng.forward(prompts[b].data(), lens[b], 0, false, b);
    double worst_cos = 1.0, worst_diff = 0.0;
    int top1_mismatch = 0;
    std::vector<double> seq_cos(B, 1.0);
    std::vector<int> seq_worst_step(B, 0);
    long long bit_identical_rows = 0;
    for (int k = 0; k < steps; ++k) {
        std::vector<int> slots(B), pos(B), in(B);
        for (int b = 0; b < B; ++b) { slots[b] = b; pos[b] = lens[b] + k; in[b] = toks[b][k]; }
        const bf16* lg = eng.forward_decode(in.data(), slots.data(), pos.data(), B);
        std::vector<float> all = to_float(lg, B * V);
        for (int b = 0; b < B; ++b) {
            const float* x = all.data() + (size_t)b * V;
            const std::vector<float>& r = ref[b][k];
            double dot = 0, nx = 0, nr = 0, md = 0;
            int ax = 0, ar = 0;
            for (int i = 0; i < V; ++i) {
                dot += (double)x[i] * r[i]; nx += (double)x[i] * x[i]; nr += (double)r[i] * r[i];
                md = std::max(md, (double)std::fabs(x[i] - r[i]));
                if (x[i] > x[ax]) ax = i;
                if (r[i] > r[ar]) ar = i;
            }
            double cos = dot / std::sqrt(nx * nr);
            worst_cos = std::min(worst_cos, cos);
            if (cos < seq_cos[b]) { seq_cos[b] = cos; seq_worst_step[b] = k; }
            bit_identical_rows += md == 0.0;
            worst_diff = std::max(worst_diff, md);
            if (ax != ar) {
                ++top1_mismatch;
                std::printf("  top-1 differs: sequence %d (prompt %d) step %d: batched %d, alone %d (alone margin %.3f)\n", b,
                            lens[b], k, ax, ar, r[ar] - r[ax]);
            }
        }
    }
    for (int b = 0; b < B; ++b)
        std::printf("  sequence %d (prompt %d): min cosine %.7f at step %d\n", b, lens[b], seq_cos[b], seq_worst_step[b]);
    const char* forced = std::getenv("QWEN3_DECODE_SPLITS");
    std::printf("BATCHED vs ALONE (QWEN3_DECODE_SPLITS=%s): %d sequences x %d steps: min cosine %.7f, max |logit diff| %.4f, "
                "top-1 mismatches %d, bit-identical rows %lld/%d\n", forced ? forced : "unset", B, steps, worst_cos, worst_diff,
                top1_mismatch, bit_identical_rows, B * steps);
    bool ok = forced ? bit_identical_rows == (long long)B * steps : top1_mismatch == 0;
    std::printf("%s (%s)\n", ok ? "PASS" : "FAIL", forced ? "criterion: every row bit-identical"
                                                         : "informational; criterion: no top-1 change");
    return ok ? 0 : 1;
}
