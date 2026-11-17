// engine_test <model_dir> <tokens.bin> <n_total> <n_prompt> <logits_out.bin> <rope_out.bin> <gen_out.bin> <n_gen>
//
// 1. Teacher forcing: prefill the first n_prompt tokens in one chunk, then feed tokens n_prompt..n_total-1 one at a time
//    (the KV-cache decode path). Writes the logits of every position, [n_total, vocab] bf16.
// 2. Greedy generation of n_gen tokens from the prompt; prints TTFT and ITL.
// 3. Dumps the engine's RoPE tables.
#include <algorithm>
#include <chrono>
#include "../engine/qwen3_engine.cuh"

struct Timer {
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start() { cudaEventRecord(a); }
    float stop() { cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms; }
};

int main(int argc, char** argv) {
    if (argc != 9) { std::fprintf(stderr, "bad usage\n"); return 1; }
    std::string dir = argv[1];
    auto tok_bytes = read_file(argv[2]);
    int n_total = std::atoi(argv[3]), n_prompt = std::atoi(argv[4]), n_gen = std::atoi(argv[8]);
    const int* tokens = (const int*)tok_bytes.data();

    auto t0 = std::chrono::steady_clock::now();
    Qwen3Engine eng(dir, n_total + n_gen + 8, n_total + 8, n_total + 8);
    double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("config: hidden=%d inter=%d layers=%d heads=%d/%d vocab=%d tied=%d\n", eng.cfg.hidden, eng.cfg.inter,
                eng.cfg.layers, eng.cfg.hq, eng.cfg.hkv, eng.cfg.vocab, (int)eng.cfg.tie_embeddings);
    std::printf("LOAD_S %.2f (weights go to the GPU once, here)\n", load_s);

    const int V = eng.cfg.vocab;
    // Only the last `keep` positions are written: for long prompts the logits of all positions would be gigabytes.
    const int keep = std::min(n_total, 64), first_keep = n_total - keep;   // keep >= number of decode steps
    std::vector<bf16> all((size_t)keep * V);
    Timer tm;

    // 1) teacher forcing
    tm.start();
    const bf16* lg = eng.forward(tokens, n_prompt, 0, true);
    float prefill_ms = tm.stop();
    if (first_keep < n_prompt)
        CK(cudaMemcpy(all.data(), lg + (size_t)first_keep * V, (size_t)(n_prompt - first_keep) * V * sizeof(bf16), cudaMemcpyDeviceToHost));
    double dec_ms = 0;
    for (int t = n_prompt; t < n_total; ++t) {
        tm.start();
        lg = eng.forward(tokens + t, 1, t, false);
        dec_ms += tm.stop();
        CK(cudaMemcpy(all.data() + (size_t)(t - first_keep) * V, lg, (size_t)V * sizeof(bf16), cudaMemcpyDeviceToHost));
    }
    write_file(argv[5], all.data(), all.size() * sizeof(bf16));
    std::printf("LOGITS_KEPT rows=%d first_position=%d\n", keep, first_keep);
    std::printf("TEACHER_FORCED prefill(%d tokens) %.1f ms, decode steps %d, avg %.1f ms\n", n_prompt, prefill_ms,
                n_total - n_prompt, n_total > n_prompt ? dec_ms / (n_total - n_prompt) : 0.0);

    // 2) greedy generation
    std::vector<int> gen;
    tm.start();
    lg = eng.forward(tokens, n_prompt, 0, false);
    int next = eng.argmax_row(lg);
    float ttft = tm.stop();
    gen.push_back(next);
    double itl = 0;
    for (int i = 1; i < n_gen; ++i) {
        tm.start();
        lg = eng.forward(&next, 1, n_prompt + i - 1, false);
        next = eng.argmax_row(lg);
        itl += tm.stop();
        gen.push_back(next);
    }
    write_file(argv[7], gen.data(), gen.size() * sizeof(int));
    std::printf("GENERATE prompt=%d tokens: TTFT_MS %.1f  ITL_MS %.1f (avg over %d steps)\n", n_prompt, ttft,
                n_gen > 1 ? itl / (n_gen - 1) : 0.0, n_gen - 1);

    // 3) RoPE tables
    auto rt = eng.rope_tables_host();
    write_file(argv[6], rt.data(), rt.size() * sizeof(bf16));
    return 0;
}
