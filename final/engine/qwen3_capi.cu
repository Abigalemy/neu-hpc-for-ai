// C interface of the engine, built as a shared library (libqwen3engine.so) and loaded by the Python HTTP server with
// ctypes. The engine has `max_seqs` slots, so up to max_seqs sequences are decoded together (continuous batching: the
// server decides which sequence occupies which slot). The KV cache is a pool of blocks of qe_block_size() tokens shared
// by all slots (PagedAttention): the server reserves blocks for a slot before it writes positions there, and releases
// them when the sequence ends or is preempted. Greedy decoding only.
//
//   void* qe_create(const char* model_dir, int max_len, int max_chunk, double yarn_factor, int max_seqs,
//                   long long kv_tokens)                             kv_tokens <= 0: max_seqs * max_len
//   int   qe_prefill(void* e, int slot, const int* tokens, int n)    -> first generated token; the prompt occupies
//                                                                       positions 0..n-1 of `slot`
//   void  qe_decode_batch(void* e, int B, const int* slots, const int* tokens, const int* positions, int* out)
//                                                                    -> out[b] = next token of the sequence in slots[b]
//                                                                       after feeding tokens[b] at positions[b]
//   int   qe_reserve(void* e, int slot, int n_tokens)                -> 1 if positions 0..n_tokens-1 of `slot` now have
//                                                                       blocks, 0 if not enough free blocks (no change)
//   void  qe_release(void* e, int slot)                              -> all blocks of `slot` back to the free list
//   int   qe_vocab(e), qe_max_len(e), qe_max_chunk(e), qe_max_seqs(e), qe_block_size(e), qe_num_blocks(e),
//         qe_free_blocks(e)
//
// The engine exits the process on invalid arguments, so the caller must check them first: 0 <= slot < max_seqs (all
// slots of a batch distinct), 1 <= n, n + max_new_tokens <= max_len, 1 <= B <= max_seqs, 0 <= token < vocab, and the
// positions written must be reserved (or the pool must have free blocks for them).
#include "qwen3_engine.cuh"

extern "C" {

void* qe_create(const char* model_dir, int max_len, int max_chunk, double yarn_factor, int max_seqs, long long kv_tokens) {
    return new Qwen3Engine(model_dir, max_len, max_chunk, 1, yarn_factor, max_seqs, kv_tokens);
}

// Expert parallelism (Part 3), one process per GPU: every rank calls qe_create_ep with the same id (from
// qe_nccl_unique_id in one process, passed to the others) and its rank; rank r holds experts
// [floor(r*128/world), floor((r+1)*128/world)). A rank without work must keep calling qe_idle_tick while any rank has
// work (every forward pass is a collective); it returns 0 once no rank has work.
int qe_nccl_id_bytes() { return (int)sizeof(ncclUniqueId); }
int qe_nccl_unique_id(unsigned char* out) {
    ncclUniqueId id;
    if (ncclGetUniqueId(&id) != ncclSuccess) return 0;
    std::memcpy(out, &id, sizeof id);
    return 1;
}
void* qe_create_ep(const char* model_dir, int max_len, int max_chunk, double yarn_factor, int max_seqs, long long kv_tokens,
                   int rank, int world, const unsigned char* nccl_id) {
    return new Qwen3Engine(model_dir, max_len, max_chunk, 1, yarn_factor, max_seqs, kv_tokens, rank, world, nccl_id);
}
int qe_idle_tick(void* e) { return static_cast<Qwen3Engine*>(e)->idle_tick(); }
int qe_world(void* e) { return static_cast<Qwen3Engine*>(e)->world(); }
int qe_rank(void* e) { return static_cast<Qwen3Engine*>(e)->rank(); }

int qe_vocab(void* e) { return static_cast<Qwen3Engine*>(e)->cfg.vocab; }
int qe_max_len(void* e) { return static_cast<Qwen3Engine*>(e)->max_len; }
int qe_max_chunk(void* e) { return static_cast<Qwen3Engine*>(e)->max_chunk; }
int qe_max_seqs(void* e) { return static_cast<Qwen3Engine*>(e)->max_seqs; }
int qe_block_size(void*) { return KV_BLOCK; }
int qe_num_blocks(void* e) { return static_cast<Qwen3Engine*>(e)->n_blocks; }
int qe_free_blocks(void* e) { return static_cast<Qwen3Engine*>(e)->free_blocks(); }
int qe_reserve(void* e, int slot, int n_tokens) { return static_cast<Qwen3Engine*>(e)->reserve(slot, n_tokens) ? 1 : 0; }
void qe_release(void* e, int slot) { static_cast<Qwen3Engine*>(e)->release(slot); }

// Prompts longer than max_chunk are prefilled in chunks; only the last chunk's last row produces logits.
int qe_prefill(void* e, int slot, const int* tokens, int n) {
    Qwen3Engine* eng = static_cast<Qwen3Engine*>(e);
    if (n == 1) return eng->argmax_row(eng->forward_decode(tokens, &slot, std::vector<int>{0}.data(), 1));
    const bf16* lg = nullptr;
    for (int start = 0; start < n; start += eng->max_chunk) {
        int len = n - start < eng->max_chunk ? n - start : eng->max_chunk;
        if (len == 1) {   // a 1-token tail chunk goes through the decode path
            int pos = start;
            lg = eng->forward_decode(tokens + start, &slot, &pos, 1);
        } else {
            lg = eng->forward(tokens + start, len, start, false, slot);
        }
    }
    return eng->argmax_row(lg);
}

void qe_decode_batch(void* e, int B, const int* slots, const int* tokens, const int* positions, int* out) {
    Qwen3Engine* eng = static_cast<Qwen3Engine*>(e);
    std::vector<int> ids = eng->argmax_rows(eng->forward_decode(tokens, slots, positions, B), B);
    for (int b = 0; b < B; ++b) out[b] = ids[b];
}

}  // extern "C"
