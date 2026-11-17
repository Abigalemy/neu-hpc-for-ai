// attention_test <float32|bfloat16> <hq> <hkv> <n_q> <n_kv> q.bin k.bin v.bin o.bin
#include <cuda_bf16.h>
#include "common.h"
#include "../kernels/qwen3_flash_attention_tile.cuh"

template <typename T, int HQ, int HKV>
static int run(int n_q, int n_kv, char** a) {
    T* q = (T*)to_dev(read_file(a[0])); T* k = (T*)to_dev(read_file(a[1])); T* v = (T*)to_dev(read_file(a[2]));
    size_t bytes = (size_t)n_q * HQ * FA_D * sizeof(T);
    T* o; CK(cudaMalloc(&o, bytes));
    if (n_q == 1 && tile_profile() == TileProfile::BIG) {
        // the engine's decode path: batched split-KV kernel, here with one sequence in slot 0 of a cache of n_kv rows
        BatchDecodeWorkspace ws;
        ws.max_parts = FD_SMAX + 1;
        CK(cudaMalloc(&ws.o, (size_t)ws.max_parts * HQ * FD_BR * FA_D * sizeof(float)));
        CK(cudaMalloc(&ws.m, (size_t)ws.max_parts * HQ * FD_BR * sizeof(float)));
        CK(cudaMalloc(&ws.l, (size_t)ws.max_parts * HQ * FD_BR * sizeof(float)));
        int h_slot = 0, h_pos = n_kv - 1, *d_slot, *d_pos;
        CK(cudaMalloc(&d_slot, sizeof(int))); CK(cudaMalloc(&d_pos, sizeof(int)));
        CK(cudaMemcpy(d_slot, &h_slot, sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pos, &h_pos, sizeof(int), cudaMemcpyHostToDevice));
        CK((launch_flash_decode_batch<T, HQ, HKV>(q, k, v, o, 1, d_slot, d_pos, n_kv, n_kv, ws)));
    } else {
        CK((launch_flash_attention<T, HQ, HKV>(q, k, v, o, n_q, n_kv)));
    }
    CK(cudaDeviceSynchronize());
    std::vector<char> h(bytes);
    CK(cudaMemcpy(h.data(), o, bytes, cudaMemcpyDeviceToHost));
    write_file(a[3], h.data(), bytes);
    return 0;
}

template <typename T>
static int dispatch(int hq, int hkv, int n_q, int n_kv, char** a) {
    if (hq == 32 && hkv == 8) return run<T, 32, 8>(n_q, n_kv, a);   // Qwen3-4B
    if (hq == 32 && hkv == 4) return run<T, 32, 4>(n_q, n_kv, a);   // Qwen3-30B-A3B
    if (hq == 16 && hkv == 8) return run<T, 16, 8>(n_q, n_kv, a);   // Qwen3-0.6B
    std::fprintf(stderr, "unsupported heads %d/%d\n", hq, hkv); return 3;
}

int main(int argc, char** argv) {
    if (argc != 10) { std::fprintf(stderr, "bad usage\n"); return 1; }
    std::string dt = argv[1]; int hq = std::atoi(argv[2]), hkv = std::atoi(argv[3]);
    int n_q = std::atoi(argv[4]), n_kv = std::atoi(argv[5]);
    if (dt == "float32") return dispatch<float>(hq, hkv, n_q, n_kv, argv + 6);
    if (dt == "bfloat16") return dispatch<__nv_bfloat16>(hq, hkv, n_q, n_kv, argv + 6);
    return 3;
}
