// embedding_test <float32|bfloat16> <hid> <vocab> <tokens> ids.bin w.bin y.bin
#include <cuda_bf16.h>
#include "common.h"
#include "../kernels/qwen3_embedding_tile.cuh"

template <typename T>
static int run(int hid, int vocab, int tokens, char** a) {
    auto hw = read_file(a[1]);
    int* ids = (int*)to_dev(read_file(a[0])); T* w = (T*)to_dev(hw);
    T* y; CK(cudaMalloc(&y, (size_t)tokens * hid * sizeof(T)));
    CK((launch_embedding<T>(ids, w, y, vocab, tokens, hid)));
    CK(cudaDeviceSynchronize());
    std::vector<char> hy((size_t)tokens * hid * sizeof(T));
    CK(cudaMemcpy(hy.data(), y, hy.size(), cudaMemcpyDeviceToHost));
    write_file(a[2], hy.data(), hy.size());
    return 0;
}
int main(int argc, char** argv) {
    if (argc != 8) { std::fprintf(stderr, "bad usage\n"); return 1; }
    std::string dt = argv[1]; int hid = std::atoi(argv[2]), vocab = std::atoi(argv[3]), tokens = std::atoi(argv[4]);
    if (dt == "float32") return run<float>(hid, vocab, tokens, argv + 5);
    if (dt == "bfloat16") return run<__nv_bfloat16>(hid, vocab, tokens, argv + 5);
    return 3;
}
