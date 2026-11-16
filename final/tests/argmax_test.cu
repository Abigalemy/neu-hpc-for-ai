// argmax_test <float32|bfloat16> <vocab> <rows> logits.bin ids.bin
#include <cuda_bf16.h>
#include "common.h"
#include "../kernels/qwen3_argmax_tile.cuh"

template <typename T>
static int run(int vocab, int rows, char** a) {
    T* l = (T*)to_dev(read_file(a[0]));
    int* ids; CK(cudaMalloc(&ids, rows * sizeof(int)));
    CK((launch_argmax<T>(l, ids, rows, vocab)));
    CK(cudaDeviceSynchronize());
    std::vector<int> h(rows);
    CK(cudaMemcpy(h.data(), ids, rows * sizeof(int), cudaMemcpyDeviceToHost));
    write_file(a[1], h.data(), rows * sizeof(int));
    return 0;
}
int main(int argc, char** argv) {
    if (argc != 6) { std::fprintf(stderr, "bad usage\n"); return 1; }
    std::string dt = argv[1]; int vocab = std::atoi(argv[2]), rows = std::atoi(argv[3]);
    if (dt == "float32") return run<float>(vocab, rows, argv + 4);
    if (dt == "bfloat16") return run<__nv_bfloat16>(vocab, rows, argv + 4);
    return 3;
}
