// linear_test <float32|bfloat16> <in> <out> <tokens> x.bin w.bin y.bin
#include <cuda_bf16.h>
#include "common.h"
#include "../kernels/qwen3_linear_tile.cuh"

template <typename T>
static int run(int in, int out, int tokens, char** a) {
    T* x = (T*)to_dev(read_file(a[0])); T* w = (T*)to_dev(read_file(a[1]));
    T* y; CK(cudaMalloc(&y, (size_t)tokens * out * sizeof(T)));
    CK((launch_linear<T>(x, w, y, tokens, in, out)));
    CK(cudaDeviceSynchronize());
    std::vector<char> hy((size_t)tokens * out * sizeof(T));
    CK(cudaMemcpy(hy.data(), y, hy.size(), cudaMemcpyDeviceToHost));
    write_file(a[2], hy.data(), hy.size());
    return 0;
}
int main(int argc, char** argv) {
    if (argc != 8) { std::fprintf(stderr, "bad usage\n"); return 1; }
    std::string dt = argv[1]; int in = std::atoi(argv[2]), out = std::atoi(argv[3]), tokens = std::atoi(argv[4]);
    if (dt == "float32") return run<float>(in, out, tokens, argv + 5);
    if (dt == "bfloat16") return run<__nv_bfloat16>(in, out, tokens, argv + 5);
    return 3;
}
