// Test harness: mlp_test <float32|bfloat16> <hidden> <inter> <tokens> x.bin wg.bin wu.bin wd.bin y.bin
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../kernels/qwen3_mlp_tile.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

static std::vector<char> read_file(const char* p) {
    FILE* f = std::fopen(p, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p); std::exit(2); }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> b(n); if (std::fread(b.data(), 1, n, f) != (size_t)n) std::exit(2);
    std::fclose(f); return b;
}
static void* to_dev(const std::vector<char>& h) {
    void* d; CK(cudaMalloc(&d, h.size())); CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice)); return d;
}

template <typename T>
static void run(int hid, int inter, int tokens, char** a) {
    auto hx = read_file(a[0]);
    T* dx = (T*)to_dev(hx); T* wg = (T*)to_dev(read_file(a[1])); T* wu = (T*)to_dev(read_file(a[2]));
    T* wd = (T*)to_dev(read_file(a[3]));
    T *dh, *dy;
    CK(cudaMalloc(&dh, (size_t)tokens * inter * sizeof(T))); CK(cudaMalloc(&dy, hx.size()));
    CK((launch_mlp<T>(dx, wg, wu, wd, dh, dy, tokens, hid, inter)));
    CK(cudaDeviceSynchronize());
    std::vector<char> hy(hx.size());
    CK(cudaMemcpy(hy.data(), dy, hy.size(), cudaMemcpyDeviceToHost));
    FILE* f = std::fopen(a[4], "wb"); std::fwrite(hy.data(), 1, hy.size(), f); std::fclose(f);
}

template <typename T>
static int dispatch(int hid, int inter, int tokens, char** a) { run<T>(hid, inter, tokens, a); return 0; }

int main(int argc, char** argv) {
    if (argc != 10) { std::fprintf(stderr, "bad usage\n"); return 1; }
    std::string dt = argv[1]; int hid = std::atoi(argv[2]), inter = std::atoi(argv[3]), tokens = std::atoi(argv[4]);
    if (dt == "float32")  return dispatch<float>(hid, inter, tokens, argv + 5);
    if (dt == "bfloat16") return dispatch<__nv_bfloat16>(hid, inter, tokens, argv + 5);
    return 3;
}
