// Test harness: rmsnorm_test <float32|bfloat16> <D> <rows> <x.bin> <w.bin> <y.bin>
// Reads raw tensors written by run_tests.py, runs the Tile kernel, writes the output tensor.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../kernels/qwen3_rmsnorm_tile.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

static std::vector<char> read_file(const char* p) {
    FILE* f = std::fopen(p, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p); std::exit(2); }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> b(n); if (std::fread(b.data(), 1, n, f) != (size_t)n) std::exit(2);
    std::fclose(f); return b;
}

template <typename T, int BLOCK_D, int D>
static void run(int rows, const std::vector<char>& hx, const std::vector<char>& hw, const char* out_path) {
    T *dx, *dy, *dw;
    CK(cudaMalloc(&dx, hx.size())); CK(cudaMalloc(&dy, hx.size())); CK(cudaMalloc(&dw, hw.size()));
    CK(cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    CK((launch_rmsnorm<T, BLOCK_D, D>(dx, dy, dw, rows)));
    CK(cudaDeviceSynchronize());
    std::vector<char> hy(hx.size());
    CK(cudaMemcpy(hy.data(), dy, hy.size(), cudaMemcpyDeviceToHost));
    FILE* f = std::fopen(out_path, "wb"); std::fwrite(hy.data(), 1, hy.size(), f); std::fclose(f);
    cudaFree(dx); cudaFree(dy); cudaFree(dw);
}

template <typename T>
static int dispatch(int D, int rows, const std::vector<char>& x, const std::vector<char>& w, const char* out) {
    if (D == 128)  { run<T, 128, 128>(rows, x, w, out); return 0; }
    if (D == 2048) { run<T, 512, 2048>(rows, x, w, out); return 0; }
    if (D == 2560) { run<T, 512, 2560>(rows, x, w, out); return 0; }
    std::fprintf(stderr, "unsupported D=%d\n", D); return 3;
}

int main(int argc, char** argv) {
    if (argc != 7) { std::fprintf(stderr, "usage: %s dtype D rows x.bin w.bin y.bin\n", argv[0]); return 1; }
    std::string dt = argv[1]; int D = std::atoi(argv[2]), rows = std::atoi(argv[3]);
    auto x = read_file(argv[4]); auto w = read_file(argv[5]);
    if (dt == "float32")  return dispatch<float>(D, rows, x, w, argv[6]);
    if (dt == "bfloat16") return dispatch<__nv_bfloat16>(D, rows, x, w, argv[6]);
    std::fprintf(stderr, "unsupported dtype %s\n", dt.c_str()); return 3;
}
