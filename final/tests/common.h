// Small helpers shared by the test harnesses: check macro, raw-file I/O, host->device copy.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

static std::vector<char> read_file(const char* p) {
    FILE* f = std::fopen(p, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p); std::exit(2); }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> b(n); if (std::fread(b.data(), 1, n, f) != (size_t)n) std::exit(2);
    std::fclose(f); return b;
}
static void write_file(const char* p, const void* data, size_t n) {
    FILE* f = std::fopen(p, "wb"); std::fwrite(data, 1, n, f); std::fclose(f);
}
static void* to_dev(const std::vector<char>& h) {
    void* d; CK(cudaMalloc(&d, h.size())); CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice)); return d;
}
