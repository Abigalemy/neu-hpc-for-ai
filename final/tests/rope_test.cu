// Test harness: rope_test <float32|bfloat16> <num_heads> <num_kv_heads> <tokens> <max_pos>
//                         q.bin k.bin cos.bin sin.bin pos.bin q_out.bin k_out.bin
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../kernels/qwen3_rope_tile.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

static std::vector<char> read_file(const char* p) {
    FILE* f = std::fopen(p, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p); std::exit(2); }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> b(n); if (std::fread(b.data(), 1, n, f) != (size_t)n) std::exit(2);
    std::fclose(f); return b;
}
static void write_file(const char* p, const std::vector<char>& b) {
    FILE* f = std::fopen(p, "wb"); std::fwrite(b.data(), 1, b.size(), f); std::fclose(f);
}
static void* to_dev(const std::vector<char>& h) {
    void* d; CK(cudaMalloc(&d, h.size())); CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice)); return d;
}

template <typename T, int NQ, int NK>
static void run(int tokens, int max_pos, char** a) {
    auto hq = read_file(a[0]), hk = read_file(a[1]), hc = read_file(a[2]), hs = read_file(a[3]), hp = read_file(a[4]);
    T* dq = (T*)to_dev(hq); T* dk = (T*)to_dev(hk);
    T* dc = (T*)to_dev(hc); T* ds = (T*)to_dev(hs); int* dp = (int*)to_dev(hp);
    CK((launch_rope<T, NQ, NK>(dq, dk, dc, ds, dp, tokens, max_pos)));
    CK(cudaDeviceSynchronize());
    std::vector<char> oq(hq.size()), ok(hk.size());
    CK(cudaMemcpy(oq.data(), dq, oq.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ok.data(), dk, ok.size(), cudaMemcpyDeviceToHost));
    write_file(a[5], oq); write_file(a[6], ok);
}

template <typename T>
static int dispatch(int nq, int nk, int tokens, int max_pos, char** a) {
    if (nq == 32 && nk == 8) { run<T, 32, 8>(tokens, max_pos, a); return 0; }
    if (nq == 32 && nk == 4) { run<T, 32, 4>(tokens, max_pos, a); return 0; }
    std::fprintf(stderr, "unsupported heads %d/%d\n", nq, nk); return 3;
}

int main(int argc, char** argv) {
    if (argc != 13) { std::fprintf(stderr, "bad usage\n"); return 1; }
    std::string dt = argv[1]; int nq = std::atoi(argv[2]), nk = std::atoi(argv[3]);
    int tokens = std::atoi(argv[4]), max_pos = std::atoi(argv[5]);
    if (dt == "float32")  return dispatch<float>(nq, nk, tokens, max_pos, argv + 6);
    if (dt == "bfloat16") return dispatch<__nv_bfloat16>(nq, nk, tokens, max_pos, argv + 6);
    return 3;
}
