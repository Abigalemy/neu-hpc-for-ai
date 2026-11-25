// moe_test router <E> <K> <hid> <n> x.bin wr.bin idx.bin w.bin
// moe_test block  <E> <K> <hid> <inter> <n> x.bin wr.bin wg.bin wu.bin wd.bin idx.bin w.bin y.bin
// moe_test debug  <same arguments as block> <dir>   as block, and also writes every intermediate to <dir>/ (count, off,
//                 sorted, inv, hbuf, ysorted) and <dir>/probe.bin, the gather_scatter_view index-semantics probe
// bf16 only (Qwen3-30B-A3B). wg, wu: [E, inter, hid]; wd: [E, hid, inter] (as stored by part1/gen_moe_cases.py).
#include <cuda_bf16.h>
#include <cstring>
#include <string>
#include "common.h"
#include "../kernels/qwen3_moe_tile.cuh"

using bf16 = __nv_bfloat16;

template <typename X> static X* alloc(size_t n) { X* p; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(X))); return p; }
template <typename X> static void save(const char* path, const X* d, size_t n) {
    std::vector<X> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(X), cudaMemcpyDeviceToHost));
    write_file(path, h.data(), n * sizeof(X));
}

// Probe: X[r, c] = r * 1000 + c (fp32, 8 x 256); gather rows {3, 1, 2, 0} with a <4, 128> tile at index 1 of the
// second dimension. Tile coordinate -> P[0, 0] = X[3, 128] = 3128; element offset -> X[3, 1] = 3001.
__tile_global__ void gs_probe(float* X, const int* IDX, float* P) {
    auto pI = ct::partition_view{ct::tensor_span{IDX, ct::extents{4}}, ct::shape<4>{}};
    auto gX = ct::gather_scatter_view{ct::tensor_span{X, ct::extents{8, 256}}, ct::shape<4, 128>{}, ct::integral_constant<0>{}};
    auto pP = ct::partition_view{ct::tensor_span{P, ct::extents{4, 128}}, ct::shape<4, 128>{}};
    pP.store(gX.load(pI.load(0), 1), 0, 0);
}

static void probe(const std::string& dir) {
    std::vector<float> x(8 * 256);
    for (int r = 0; r < 8; ++r) for (int c = 0; c < 256; ++c) x[r * 256 + c] = r * 1000.0f + c;
    std::vector<int> idx = {3, 1, 2, 0};
    float *dx, *dp; int* di;
    CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dp, 4 * 128 * 4)); CK(cudaMalloc(&di, 16));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(di, idx.data(), 16, cudaMemcpyHostToDevice));
    gs_probe<<<1, 1>>>(dx, di, dp);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    save((dir + "/probe.bin").c_str(), dp, 4 * 128);
}

template <int E, int K>
static int run(bool block, int hid, int inter, int n, char** f, const char* dbg = nullptr) {
    bf16* x = (bf16*)to_dev(read_file(f[0]));
    bf16* wr = (bf16*)to_dev(read_file(f[1]));
    MoeWorkspace<bf16> ws;
    ws.max_tokens = n;
    ws.idx = alloc<int>((size_t)n * K); ws.w = alloc<bf16>((size_t)n * K);
    if (!block) {
        CK((launch_moe_router<bf16, E, K>(x, wr, ws.idx, ws.w, n, hid, true)));
        CK(cudaDeviceSynchronize());
        save(f[2], ws.idx, (size_t)n * K); save(f[3], ws.w, (size_t)n * K);
        return 0;
    }
    bf16 *wg = (bf16*)to_dev(read_file(f[2])), *wu = (bf16*)to_dev(read_file(f[3])), *wd = (bf16*)to_dev(read_file(f[4]));
    const size_t A = (size_t)n * K;
    ws.count = alloc<int>(E); ws.off = alloc<int>(E); ws.sorted = alloc<int>(A); ws.inv = alloc<int>(A);
    ws.hbuf = alloc<bf16>(A * inter); ws.ysorted = alloc<bf16>(A * hid);
    bf16* y = alloc<bf16>((size_t)n * hid);
    CK((launch_moe<bf16, E, K>(x, wr, wg, wu, wd, y, n, hid, inter, true, ws)));
    CK(cudaDeviceSynchronize());
    save(f[5], ws.idx, A); save(f[6], ws.w, A); save(f[7], y, (size_t)n * hid);
    if (dbg) {
        std::string d = dbg;
        save((d + "/count.bin").c_str(), ws.count, E); save((d + "/off.bin").c_str(), ws.off, E);
        save((d + "/sorted.bin").c_str(), ws.sorted, A); save((d + "/inv.bin").c_str(), ws.inv, A);
        save((d + "/hbuf.bin").c_str(), ws.hbuf, A * inter); save((d + "/ysorted.bin").c_str(), ws.ysorted, A * hid);
        probe(d);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: see the file comment\n"); return 1; }
    bool debug = std::strcmp(argv[1], "debug") == 0;
    bool block = debug || std::strcmp(argv[1], "block") == 0;
    if ((debug && argc != 16) || (block && !debug && argc != 15) || (!block && argc != 10)) {
        std::fprintf(stderr, "bad arguments\n");
        return 1;
    }
    const char* dbg = debug ? argv[15] : nullptr;
    int E = std::atoi(argv[2]), K = std::atoi(argv[3]), hid = std::atoi(argv[4]);
    int inter = block ? std::atoi(argv[5]) : 0, n = std::atoi(argv[block ? 6 : 5]);
    char** f = argv + (block ? 7 : 6);
    if (K != 8) { std::fprintf(stderr, "only top-8 is compiled\n"); return 3; }
    if (E == 128) return run<128, 8>(block, hid, inter, n, f, dbg);
    if (E == 16) return run<16, 8>(block, hid, inter, n, f, dbg);
    std::fprintf(stderr, "unsupported number of experts %d\n", E);
    return 3;
}
