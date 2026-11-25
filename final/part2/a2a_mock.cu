// Part 2: all-to-all dispatch and combine of an expert-parallel MoE layer over two GPUs, with the tokens, the routing
// and the experts mocked as in the Week 8 figure (see a2a_plan.h). One host thread per GPU plays one worker (one MPI
// rank in the Week 5 "MPI with CUDA" notebook); NCCL moves the tokens between the GPUs.
//   permutation / un-permutation: CUDA Tile gather_scatter_view (gather rows by index, scatter rows by index)
//   dispatch / combine:           NCCL all-to-all as grouped ncclSend / ncclRecv (counts differ per peer)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <cuda_runtime.h>
#include <nccl.h>
#include "cuda_tile.h"
#include "a2a_plan.h"

namespace ct = cuda::tiles;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("CUDA %s at line %d\n", cudaGetErrorString(e), __LINE__); std::exit(2); } } while (0)
#define NK(x) do { ncclResult_t e = (x); if (e != ncclSuccess) { std::printf("NCCL %s at line %d\n", ncclGetErrorString(e), __LINE__); std::exit(2); } } while (0)

constexpr int BR = 8;                                // rows per tile
using idx_t = ct::tile<int, ct::shape<BR>>;

// row indices of tile b; rows past n get an out-of-range index (n_rows) so the masked gather / scatter skips them
__tile__ idx_t row_indices(const int* IDX, int n, int n_rows, int b) {
    auto pI = ct::partition_view{ct::tensor_span{IDX, ct::extents{n}}, ct::shape<BR>{}};
    return ct::select(b * BR + ct::iota<idx_t>() < n, pI.load_masked(b), ct::full<idx_t>(n_rows));
}
__tile__ auto rows_view(float* P, int n) {
    return ct::partition_view{ct::tensor_span{P, ct::extents{n, ct::integral_constant<D>{}}}, ct::shape<BR, D>{}};
}

// permutation: Y[j] = X[IDX[j]]
__tile_global__ void gather_rows(float* X, int n_x, const int* IDX, float* Y, int n) {
    int b = static_cast<int>(ct::bid().x);
    auto gX = ct::gather_scatter_view{ct::tensor_span{X, ct::extents{n_x, ct::integral_constant<D>{}}}, ct::shape<BR, D>{},
                                      ct::integral_constant<0>{}};
    rows_view(Y, n).store_masked(gX.load_masked(row_indices(IDX, n, n_x, b), 0), b, 0);
}

// the mocked experts: row j goes through expert EXP[j], e_k(x) = (k + 1) x + k
__tile_global__ void run_experts(float* Y, const int* EXP, int n) {
    int b = static_cast<int>(ct::bid().x);
    auto pE = ct::partition_view{ct::tensor_span{EXP, ct::extents{n}}, ct::shape<BR>{}};
    auto k = ct::reshape<ct::shape<BR, 1>>(ct::element_cast<float>(pE.load_masked(b)));
    auto pY = rows_view(Y, n);
    pY.store_masked((k + 1.0f) * pY.load_masked(b, 0) + k, b, 0);
}

// un-permutation and scale: OUT[IDX[j]] = W[j] * Y[j]
__tile_global__ void scatter_scale(float* Y, const float* W, const int* IDX, int n, float* OUT, int n_out) {
    int b = static_cast<int>(ct::bid().x);
    auto pW = ct::partition_view{ct::tensor_span{W, ct::extents{n}}, ct::shape<BR>{}};
    auto w = ct::reshape<ct::shape<BR, 1>>(pW.load_masked(b));
    auto gO = ct::gather_scatter_view{ct::tensor_span{OUT, ct::extents{n_out, ct::integral_constant<D>{}}}, ct::shape<BR, D>{},
                                      ct::integral_constant<0>{}};
    gO.store_masked(rows_view(Y, n).load_masked(b, 0) * w, row_indices(IDX, n, n_out, b), 0);
}

template <typename X> X* dev(const std::vector<X>& h, size_t n = 0) {
    X* p; CK(cudaMalloc(&p, std::max<size_t>(1, std::max(n, h.size())) * sizeof(X)));
    if (!h.empty()) CK(cudaMemcpy(p, h.data(), h.size() * sizeof(X), cudaMemcpyHostToDevice));
    return p;
}
static int tiles(int n) { return (n + BR - 1) / BR; }

// all-to-all with per-peer counts: send rows [soff[p], soff[p]+scnt[p]) to p, receive rcnt[p] rows at roff[p]
template <typename X>
static void all_to_all(const X* send, const std::vector<int>& scnt, const std::vector<int>& soff, X* recv,
                       const std::vector<int>& rcnt, const std::vector<int>& roff, int width, ncclDataType_t type,
                       ncclComm_t comm, cudaStream_t s) {
    NK(ncclGroupStart());
    for (int p = 0; p < WORLD; ++p) {
        if (scnt[p]) NK(ncclSend(send + (size_t)soff[p] * width, (size_t)scnt[p] * width, type, p, comm, s));
        if (rcnt[p]) NK(ncclRecv(recv + (size_t)roff[p] * width, (size_t)rcnt[p] * width, type, p, comm, s));
    }
    NK(ncclGroupEnd());
}

static void worker(int rank, ncclUniqueId id, std::string* log, int* bad) {
    CK(cudaSetDevice(rank));
    ncclComm_t comm; NK(ncclCommInitRank(&comm, WORLD, id, rank));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    std::vector<Token> tok = tokens_of(rank);
    Plan p = make_plan(tok);
    const int n = (int)tok.size();
    std::vector<float> x;                                       // routing input: this worker's token embeddings
    for (auto& t : tok) for (int d = 0; d < D; ++d) x.push_back(embed(t.word, d));
    float *dX = dev(x), *dSend = dev<float>({}, (size_t)n * D);
    int *dPerm = dev(p.perm), *dPermExp = dev(p.perm_expert);
    float* dPermW = dev(p.perm_weight);
    gather_rows<<<tiles(n), 1, 0, s>>>(dX, n, dPerm, dSend, n);                      // permutation (tile gather)
    CK(cudaGetLastError());

    std::vector<int> one(WORLD, 1), iota(WORLD), rcnt(WORLD), roff(WORLD, 0);       // exchange the counts first
    std::iota(iota.begin(), iota.end(), 0);
    int *dScnt = dev(p.send_count), *dRcnt = dev<int>({}, WORLD);
    all_to_all(dScnt, one, iota, dRcnt, one, iota, 1, ncclInt32, comm, s);
    CK(cudaMemcpyAsync(rcnt.data(), dRcnt, WORLD * sizeof(int), cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    for (int q = 1; q < WORLD; ++q) roff[q] = roff[q - 1] + rcnt[q - 1];
    const int m = roff[WORLD - 1] + rcnt[WORLD - 1];          // rows this worker's experts receive

    float *dRecv = dev<float>({}, (size_t)m * D), *dBack = dev<float>({}, (size_t)n * D), *dOut = dev<float>({}, (size_t)n * D);
    int* dRecvExp = dev<int>({}, m);
    all_to_all(dSend, p.send_count, p.send_off, dRecv, rcnt, roff, D, ncclFloat32, comm, s);       // dispatch
    all_to_all(dPermExp, p.send_count, p.send_off, dRecvExp, rcnt, roff, 1, ncclInt32, comm, s);   //  + expert ids
    run_experts<<<tiles(m), 1, 0, s>>>(dRecv, dRecvExp, m);                                       // computation
    CK(cudaGetLastError());
    all_to_all(dRecv, rcnt, roff, dBack, p.send_count, p.send_off, D, ncclFloat32, comm, s);       // combine
    scatter_scale<<<tiles(n), 1, 0, s>>>(dBack, dPermW, dPerm, n, dOut, n);                        // un-permute + scale
    CK(cudaGetLastError());

    std::vector<float> out((size_t)n * D);
    std::vector<int> rexp(m);
    CK(cudaMemcpyAsync(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(rexp.data(), dRecvExp, m * sizeof(int), cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    char buf[256];
    std::snprintf(buf, sizeof buf, "worker %d: permuted order", rank); *log += buf;
    for (int i : p.perm) *log += " " + tok[i].word + "(e" + std::to_string(tok[i].expert) + ")";
    std::snprintf(buf, sizeof buf, "\nworker %d: sends %d/%d rows to workers 0/1, its experts receive %d rows (experts",
                  rank, p.send_count[0], p.send_count[1], m); *log += buf;
    for (int e : rexp) *log += " " + std::to_string(e);
    *log += ")\n";
    for (int i = 0; i < n; ++i) {
        int wrong = 0;
        for (int d = 0; d < D; ++d) wrong += out[(size_t)i * D + d] != expected(tok[i], d);
        *bad += wrong;
        std::snprintf(buf, sizeof buf, "worker %d: %-6s -> %.2f * e%d(x): %s\n", rank, tok[i].word.c_str(), tok[i].weight,
                      tok[i].expert, wrong ? "WRONG" : "exact");
        *log += buf;
    }
    NK(ncclCommDestroy(comm));
}

int main() {
    int ngpu = 0;
    CK(cudaGetDeviceCount(&ngpu));
    if (ngpu < WORLD) { std::printf("needs %d GPUs, found %d\n", WORLD, ngpu); return 2; }
    ncclUniqueId id; NK(ncclGetUniqueId(&id));        // in MPI this id is broadcast from rank 0; here the threads share it
    std::string log[WORLD];
    int bad[WORLD] = {};
    std::vector<std::thread> th;
    for (int r = 0; r < WORLD; ++r) th.emplace_back(worker, r, id, &log[r], &bad[r]);
    for (auto& t : th) t.join();
    for (int r = 0; r < WORLD; ++r) std::printf("%s", log[r].c_str());
    std::printf(bad[0] + bad[1] ? "FAIL\n" : "PASS: all 6 tokens exact after dispatch, experts, combine\n");
    return bad[0] + bad[1] != 0;
}
