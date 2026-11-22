// Part 3: expert parallelism of one MoE layer over `world` GPUs (one process per GPU, NCCL between them), as in the
// Week 8 handout ("Data and Expert Parallelization"): routing -> permutation (local sort + all-to-all) -> computation
// -> un-permutation (all-to-all back + local gather) -> scaling. Data parallelism: every rank runs its own requests;
// the router, attention and every non-expert weight are replicated; rank r holds only the experts
// [floor(r*E/world), floor((r+1)*E/world)).
//
// One layer on rank r with n local tokens (n may be 0: the rank still serves the other ranks' rows):
//   1. moe_route: router, count, stable sort by global expert id (Part 1). Sorted by expert = grouped by owner rank.
//   2. all-gather of every rank's COUNT[E] -> [world, E]; ONE copy to the host gives both what this rank sends to each
//      rank (its counts summed over that rank's experts) and what it receives (every rank's counts over its own).
//   3. send buffer in sorted order: rows x[SORTED[i]/K] and their expert ids IDX[SORTED[i]] (tile gathers).
//   4. dispatch: grouped ncclSend / ncclRecv of the rows and the ids (counts differ per peer, as in part2/a2a_mock.cu).
//   5. computation on the received rows: count + sort them by expert (K = 1), grouped gate/up + down with ONLY the local
//      experts' weights, then back to received order (combine with K = 1 and weight 1: a permutation, exact).
//   6. combine: all-to-all back with the counts swapped; the rows come back in this rank's sorted order, i.e. they ARE
//      the single-GPU "ysorted", so Part 1's weighted combine (K = 8) finishes the layer.
// The receive side is sized for the worst case: every rank's tokens on this rank's experts, world * N * K rows
// (part3/ep_sim.py, check_capacity). The design was checked on a CPU first (part3/ep_sim.py: bit-for-bit equal to the
// single-GPU layer for 1..8 workers, every assignment traced to its owner and back).
#pragma once
#include <algorithm>
#include <cstdio>
#include <vector>
#include <nccl.h>
#include "../kernels/qwen3_moe_tile.cuh"

#define NCK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { std::fprintf(stderr, "NCCL %s at %s:%d\n", ncclGetErrorString(r_), __FILE__, __LINE__); std::exit(2); } } while (0)

struct EpGroup {
    int rank = 0, world = 1;
    ncclComm_t comm = nullptr;
    std::vector<int> first;      // first[r] = first expert of rank r; first[world] = E (the one ownership table)
    int e0() const { return first[rank]; }
    int e_local() const { return first[rank + 1] - first[rank]; }
    static std::vector<int> table(int E, int world) {
        std::vector<int> f(world + 1);
        for (int r = 0; r <= world; ++r) f[r] = (int)((long long)r * E / world);
        return f;
    }
};

template <typename T>
struct EpWorkspace {
    int max_tokens = 0, max_recv = 0;      // per rank: N local tokens; up to world * N * K received rows
    T* send_rows = nullptr;  int* send_ids = nullptr;      // [N*K]
    T* recv_rows = nullptr;  int* recv_ids = nullptr;      // [world*N*K]
    int *r_count = nullptr, *r_off = nullptr, *r_sorted = nullptr, *r_inv = nullptr;   // receiver's own sort (K = 1)
    T *r_hbuf = nullptr, *r_ysorted = nullptr, *r_y = nullptr, *ones = nullptr;
    int* r_tiles = nullptr;                // work list of the expert GEMMs on the received rows (Part 4)
    int* all_counts = nullptr;             // [world, E] on the device
    std::vector<int> h_counts;             // its host copy
};

// Default stage hook of moe_ep_layer: just run the step. The engine passes one that times each step (its profile).
struct EpNoStage {
    template <typename F> void operator()(const char*, F&& f) const { f(); }
};

// One MoE layer with expert parallelism. x [n, hid] -> y [n, hid]; n may be 0. ws = the single-GPU workspace (router
// and sort of the local tokens, the returned rows in ws.ysorted); wg/wu/wd = THIS rank's experts only.
// st(name, fn) runs each step (named ep_*): the engine's profile times them; the host wait for the counts is between
// ep_pack and ep_dispatch, outside every step, so it shows up as time not in any stage.
template <typename T, int E, int K, typename St = EpNoStage>
void moe_ep_layer(const T* x, const T* wr, const T* wg, const T* wu, const T* wd, T* y, int n, int hid, int inter,
                  bool norm_topk, const MoeWorkspace<T>& ws, EpWorkspace<T>& ep, const EpGroup& g, cudaStream_t s,
                  ncclDataType_t nccl_t, St st = St{}) {
    auto CKE = [](cudaError_t e, const char* what) {
        if (e != cudaSuccess) { std::fprintf(stderr, "moe_ep %s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
    };
    if (n < 0 || n > ws.max_tokens || n > ep.max_tokens) { std::fprintf(stderr, "moe_ep: %d tokens\n", n); std::exit(2); }
    const int A = n * K, W = g.world;
    // 1. local routing and sort (nothing to route on an idle rank: its counts are zero)
    st("ep_route", [&] {
        if (n > 0) CKE((moe_route<T, E, K>(x, wr, n, hid, norm_topk, ws, s)), "route");
        else CKE(cudaMemsetAsync(ws.count, 0, E * sizeof(int), s), "zero counts");
    });
    // 2. everyone's counts, one host copy
    st("ep_allgather", [&] { NCK(ncclAllGather(ws.count, ep.all_counts, E, ncclInt32, g.comm, s)); });
    ep.h_counts.resize((size_t)W * E);
    st("ep_pack", [&] {
        CKE(cudaMemcpyAsync(ep.h_counts.data(), ep.all_counts, (size_t)W * E * sizeof(int), cudaMemcpyDeviceToHost, s), "counts");
        // 3. send buffer (overlaps the copy above on the same stream; the host waits once, below)
        if (A > 0) {
            constexpr int BR = 16, BN = 256, BA = 128;
            moe_gather_rows_kernel<T, K, BR, BN><<<dim3((A + BR - 1) / BR, (hid + BN - 1) / BN, 1), 1, 0, s>>>(x, n, ws.sorted, A,
                                                                                                      ep.send_rows, hid);
            moe_gather_ids_kernel<BA><<<(A + BA - 1) / BA, 1, 0, s>>>(ws.idx, ws.sorted, A, ep.send_ids);
            CKE(cudaGetLastError(), "gather");
        }
    });
    CKE(cudaStreamSynchronize(s), "sync");
    std::vector<int> scnt(W), soff(W), rcnt(W), roff(W);
    for (int q = 0; q < W; ++q) {
        long long sc = 0, rc = 0;
        for (int e = g.first[q]; e < g.first[q + 1]; ++e) sc += ep.h_counts[(size_t)g.rank * E + e];
        for (int e = g.e0(); e < g.e0() + g.e_local(); ++e) rc += ep.h_counts[(size_t)q * E + e];
        scnt[q] = (int)sc; rcnt[q] = (int)rc;
    }
    for (int q = 1; q < W; ++q) { soff[q] = soff[q - 1] + scnt[q - 1]; roff[q] = roff[q - 1] + rcnt[q - 1]; }
    const int R = roff[W - 1] + rcnt[W - 1];
    if (soff[W - 1] + scnt[W - 1] != A || R > ep.max_recv) {
        std::fprintf(stderr, "moe_ep: send %d of %d rows, receive %d (capacity %d)\n", soff[W - 1] + scnt[W - 1], A, R, ep.max_recv);
        std::exit(2);
    }
    // 4. dispatch: rows and expert ids
    auto a2a = [&](const void* sbuf, void* rbuf, size_t width, ncclDataType_t t, size_t esz, const std::vector<int>& sc,
                   const std::vector<int>& so, const std::vector<int>& rc, const std::vector<int>& ro) {
        NCK(ncclGroupStart());
        for (int q = 0; q < W; ++q) {
            if (sc[q]) NCK(ncclSend((const char*)sbuf + (size_t)so[q] * width * esz, (size_t)sc[q] * width, t, q, g.comm, s));
            if (rc[q]) NCK(ncclRecv((char*)rbuf + (size_t)ro[q] * width * esz, (size_t)rc[q] * width, t, q, g.comm, s));
        }
        NCK(ncclGroupEnd());
    };
    st("ep_dispatch", [&] {
        a2a(ep.send_rows, ep.recv_rows, hid, nccl_t, sizeof(T), scnt, soff, rcnt, roff);
        a2a(ep.send_ids, ep.recv_ids, 1, ncclInt32, sizeof(int), scnt, soff, rcnt, roff);
    });
    // 5. computation on the received rows, with this rank's experts only
    if (R > 0) {
        st("ep_recv_sort", [&] {
            moe_count_kernel<E><<<E, 1, 0, s>>>(ep.recv_ids, R, ep.r_count);
            moe_sort_kernel<T, E><<<E, 1, 0, s>>>(ep.recv_ids, R, ep.r_count, ep.r_sorted, ep.r_inv, ep.r_off);
            CKE(cudaGetLastError(), "recv sort");
        });
        // any one expert has at most W * n_max rows (a token picks an expert once): R bounds it
        const long long avg = (long long)R / std::max(1, g.e_local());
        // Part 4 (C3): a block per non-empty (expert, row tile) instead of e_local x R / BM (the 2x H200 profile)
        st("ep_tiles", [&] { CKE((moe_tile_list<E>(ep.r_count, g.e0(), g.e_local(), R, avg, ep.r_tiles, s)), "tiles"); });
        st("ep_gate_up", [&] {
            CKE((moe_experts_part<T, 1>(true, ep.recv_rows, R, wg, wu, wd, ep.r_count, ep.r_off, ep.r_sorted, ep.r_hbuf,
                                        ep.r_ysorted, g.e0(), g.e_local(), R, avg, hid, inter, s, ep.r_tiles, R)), "gate_up");
        });
        st("ep_down", [&] {
            CKE((moe_experts_part<T, 1>(false, ep.recv_rows, R, wg, wu, wd, ep.r_count, ep.r_off, ep.r_sorted, ep.r_hbuf,
                                        ep.r_ysorted, g.e0(), g.e_local(), R, avg, hid, inter, s, ep.r_tiles, R)), "down");
        });
        st("ep_unsort", [&] { CKE((moe_combine<T, 1>(ep.r_ysorted, R, ep.r_inv, ep.ones, ep.r_y, R, hid, s)), "unsort"); });
    }
    // 6. back to the owners (counts swapped), then the weighted combine of Part 1
    st("ep_return", [&] { a2a(ep.r_y, ws.ysorted, hid, nccl_t, sizeof(T), rcnt, roff, scnt, soff); });
    if (n > 0) st("ep_combine", [&] { CKE((moe_combine<T, K>(ws.ysorted, A, ws.inv, ws.w, y, n, hid, s)), "combine"); });
}

template <typename T>
void alloc_ep_workspace(EpWorkspace<T>& ep, int N, int K, int E, int hid, int inter, int world, size_t* bytes) {
    auto A = [&](auto*& p, size_t n) {
        void* q;
        if (cudaMalloc(&q, std::max<size_t>(1, n) * sizeof(*p)) != cudaSuccess) { std::fprintf(stderr, "out of GPU memory (EP workspace)\n"); std::exit(2); }
        p = (std::remove_reference_t<decltype(p)>)q;
        *bytes += n * sizeof(*p);
    };
    const size_t S = (size_t)N * K, R = (size_t)world * N * K;   // N = the largest rank's tokens per pass
    if (R > 0x7fffffffULL) { std::fprintf(stderr, "EP workspace: %zu rows overflow int\n", R); std::exit(2); }
    ep.max_tokens = N; ep.max_recv = (int)R;
    A(ep.send_rows, S * hid); A(ep.send_ids, S);
    A(ep.recv_rows, R * hid); A(ep.recv_ids, R);
    A(ep.r_count, E); A(ep.r_off, E); A(ep.r_sorted, R); A(ep.r_inv, R);
    A(ep.r_hbuf, R * inter); A(ep.r_ysorted, R * hid); A(ep.r_y, R * hid); A(ep.ones, R);
    A(ep.all_counts, (size_t)world * E);
    A(ep.r_tiles, 2 * (size_t)moe_tile_cap((long long)R, 16, E));
    std::vector<T> one(R, T(1.0f));
    if (cudaMemcpy(ep.ones, one.data(), R * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) std::exit(2);
}
