// Part 4, C3: the expert GEMMs with the work list (moe_tile_list) against the full grid they replace, bit for bit.
// Both runs compute the same (expert, row tile, column tile) blocks with the same kernels; the work list only skips
// the blocks that have no rows. Checked on hbuf (gate/up + SiLU) and ysorted (down), every row of every case, with the
// outputs pre-filled with a sentinel so an unwritten row would show. Small hid / inter (the kernels take them at run
// time); 128 experts as the model.
//   one GPU (KDIV 8): n tokens, 8 distinct experts each, uniform or all on experts 0..7
//   expert parallel (KDIV 1): R received rows, e_base 0 / 64 with 128 / 64 local experts, uniform, skewed, or all on
//   one expert
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include "common.h"
#include "../kernels/qwen3_moe_tile.cuh"

using bf16 = __nv_bfloat16;

constexpr int E = 128, HID = 256, INTER = 128;

template <typename V>
static V* dev(size_t n) { V* p; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(V))); return p; }

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 0.5f);
    auto rnd = [&](size_t n) { std::vector<bf16> v(n); for (auto& x : v) x = __float2bfloat16(nd(rng)); return v; };
    const int MAXR = 4096 * 8;
    auto hx = rnd((size_t)MAXR * HID), hg = rnd((size_t)E * INTER * HID), hu = rnd((size_t)E * INTER * HID),
         hd = rnd((size_t)E * HID * INTER);
    bf16 *x = dev<bf16>(hx.size()), *wg = dev<bf16>(hg.size()), *wu = dev<bf16>(hu.size()), *wd = dev<bf16>(hd.size());
    CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(wg, hg.data(), hg.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(wu, hu.data(), hu.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(wd, hd.data(), hd.size() * 2, cudaMemcpyHostToDevice));
    MoeWorkspace<bf16> ws;
    ws.idx = dev<int>(MAXR); ws.count = dev<int>(E); ws.off = dev<int>(E); ws.sorted = dev<int>(MAXR); ws.inv = dev<int>(MAXR);
    ws.hbuf = dev<bf16>((size_t)MAXR * INTER); ws.ysorted = dev<bf16>((size_t)MAXR * HID);
    ws.tiles = dev<int>(2 * moe_tile_cap(MAXR, 16, E));
    ws.max_tokens = MAXR;

    struct Case { const char* name; int kdiv, rows, e_base, e_local; int kind; };   // kind 0 uniform 1 hot 2 one 3 skewed
    std::vector<Case> cases;
    for (int n : {1, 5, 64, 300, 2048}) { cases.push_back({"1gpu uniform", 8, n, 0, 128, 0}); cases.push_back({"1gpu hot8", 8, n, 0, 128, 1}); }
    for (int r : {1, 7, 512, 4096, 16384})
        for (auto [eb, el] : {std::pair{0, 128}, std::pair{64, 64}})
            for (int k : {0, 2, 3}) cases.push_back({k == 0 ? "ep uniform" : k == 2 ? "ep one expert" : "ep skewed", 1, r, eb, el, k});

    int failed = 0;
    for (auto& c : cases) {
        const int n = c.rows, A = n * c.kdiv;
        std::vector<int> ids(A);
        if (c.kdiv == 8) {
            std::vector<int> perm(E);
            for (int t = 0; t < n; ++t) {
                std::iota(perm.begin(), perm.end(), 0);
                if (c.kind == 0) std::shuffle(perm.begin(), perm.end(), rng);
                for (int j = 0; j < 8; ++j) ids[t * 8 + j] = perm[j];   // distinct experts per token, as the router
            }
        } else {
            std::geometric_distribution<int> geo(0.08);
            for (auto& v : ids) v = c.e_base + (c.kind == 0 ? (int)(rng() % c.e_local) : c.kind == 2 ? c.e_local - 1
                                                : std::min(geo(rng), c.e_local - 1));
        }
        CK(cudaMemcpy(ws.idx, ids.data(), A * 4, cudaMemcpyHostToDevice));
        if (c.kdiv == 8) CK((moe_sort<bf16, E, 8>(n, ws))); else CK((moe_sort<bf16, E, 1>(n, ws)));
        const long long avg = (long long)A / c.e_local;
        // the list as the sort kernel writes it (what the engine uses) must equal moe_tile_list's, entry for entry
        const int bm = moe_bm(avg), cap = (int)moe_tile_cap(A, bm, c.e_local);
        std::vector<int> l1(2 * (size_t)cap), l2(2 * (size_t)cap);
        if (cap > 0) {
            CK(cudaMemset(ws.tiles, 0x7F, 2 * (size_t)cap * 4));
            CK((moe_tile_list<E>(ws.count, c.e_base, c.e_local, A, avg, ws.tiles)));
            CK(cudaMemcpy(l1.data(), ws.tiles, l1.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemset(ws.tiles, 0x7F, 2 * (size_t)cap * 4));
            moe_sort_kernel<bf16, E, true><<<E, 1>>>(ws.idx, A, ws.count, ws.sorted, ws.inv, ws.off, bm, c.e_base, c.e_local,
                                                     cap, ws.tiles);
            CK(cudaGetLastError());
            CK(cudaMemcpy(l2.data(), ws.tiles, l2.size() * 4, cudaMemcpyDeviceToHost));
        }
        const bool same_list = l1 == l2;
        const int max_rows = c.kdiv == 8 ? n : A;           // what the engine passes today (n, or every received row)
        auto run = [&](bool list, std::vector<bf16>& h, std::vector<bf16>& y) {
            CK(cudaMemset(ws.hbuf, 0xFF, (size_t)A * INTER * 2));   // sentinel: NaN pattern, an unwritten row would differ
            CK(cudaMemset(ws.ysorted, 0xFF, (size_t)A * HID * 2));
            // list: the sort-built list from above is still in ws.tiles (the sort itself is deterministic)
            for (bool gu : {true, false}) {
                cudaError_t e = c.kdiv == 8
                    ? moe_experts_part<bf16, 8>(gu, x, n, wg, wu, wd, ws.count, ws.off, ws.sorted, ws.hbuf, ws.ysorted, c.e_base,
                                                c.e_local, max_rows, avg, HID, INTER, 0, list ? ws.tiles : nullptr, A)
                    : moe_experts_part<bf16, 1>(gu, x, n, wg, wu, wd, ws.count, ws.off, ws.sorted, ws.hbuf, ws.ysorted, c.e_base,
                                                c.e_local, max_rows, avg, HID, INTER, 0, list ? ws.tiles : nullptr, A);
                CK(e);
            }
            CK(cudaDeviceSynchronize());
            h.resize((size_t)A * INTER); y.resize((size_t)A * HID);
            CK(cudaMemcpy(h.data(), ws.hbuf, h.size() * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(y.data(), ws.ysorted, y.size() * 2, cudaMemcpyDeviceToHost));
        };
        std::vector<bf16> h0, y0, h1, y1;
        run(false, h0, y0);
        run(true, h1, y1);
        long long bad = 0, unwritten = 0;
        for (size_t i = 0; i < h0.size(); ++i) bad += std::memcmp(&h0[i], &h1[i], 2) != 0;
        for (size_t i = 0; i < y0.size(); ++i) bad += std::memcmp(&y0[i], &y1[i], 2) != 0;
        uint16_t sentinel = 0xFFFF;
        for (auto& v : y1) unwritten += std::memcmp(&v, &sentinel, 2) == 0;
        const int BM = moe_bm(avg);
        std::printf("%-14s kdiv %d rows %5d experts %3d..%3d  BM %2d  blocks per column tile: grid %6lld, list %5lld | %s | list %s",
                    c.name, c.kdiv, n, c.e_base, c.e_base + c.e_local - 1, BM,
                    (long long)c.e_local * ((max_rows + BM - 1) / BM), moe_tile_cap(A, BM, c.e_local),
                    bad || unwritten ? "DIFFERS" : "bit-exact", same_list ? "from the sort = moe_tile_list" : "FROM THE SORT DIFFERS");
        if (bad || unwritten) std::printf(" (%lld values differ, %lld output values unwritten)", bad, unwritten);
        std::printf("\n");
        failed += bad || unwritten || !same_list;
    }
    std::printf(failed ? "MOE TILES: %d case(s) FAILED\n" : "MOE TILES: all %zu cases bit-exact (work list = full grid)\n",
                failed ? failed : cases.size());
    return failed != 0;
}
