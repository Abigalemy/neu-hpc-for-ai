"""Part 4, C3: CPU model of the MoE work list, checked before the GPU kernel is written.

  python part4/tile_list_sim.py

Today the grouped gate/up and down GEMMs launch a block for every (local expert, row tile up to max_rows, column
tile): most row tiles of most experts are empty (the 1x H200 profile: 30-40 TFLOP/s at prefill; with expert
parallelism max_rows is every received row, so the grid is larger still). The work list holds only the non-empty
(expert, row tile) pairs; the GEMM grid is x = list position, z = column tile, and each block computes the same tile
as before (same BM, same K loop), so the result is bit-for-bit the old one.

The list is built by one block of moe_tile_list_kernel with tile operations only (as the count / sort kernels), and
this file does the same arithmetic in numpy:
  tiles[e]  = ceil(COUNT[e_base + e] / BM) for the e_local local experts (0 for padding lanes up to E)
  excl[e]   = sum of tiles[e'] for e' < e            ([E, E] masked sum, as OFF in moe_sort_kernel)
  for list positions p in chunks of BP:
    e(p)    = number of experts with excl[e] + tiles[e] <= p   (the first expert whose range reaches past p)
    mb(p)   = p - excl[e(p)]                                    ([BP, E] masked sum picks excl[e(p)])
    entry   = (e(p), mb(p)) if p < total tiles, else (-1, 0)    (a block that does nothing)
Capacity (the grid size, known on the host without the counts): ceil(A / BM) + min(e_local, A), because
sum ceil(c/BM) <= sum (c/BM + 1 over the non-empty experts) <= A/BM + (number of non-empty experts).
Checked: every non-empty tile exactly once, nothing else, total <= capacity, for random, skewed, single-expert,
empty and expert-parallel (e_base > 0) count vectors, both row tiles (16, 64), and the chunk edges.
"""
import numpy as np

E, BP = 128, 128


def capacity(A, BM, e_local):
    return (A + BM - 1) // BM + min(e_local, A)


def tile_list(count, e_base, e_local, A, BM):
    """The kernel's arithmetic: count = the [E] global COUNT array."""
    lane = np.arange(E)
    c = np.where(lane < e_local, count[(e_base + lane) % E], 0) if e_local < E else count[e_base + lane]
    tiles = (c + BM - 1) // BM
    lower = lane[None, :] < lane[:, None]                       # [e, e']: e' < e
    excl = np.where(lower, tiles[None, :], 0).sum(1)
    incl = excl + tiles
    total = tiles.sum()
    cap = capacity(A, BM, e_local)
    out = []
    for chunk in range((cap + BP - 1) // BP):
        p = chunk * BP + np.arange(BP)
        e_of = (incl[None, :] <= p[:, None]).sum(1)              # [BP]
        pick = lane[None, :] == e_of[:, None]                    # [BP, E]; e_of == E picks nothing
        mb = p - np.where(pick, excl[None, :], 0).sum(1)
        valid = (p < total) & (p < cap)
        for i in range(BP):
            if p[i] < cap:
                out.append((int(e_of[i]), int(mb[i])) if valid[i] else (-1, 0))
    return out, int(total), cap


def check(count, e_base, e_local, A, BM, what):
    lst, total, cap = tile_list(count, e_base, e_local, A, BM)
    want = [(e, m) for e in range(e_local) for m in range((count[e_base + e] + BM - 1) // BM)]
    got = [t for t in lst if t[0] >= 0]
    assert len(lst) == cap, (what, len(lst), cap)
    assert total <= cap, (what, total, cap)
    assert got == want, (what, got[:5], want[:5])               # same tiles, same (expert-major) order
    assert all(t == (-1, 0) for t in lst[len(want):]), what
    return total, cap


def main():
    rng = np.random.default_rng(0)
    n_cases, waste_old, waste_new = 0, [], []
    for BM in (16, 64):
        for n in (1, 2, 8, 33, 64, 127, 128, 129, 1024, 2048, 3072):
            for kind in ("uniform", "skewed", "one_expert", "hot8"):
                for e_base, e_local, kdiv in ((0, 128, 8), (0, 64, 1), (64, 64, 1), (32, 32, 1)):
                    A = n * 8 if kdiv == 8 else n * 8          # rows into the GEMM: assignments (or received rows)
                    count = np.zeros(E, dtype=np.int64)
                    if kind == "uniform":
                        ids = rng.integers(0, E, size=A)
                    elif kind == "skewed":
                        ids = np.minimum(rng.geometric(0.05, size=A) - 1, E - 1)
                    elif kind == "one_expert":
                        ids = np.full(A, e_base + e_local - 1)
                    else:
                        ids = rng.integers(0, 8, size=A) + e_base
                    for i in ids:
                        if e_base <= i < e_base + e_local:
                            count[i] += 1
                    rows = int(count[e_base:e_base + e_local].sum())   # what this rank's GEMM gets
                    total, cap = check(count, e_base, e_local, rows, BM, (BM, n, kind, e_base, e_local))
                    # old grid: max_rows = n tokens on one GPU (KDIV 8), every received row with expert parallelism
                    old = e_local * ((max(n if kdiv == 8 else rows, 1) + BM - 1) // BM)
                    if kind == "uniform" and e_local == 128 and kdiv == 8:
                        waste_old.append((BM, n, old, total))
                    n_cases += 1
    # empty layer (an idle expert-parallel rank receives nothing): no list positions
    assert tile_list(np.zeros(E, dtype=np.int64), 0, 128, 0, 16)[2] == 0
    print(f"{n_cases} count vectors: every non-empty tile exactly once, in expert order, total <= capacity")
    print("row-tile blocks launched (per column tile), uniform routing on one GPU: old grid -> work list")
    for BM, n, old, new in waste_old:
        if n in (1, 64, 2048, 3072):
            print(f"  BM {BM:2d}, {n:4d} tokens: {old:6d} -> {new:4d}")
    print("TILE LIST SIM PASSED")


if __name__ == "__main__":
    main()
