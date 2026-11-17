"""Simulate the split-KV decode exactly as launch_flash_decode + the two kernels do it (same split rule, same online
softmax per split, same combine), in float64 and in float32, and compare with a direct softmax."""
import numpy as np

FD_BC, FD_SMAX, D = 128, 64, 128

def split_rule(n_kv, HQ):
    n_tiles = (n_kv + FD_BC - 1) // FD_BC
    want = (512 + HQ - 1) // HQ
    n_splits = min(n_tiles, want, FD_SMAX)
    tps = (n_tiles + n_splits - 1) // n_splits
    n_splits = (n_tiles + tps - 1) // tps
    return n_tiles, n_splits, tps

def partial(q, K, V, t0, t1, n_kv, scale, dt):
    m, l, acc = dt(-np.finfo(np.float32).max), dt(0), np.zeros(D, dt)
    qpos = n_kv - 1
    for kb in range(t0, t1):
        idx = np.arange(kb * FD_BC, (kb + 1) * FD_BC)
        valid = idx < n_kv                       # load_masked padding beyond n_kv
        k = np.zeros((FD_BC, D), dt); v = np.zeros((FD_BC, D), dt)
        k[valid] = K[idx[valid]]; v[valid] = V[idx[valid]]
        s = (k @ q).astype(dt) * dt(scale)
        s = np.where(idx <= qpos, s, dt(-np.finfo(np.float32).max))
        new_m = max(m, s.max())
        alpha = np.exp(dt(m - new_m))
        p = np.exp((s - new_m).astype(dt))
        l = l * alpha + p.sum(dtype=dt)
        acc = acc * alpha + (p @ v).astype(dt)
        m = new_m
    return m, l, acc

def split_decode(q, K, V, HQ, dt):
    n_kv = K.shape[0]
    scale = 1 / np.sqrt(D)
    n_tiles, n_splits, tps = split_rule(n_kv, HQ)
    parts = [partial(q, K, V, s * tps, min(s * tps + tps, n_tiles), n_kv, scale, dt) for s in range(n_splits)]
    ms = np.array([p[0] for p in parts], dt)
    mstar = ms.max()
    w = np.exp((ms - mstar).astype(dt))
    L = (w * np.array([p[1] for p in parts], dt)).sum(dtype=dt)
    O = (w[:, None] * np.stack([p[2] for p in parts])).sum(0, dtype=dt) / L
    # every split must hold at least one real key and splits must tile [0, n_tiles) exactly
    cover = [t for s in range(n_splits) for t in range(s * tps, min(s * tps + tps, n_tiles))]
    assert cover == list(range(n_tiles)), "split coverage"
    assert n_splits <= FD_SMAX and all(s * tps < n_tiles for s in range(n_splits)), "empty split"
    return O, n_splits, tps

def direct(q, K, V):
    s = K.astype(np.float64) @ q.astype(np.float64) / np.sqrt(D)
    p = np.exp(s - s.max()); p /= p.sum()
    return p @ V.astype(np.float64)

rng = np.random.default_rng(0)
worst = 0.0
for n_kv in (1, 2, 127, 128, 129, 255, 1000, 3072, 3073, 8192, 65536, 130944, 131072):
    for HQ in (16, 32):
        for peak in (1.0, 6.0):            # peak > 1 makes the softmax sharp, like real attention with YaRN scaling
            q = rng.standard_normal(D) * peak
            K = rng.standard_normal((n_kv, D)); V = rng.standard_normal((n_kv, D))
            ref = direct(q, K, V)
            o64, ns, tps = split_decode(q, K, V, HQ, np.float64)
            o32, _, _ = split_decode(q.astype(np.float32), K.astype(np.float32), V.astype(np.float32), HQ, np.float32)
            e64 = np.abs(o64 - ref).max() / np.abs(ref).max()
            e32 = np.abs(o32 - ref).max() / np.abs(ref).max()
            worst = max(worst, e32)
            assert e64 < 1e-12, (n_kv, HQ, e64)
            if n_kv in (1, 129, 3072, 131072) and peak == 6.0:
                print(f"n_kv={n_kv:7d} HQ={HQ}: splits={ns:2d} tiles/split={tps:4d} blocks={ns*HQ:4d} | "
                      f"rel. error float64 {e64:.1e}, float32 {e32:.1e}")
print(f"ALL OK: algebra exact in float64 for every case; worst float32 relative error {worst:.1e}")
