"""Stage-by-stage check of the MoE kernels on one GPU (runs inside the Modal test container):

  python tools/moe_debug.py test_cases_moe.db build/moe_test

Runs `moe_test debug` on a few cases and checks, in pipeline order, which stage first disagrees with numpy:
probe (gather_scatter_view index semantics), count, off, sorted (stable by expert), inv, hbuf (gate/up + SiLU rows),
ysorted (expert outputs) and y. Prints the first wrong stage with a few values.
"""
import os
import sqlite3
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_tests import case_tensor, load, param, write  # noqa: E402


def bf16(a):   # round fp32 to bf16 (nearest even), back to fp32
    u = np.asarray(a, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def rd(path, dtype, shape):
    data = open(path, "rb").read()
    return load(data, dtype, "x".join(map(str, shape)))


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name} {detail}")
    return ok


def main(dbp, binary):
    db = sqlite3.connect(dbp)
    d = "/tmp/moedbg"
    os.makedirs(d, exist_ok=True)
    for kind, label in (("moe_block_designed", "two_tokens_overlap"), ("moe_block", "t5")):
        mid = db.execute("SELECT id FROM module WHERE kind=?", (kind,)).fetchone()[0]
        E, K, H, I = (param(db, mid, k) for k in ("num_experts", "top_k", "hidden_size", "moe_intermediate_size"))
        W = {}
        for i, nm in enumerate(("router", "gate_proj", "up_proj", "down_proj")):
            data, shape = db.execute("SELECT data, shape FROM module_tensor WHERE module_id=? AND name=?", (mid, nm)).fetchone()
            write(f"/tmp/m{i}.bin", data)
            W[nm] = load(data, "bfloat16", shape)
        cid = db.execute("SELECT id FROM test_case WHERE module_id=? AND label=?", (mid, label)).fetchone()[0]
        xd, xs, _ = case_tensor(db, cid, "input", "x")
        write("/tmp/x.bin", xd)
        x = load(xd, "bfloat16", xs)
        T = x.shape[0]
        A = T * K
        print(f"\n{kind} {label}: T={T} E={E} K={K} H={H} I={I}")
        r = subprocess.run([binary, "debug", str(E), str(K), str(H), str(I), str(T), "/tmp/x.bin", "/tmp/m0.bin", "/tmp/m1.bin",
                            "/tmp/m2.bin", "/tmp/m3.bin", "/tmp/idx.bin", "/tmp/w.bin", "/tmp/y.bin", d], capture_output=True, text=True)
        if r.returncode:
            print("  binary failed:", r.stderr[-500:])
            return 1
        p = rd(f"{d}/probe.bin", "float32", (4, 128))
        sem = "tile coordinate" if p[0, 0] == 3128 else "element offset" if p[0, 0] == 3001 else f"unknown ({p[0, 0]})"
        print(f"  probe: gather_scatter_view index of the 2nd dimension is a {sem}; rows {p[:, 0].tolist()}")
        idx = np.frombuffer(open("/tmp/idx.bin", "rb").read(), np.int32).reshape(T, K)
        w = rd("/tmp/w.bin", "bfloat16", (T, K))
        cnt = np.frombuffer(open(f"{d}/count.bin", "rb").read(), np.int32)
        off = np.frombuffer(open(f"{d}/off.bin", "rb").read(), np.int32)
        srt = np.frombuffer(open(f"{d}/sorted.bin", "rb").read(), np.int32)
        inv = np.frombuffer(open(f"{d}/inv.bin", "rb").read(), np.int32)
        flat = idx.reshape(-1)
        want_cnt = np.bincount(flat, minlength=E)
        if not check("count", np.array_equal(cnt, want_cnt), f"got {cnt.tolist()} want {want_cnt.tolist()}"):
            continue
        want_off = np.concatenate([[0], np.cumsum(want_cnt)[:-1]])
        if not check("off", np.array_equal(off, want_off), f"got {off.tolist()} want {want_off.tolist()}"):
            continue
        want_srt = np.argsort(flat, kind="stable").astype(np.int32)
        if not check("sorted", np.array_equal(srt, want_srt), f"got {srt[:24].tolist()} want {want_srt[:24].tolist()}"):
            continue
        want_inv = np.empty(A, np.int32); want_inv[want_srt] = np.arange(A)
        if not check("inv", np.array_equal(inv, want_inv), f"got {inv[:24].tolist()} want {want_inv[:24].tolist()}"):
            continue
        hb = rd(f"{d}/hbuf.bin", "bfloat16", (A, I))
        ys = rd(f"{d}/ysorted.bin", "bfloat16", (A, H))
        href = np.zeros((A, I), np.float32); yref = np.zeros((A, H), np.float32)
        for pos in range(A):
            a = want_srt[pos]; t, e = a // K, flat[a]
            g = bf16(x[t] @ W["gate_proj"][e].T); u = bf16(x[t] @ W["up_proj"][e].T)
            href[pos] = bf16(bf16(g / (1 + np.exp(-g))) * u)
            yref[pos] = bf16(href[pos] @ W["down_proj"][e].T)
        eh = np.abs(hb - href).max() / np.abs(href).max()
        if not check("hbuf", eh < 2e-2, f"max err {eh:.2e} of max|h|; row0 got {hb[0, :4]} want {href[0, :4]}"):
            # which rows / columns: does a row of hbuf match the reference for another token?
            for pos in range(min(A, 4)):
                errs = [np.abs(hb[pos] - href[q]).max() for q in range(A)]
                print(f"    hbuf row {pos}: closest reference row {int(np.argmin(errs))} (err {min(errs):.3g})")
            continue
        ey = np.abs(ys - yref).max() / np.abs(yref).max()
        if not check("ysorted", ey < 2e-2, f"max err {ey:.2e} of max|y|; row0 got {ys[0, :4]} want {yref[0, :4]}"):
            continue
        y = rd("/tmp/y.bin", "bfloat16", (T, H))
        want_y = np.zeros((T, H));
        for a in range(A):
            want_y[a // K] += w.reshape(-1)[a] * yref[want_inv[a]]
        e_ = np.abs(y - want_y).max() / np.abs(want_y).max()
        check("combine (y)", e_ < 2e-2, f"max err {e_:.2e} of max|y|; row0 got {y[0, :4]} want {want_y[0, :4]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
