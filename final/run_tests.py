"""Run the CUDA Tile kernels against the SQLite reference cases. Runs inside the Modal GPU container.

  python run_tests.py test_cases.db build [rmsnorm rope]

For every case: dump the saved weights/inputs to raw files, run the compiled test binary, read its output and compare
with the expected tensors stored in the database.

Pass rule (fixed here, not in the database):
  float32  : |y - e| <= 1e-5 + 1e-5*|e|
  bfloat16 : |y - e| <= 1e-3 + 2**-7 * |e|   (one bf16 rounding step is 2**-8 relative; allow two)
Both the tolerance verdict and the strict (bit-identical) verdict are reported.
"""
import sqlite3
import subprocess
import sys

import numpy as np


def load(data, dtype, shape):
    shape = tuple(int(s) for s in shape.split("x"))
    if dtype == "float32":
        return np.frombuffer(data, dtype=np.float32).reshape(shape)
    if dtype == "bfloat16":  # stored as 16-bit patterns; widen to float32 by shifting into the high half
        u = np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(shape)
    if dtype == "int32":
        return np.frombuffer(data, dtype=np.int32).reshape(shape)
    raise ValueError(dtype)


def compare(dtype, y, e):
    d = np.abs(y.astype(np.float64) - e.astype(np.float64))
    if dtype == "float32":
        ok = d <= 1e-5 + 1e-5 * np.abs(e)
    else:
        ok = d <= 1e-3 + 2.0 ** -7 * np.abs(e)
    strict = np.array_equal(y.view(np.uint32), e.view(np.uint32))
    return bool(ok.all()), strict, float(d.max())


def compare_matmul(dtype, y, e):
    """Tolerance for modules with matmuls (different summation order on GPU vs CPU, so scale the error by the
    output magnitude rather than by each element):
      float32  : |d| <= 1e-4 * (|e| + max|e|)
      bfloat16 : |d| <= 2**-6 * (|e| + 0.05 * max|e|)
    """
    d = np.abs(y.astype(np.float64) - e.astype(np.float64))
    m = float(np.abs(e).max())
    ok = d <= (1e-4 * (np.abs(e) + m) if dtype == "float32" else 2.0 ** -6 * (np.abs(e) + 0.05 * m))
    strict = np.array_equal(y.view(np.uint32), e.view(np.uint32))
    return bool(ok.all()), strict, float(d.max())


def compare_attention(dtype, y, e, vmax):
    """Attention tolerance, derived from the arithmetic rather than tuned on observed errors.
    float32 : same as compare_matmul.
    bfloat16: the fused kernel rounds P = exp(s - m) to bf16 before P@V. bf16 has 8 significant bits, so one rounding
              has relative error <= u = 2**-8 (NOT 2**-9: an earlier version of this comment had that off by one bit).
              Each weight then has relative error <= u, so |dO| <= u * sum_j p_j |v_j| / l <= u * max|V|. The output and
              the reference are each rounded to bf16 as well (<= u |e| each).  =>  |y - e| <= 2**-8 * (max|V| + 2|e|).
    """
    if dtype == "float32":
        return compare_matmul(dtype, y, e)
    d = np.abs(y.astype(np.float64) - e.astype(np.float64))
    ok = d <= 2.0 ** -8 * (vmax + 2.0 * np.abs(e))
    strict = np.array_equal(y.view(np.uint32), e.view(np.uint32))
    return bool(ok.all()), strict, float(d.max())


def param(db, mid, name):
    return db.execute("SELECT COALESCE(int_value, real_value) FROM module_param WHERE module_id=? AND name=?",
                      (mid, name)).fetchone()[0]


def case_tensor(db, cid, role, name):
    return db.execute("SELECT data, shape, dtype FROM case_tensor WHERE case_id=? AND role=? AND name=?",
                      (cid, role, name)).fetchone()


def write(path, data):
    open(path, "wb").write(data)


def report(title, rows):
    print(f"\n{title}: cases / pass(tolerance) / pass(bit-identical) / worst abs err")
    for desc, n, n_ok, n_strict, worst in rows:
        print("  %-28s cases=%-3d tol=%-3d strict=%-3d worst=%.3e" % (desc, n, n_ok, n_strict, worst))
    return all(r[1] == r[2] for r in rows)


def run_rmsnorm(db, binary):
    out = []
    for mid, dtype in db.execute("SELECT id, dtype FROM module WHERE kind='rmsnorm'").fetchall():
        hidden = param(db, mid, "hidden_size")
        write("/tmp/w.bin", db.execute("SELECT data FROM module_tensor WHERE module_id=? AND name='weight'", (mid,)).fetchone()[0])
        n = n_ok = n_strict = 0
        worst = 0.0
        for cid, label in db.execute("SELECT id, label FROM test_case WHERE module_id=?", (mid,)).fetchall():
            xd, xs, _ = case_tensor(db, cid, "input", "x")
            ed, es, _ = case_tensor(db, cid, "expected", "y")
            write("/tmp/x.bin", xd)
            r = subprocess.run([binary, dtype, str(hidden), xs.split("x")[0], "/tmp/x.bin", "/tmp/w.bin", "/tmp/y.bin"],
                               capture_output=True, text=True)
            n += 1
            if r.returncode != 0:
                print(f"  ! {dtype} D={hidden} {label}: binary failed: {r.stderr.strip()[:300]}")
                continue
            ok, strict, maxabs = compare(dtype, load(open("/tmp/y.bin", "rb").read(), dtype, xs), load(ed, dtype, es))
            n_ok += ok; n_strict += strict; worst = max(worst, maxabs)
            if not ok:
                print(f"  FAIL rmsnorm {dtype} D={hidden} {label}: max abs err {maxabs:.3e}")
        out.append((f"rmsnorm {dtype} D={hidden}", n, n_ok, n_strict, worst))
    return report("RMSNorm", out)


def run_rope(db, binary):
    out = []
    for mid, dtype in db.execute("SELECT id, dtype FROM module WHERE kind='rope'").fetchall():
        nh, nkv, maxp = param(db, mid, "num_heads"), param(db, mid, "num_kv_heads"), param(db, mid, "max_pos")
        for name in ("cos", "sin"):
            write(f"/tmp/{name}.bin", db.execute("SELECT data FROM module_tensor WHERE module_id=? AND name=?", (mid, name)).fetchone()[0])
        n = n_ok = n_strict = 0
        worst = 0.0
        for cid, label in db.execute("SELECT id, label FROM test_case WHERE module_id=?", (mid,)).fetchall():
            qd, qs, _ = case_tensor(db, cid, "input", "q")
            kd, ks, _ = case_tensor(db, cid, "input", "k")
            pd, ps, _ = case_tensor(db, cid, "input", "positions")
            write("/tmp/q.bin", qd); write("/tmp/k.bin", kd); write("/tmp/pos.bin", pd)
            tokens = qs.split("x")[0]
            r = subprocess.run([binary, dtype, str(nh), str(nkv), tokens, str(maxp), "/tmp/q.bin", "/tmp/k.bin",
                                "/tmp/cos.bin", "/tmp/sin.bin", "/tmp/pos.bin", "/tmp/qo.bin", "/tmp/ko.bin"],
                               capture_output=True, text=True)
            n += 1
            if r.returncode != 0:
                print(f"  ! rope {dtype} {nh}/{nkv} {label}: binary failed: {r.stderr.strip()[:300]}")
                continue
            res = []
            for fname, shape, exp in (("qo", qs, "q_out"), ("ko", ks, "k_out")):
                ed, es, _ = case_tensor(db, cid, "expected", exp)
                res.append(compare(dtype, load(open(f"/tmp/{fname}.bin", "rb").read(), dtype, shape), load(ed, dtype, es)))
            ok, strict, maxabs = all(r_[0] for r_ in res), all(r_[1] for r_ in res), max(r_[2] for r_ in res)
            n_ok += ok; n_strict += strict; worst = max(worst, maxabs)
            if not ok:
                print(f"  FAIL rope {dtype} heads={nh}/{nkv} {label}: max abs err {maxabs:.3e}")
        out.append((f"rope {dtype} heads={nh}/{nkv}", n, n_ok, n_strict, worst))
    return report("RoPE", out)


def run_mlp(db, binary):
    out = []
    for mid, dtype in db.execute("SELECT id, dtype FROM module WHERE kind='mlp'").fetchall():
        hid, inter = param(db, mid, "hidden_size"), param(db, mid, "intermediate_size")
        for name, f in (("gate_proj", "wg"), ("up_proj", "wu"), ("down_proj", "wd")):
            write(f"/tmp/{f}.bin", db.execute("SELECT data FROM module_tensor WHERE module_id=? AND name=?", (mid, name)).fetchone()[0])
        n = n_ok = n_strict = 0
        worst = 0.0
        for cid, label in db.execute("SELECT id, label FROM test_case WHERE module_id=?", (mid,)).fetchall():
            xd, xs, _ = case_tensor(db, cid, "input", "x")
            ed, es, _ = case_tensor(db, cid, "expected", "y")
            write("/tmp/x.bin", xd)
            r = subprocess.run([binary, dtype, str(hid), str(inter), xs.split("x")[0], "/tmp/x.bin", "/tmp/wg.bin",
                                "/tmp/wu.bin", "/tmp/wd.bin", "/tmp/y.bin"], capture_output=True, text=True)
            n += 1
            if r.returncode != 0:
                print(f"  ! mlp {dtype} {hid}/{inter} {label}: binary failed: {r.stderr.strip()[:300]}")
                continue
            e = load(ed, dtype, es)
            ok, strict, maxabs = compare_matmul(dtype, load(open("/tmp/y.bin", "rb").read(), dtype, xs), e)
            n_ok += ok; n_strict += strict; worst = max(worst, maxabs / float(np.abs(e).max()))
            if not ok:
                print(f"  FAIL mlp {dtype} {hid}/{inter} {label}: max abs err {maxabs:.3e} (max|e| {float(np.abs(e).max()):.3e})")
        out.append((f"mlp {dtype} {hid}/{inter} (worst = rel. to max|e|)", n, n_ok, n_strict, worst))
    return report("MLP", out)


def run_embedding(db, binary):
    out = []
    for mid, dtype in db.execute("SELECT id, dtype FROM module WHERE kind='embedding'").fetchall():
        vocab, hid = param(db, mid, "vocab_size"), param(db, mid, "hidden_size")
        write("/tmp/w.bin", db.execute("SELECT data FROM module_tensor WHERE module_id=? AND name='weight'", (mid,)).fetchone()[0])
        n = n_ok = 0
        for cid, label in db.execute("SELECT id, label FROM test_case WHERE module_id=?", (mid,)).fetchall():
            idd, ids_, _ = case_tensor(db, cid, "input", "ids")
            ed, es, _ = case_tensor(db, cid, "expected", "y")
            write("/tmp/ids.bin", idd)
            r = subprocess.run([binary, dtype, str(hid), str(vocab), ids_, "/tmp/ids.bin", "/tmp/w.bin", "/tmp/y.bin"],
                               capture_output=True, text=True)
            n += 1
            if r.returncode != 0:
                print(f"  ! embedding {dtype} {vocab}x{hid} {label}: binary failed: {r.stderr.strip()[:300]}")
                continue
            y = load(open("/tmp/y.bin", "rb").read(), dtype, es)
            ok = np.array_equal(y.view(np.uint32), load(ed, dtype, es).view(np.uint32))  # a copy must be bit-exact
            n_ok += ok
            if not ok:
                print(f"  FAIL embedding {dtype} {vocab}x{hid} {label}")
        out.append((f"embedding {dtype} {vocab}x{hid}", n, n_ok, n_ok, 0.0))
    return report("Embedding (bit-exact required)", out)


def run_linear(db, binary):
    out = []
    for mid, dtype in db.execute("SELECT id, dtype FROM module WHERE kind='linear'").fetchall():
        nin, nout = param(db, mid, "in_features"), param(db, mid, "out_features")
        write("/tmp/w.bin", db.execute("SELECT data FROM module_tensor WHERE module_id=? AND name='weight'", (mid,)).fetchone()[0])
        n = n_ok = n_strict = 0
        worst = 0.0
        for cid, label in db.execute("SELECT id, label FROM test_case WHERE module_id=?", (mid,)).fetchall():
            xd, xs, _ = case_tensor(db, cid, "input", "x")
            ed, es, _ = case_tensor(db, cid, "expected", "y")
            write("/tmp/x.bin", xd)
            r = subprocess.run([binary, dtype, str(nin), str(nout), xs.split("x")[0], "/tmp/x.bin", "/tmp/w.bin", "/tmp/y.bin"],
                               capture_output=True, text=True)
            n += 1
            if r.returncode != 0:
                print(f"  ! linear {dtype} {nin}->{nout} {label}: binary failed: {r.stderr.strip()[:300]}")
                continue
            e = load(ed, dtype, es)
            ok, strict, maxabs = compare_matmul(dtype, load(open("/tmp/y.bin", "rb").read(), dtype, es), e)
            n_ok += ok; n_strict += strict; worst = max(worst, maxabs / float(np.abs(e).max()))
            if not ok:
                print(f"  FAIL linear {dtype} {nin}->{nout} {label}: max abs err {maxabs:.3e}")
        out.append((f"linear {dtype} {nin}->{nout} (worst = rel. to max|e|)", n, n_ok, n_strict, worst))
    return report("Linear / lm_head", out)


def run_argmax(db, binary):
    out = []
    for mid, dtype in db.execute("SELECT id, dtype FROM module WHERE kind='argmax'").fetchall():
        vocab = param(db, mid, "vocab_size")
        n = n_ok = 0
        for cid, label in db.execute("SELECT id, label FROM test_case WHERE module_id=?", (mid,)).fetchall():
            ld, ls, _ = case_tensor(db, cid, "input", "logits")
            ed, es, _ = case_tensor(db, cid, "expected", "ids")
            write("/tmp/l.bin", ld)
            r = subprocess.run([binary, dtype, str(vocab), ls.split("x")[0], "/tmp/l.bin", "/tmp/ids.bin"],
                               capture_output=True, text=True)
            n += 1
            if r.returncode != 0:
                print(f"  ! argmax {dtype} V={vocab} {label}: binary failed: {r.stderr.strip()[:300]}")
                continue
            got = np.frombuffer(open("/tmp/ids.bin", "rb").read(), dtype=np.int32)
            exp = np.frombuffer(ed, dtype=np.int32)
            ok = np.array_equal(got, exp)
            n_ok += ok
            if not ok:
                print(f"  FAIL argmax {dtype} V={vocab} {label}: got {got.tolist()[:8]} expected {exp.tolist()[:8]}")
        out.append((f"argmax {dtype} V={vocab}", n, n_ok, n_ok, 0.0))
    return report("Greedy argmax (exact ids required)", out)


def run_attention(db, binary):
    out = []
    for mid, dtype in db.execute("SELECT id, dtype FROM module WHERE kind='attention'").fetchall():
        hq, hkv = param(db, mid, "num_heads"), param(db, mid, "num_kv_heads")
        n = n_ok = n_strict = 0
        worst = 0.0
        for cid, label in db.execute("SELECT id, label FROM test_case WHERE module_id=?", (mid,)).fetchall():
            qd, qs, _ = case_tensor(db, cid, "input", "q")
            kd, ks, _ = case_tensor(db, cid, "input", "k")
            vd, vs, _ = case_tensor(db, cid, "input", "v")
            ed, es, _ = case_tensor(db, cid, "expected", "o")
            write("/tmp/q.bin", qd); write("/tmp/k.bin", kd); write("/tmp/v.bin", vd)
            n_q, n_kv = qs.split("x")[0], ks.split("x")[0]
            r = subprocess.run([binary, dtype, str(hq), str(hkv), n_q, n_kv, "/tmp/q.bin", "/tmp/k.bin", "/tmp/v.bin", "/tmp/o.bin"],
                               capture_output=True, text=True)
            n += 1
            if r.returncode != 0:
                print(f"  ! attention {dtype} {hq}/{hkv} {label}: binary failed: {r.stderr.strip()[:300]}")
                continue
            e = load(ed, dtype, es)
            vmax = float(np.abs(load(vd, dtype, vs)).max())
            ok, strict, maxabs = compare_attention(dtype, load(open("/tmp/o.bin", "rb").read(), dtype, es), e, vmax)
            n_ok += ok; n_strict += strict; worst = max(worst, maxabs / float(np.abs(e).max()))
            if not ok:
                print(f"  FAIL attention {dtype} {hq}/{hkv} {label}: max abs err {maxabs:.3e} (max|e| {float(np.abs(e).max()):.3e}, max|V| {vmax:.2f}, bound at max|e| {2.0 ** -8 * (vmax + 2 * float(np.abs(e).max())):.3e})")
        out.append((f"attention {dtype} heads={hq}/{hkv} (worst = rel. to max|e|)", n, n_ok, n_strict, worst))
    return report("Flash attention (causal GQA over KV cache)", out)


if __name__ == "__main__":
    db = sqlite3.connect(sys.argv[1])
    build = sys.argv[2]
    which = sys.argv[3:] or ["rmsnorm", "rope", "mlp", "embedding", "linear", "argmax", "attention"]
    results = []
    if "rmsnorm" in which:
        results.append(run_rmsnorm(db, f"{build}/rmsnorm_test"))
    if "rope" in which:
        results.append(run_rope(db, f"{build}/rope_test"))
    if "mlp" in which:
        results.append(run_mlp(db, f"{build}/mlp_test"))
    if "embedding" in which:
        results.append(run_embedding(db, f"{build}/embedding_test"))
    if "linear" in which:
        results.append(run_linear(db, f"{build}/linear_test"))
    if "argmax" in which:
        results.append(run_argmax(db, f"{build}/argmax_test"))
    if "attention" in which:
        results.append(run_attention(db, f"{build}/attention_test"))
    ok = all(results)
    print("\nALL PASS (tolerance)" if ok else "\nSOME CASES FAILED")
    sys.exit(0 if ok else 1)
