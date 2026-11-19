"""Part 3 design check on a CPU (numpy, no GPU): data parallelism + expert parallelism of one MoE layer over K workers,
stage by stage as the engine will run it, against the single-GPU MoE layer of Part 1.

The phases are the ones of the Week 8 handout ("Data and Expert Parallelization"):
  routing        each worker runs the (replicated) router on ITS OWN tokens (data parallelism)
  permutation    local stable sort of the n*K token copies by expert id; experts are owned in contiguous ranges, so
                 sorted by expert = sorted by destination worker; gather the rows in that order into a send buffer
  dispatch       all-to-all: to worker q the rows (and expert ids) for q's experts
  computation    worker q sorts what it received by its local expert (K=1 sort), runs gate/up + SiLU and down per
                 expert, and puts the outputs back in received order
  combine        all-to-all back: each worker gets its rows' outputs in its own send (= sorted) order
  un-permutation + scaling: out[t] = sum_k w[t,k] * y[sorted position of copy (t,k)], as the single-GPU combine

The single-GPU path is the same code with K=1 worker, so every stage of Part 1 is reused; the only new pieces are
the send gather, the two all-to-alls and the receiver's local sort. Expert ownership: worker r owns experts
[r*E//W, (r+1)*E//W) (W = number of workers, 1..8; uneven when E % W != 0).

Second part: the lock-step protocol. An all-to-all is a collective: every worker must take part in every layer of
every forward pass, also a worker with no requests. Each forward pass starts with a "tick": an all-reduce(max) of
"I have work"; if the max is 1, every worker runs the 48 layers, idle ones with 0 tokens. Simulated with threads and a
barrier-based collective, with random workloads, to show that it neither deadlocks nor mixes up passes.

  python part3/ep_sim.py
"""
import threading

import numpy as np

H, I, E, TOPK = 64, 32, 16, 4        # small stand-ins for 2048, 768, 128, 8


def silu(v):
    return v / (1.0 + np.exp(-v))


def router(x, wr):
    logits = (x @ wr.T).astype(np.float32)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    idx = np.argsort(-p, axis=1, kind="stable")[:, :TOPK]
    w = np.take_along_axis(p, idx, 1)
    return idx.astype(np.int32), (w / w.sum(1, keepdims=True)).astype(np.float32)


def experts(rows, eids, wg, wu, wd):
    """Grouped gate/up + SiLU + down for rows of one worker: sort by expert (stable), run, put back (receiver side)."""
    order = np.argsort(eids, kind="stable")
    out = np.zeros_like(rows)
    for j in order:                      # per row, so the arithmetic is identical whatever the grouping
        e = eids[j]
        out[j] = (silu(rows[j] @ wg[e].T) * (rows[j] @ wu[e].T)) @ wd[e].T
    return out


def moe_single(x, wr, wg, wu, wd):
    """Part 1, one GPU: router -> sort copies by expert -> experts -> combine."""
    idx, w = router(x, wr)
    flat = idx.reshape(-1)
    order = np.argsort(flat, kind="stable")              # sorted position -> copy
    ysorted = experts(x[order // TOPK], flat[order], wg, wu, wd)
    inv = np.empty_like(order); inv[order] = np.arange(order.size)
    return (w[..., None] * ysorted[inv].reshape(len(x), TOPK, H)).sum(1)


def owner_ranges(W, n_exp=None):
    """Expert ownership: worker r owns [floor(r*E/W), floor((r+1)*E/W)); the one table used for routing and indexing."""
    n_exp = n_exp or E
    return [(r * n_exp // W, (r + 1) * n_exp // W) for r in range(W)]


def moe_ep(xs, wr=None, wg=None, wu=None, wd=None, routes=None, apply=None, n_exp=None, topk=None):
    """W workers; xs[r] = worker r's own tokens (may be empty). Returns (outputs per worker, stats).

    routes: optional fixed (idx, w) per worker instead of the router (forced routing for tests).
    apply(rows, global_eids, q): optional expert function on worker q instead of the real experts (tracing tests);
    the real path uses ONLY q's own experts' weights, indexed by local id = global id - first owned id.
    """
    n_exp, topk = n_exp or E, topk or TOPK
    W = len(xs)
    rng = owner_ranges(W, n_exp)
    send = []
    for r, x in enumerate(xs):                           # routing + permutation, per worker
        if routes is not None:
            idx, w = routes[r]
        elif len(x):
            idx, w = router(x, wr)
        else:
            idx, w = np.zeros((0, topk), np.int32), np.zeros((0, topk), np.float32)
        flat = idx.reshape(-1)
        order = np.argsort(flat, kind="stable")          # send order: sorted by expert = grouped by destination
        counts = [int(((flat >= a) & (flat < b)).sum()) for a, b in rng]   # rows per destination worker
        send.append(dict(idx=idx, w=w, order=order, rows=x[order // topk] if len(x) else np.zeros((0, x.shape[1]), x.dtype),
                         eids=flat[order], counts=counts))
    for p in range(W):                                   # the send plan covers every assignment exactly once
        assert sum(send[p]["counts"]) == send[p]["order"].size
    # dispatch: worker q receives from every p the slice of p's send buffer for q's experts, in source order
    recv, stats = [], dict(recv_rows=[], max_rows_per_expert=[])
    for q in range(W):
        parts, eparts = [], []
        for p in range(W):
            off = sum(send[p]["counts"][:q])
            parts.append(send[p]["rows"][off:off + send[p]["counts"][q]])
            eparts.append(send[p]["eids"][off:off + send[p]["counts"][q]])
        rows = np.concatenate(parts)
        eids = np.concatenate(eparts)
        lo, hi = rng[q]
        assert ((eids >= lo) & (eids < hi)).all(), "a row reached a worker that does not own its expert"
        stats["recv_rows"].append(len(rows))
        stats["max_rows_per_expert"].append(int(np.bincount(eids - lo, minlength=hi - lo).max()) if len(eids) else 0)
        if apply is not None:
            y = apply(rows, eids, q)
        else:                                            # computation with ONLY the local experts' weights
            y = experts(rows, eids - lo, wg[lo:hi], wu[lo:hi], wd[lo:hi]) if len(rows) else rows
        recv.append(y)
    # combine: worker p gets back, from each q, its own rows for q's experts, in the order it sent them
    outs, slots = [], []
    for p in range(W):
        back = []
        for q in range(W):
            off = sum(send[r]["counts"][q] for r in range(p))          # p's rows start after workers 0..p-1's in q
            back.append(recv[q][off:off + send[p]["counts"][q]])
        ysorted = np.concatenate(back)                   # = p's "ysorted" of the single-GPU path
        s = send[p]
        n = len(xs[p])
        inv = np.empty_like(s["order"]); inv[s["order"]] = np.arange(s["order"].size)
        slot = ysorted[inv].reshape(n, topk, ysorted.shape[1])   # slot[t, k] = output for (token t, k-th choice)
        slots.append(slot)
        outs.append((s["w"][..., None] * slot).sum(1))
    stats["slots"] = slots
    return outs, stats


def check_layer():
    rs = np.random.default_rng(0)
    wr = rs.standard_normal((E, H)).astype(np.float32)
    wg, wu = (rs.standard_normal((E, I, H)).astype(np.float32) * 0.1 for _ in range(2))
    wd = rs.standard_normal((E, H, I)).astype(np.float32) * 0.1
    cases = 0
    for W in range(1, 9):
        for trial in range(4):
            ns = rs.integers(0, 7, W)
            if trial == 0:
                ns[:] = 0; ns[0] = 5                     # only one worker has tokens; the others take part empty
            xs = [rs.standard_normal((int(n), H)).astype(np.float32) for n in ns]
            got, _ = moe_ep(xs, wr, wg, wu, wd)
            for x, y in zip(xs, got):
                if len(x):
                    ref = moe_single(x, wr, wg, wu, wd)
                    assert np.array_equal(ref, y), (W, trial, np.abs(ref - y).max())
            cases += 1
    print(f"EP layer == single-GPU layer, bit for bit: {cases} cases, W = 1..8 workers (uneven expert split for W = "
          f"3, 5, 6, 7), tokens per worker 0..6, incl. one worker with all the tokens")


def check_ownership():
    for W in range(1, 9):
        rng = owner_ranges(W, 128)
        owners = [sum(a <= e < b for a, b in rng) for e in range(128)]
        assert owners == [1] * 128 and rng[0][0] == 0 and rng[-1][1] == 128, W
        sizes = sorted({b - a for a, b in rng})
        assert sizes[-1] - sizes[0] <= 1, (W, sizes)
    print("ownership: W = 1..8, each of 128 experts has exactly one owner, shares differ by at most 1 (e.g. 42/43 at W=3)")


def check_tags():
    """Every assignment (source worker, token, k-th choice) reaches the owner of its expert and comes back to the same
    (token, k) slot of its source, exactly once; payloads are tags, so equal token values cannot hide a mix-up."""
    rs = np.random.default_rng(1)
    n_exp, topk, cases = 128, 8, 0
    for W in range(1, 9):
        rng = owner_ranges(W, n_exp)
        for kind in ("random", "all_local", "all_remote", "hotspot", "zero_pair"):
            ns = [int(v) for v in rs.integers(0, 40, W)]
            routes, xs = [], []
            for p, n in enumerate(ns):
                if kind == "random":
                    idx = np.stack([rs.choice(n_exp, topk, replace=False) for _ in range(n)]) if n else np.zeros((0, topk), int)
                elif kind == "all_local":
                    a, b = rng[p]; idx = np.stack([a + rs.choice(b - a, topk, replace=b - a < topk) for _ in range(n)]) if n else np.zeros((0, topk), int)
                    if b - a < topk: idx = np.zeros((0, topk), int); n = ns[p] = 0          # W > 16: fewer than 8 local
                elif kind == "all_remote":
                    q = (p + 1) % W; a, b = rng[q]
                    idx = np.stack([a + rs.choice(b - a, topk, replace=False) for _ in range(n)]) if n and b - a >= topk else np.zeros((0, topk), int)
                    n = ns[p] = len(idx)
                elif kind == "hotspot":                  # every token of every worker picks the same 8 experts
                    idx = np.tile(np.arange(topk), (n, 1))
                else:                                    # zero_pair: worker p never sends to worker (p+1) % W
                    q = (p + 1) % W if W > 1 else -1
                    allowed = [e for e in range(n_exp) if not (q >= 0 and rng[q][0] <= e < rng[q][1])]
                    idx = np.stack([rs.choice(allowed, topk, replace=False) for _ in range(n)]) if n else np.zeros((0, topk), int)
                idx = idx.astype(np.int32)
                routes.append((idx, np.full(idx.shape, 1.0 / topk, np.float32)))
                xs.append(np.array([[p, t] for t in range(n)], np.int64).reshape(n, 2))
            def apply(rows, eids, q):                    # tag in -> (source, token, expert, receiving worker) out
                return np.concatenate([rows, eids[:, None], np.full((len(rows), 1), q)], 1)
            _, st = moe_ep(xs, routes=routes, apply=apply, n_exp=n_exp, topk=topk)
            for p in range(W):
                idx = routes[p][0]
                for t in range(len(xs[p])):
                    for k in range(topk):
                        src, tok, e, q = st["slots"][p][t, k]
                        assert (src, tok, e) == (p, t, idx[t, k]), (W, kind, p, t, k, st["slots"][p][t, k])
                        assert rng[q][0] <= e < rng[q][1]
            cases += 1
    print(f"tags: {cases} cases (W = 1..8 x random / all local / all remote / hotspot / one zero-traffic pair): every "
          f"(worker, token, k) reaches its expert's owner and returns to its own slot")


def check_capacity():
    """Workspace sizes the receiver needs: rows received up to W * max_tokens * K, rows per expert up to W * max_tokens
    (each token picks an expert once) - not the sender's n * K and n. Hotspot at the real shape."""
    n_exp, topk, W, n = 128, 8, 2, 1024
    idx = np.tile(np.arange(topk, dtype=np.int32), (n, 1))       # experts 0..7, all owned by worker 0
    routes = [(idx, np.full(idx.shape, 1.0 / topk, np.float32))] * W
    xs = [np.zeros((n, 1), np.float32)] * W
    _, st = moe_ep(xs, routes=routes, apply=lambda rows, e, q: rows, n_exp=n_exp, topk=topk)
    assert st["recv_rows"] == [W * n * topk, 0], st["recv_rows"]
    assert st["max_rows_per_expert"][0] == W * n, st["max_rows_per_expert"]
    print(f"capacity: 2 workers x 1024 tokens, all on worker 0's experts 0-7: worker 0 receives {st['recv_rows'][0]} rows, "
          f"{st['max_rows_per_expert'][0]} per expert (= W*n*K and W*n); worker 1 receives 0")


def check_lockstep(W=4, layers=6, seconds=2.0):
    """Threads as workers; a collective = barrier (every worker must call it, in the same order)."""
    bar = threading.Barrier(W, timeout=5)
    flags = [0] * W
    log = [[] for _ in range(W)]
    planned, errors = [None] * W, []
    stop = threading.Event()

    def allreduce_max(r, v):
        flags[r] = v
        bar.wait()
        m = max(flags)
        bar.wait()
        return m

    def forward(r, n):                       # one pass: every MoE layer is a collective all-to-all
        for layer in range(layers):
            bar.wait()
        log[r].append(n)

    def worker(r):
        try:
            run(r)
        except Exception as e:                   # a broken barrier = a worker left a collective: report, do not hide
            errors.append(f"worker {r}: {type(e).__name__}")
            bar.abort()

    def run(r):
        rs = np.random.default_rng(r)
        work = [int(v) for v in rs.integers(1, 5, rs.integers(0, 12))]   # forward passes this worker must run
        planned[r] = work
        done = 0
        while True:
            have = 1 if done < len(work) else 0
            if allreduce_max(r, have) == 0:      # tick: nobody has work
                if stop.is_set():
                    return
                continue
            forward(r, work[done] if have else 0)   # idle workers run the pass with 0 tokens
            done += have

    th = [threading.Thread(target=worker, args=(r,)) for r in range(W)]
    for t in th:
        t.start()
    threading.Event().wait(seconds)
    stop.set()
    for t in th:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in th), "deadlock"
    assert not errors, f"collective broken: {errors}"
    for r in range(W):                           # every worker ran exactly its own passes, in order
        assert [n for n in log[r] if n] == planned[r], (r, log[r], planned[r])
    lens = {len(l) for l in log}
    assert len(lens) == 1, f"workers ran different numbers of passes: {[len(l) for l in log]}"
    real = [len(w) for w in planned]
    print(f"lock-step: {W} workers ran {lens.pop()} passes each (own work per worker {real}, the rest empty), "
          f"no deadlock")


if __name__ == "__main__":
    check_ownership()
    check_layer()
    check_tags()
    check_capacity()
    check_lockstep()
