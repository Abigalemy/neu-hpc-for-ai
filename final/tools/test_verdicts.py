"""Local CPU checks (no GPU, no Modal call) of the pass/fail decisions: each acceptance function must FAIL on a
constructed counterexample, not only pass on good data.

  .venv/bin/python tools/test_verdicts.py
"""
import asyncio
import io
import itertools
import os
import sys
from contextlib import redirect_stdout

import tempfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [HERE, os.path.join(HERE, "server")]
from e2e_test import dense_verdict, judge_variants, pinned_snapshot, output_problems, MAX_GEN   # noqa: E402
sys.path.insert(0, os.path.join(HERE, "part1"))
from moe_e2e import verdict as moe_verdict  # noqa: E402
import run_tests                           # noqa: E402
from api_server import EngineClient        # noqa: E402

failed = []


def check(name, cond):
    print(("ok    " if cond else "FAIL  ") + name)
    if not cond:
        failed.append(name)


def quiet(f, *a):
    with redirect_stdout(io.StringIO()):
        return f(*a)


# ---- dense_verdict (e2e_test.py) ----
# Each counterexample is built so that only the check it targets can catch it: the other checks pass on it (asserted
# with `holds`, which stops the script if the construction is wrong, so a test cannot pass for the wrong reason).
def holds(name, cond):
    if not cond:
        raise SystemExit(f"construction of '{name}' is wrong: the counterexample does not isolate its check")


def cosines(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


rng = np.random.default_rng(0)
R, V = 8, 50
ref = rng.normal(0, 3, (R, V)).astype(np.float32)
exp = ref.argmax(1)[3:]                  # rows 3..7 predict the 5 greedy tokens (gen_row0 = 3)
good, _ = dense_verdict(ref, ref + 1e-3 * rng.normal(size=(R, V)).astype(np.float32), exp.copy(), exp, 3, 0)
check("dense: engine equal to HF up to noise passes", good)

# A/B/C: HF ties A and B; the engine picks C, 1.0 below HF's best. HF's own top-2 gap is 0 (the old rule passed it)
# and the cosine stays > 0.999 (so the cosine check cannot be what catches it).
abc = np.full((4, 3), 1000.0, np.float32); abc[:, 2] = 999
m = abc.copy(); m[0, 2] = 1001
top2 = np.sort(abc[0])[::-1]
holds("A/B/C", cosines(abc, m).min() > 0.999 and top2[0] - top2[1] == 0 and abc[0].max() - abc[0, 2] >= 0.5)
e1 = np.array([0])
ok, _ = dense_verdict(abc, m, e1, e1, 1, 0)
check("dense: HF tied on A, B; engine picks C 1.0 below (cosine > 0.999, HF top-2 gap 0) -> FAIL", not ok)
m = abc.copy(); m[0, 1] = 1000.5
ok, _ = dense_verdict(abc, m, e1, e1, 1, 0)
check("dense: HF tied on A, B; engine picks B -> pass (near tie)", ok)

g = exp.copy(); g[2] = int(np.argsort(ref[5])[0])     # step 2 (row 5): HF's worst token, far below its best
holds("greedy clear", ref[5].max() - ref[5, g[2]] >= 0.5)
ok, _ = dense_verdict(ref, ref, g, exp, 3, 0)
check("dense: greedy differs at a clear margin -> FAIL", not ok)
near = ref.copy(); other = (int(exp[2]) + 1) % V
near[5, other] = near[5].max() - 0.1                  # a token that is NOT HF's best, 0.1 below it
holds("greedy near", other != exp[2] and near[5].argmax() == exp[2])
g = exp.copy(); g[2] = other
ok, _ = dense_verdict(near, near, g, exp, 3, 0)
check("dense: greedy differs at a near tie -> pass", ok)

# output length and ids: the first token tied with HF's best, so the near-tie path alone would accept it
tied = ref.copy(); alt = (int(exp[0]) + 1) % V; tied[3, alt] = tied[3].max()
ok, _ = dense_verdict(tied, tied, np.array([alt]), exp, 3, 0)
check("dense: engine stops after 1 token that ties HF's best -> FAIL", not ok)
ok, _ = dense_verdict(ref, ref, np.append(exp, exp[0]), exp, 3, 0)
check("dense: engine produces one extra token -> FAIL", not ok)
ok, _ = dense_verdict(ref, ref, exp[:3].copy(), exp, 3, 0)
check("dense: engine stops early (no difference before) -> FAIL", not ok)
bad = ref.copy(); g = exp.copy(); g[1] = -2; bad[4, -2] = bad[4].max()   # numpy would read -2 as column V-2
ok, _ = dense_verdict(bad, bad, g, exp, 3, 0)
check("dense: id -2 (a valid column for numpy) -> FAIL", not ok)
g = exp.copy(); g[1] = V
ok, _ = dense_verdict(ref, ref, g, exp, 3, 0)
check("dense: id = vocab size -> FAIL", not ok)

ok, _ = dense_verdict(ref, ref, exp.copy(), exp, 3, 1)
check("dense: one RoPE table entry differs -> FAIL", not ok)
low = ref.copy(); low[4] = rng.normal(0, 3, V)
low[4, ref[4].argmax()] = low[4].max() + 1           # the engine still picks HF's best: only the cosine is wrong
holds("low cosine", low[4].argmax() == ref[4].argmax() and cosines(ref, low).min() < 0.999)
ok, _ = dense_verdict(ref, low, exp.copy(), exp, 3, 0)
check("dense: one row with low cosine -> FAIL", not ok)

# several configurations: each judged with ITS OWN RoPE count; a bad first and a good last must fail
ok, _ = judge_variants(ref, [("first", ref, exp.copy(), 5), ("last", ref, exp.copy(), 0)], exp, 3)
check("dense A/B: first variant's RoPE wrong, last right -> FAIL", not ok)
ok, _ = judge_variants(ref, [("first", ref, exp.copy(), 0), ("last", ref, exp.copy(), 0)], exp, 3)
check("dense A/B: both right -> pass", ok)
check(f"dense: generation limit is 63 (64 kept rows, the first prediction is the prompt's last row)", MAX_GEN == 63)

# a run that wrote nothing, or too little: a clear problem list, not an exception later
with tempfile.TemporaryDirectory() as d:
    a, b = os.path.join(d, "rope.bin"), os.path.join(d, "gen.bin")
    open(b, "wb").write(b"\0" * 12)
    probs = output_problems({a: 16, b: 16})
    check("dense: missing and short output files -> both named", len(probs) == 2 and "not written" in probs[0])
    open(a, "wb").write(b"\0" * 16); open(b, "wb").write(b"\0" * 16)
    check("dense: output files of the right size -> no problem", output_problems({a: 16, b: 16}) == [])

# the dense branch of part1/moe_e2e.py's verdict (the 0.6B check run with the MoE model)
ok, _ = moe_verdict(ref, ref, 3, 0, exp.copy(), exp)
check("moe_e2e dense: identical passes", ok)
ok, _ = moe_verdict(ref, ref, 3, 0, np.append(exp, exp[0]), exp)
check("moe_e2e dense: one extra greedy token -> FAIL", not ok)
g = exp.copy(); g[2] = int(np.argsort(ref[5])[0])
ok, _ = moe_verdict(ref, ref, 3, 0, g, exp)
check("moe_e2e dense: greedy differs -> FAIL", not ok)

# pinned snapshot: two revisions cached, refs/main on the other one
with tempfile.TemporaryDirectory() as hub:
    base = os.path.join(hub, "models--Qwen--Qwen3-4B")
    for rev in ("1cfa9a7208912126459214e8b04321603b3df60c", "ffff"):
        os.makedirs(os.path.join(base, "snapshots", rev))
    os.makedirs(os.path.join(base, "refs")); open(os.path.join(base, "refs", "main"), "w").write("ffff")
    snap = pinned_snapshot(hub, "Qwen/Qwen3-4B")
    check("pinned snapshot: two revisions cached, refs/main elsewhere -> the pinned one",
          snap.endswith("1cfa9a7208912126459214e8b04321603b3df60c"))
    try:
        pinned_snapshot(hub, "Qwen/Qwen3-4B", {"Qwen/Qwen3-4B": "0000"})
        missing = False
    except SystemExit:
        missing = True
    check("pinned snapshot: pinned revision not cached -> error", missing)


# ---- run_tests.report ----
check("report: no module -> FAIL", not quiet(run_tests.report, "x", []))
check("report: module with 0 cases -> FAIL", not quiet(run_tests.report, "x", [("m", 0, 0, 0, 0.0)]))
check("report: 3/3 pass", quiet(run_tests.report, "x", [("m", 3, 3, 3, 0.0)]))
check("report: 2/3 -> FAIL", not quiet(run_tests.report, "x", [("m", 3, 2, 2, 0.0)]))
out = io.StringIO()
with redirect_stdout(out):
    run_tests.report("x", [("moe", 10, 10, None, 1e-3)])
check("report: MoE strict column is N/A, not the tolerance count", "strict=N/A" in out.getvalue())
old_argv = sys.argv
sys.argv = ["run_tests.py", ":memory:", "build", "atention"]
try:
    with redirect_stdout(io.StringIO()):
        exec(compile(open(os.path.join(HERE, "run_tests.py")).read(), "run_tests.py", "exec"), {"__name__": "__main__"})
    code = 0
except SystemExit as e:
    code = e.code
sys.argv = old_argv
check("run_tests: misspelt test name -> non-zero exit", code not in (0, None))


# ---- EngineClient.tokens: the first send fails ----
def bare_client(send):
    c = EngineClient.__new__(EngineClient)
    c.api, c.world, c.pending, c.ids = 0, 1, {}, itertools.count()
    c.sent, c.loads, c.since, c.rr = [0], [0], [0], itertools.count()
    c._send = send
    return c


sent = []


def boom(e, msg):
    sent.append(msg["type"])
    raise OSError("zmq send failed")


c = bare_client(boom)


async def first():
    async for _ in c.tokens([1, 2], 4, 0, True):
        pass

try:
    asyncio.run(first())
    raised = False
except OSError:
    raised = True
check("api: first ADD fails -> the error reaches the caller", raised)
check("api: first ADD fails -> pending is empty again", c.pending == {})
check("api: first ADD fails -> no ABORT for a request never added", sent == ["ADD"])

# ---- EngineClient.tokens: the consumer closes the generator right after the last token ----
abort_sent = []
c2 = bare_client(lambda e, msg: abort_sent.append(msg["type"]))


async def last_then_close():
    gen = c2.tokens([1], 1, 0, True)
    first = asyncio.ensure_future(gen.__anext__())
    await asyncio.sleep(0)
    rid = next(iter(c2.pending))
    c2.pending[rid][1].put_nowait((7, "length"))   # the engine core's last token
    tok = await first
    await gen.aclose()                             # closed on the final token, before asking for more
    return tok

tok = asyncio.run(last_then_close())
check("api: generator closed on the final token -> no ABORT sent", abort_sent == ["ADD"] and tok == (7, "length"))
check("api: generator closed on the final token -> pending empty, request counted once",
      c2.pending == {} and c2.sent == [1])
c3 = bare_client(boom)
try:
    asyncio.run((lambda: (lambda g: g.__anext__())(c3.tokens([1], 1, 0, True)))())
except OSError:
    pass
check("api: first ADD fails -> not counted as sent", c3.sent == [0] and c3.since == [0])


# ---- WorkerProxy: the worker stays alive but never answers a call ----
import subprocess   # noqa: E402
import time         # noqa: E402
HUNG = r"""
import multiprocessing as mp, os, sys, time
sys.path.insert(0, sys.argv[1])
def hung_worker(pipe):
    pipe.send({"vocab": 1, "max_len": 1, "max_seqs": 1, "block_size": 1, "num_blocks": 1, "world": 1})
    pipe.recv()
    time.sleep(20)            # alive, no reply
if __name__ == "__main__":
    import gpu_worker
    a, b = mp.Pipe()
    p = mp.Process(target=hung_worker, args=(b,), daemon=True); p.start()
    w = gpu_worker.WorkerProxy(a, p)
    w._call("decode_batch", [])
"""
import tempfile   # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    script = os.path.join(tmp, "hung.py")       # a file, not -c: multiprocessing's spawn re-imports the main module
    open(script, "w").write(HUNG)
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, script, os.path.join(HERE, "server")], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, env={**os.environ, "QWEN3_WORKER_TIMEOUT": "1"})
    code = proc.wait(timeout=60)
    dt = time.time() - t0
check(f"worker alive but hung: engine core exits (code {code}) after {dt:.1f} s", code == 5 and dt < 15)

print("\nALL VERDICT CHECKS PASS" if not failed else f"\n{len(failed)} CHECK(S) FAILED")
sys.exit(1 if failed else 0)
