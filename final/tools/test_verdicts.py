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

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [HERE, os.path.join(HERE, "server")]
from e2e_test import dense_verdict          # noqa: E402
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
rng = np.random.default_rng(0)
R, V = 8, 50
ref = rng.normal(0, 3, (R, V)).astype(np.float32)
exp = ref.argmax(1)[3:]                  # rows 3..7 predict the 5 greedy tokens (gen_row0 = 3)
good, _ = dense_verdict(ref, ref + 1e-3 * rng.normal(size=(R, V)).astype(np.float32), exp.copy(), exp, 3, 0)
check("dense: engine equal to HF up to noise passes", good)

# A/B/C: HF ties A and B at the top; the engine picks C, far below. HF's own top-2 gap is 0, so the old rule passed.
tie = ref.copy()
a, b, c = 1, 2, 3
tie[0] = 0; tie[0, a] = tie[0, b] = 10; tie[0, c] = 2
mine = tie.copy(); mine[0, c] = 20
ok, _ = dense_verdict(tie, mine, exp.copy(), exp, 3, 0)
check("dense: HF tied on A, B; engine picks C 8 below -> FAIL", not ok)
mine = tie.copy(); mine[0, b] = 10.5
ok, _ = dense_verdict(tie, mine, exp.copy(), exp, 3, 0)
check("dense: HF tied on A, B; engine picks B -> pass (near tie)", ok)

g = exp.copy(); g[2] = np.argsort(ref[5])[0]          # the engine's greedy token at step 2 is HF's worst there
ok, _ = dense_verdict(ref, ref, g, exp, 3, 0)
check("dense: greedy differs at a clear margin -> FAIL", not ok)
near = ref.copy(); near[5, 0] = near[5].max() - 0.1   # step 2 predicted by row 5: token 0 just below HF's best
g = exp.copy(); g[2] = 0
ok, _ = dense_verdict(near, near, g, exp, 3, 0)
check("dense: greedy differs at a near tie -> pass", ok or exp[2] == 0)
ok, _ = dense_verdict(ref, ref, exp[:3].copy(), exp, 3, 0)
check("dense: engine stops early -> FAIL", not ok)
ok, _ = dense_verdict(ref, ref, exp.copy(), exp, 3, 1)
check("dense: one RoPE table entry differs -> FAIL", not ok)
low = ref.copy(); low[4] = rng.normal(0, 3, V)
ok, _ = dense_verdict(ref, low, exp.copy(), exp, 3, 0)
check("dense: one row with low cosine -> FAIL", not ok)

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
