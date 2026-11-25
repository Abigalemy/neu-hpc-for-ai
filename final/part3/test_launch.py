"""CPU test of part3/launch.py with fake engines in real processes (no GPU): load balancer -> K rank servers, the
coordinator's lock step (fake ticks answered like the engine's all-reduce), park / wake, and the watchdog.

  python part3/test_launch.py

Checked: (1) concurrent streamed requests through the balancer all give exactly the tokens of the same request run
alone on one fake engine, and every rank got its share; (2) lock step: the coordinator answers a tick only when every
rank has sent its own (as the engine's all-reduce), so a rank that skipped a pass would hang the test; (3) once idle, no tick happens (the group is parked); (4) a request after
the idle period wakes the group and completes; (5) killing one rank stops the launcher and every rank.
"""
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time

import httpx

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "server"))
from server import FakeBackend  # noqa: E402

W, PORT, N_REQ, OUT = 3, 18700, 12, 8
PROMPT = [11, 12, 13, 14]


def alone(prompt, n):
    """The same request on one fake engine, no server: prefill, then n - 1 decode steps."""
    be = FakeBackend(max_seqs=1)
    be.reserve(0, len(prompt) + n)
    toks = [be.prefill(0, prompt)]
    for i in range(n - 1):
        toks += be.decode_batch([0], [toks[-1]], [len(prompt) + i])
    return toks


async def one(c, prompt):
    text = ""
    async with c.stream("POST", f"http://127.0.0.1:{PORT}/v1/completions",
                        json={"prompt": prompt, "max_tokens": OUT, "stream": True, "ignore_eos": True}) as r:
        assert r.status_code == 200, r.status_code
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                ev = json.loads(line[6:])
                if ev["choices"]:
                    text += ev["choices"][0]["text"]
    return text


async def main():
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "part3", "launch.py"), "--world", str(W), "--fake",
                             "--port", str(PORT), "--max-seqs", "4", "--fake-delay", "0.01"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []

    def reader():
        for line in proc.stdout:
            lines.append(line.rstrip())
    import threading
    threading.Thread(target=reader, daemon=True).start()
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            t0 = time.time()
            while True:
                try:
                    if (await c.get(f"http://127.0.0.1:{PORT}/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                assert time.time() - t0 < 60 and proc.poll() is None, "\n".join(lines)
                await asyncio.sleep(0.2)
            # (1) concurrent requests through the balancer = the request alone
            want = "".join(f" w{t}" for t in alone(PROMPT, OUT))
            got = await asyncio.gather(*[one(c, PROMPT) for _ in range(N_REQ)])
            assert all(g == want for g in got), (want, got)
            st = (await c.get(f"http://127.0.0.1:{PORT}/stats")).json()
            assert st["requests_sent"] == [N_REQ // W] * W, st["requests_sent"]
            assert all(r["batch_steps"] > 0 for r in st["ranks"]), st
            print(f"(1) {N_REQ} concurrent requests via the balancer: all equal to the request alone; per rank "
                  f"{st['requests_sent']}")
            # (3) idle: no tick while parked
            await asyncio.sleep(1.0)
            t1 = (await c.get(f"http://127.0.0.1:{PORT}/stats")).json()["coordinator"]
            await asyncio.sleep(1.5)
            t2 = (await c.get(f"http://127.0.0.1:{PORT}/stats")).json()
            assert t2["coordinator"]["fake_ticks"] == t1["fake_ticks"], (t1, t2["coordinator"])
            assert t2["coordinator"]["parked_now"] == W, t2["coordinator"]
            assert all(r["parks"] >= 1 for r in t2["ranks"]), t2["ranks"]
            print(f"(2, 3) every tick answered for all {W} ranks ({t1['fake_ticks']} ticks); idle for 1.5 s: 0 new "
                  f"ticks, all {W} ranks parked")
            # (4) wake from idle
            g = await one(c, PROMPT)
            assert g == want
            t3 = (await c.get(f"http://127.0.0.1:{PORT}/stats")).json()["coordinator"]
            assert t3["wakes"] > t1["wakes"] and t3["fake_ticks"] > t1["fake_ticks"], (t1, t3)
            print(f"(4) a request after the idle period woke the group ({t3['wakes']} wakes so far) and completed")
        # (5) a rank dies: the launcher stops everything
        pids = [int(m.group(1)) for l in lines for m in [re.search(r"rank \d+: pid (\d+)", l)] if m]
        assert len(pids) == W, lines
        os.kill(pids[1], signal.SIGKILL)
        t0 = time.time()
        rc = proc.wait(timeout=20)
        dt = time.time() - t0
        alive = [p for p in pids if subprocess.run(["kill", "-0", str(p)], capture_output=True).returncode == 0]
        assert rc != 0 and not alive, (rc, alive)
        print(f"(5) rank 1 killed: launcher exited with code {rc} after {dt:.1f} s, no rank left running")
        print("LAUNCH TEST PASSED")
    finally:
        if proc.poll() is None:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
