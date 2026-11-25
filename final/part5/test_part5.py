"""CPU test of the Part 5 process layout (part5/launch.py) with fake engines in real processes: balancer -> K API servers
-> (ZMQ, many-to-many) -> K engine cores -> (Pipe) -> K GPU workers, plus the DP coordinator.

  python part5/test_part5.py

For K = 1 and K = 3:
  (1) concurrent streamed requests through the balancer give exactly the tokens of the same request run alone on one
      fake engine; for K = 3 every engine core ran some, and API servers sent requests to engine cores other than
      their own (many-to-many); a non-streamed chat request works
  (2) a client that leaves mid-stream: its ABORT reaches the engine core, every KV block is free again, load 0
  (3) K = 3: once idle no fake tick happens and every engine core is parked; (4) a request then wakes the group
Then for K = 2, once per process kind: (5) killing an API server, an engine core, or a GPU worker stops the launcher
and every process (no orphan worker left).
"""
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

import httpx

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "server"))
from server import FakeBackend  # noqa: E402

OUT = 8
PROMPT = [11, 12, 13, 14]


def alone(prompt, n):
    be = FakeBackend(max_seqs=1)
    be.reserve(0, len(prompt) + n)
    toks = [be.prefill(0, prompt)]
    for i in range(n - 1):
        toks += be.decode_batch([0], [toks[-1]], [len(prompt) + i])
    return toks


class Group:
    def __init__(self, world, port):
        self.world, self.port, self.lines = world, port, []
        self.proc = subprocess.Popen([sys.executable, os.path.join(HERE, "part5", "launch.py"), "--world", str(world),
                                      "--fake", "--port", str(port), "--max-seqs", "4", "--fake-delay", "0.01"],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        threading.Thread(target=self._read, daemon=True).start()
        self.url = f"http://127.0.0.1:{port}"

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line.rstrip())

    async def ready(self, c):
        t0 = time.time()
        while True:
            try:
                if (await c.get(self.url + "/health")).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            assert time.time() - t0 < 60 and self.proc.poll() is None, "\n".join(self.lines)
            await asyncio.sleep(0.2)

    def pids(self, pattern):
        return [int(m.group(1)) for l in self.lines for m in [re.search(pattern, l)] if m]

    def all_pids(self):
        return sorted(set(self.pids(r": pid (\d+)") + self.pids(r"GPU worker pid (\d+)")))


async def stream(c, url, n=OUT, stop_after=None):
    text, k = "", 0
    async with c.stream("POST", url + "/v1/completions",
                        json={"prompt": PROMPT, "max_tokens": n, "stream": True, "ignore_eos": True}) as r:
        assert r.status_code == 200, r.status_code
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                ev = json.loads(line[6:])
                if ev["choices"]:
                    text += ev["choices"][0]["text"]
                    k += 1
                    if stop_after and k >= stop_after:
                        return text          # leaving the block closes the connection mid-stream
    return text


async def until(cond, what, timeout=10):
    t0 = time.time()
    while not await cond():
        assert time.time() - t0 < timeout, what
        await asyncio.sleep(0.1)


async def serve_checks(world, port):
    g = Group(world, port)
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            await g.ready(c)
            stats = lambda: c.get(g.url + "/stats")
            # (1)
            async def reported():
                return all((await stats()).json()["coordinator"]["cores"])
            await until(reported, "an engine core never reported")
            nb = [x["free_blocks"] for x in (await stats()).json()["coordinator"]["cores"]]   # all free at start
            want = "".join(f" w{t}" for t in alone(PROMPT, OUT))
            n_req = 4 * world
            got = await asyncio.gather(*[stream(c, g.url) for _ in range(n_req)])
            assert all(x == want for x in got), (want, got)
            st = (await stats()).json()
            sent = [r["sent"] for r in st["ranks"]]
            cores = st["coordinator"]["cores"]
            assert sum(map(sum, sent)) == n_req, sent
            if world > 1:
                assert all(cs["batch_steps"] > 0 for cs in cores), cores
                assert any(sent[i][e] for i in range(world) for e in range(world) if e != i), sent
            r = await c.post(g.url + "/v1/chat/completions",
                             json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
            assert r.status_code == 200 and r.json()["usage"]["completion_tokens"] >= 1, r.text
            print(f"K={world} (1) {n_req} concurrent requests equal the request alone; API server -> engine core "
                  f"counts {sent}; chat ok")
            # (2) leave mid-stream
            await stream(c, g.url, n=2000, stop_after=3)

            async def released():
                cs = (await stats()).json()["coordinator"]["cores"]
                return all(x["load"] == 0 and x["free_blocks"] == x_nb for x, x_nb in zip(cs, nb))
            try:
                await until(released, "blocks not released")
            except AssertionError:
                print((await stats()).json()); print("\n".join(g.lines)); raise
            print(f"K={world} (2) a client that left mid-stream: ABORT reached its engine core, all KV blocks free")
            if world > 1:
                # (3) idle
                await asyncio.sleep(1.0)
                t1 = (await stats()).json()["coordinator"]
                await asyncio.sleep(1.5)
                t2 = (await stats()).json()["coordinator"]
                assert t2["fake_ticks"] == t1["fake_ticks"] and t2["parked_now"] == world, (t1, t2)
                # (4) wake
                assert await stream(c, g.url) == want
                t3 = (await stats()).json()["coordinator"]
                assert t3["wakes"] > t1["wakes"], (t1, t3)
                print(f"K={world} (3, 4) idle 1.5 s: 0 ticks, all {world} engine cores parked; a request woke them "
                      f"({t3['wakes']} wakes) and completed")
    finally:
        g.proc.terminate()
        g.proc.wait(timeout=20)
    # each process leaves by itself within ~0.2 s of its parent (ep_link.exit_with_parent): workers after their core
    alive = lambda: [p for p in g.all_pids() if subprocess.run(["kill", "-0", str(p)], capture_output=True).returncode == 0]
    t0 = time.time()
    while alive() and time.time() - t0 < 3:
        time.sleep(0.1)
    assert not alive(), f"processes left 3 s after SIGTERM: {alive()}"
    print(f"K={world} SIGTERM to the launcher: all {len(g.all_pids())} processes gone after {time.time() - t0:.1f} s")


async def kill_check(kind, port):
    g = Group(2, port)
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            await g.ready(c)
            assert await stream(c, g.url) == "".join(f" w{t}" for t in alone(PROMPT, OUT))
        pids = g.all_pids()
        victim = {"api server": g.pids(r"api server 1: pid (\d+),")[0],
                  "engine core": g.pids(r"engine core 1: pid (\d+),")[0],
                  "GPU worker": g.pids(r"engine core 1: pid \d+, GPU worker pid (\d+)")[0]}[kind]
        os.kill(victim, signal.SIGKILL)
        t0 = time.time()
        rc = g.proc.wait(timeout=30)
        dt = time.time() - t0
        await asyncio.sleep(1.5)        # a worker notices a dead parent within 0.2 s
        left = [p for p in pids if p != victim and
                subprocess.run(["kill", "-0", str(p)], capture_output=True).returncode == 0]
        assert rc != 0 and not left, (kind, rc, left)
        print(f"K=2 (5) {kind} 1 killed: launcher exited ({rc}) after {dt:.1f} s, none of the other {len(pids) - 1} "
              f"processes left")
    finally:
        if g.proc.poll() is None:
            g.proc.kill()


async def main():
    await serve_checks(1, 18900)
    await serve_checks(3, 18910)
    for i, kind in enumerate(("api server", "engine core", "GPU worker")):
        await kill_check(kind, 18920 + 10 * i)
    print("PART 5 TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())
