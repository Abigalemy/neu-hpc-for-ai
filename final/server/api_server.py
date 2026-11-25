"""Part 5: the API server process (vLLM's "API server": HTTP, tokenization, streaming; one per DP rank).

  python api_server.py --api i --world K --ipc DIR --port P [--coord host:port] (--fake | --model-dir ...)

The HTTP side is server.py's create_app unchanged (OpenAI endpoints, validation, chat template, incremental
detokenizer, SSE). Instead of a scheduler in this process it gets EngineClient, which sends each request to one of the
K engine cores over ZMQ, as in the course demo (suhabe/vllm-zmq-moe): a ROUTER (bound here) addresses an engine core by
its identity, a PULL (bound here) receives every engine core's OUTPUT messages. Any API server can use any engine core
(many-to-many, as in vLLM); the engine core is the least loaded one according to the coordinator's table (the engine
cores report their running + waiting requests), counting the requests this API server sent since the last table.

Startup: the ZMQ endpoints are bound first; HTTP starts only once every engine core has sent READY (its model is
loaded), so /health is ok only when the whole group can serve.
"""
import argparse
import asyncio
import itertools
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import zmq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_core import endpoints  # noqa: E402


class EngineClient:
    def __init__(self, ipc_dir: str, api: int, world: int, link=None):
        self.api, self.world, self.link = api, world, link
        self.ctx = zmq.Context()
        req_ep, out_ep = endpoints(ipc_dir, api)
        self.req = self.ctx.socket(zmq.ROUTER)
        self.req.setsockopt(zmq.ROUTER_MANDATORY, 1)      # an unknown engine identity is an error, not a silent drop
        self.req.bind(req_ep)
        self.out = self.ctx.socket(zmq.PULL)
        self.out.bind(out_ep)
        self.pending = {}                                # request id -> (event loop, asyncio.Queue)
        self.ids = itertools.count()
        self.sent = [0] * world                          # requests sent to each engine core (statistics)
        self.loads, self.since = [0] * world, [0] * world
        self.rr = itertools.count(api)                   # ties: start at a different engine core per API server

    def wait_ready(self) -> SimpleNamespace:
        """Block until every engine core has said READY; returns the engine limits (the same on every rank)."""
        seen = {}
        while len(seen) < self.world:
            ident, body = self.req.recv_multipart()
            msg = json.loads(body)
            assert msg["type"] == "READY", msg
            seen[int(ident)] = msg
        lim = {k: seen[0][k] for k in ("vocab", "max_len", "max_seqs", "block_size", "num_blocks")}
        for r, m in seen.items():
            if any(m[k] != v for k, v in lim.items()):
                raise SystemExit(f"api {self.api}: engine core {r} has other limits than core 0: {m} vs {lim}")
        threading.Thread(target=self._receive, daemon=True).start()
        if self.link is not None:
            threading.Thread(target=self._poll_loads, daemon=True).start()
        return SimpleNamespace(**lim)

    def _receive(self):
        while True:
            msg = self.out.recv_json()
            p = self.pending.get(msg["id"])
            if p is None:                                # its client already left (an ABORT is on its way)
                continue
            try:
                p[0].call_soon_threadsafe(p[1].put_nowait, (msg["token"], msg["finish"]))
            except RuntimeError:                         # that event loop is closed
                pass

    def _poll_loads(self):
        while True:
            loads = self.link.loads()
            self.loads, self.since = loads, [0] * self.world
            time.sleep(0.02)

    def _pick(self) -> int:
        score = [l + s for l, s in zip(self.loads, self.since)]
        best = min(score)
        start = next(self.rr)
        for i in range(self.world):
            e = (start + i) % self.world
            if score[e] == best:
                return e

    def _send(self, engine: int, msg: dict):
        self.req.send_multipart([str(engine).encode(), json.dumps(msg).encode()])

    async def tokens(self, prompt_ids, max_new, min_new, ignore_eos):
        """Async generator of (token_id, finish_reason or None), as server.Scheduler.tokens."""
        rid = f"{self.api}-{next(self.ids)}"
        q = asyncio.Queue()
        self.pending[rid] = (asyncio.get_running_loop(), q)
        finished = added = False
        try:                                             # from here on, pending[rid] is removed whatever happens
            e = self._pick()
            self._send(e, {"type": "ADD", "id": rid, "prompt_ids": list(prompt_ids), "max_new": max_new,
                           "min_new": min_new, "ignore_eos": ignore_eos})
            added = True
            self.sent[e] += 1                            # counted once the request was actually sent
            self.since[e] += 1
            while True:
                tok, fin = await q.get()
                if fin:                                  # set before the yield: a consumer that closes the generator
                    finished = True                      # on the last token must not send an ABORT for it
                yield tok, fin
                if fin:
                    return
        finally:
            self.pending.pop(rid, None)
            if added and not finished:                   # the client left early: the engine core frees its blocks
                self._send(e, {"type": "ABORT", "id": rid})

    def stats(self) -> dict:
        return {"api": self.api, "world": self.world, "sent": list(self.sent), "loads": list(self.loads),
                "in_flight": len(self.pending)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--ipc", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--coord", default="")
    ap.add_argument("--model-dir")
    ap.add_argument("--served-model-name", default="Qwen/Qwen3-4B")
    ap.add_argument("--fake", action="store_true")
    a, _ = ap.parse_known_args()          # the launcher gives every process the same arguments

    import faulthandler, signal
    faulthandler.register(signal.SIGUSR1)
    from ep_link import exit_with_parent
    exit_with_parent()                           # the launcher is the parent
    from server import FakeTokenizer, create_app, load_tokenizer
    link = None
    if a.coord:
        from ep_link import EpLink
        host, port = a.coord.rsplit(":", 1)
        link = EpLink((host, int(port)), bytes.fromhex(os.environ["QWEN3_COORD_KEY"]), a.api, role="api")
    tok, eos = (FakeTokenizer(), [FakeTokenizer.eos_token_id]) if a.fake else load_tokenizer(a.model_dir)
    client = EngineClient(a.ipc, a.api, a.world, link if a.world > 1 else None)
    print(f"api server {a.api}: pid {os.getpid()}, waiting for {a.world} engine core(s)", flush=True)
    limits = client.wait_ready()
    print(f"api server {a.api}: all engine cores ready; HTTP on port {a.port}", flush=True)
    import uvicorn
    uvicorn.run(create_app(limits, tok, a.served_model_name, eos, sched=client), host="0.0.0.0", port=a.port,
                log_level="warning")


if __name__ == "__main__":
    main()
