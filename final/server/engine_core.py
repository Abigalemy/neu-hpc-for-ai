"""Part 5: the engine core process (vLLM's "engine core": scheduler and KV cache management, one per DP rank).

  python engine_core.py --rank r --world K --ipc DIR [--coord host:port] (--fake | --model-dir ... --lib ...) [sizes]

What runs here:
  - the GPU worker, started as a child process (gpu_worker.py) and driven over a Pipe;
  - the Scheduler of server.py, unchanged: continuous batching, admission by free KV blocks, recompute preemption,
    and with K > 1 the lock step (idle_tick / park) of Part 3. Its backend is the worker proxy;
  - ZMQ to every API server, as in the course demo (suhabe/vllm-zmq-moe): a DEALER (identity = rank) connected to each
    API server's ROUTER brings ADD / ABORT, and a PUSH connected to each API server's PULL takes OUTPUT back to the API
    that sent the request (many-to-many: any API server can use any engine core);
  - with a coordinator: load reports (running + waiting requests, and statistics) for the API servers' routing.

Threads: the main thread receives (Poller over the DEALERs); the scheduler thread is the only one that sends results
(GenRequest.emit is only called there), so each ZMQ socket is used by one thread.
Messages (JSON): ADD {id, prompt_ids, max_new, min_new, ignore_eos}, ABORT {id};
OUTPUT {id, token, finish} with finish null / "stop" / "length" / "error".
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time

import zmq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def endpoints(ipc_dir: str, api: int):
    return f"ipc://{ipc_dir}/req-{api}", f"ipc://{ipc_dir}/out-{api}"


class RemoteRequest:
    """What the Scheduler needs of a request (the fields of server.GenRequest); emit sends the token to the API."""

    def __init__(self, rid, ids, max_new, min_new, ignore_eos, out_sock, done):
        self.rid, self.ids, self.max_new, self.min_new, self.ignore_eos = rid, ids, max_new, min_new, ignore_eos
        self.out, self.done = out_sock, done
        self.slot = self.pos = self.last_tok = None
        self.n_gen = 0
        self.gen = []
        self.cancelled = False

    def emit(self, tok, finish):
        self.out.send_json({"id": self.rid, "token": tok, "finish": finish})
        if finish:
            self.done(self.rid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--ipc", required=True, help="directory of the API servers' ZMQ endpoints")
    ap.add_argument("--coord", default="", help="host:port of the coordinator (key in QWEN3_COORD_KEY)")
    ap.add_argument("--model-dir")
    ap.add_argument("--lib", default=os.path.join(os.path.dirname(__file__), "..", "build", "libqwen3engine.so"))
    ap.add_argument("--max-len", type=int, default=4096 + 1024 + 64)
    ap.add_argument("--max-chunk", type=int, default=4096)
    ap.add_argument("--yarn", type=float, default=0.0)
    ap.add_argument("--max-seqs", type=int, default=1)
    ap.add_argument("--kv-tokens", type=int, default=0)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--fake-delay", type=float, default=0.0)
    a, _ = ap.parse_known_args()          # the launcher gives every process the same arguments

    import faulthandler, signal
    faulthandler.register(signal.SIGUSR1)        # kill -USR1 <pid>: every thread's stack (debugging a stuck core)
    from ep_link import exit_with_parent
    exit_with_parent()                           # the launcher is the parent
    from server import FakeTokenizer, Scheduler, load_tokenizer
    from gpu_worker import WorkerProxy, run_worker
    link = None
    if a.coord:
        from ep_link import EpLink
        host, port = a.coord.rsplit(":", 1)
        link = EpLink((host, int(port)), bytes.fromhex(os.environ["QWEN3_COORD_KEY"]), a.rank, role="core")
    eos = [FakeTokenizer.eos_token_id] if a.fake else load_tokenizer(a.model_dir)[1]

    # the GPU worker: a child process on this rank's GPU (CUDA_VISIBLE_DEVICES is set by the launcher)
    cfg = {"fake": a.fake, "rank": a.rank, "world": a.world, "coord": a.coord,
           "coord_key": os.environ.get("QWEN3_COORD_KEY", ""), "nccl_id": os.environ.get("QWEN3_NCCL_ID", ""),
           "lib": a.lib, "model_dir": a.model_dir, "max_len": a.max_len, "max_chunk": a.max_chunk, "yarn": a.yarn,
           "max_seqs": a.max_seqs, "kv_tokens": a.kv_tokens, "fake_delay": a.fake_delay}
    ctx_mp = mp.get_context("spawn")
    mine, theirs = ctx_mp.Pipe()
    proc = ctx_mp.Process(target=run_worker, args=(theirs, cfg), daemon=False)
    proc.start()
    theirs.close()
    print(f"engine core {a.rank}: pid {os.getpid()}, GPU worker pid {proc.pid}", flush=True)
    backend = WorkerProxy(mine, proc)
    print(f"engine core {a.rank}: worker ready (max_seqs {backend.max_seqs}, KV pool {backend.num_blocks} blocks x "
          f"{backend.block_size} tokens)", flush=True)

    def watch_worker():           # a worker that dies while this core waits elsewhere (parked) ends the core too
        proc.join()
        print(f"engine core {a.rank}: GPU worker exited with code {proc.exitcode}", file=sys.stderr, flush=True)
        os._exit(4)
    threading.Thread(target=watch_worker, daemon=True).start()

    sched = Scheduler(backend, eos, link if a.world > 1 else None)

    ctx = zmq.Context()
    poller = zmq.Poller()
    out_of = {}                   # request socket (one per API server) -> the PUSH back to that API server
    for api in range(a.world):
        req_ep, out_ep = endpoints(a.ipc, api)
        rq = ctx.socket(zmq.DEALER)
        rq.setsockopt(zmq.IDENTITY, str(a.rank).encode())
        rq.connect(req_ep)
        out = ctx.socket(zmq.PUSH)
        out.connect(out_ep)
        out_of[rq] = out
        poller.register(rq, zmq.POLLIN)
        info = {k: getattr(backend, k) for k in ("vocab", "max_len", "max_seqs", "block_size", "num_blocks")}
        rq.send_json({"type": "READY", "rank": a.rank, **info})

    live = {}                     # (api socket, request id) -> RemoteRequest, until its final token is sent
    lock = threading.Lock()

    def report_loop():            # load for the API servers' routing (DP coordinator), and statistics for the tests
        last = None
        while True:
            st = sched.stats()
            st["load"] = st["running"] + st["waiting"]
            st["worker_calls"] = backend.calls
            if st != last:
                link.report(st)
                last = st
            time.sleep(0.02)
    if link is not None:
        threading.Thread(target=report_loop, daemon=True).start()

    while True:
        for rq, _ in poller.poll():
            msg = rq.recv_json()
            if msg["type"] == "ADD":
                key = (rq, msg["id"])

                def done(rid, rq=rq):
                    with lock:
                        live.pop((rq, rid), None)
                r = RemoteRequest(msg["id"], msg["prompt_ids"], msg["max_new"], msg["min_new"], msg["ignore_eos"],
                                  out_of[rq], done)
                with lock:
                    live[key] = r
                sched.submit(r)
            elif msg["type"] == "ABORT":       # the client went away: the scheduler drops it and frees its blocks
                with lock:
                    r = live.pop((rq, msg["id"]), None)
                if r is not None:
                    r.cancelled = True


if __name__ == "__main__":
    main()
