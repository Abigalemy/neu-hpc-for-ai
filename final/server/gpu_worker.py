"""Part 5: the GPU worker process (vLLM's "GPU worker": one per GPU, loads the weights, runs the forward passes).

The engine core (engine_core.py) starts it as a child process and talks to it over a Pipe, as in the course demo
(suhabe/vllm-zmq-moe, moe_cutile_nccl_demo.py). The worker only executes: every decision (admission, preemption,
which sequences run in a step) is the engine core's scheduler. On the GPU it is the Part 3 engine, so the workers of
the K ranks run the MoE layers together over NCCL.

  WorkerProxy  engine-core side: the same methods as server.CBackend / FakeBackend, each one call over the Pipe
  run_worker   worker side: builds the backend, then answers calls until the Pipe closes
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# seconds a call may take before the worker counts as hung (alive but making no progress, e.g. stuck in a collective):
# the engine core then exits, which the launcher's watchdog sees, and the group stops. Loading the weights (the first
# message) has its own, longer limit. Neither applies while the core is parked: then no call is in flight.
CALL_TIMEOUT = float(os.environ.get("QWEN3_WORKER_TIMEOUT", "600"))
LOAD_TIMEOUT = float(os.environ.get("QWEN3_WORKER_LOAD_TIMEOUT", "3600"))

INFO = ("vocab", "max_len", "max_seqs", "block_size", "num_blocks", "world")
CALLS = ("idle_tick", "reserve", "release", "prefill", "decode_batch")


def run_worker(pipe, cfg: dict):
    """cfg: the engine core's arguments (fake or real engine, sizes, rank / world / NCCL id / coordinator)."""
    from ep_link import exit_with_parent
    exit_with_parent()            # the engine core is the parent
    from server import CBackend, FakeBackend
    if cfg["fake"]:
        link = None
        if cfg["world"] > 1:      # the fake engine's "all-reduce" is answered by the coordinator
            from ep_link import EpLink
            host, port = cfg["coord"].rsplit(":", 1)
            link = EpLink((host, int(port)), bytes.fromhex(cfg["coord_key"]), cfg["rank"], role="worker")
        be = FakeBackend(max_len=cfg["max_len"], max_seqs=cfg["max_seqs"], delay=cfg["fake_delay"],
                         num_blocks=-(-cfg["kv_tokens"] // 128) if cfg["kv_tokens"] > 0 else None,
                         link=link, world=cfg["world"])
    else:
        be = CBackend(cfg["lib"], cfg["model_dir"], cfg["max_len"], cfg["max_chunk"], cfg["yarn"], cfg["max_seqs"],
                      cfg["kv_tokens"], cfg["rank"], cfg["world"], bytes.fromhex(cfg["nccl_id"]))
    pipe.send({k: getattr(be, k, 1) for k in INFO})
    while True:
        try:
            method, args = pipe.recv()
        except EOFError:          # the engine core is gone
            return
        try:
            value = getattr(be, method)(*args)
            pipe.send(("ok", value, be.free_blocks()))      # free blocks after every call: read without a round trip
        except Exception as e:    # the engine core turns this into a failed step (its scheduler's error path)
            pipe.send(("err", f"{type(e).__name__}: {e}"))


class WorkerProxy:
    """The backend as the engine core's scheduler sees it: attributes from the worker's first message, and every call
    a round trip over the Pipe (only the scheduler thread calls)."""

    def __init__(self, pipe, proc):
        self.pipe, self.proc = pipe, proc
        info = self._recv(LOAD_TIMEOUT, "loading")
        for k in INFO:
            setattr(self, k, info[k])
        self.calls = 0
        self.free = self.num_blocks       # the worker's free KV blocks after the last call

    def free_blocks(self) -> int:
        """Any thread may read this (statistics): it is the value sent with the last reply, no Pipe traffic."""
        return self.free

    def _recv(self, timeout=None, what="a call"):
        timeout = CALL_TIMEOUT if timeout is None else timeout
        try:
            if not self.pipe.poll(timeout):
                print(f"GPU worker (pid {self.proc.pid}): no reply to {what} in {timeout:.0f} s (alive but hung?); "
                      f"engine core exiting", file=sys.stderr, flush=True)
                os._exit(5)
            return self.pipe.recv()
        except EOFError:          # the worker died: so does this engine core (the launcher then stops the group)
            print(f"GPU worker (pid {self.proc.pid}) is gone; engine core exiting", file=sys.stderr, flush=True)
            os._exit(4)

    def _call(self, method, *args):
        self.pipe.send((method, args))
        self.calls += 1
        reply = self._recv(what=method)
        if reply[0] != "ok":
            raise RuntimeError(f"GPU worker: {reply[1]}")
        self.free = reply[2]
        return reply[1]

    def __getattr__(self, name):
        if name in CALLS:
            return lambda *args: self._call(name, *args)
        raise AttributeError(name)
