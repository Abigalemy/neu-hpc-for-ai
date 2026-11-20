"""Part 3 / Part 5: the lock-step coordination of K data + expert parallel ranks (one server process per GPU).

Every MoE layer is a collective (NCCL all-to-all), so every rank must run every forward pass. While any rank has work,
the engine starts each forward pass with a tick (an NCCL all-reduce(max) of "this rank has work", in the engine);
ranks without work call idle_tick(), which runs the pass with 0 local tokens. When a tick returns 0 nobody has work:
all ranks saw the same 0 at the same tick, so all of them park (no GPU work while idle) until the coordinator wakes
them. A request arriving at a rank sends WANT; the coordinator sends WAKE to every rank once ALL ranks are parked and
someone wants to run, so a request that arrives between the last tick and the park is not lost. (Checked with real
processes in part3/protocol_sim.py.)

  EpLink       rank side: want() from the HTTP thread, park() / fake_tick() from the scheduler thread
  Coordinator  launcher side (part3/launch.py): one connection per rank, one thread that answers

fake_tick: the fake engine (server.py --fake) has no NCCL; its tick goes to the coordinator, which answers once all
ranks have sent theirs with the max, the same semantics as the engine's all-reduce, so the launcher, the load balancer
and park / wake can be tested with real processes on a CPU.
"""
import os
import queue
import sys
import threading
from multiprocessing.connection import Client, Listener


def exit_with_parent(code: int = 5):
    """Exit when this process's parent is gone (even after SIGKILL, or a uvicorn that re-raises SIGTERM before the
    launcher's cleanup runs): a process left behind would hold its port, its GPU, or wait in a collective forever."""
    import time
    parent = os.getppid()

    def watch():
        while True:
            if os.getppid() != parent:
                os._exit(code)
            time.sleep(0.2)
    threading.Thread(target=watch, daemon=True).start()


class EpLink:
    """role: "rank" (Part 3: one process does everything), or in Part 5 "core" (engine core: want / park / report),
    "worker" (fake GPU worker: fake_tick), "api" (API server: load queries)."""

    def __init__(self, address, authkey: bytes, rank: int, role: str = "rank"):
        self.conn = Client(address, authkey=authkey)
        self.rank, self.role = rank, role
        self.lock = threading.Lock()          # want() comes from HTTP / IO threads, the rest from the scheduler thread
        self.parks = 0
        self._send(("HELLO", rank, role))

    def _gone(self, e):
        # the coordinator (launcher) is gone: this rank cannot take part in the group any more; leaving at once is
        # better than a scheduler loop that keeps failing (the other ranks are stopped by the launcher)
        print(f"rank {self.rank}: coordinator connection lost ({e!r}); exiting", file=sys.stderr, flush=True)
        os._exit(3)

    def _send(self, msg):
        try:
            with self.lock:
                self.conn.send(msg)
        except (EOFError, OSError) as e:
            self._gone(e)

    def _recv(self):
        try:
            return self.conn.recv()
        except (EOFError, OSError) as e:
            self._gone(e)

    def want(self):
        self._send(("WANT", self.rank))

    def park(self):
        """All ranks are idle: sleep until the coordinator wakes every rank."""
        self.parks += 1
        self._send(("PARKED", self.rank))
        msg = self._recv()
        if msg != "WAKE":
            raise RuntimeError(f"rank {self.rank}: expected WAKE, got {msg!r}")

    def report(self, stats: dict):
        """Engine core -> coordinator: its load (stats["load"] = running + waiting requests) and statistics."""
        self._send(("REPORT", self.rank, stats))

    def loads(self) -> list:
        """API server -> coordinator: every engine core's last reported load (only this thread receives on an api link)."""
        self._send(("LOADQ", self.rank))
        msg = self._recv()
        if not (isinstance(msg, tuple) and msg[0] == "LOAD"):
            raise RuntimeError(f"api {self.rank}: expected loads, got {msg!r}")
        return msg[1]

    def fake_tick(self, have: int) -> int:
        self._send(("TICK", self.rank, have))
        msg = self._recv()
        if not (isinstance(msg, tuple) and msg[0] == "TICK"):
            raise RuntimeError(f"rank {self.rank}: expected a tick result, got {msg!r}")
        return msg[1]


class Coordinator:
    """Accepts `world` connections of each role in `roles`, then answers in one thread (the only sender):
      PARKED / WANT   from "rank" or "core": when every one of them is parked and someone wants to run, WAKE them all
      TICK            fake all-reduce(max) over the `world` senders, answered to each sender
      REPORT          an engine core's load and statistics (Part 5's DP coordinator: load balancing)
      LOADQ           an API server asks for every engine core's load
      GONE            any process ended: stop (the launcher's watchdog stops the group)"""

    def __init__(self, world: int, authkey: bytes, host: str = "127.0.0.1", roles=("rank",)):
        self.world, self.roles = world, tuple(roles)
        self.listener = Listener((host, 0), authkey=authkey)
        self.address = self.listener.address
        self.conns = {}                       # (role, rank) -> connection
        self.inbox = queue.Queue()
        self.parked, self.want, self.ticks = set(), False, {}
        self.loads = [0] * world
        self.stats = {"wakes": 0, "fake_ticks": 0, "parked_now": 0, "cores": [None] * world}
        self.error = None

    def accept_all(self):
        while len(self.conns) < self.world * len(self.roles):
            c = self.listener.accept()
            msg = c.recv()
            key = (msg[2] if len(msg) > 2 else "rank", msg[1])
            if msg[0] != "HELLO" or key in self.conns or key[0] not in self.roles or not 0 <= key[1] < self.world:
                raise RuntimeError(f"bad hello {msg!r}")
            self.conns[key] = c
            threading.Thread(target=self._reader, args=(key, c), daemon=True).start()

    def _reader(self, key, c):
        try:
            while True:
                self.inbox.put((key, c.recv()))
        except (EOFError, OSError) as e:          # the process ended: the launcher's watchdog acts on it
            self.inbox.put((key, ("GONE", key[1], repr(e))))

    def serve(self):
        parkers = [k for k in self.conns if k[0] in ("rank", "core")]
        while True:
            key, msg = self.inbox.get()
            kind = msg[0]
            if kind == "PARKED":
                self.parked.add(key)
            elif kind == "WANT":
                self.want = True
            elif kind == "TICK":
                self.ticks[key] = msg[2]
                if len(self.ticks) == self.world:
                    m = max(self.ticks.values())
                    for k in self.ticks:
                        self.conns[k].send(("TICK", m))
                    self.ticks.clear()
                    self.stats["fake_ticks"] += 1
            elif kind == "REPORT":
                self.loads[msg[1]] = msg[2].get("load", 0)
                self.stats["cores"][msg[1]] = msg[2]
            elif kind == "LOADQ":
                self.conns[key].send(("LOAD", list(self.loads)))
            elif kind == "GONE":
                self.error = f"{key[0]} {key[1]} disconnected ({msg[2]})"
                return
            if len(self.parked) == len(parkers) and self.want:
                self.want = False
                self.parked.clear()
                self.stats["wakes"] += 1
                for k in parkers:
                    self.conns[k].send("WAKE")
            self.stats["parked_now"] = len(self.parked)
