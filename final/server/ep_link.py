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


class EpLink:
    def __init__(self, address, authkey: bytes, rank: int):
        self.conn = Client(address, authkey=authkey)
        self.rank = rank
        self.lock = threading.Lock()          # want() comes from HTTP threads, the rest from the scheduler thread
        self.parks = 0
        self._send(("HELLO", rank))

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

    def fake_tick(self, have: int) -> int:
        self._send(("TICK", self.rank, have))
        msg = self._recv()
        if not (isinstance(msg, tuple) and msg[0] == "TICK"):
            raise RuntimeError(f"rank {self.rank}: expected a tick result, got {msg!r}")
        return msg[1]


class Coordinator:
    """Accepts `world` ranks, then answers PARKED / WANT / TICK in one thread (the only sender)."""

    def __init__(self, world: int, authkey: bytes, host: str = "127.0.0.1"):
        self.world = world
        self.listener = Listener((host, 0), authkey=authkey)
        self.address = self.listener.address
        self.conns = {}
        self.inbox = queue.Queue()
        self.parked, self.want, self.ticks = set(), False, {}
        self.stats = {"wakes": 0, "fake_ticks": 0, "parked_now": 0}
        self.error = None

    def accept_all(self):
        while len(self.conns) < self.world:
            c = self.listener.accept()
            kind, r = c.recv()
            if kind != "HELLO" or r in self.conns or not 0 <= r < self.world:
                raise RuntimeError(f"bad hello {kind!r} {r!r}")
            self.conns[r] = c
            threading.Thread(target=self._reader, args=(r, c), daemon=True).start()

    def _reader(self, r, c):
        try:
            while True:
                self.inbox.put(c.recv())
        except (EOFError, OSError) as e:          # the rank's process ended: the launcher's watchdog acts on it
            self.inbox.put(("GONE", r, repr(e)))

    def serve(self):
        while True:
            msg = self.inbox.get()
            kind = msg[0]
            if kind == "PARKED":
                self.parked.add(msg[1])
            elif kind == "WANT":
                self.want = True
            elif kind == "TICK":
                self.ticks[msg[1]] = msg[2]
                if len(self.ticks) == self.world:
                    m = max(self.ticks.values())
                    self.ticks.clear()
                    self.stats["fake_ticks"] += 1
                    for c in self.conns.values():
                        c.send(("TICK", m))
            elif kind == "GONE":
                self.error = f"rank {msg[1]} disconnected ({msg[2]})"
                return
            if len(self.parked) == self.world and self.want:
                self.want = False
                self.parked.clear()
                self.stats["wakes"] += 1
                for c in self.conns.values():
                    c.send("WAKE")
            self.stats["parked_now"] = len(self.parked)
