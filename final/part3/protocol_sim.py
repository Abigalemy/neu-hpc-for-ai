"""Part 3: the lock-step protocol of the data + expert parallel server, with REAL processes on a CPU (no GPU, no NCCL).

The rule: an MoE all-to-all is a collective, so every rank takes part in every layer of every forward pass. While any
rank has work, each forward pass starts with a "tick" (in the engine: an NCCL all-reduce(max) of "I have work"); if
the max is 1 every rank runs the pass, idle ranks with 0 local tokens. When a tick returns 0 (nobody has work), every
rank saw the same 0 at the same tick, so all ranks park together: each tells the coordinator PARKED and sleeps until
WAKE (no GPU work while everyone is idle). A request reaching a parked rank sends WANT; the coordinator sends WAKE to
all once every rank has parked and someone wants to run (so a WANT arriving between the last tick and the park is not
lost). A parent process watches the ranks: if one exits, it stops the whole group (the others would wait forever in
a collective).

Here the NCCL collectives are a multiprocessing.Barrier with shared memory, the forward pass is LAYERS barrier waits,
and the coordinator is the parent. Checked: (1) random arrivals with idle gaps: every rank runs exactly its own
requests in order, all ranks run the same number of passes, and no tick happens while the group is parked; (2) a
request that arrives right after the idle tick and before the park still runs; (3) a rank that dies mid-pass stops
the whole group within the timeout.

  python part3/protocol_sim.py
"""
import multiprocessing as mp
import queue
import random
import time

LAYERS = 4


def allreduce_max(bar, flags, r, v):
    flags[r] = v
    bar.wait()
    m = max(flags[:])
    bar.wait()
    return m


def rank_main(r, W, bar, flags, ticks, inbox, to_coord, wakeq, jobs, log, die_at_pass, late_job_at_idle):
    """One rank: the scheduler loop of server.py, reduced to the protocol."""
    pending = []                 # local requests, in arrival order
    done_passes = 0
    idle_ticks = 0

    def drain():
        while True:
            try:
                pending.append(inbox.get_nowait())
            except queue.Empty:
                return

    while True:
        drain()
        if pending and pending[0] == "STOP":
            return
        have = 1 if pending else 0
        with ticks.get_lock():
            ticks.value += 1
        if allreduce_max(bar, flags, r, have) == 0:
            idle_ticks += 1
            if late_job_at_idle and r == 0 and idle_ticks == 1:
                inbox.put(("late", 999))             # arrives after the idle tick, before the park (the race)
            to_coord.put(("PARKED", r))
            assert wakeq.get() == "WAKE"             # this rank's own wake message: nothing to clear, no race
            continue
        if die_at_pass is not None and r == W - 1 and done_passes == die_at_pass:
            raise SystemExit(3)                      # a rank dies inside a pass
        for _ in range(LAYERS):                      # every layer: a collective all-to-all
            bar.wait()
        done_passes += 1
        if have:
            job = pending.pop(0)
            log.put((r, job))
        else:
            log.put((r, None))


def submit(r, job, inboxes, to_coord):
    inboxes[r].put(job)
    to_coord.put(("WANT", r))


def coordinator_step(to_coord, wakeqs, parked, want, W, drop_early_want=False):
    """Handle one message; WAKE every rank (one message each) once all are parked and someone wants to run. A WANT
    that arrives before the last PARKED is remembered (drop_early_want=True forgets it: the lost-wake bug)."""
    try:
        kind, r = to_coord.get(timeout=0.05)
    except queue.Empty:
        return
    if kind == "PARKED":
        parked.add(r)
    elif len(parked) == W or not drop_early_want:
        want.add(r)
    if len(parked) == W and want:
        want.clear()
        parked.clear()
        for q in wakeqs:
            q.put("WAKE")


def run(W=3, scenario="random", seed=0, timeout=20, drop_early_want=False):
    ctx = mp.get_context("spawn")
    bar = ctx.Barrier(W, timeout=5)
    flags = ctx.Array("i", W)
    ticks = ctx.Value("i", 0)
    to_coord, log = ctx.Queue(), ctx.Queue()
    inboxes = [ctx.Queue() for _ in range(W)]
    wakeqs = [ctx.Queue() for _ in range(W)]
    procs = [ctx.Process(target=rank_main, args=(r, W, bar, flags, ticks, inboxes[r], to_coord, wakeqs[r], None, log,
                                                 2 if scenario == "die" else None, scenario == "late"))
             for r in range(W)]
    for p in procs:
        p.start()
    rs = random.Random(seed)
    plan = {r: [] for r in range(W)}
    # arrivals in bursts separated by idle gaps (the group parks in between)
    bursts = [[(rs.randrange(W), (b, i)) for i in range(rs.randrange(1, 6))] for b in range(4)]
    if scenario == "late":
        bursts = [[(1, (0, 0))]]
    parked, want = set(), set()
    t0 = time.time()
    ticks_while_parked = []
    try:
        for burst in bursts:
            for r, job in burst:
                plan[r].append(job)
                submit(r, job, inboxes, to_coord)
            t_end = time.time() + 0.5
            while time.time() < t_end:               # run until the group parks again
                coordinator_step(to_coord, wakeqs, parked, want, W, drop_early_want)
                if any(p.exitcode not in (None, 0) for p in procs):
                    raise RuntimeError(f"rank exited: {[p.exitcode for p in procs]}")
            if len(parked) == W:                     # everyone parked: no tick may happen while idle
                before = ticks.value
                time.sleep(0.3)
                ticks_while_parked.append(ticks.value - before)
        if scenario == "late":
            plan[0].append(("late", 999))
            t_end = time.time() + 1.0
            while time.time() < t_end:
                coordinator_step(to_coord, wakeqs, parked, want, W, drop_early_want)
        for r in range(W):
            inboxes[r].put("STOP")
            to_coord.put(("WANT", r))
        while any(p.is_alive() for p in procs) and time.time() - t0 < timeout:
            coordinator_step(to_coord, wakeqs, parked, want, W, drop_early_want)
            if any(p.exitcode not in (None, 0) for p in procs):
                raise RuntimeError(f"rank exited: {[p.exitcode for p in procs]}")
    except RuntimeError as e:
        t_fail = time.time()
        for p in procs:                              # the parent stops the whole group
            p.terminate()
        for p in procs:
            p.join(timeout=5)
        return dict(failed=str(e), stop_s=time.time() - t_fail, alive=[p.is_alive() for p in procs])
    for p in procs:
        p.join(timeout=5)
    got = {r: [] for r in range(W)}
    passes = {r: 0 for r in range(W)}
    while True:
        try:
            r, job = log.get(timeout=0.2)
        except queue.Empty:
            break
        passes[r] += 1
        if job is not None:
            got[r].append(job)
    return dict(plan=plan, got=got, passes=passes, ticks_while_parked=ticks_while_parked,
                alive=[p.is_alive() for p in procs])


if __name__ == "__main__":
    for seed in range(3):
        res = run(W=3, scenario="random", seed=seed)
        assert not any(res["alive"]), res
        assert res["got"] == res["plan"], (res["got"], res["plan"])
        assert len(set(res["passes"].values())) == 1, res["passes"]
        assert res["ticks_while_parked"] and all(t == 0 for t in res["ticks_while_parked"]), res["ticks_while_parked"]
    print(f"bursts with idle gaps (3 processes, 3 seeds): every rank ran exactly its own requests in order, the same "
          f"number of passes on all ranks, 0 ticks while parked ({res['ticks_while_parked']})")
    res = run(W=3, scenario="late")
    assert not any(res["alive"]) and res["got"] == res["plan"], res
    print("late request (after the idle tick, before the park): woken and run")
    res = run(W=3, scenario="late", drop_early_want=True, timeout=5)
    assert res.get("got") != res.get("plan") or "failed" in res, res
    print("  (the same with a coordinator that drops an early WANT: the late request is lost, so the check works)")
    res = run(W=3, scenario="die")
    assert "failed" in res and not any(res["alive"]) and res["stop_s"] < 6, res
    print(f"a rank dies inside a pass: the parent stopped all ranks in {res['stop_s']:.2f} s ({res['failed']})")
