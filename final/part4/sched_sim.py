"""Part 4: would chunked prefill lower the mean ITL of prefill_heavy (30B, 1x H200)? A scheduler simulation on the CPU.

  .venv/bin/python part4/sched_sim.py      # output: results/part4/sched_sim_2026-11-25.txt

Step-time model from the 1x H200 per-stage profile after the GEMM tiles (results/part4/profile_30B_h200x1_tiles_2026-11-21.txt):
prefill(n tokens) and decode(batch b) in ms, linear between the measured points. Workload as AIPerf's prefill_heavy:
2048 in / 32 out, concurrency 8, a new request as soon as one finishes. Policies:
  now          our scheduler: all waiting prefills (whole), then one decode step of the running sequences
  interleaved  one prefill chunk per step, then a decode step of the running sequences
  mixed        one prefill chunk per step with the running sequences' decode tokens in the same forward
               (assumed cost: the chunk's prefill time + 0.15 ms per decode row)
A model, not a measurement: "now" gives 389 ms / 25.6 ms against the measured 391 / 24.4, the check that the model is
close enough to compare policies. In it every chunked policy raises the mean ITL: each chunk is a forward of its own
with the fixed cost of a forward (34 ms at 256 tokens: the 48 MoE layers' expert weights, of which a chunk of a few
hundred tokens x top-8 touches most), and the running sequences wait for each of them.
"""
import numpy as np


def prefill(n):
    return np.interp(n, [256, 2048, 3072], [34.2, 86.5, 124.9]) if n >= 256 else 34.2 * (0.5 + n / 512)


def decode(b):
    return np.interp(b, [1, 8, 32, 64], [13.6, 15.8, 20.5, 26.3])


def sim(policy, isl=2048, osl=32, conc=8, nreq=16, chunk=512):
    t, waiting, running, done, issued = 0.0, [], [], [], 0

    def issue():
        nonlocal issued
        while issued < nreq and len(waiting) + len(running) < conc:
            waiting.append({"left": isl, "toks": [], "start": t})
            issued += 1
    issue()
    while len(done) < nreq:
        if policy == "now":
            for r in waiting:
                t += prefill(isl)
                r["left"] = 0
                r["toks"].append(t)
                running.append(r)
            waiting.clear()
            dec = [r for r in running if 0 < len(r["toks"]) < osl]
            if dec:
                t += decode(len(dec))
                for r in dec:
                    r["toks"].append(t)
        else:
            pf = waiting[0] if waiting else None
            dec = [r for r in running if 0 < len(r["toks"]) < osl]
            n = min(chunk, pf["left"]) if pf else 0
            if policy == "mixed":
                t += (prefill(n) + 0.15 * len(dec)) if pf else decode(len(dec))
                for r in dec:
                    r["toks"].append(t)
            else:
                if pf:
                    t += prefill(n)
                if dec:
                    t += decode(len(dec))
                    for r in dec:
                        r["toks"].append(t)
            if pf:
                pf["left"] -= n
                if pf["left"] == 0:
                    pf["toks"].append(t)
                    waiting.pop(0)
                    running.append(pf)
        for r in [r for r in running if len(r["toks"]) >= osl]:
            running.remove(r)
            done.append(r)
        issue()
    itl = np.mean([(r["toks"][-1] - r["toks"][0]) / (osl - 1) for r in done])
    ttft = np.mean([r["toks"][0] - r["start"] for r in done])
    return f"TTFT {ttft:.0f} ms, ITL {itl:.1f} ms"


if __name__ == "__main__":
    print("prefill_heavy (2048 in, 32 out, concurrency 8), 30B 1x H200; measured then: ours 391 / 24.4 ms, vLLM 245 / 8.2 ms")
    print("  now (whole prefills, then decode):", sim("now"))
    for c in (2048, 1024, 512):
        print(f"  chunk {c}: interleaved {sim('inter', chunk=c)};  mixed {sim('mixed', chunk=c)}")
