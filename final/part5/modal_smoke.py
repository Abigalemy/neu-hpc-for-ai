"""Part 5 smoke test on 2 GPUs: the real 30B engine through the whole process layout, against the 1-GPU server.

  modal run part5/modal_smoke.py            # plan only
  modal run part5/modal_smoke.py --yes      # 2x H100

In one container with 2 GPUs (the image, with the engine library built on a CPU, is modal_aipref.py's):
  1. the single-process server (Part 1) on GPU 0: a few greedy requests, one at a time; stopped;
  2. part5/launch.py --world 2 (balancer, 2 API servers, 2 engine cores, 2 GPU workers, coordinator; NCCL between
     the workers): the SAME requests, one at a time. Each one runs on one engine core while the other core's worker
     joins every pass with 0 tokens (idle_tick), so this also runs the idle path on GPUs. The texts must be
     identical (the EP layer and the whole model on 2 GPUs were bit-exact vs 1 GPU in Part 3);
  3. the same requests all at once (both cores busy): reported, not required to match (other batch sizes).
"""
import os
import subprocess
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # locally: final/
sys.path.insert(0, "/root/final")                                                  # in the container
from modal_aipref import HF_DIR, hf_cache, image, local_snapshot  # noqa: E402

app = modal.App("final-part5-smoke")
MODEL, GPU, MINUTES = "Qwen/Qwen3-30B-A3B", "H100:2", 20
PROMPTS = [[9707, 11, 847, 829, 374], [785, 6722, 315, 9625, 374], [40, 1079, 264, 5458, 504, 4492, 13, 358],
           list(range(1000, 1300))]                 # three short prompts and one of 300 tokens
OUT = 32
ARGS = ["--max-len", "2048", "--max-chunk", "1024", "--max-seqs", "8", "--kv-tokens", "16384",
        "--served-model-name", MODEL]


def wait_health(proc, port, what, limit=900):
    import urllib.request
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"{what} exited early (code {proc.returncode})")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            print(f"{what}: healthy after {time.time() - t0:.0f} s", flush=True)
            return
        except Exception:
            if time.time() - t0 > limit:
                raise RuntimeError(f"{what}: not healthy after {limit} s")
            time.sleep(2)


def complete(port, prompt):
    import json
    import urllib.request
    body = json.dumps({"prompt": prompt, "max_tokens": OUT, "ignore_eos": True, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, {"content-type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    assert r["usage"]["completion_tokens"] == OUT, r["usage"]
    return r["choices"][0]["text"]


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(3)                                      # children leave within ~0.2 s of their parent


@app.function(image=image, gpu=GPU, volumes={HF_DIR: hf_cache}, timeout=MINUTES * 60)
def smoke():
    import json
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    os.chdir("/root/final")
    snap = local_snapshot(MODEL)
    lib = "build/libqwen3engine.so"
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout, flush=True)

    one = subprocess.Popen(["python", "server/server.py", "--model-dir", snap, "--lib", lib, "--port", "8100"] + ARGS,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    try:
        wait_health(one, 8100, "1-GPU server")
        ref = [complete(8100, p) for p in PROMPTS]
    finally:
        stop(one)
    print("1-GPU texts:", json.dumps(ref)[:400], flush=True)

    grp = subprocess.Popen(["python", "part5/launch.py", "--world", "2", "--lib", lib, "--model-dir", snap,
                            "--port", "8000"] + ARGS)
    try:
        wait_health(grp, 8000, "Part 5 group (2 GPUs)")
        alone = [complete(8000, p) for p in PROMPTS]
        st1 = json.load(urllib.request.urlopen("http://127.0.0.1:8000/stats"))
        with ThreadPoolExecutor(2 * len(PROMPTS)) as ex:
            together = list(ex.map(lambda p: complete(8000, p), PROMPTS * 2))
        st2 = json.load(urllib.request.urlopen("http://127.0.0.1:8000/stats"))
    finally:
        stop(grp)
    same = [a == r for a, r in zip(alone, ref)]
    print(f"one at a time, 2 GPUs vs 1 GPU: {sum(same)}/{len(same)} texts identical {same}", flush=True)
    print(f"all at once ({len(together)} requests): {sum(t == r for t, r in zip(together, ref * 2))}/{len(together)} "
          f"identical to 1 GPU", flush=True)
    for name, st in (("after one at a time", st1), ("after all at once", st2)):
        cores = [{k: c[k] for k in ("batch_steps", "idle_passes", "parks", "free_blocks")} for c in st["coordinator"]["cores"]]
        print(f"stats {name}: sent per API server {[r['sent'] for r in st['ranks']]}; cores {cores}; "
              f"wakes {st['coordinator']['wakes']}", flush=True)
    ok = all(same)
    print("PART 5 GPU SMOKE: " + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


@app.local_entrypoint()
def main(yes: bool = False):
    usd = 2 * 3.95 * MINUTES / 60
    print(f"PLAN: Part 5 smoke test, {MODEL}, {GPU}, 1-GPU server then the 2-GPU group, {len(PROMPTS)} prompts x "
          f"{OUT} tokens; container cap {MINUTES} min -> at most ~${usd:.2f} of shared credit")
    if not yes:
        print("dry run: add --yes to run it")
        return
    raise SystemExit(smoke.remote())
