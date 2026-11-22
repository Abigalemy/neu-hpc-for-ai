"""Part 4, C1: per-stage profile of Qwen3-30B-A3B in our engine (tests/moe_profile_test.cu), on 1 or 2 GPUs.

  modal run part4/modal_profile.py                       # plan only
  modal run part4/modal_profile.py --yes                 # 1x H200
  modal run part4/modal_profile.py --yes --world 2       # 2x H200, expert parallelism (one process per GPU)

Prefill of 256 / 2048 / 3072 tokens (TTFT's forward: the suite's prompt lengths) and decode at batch 1 / 8 / 32 / 64
with ~256 tokens of context (ITL's forward at the suite's concurrencies). The output is saved to final-results.
"""
import os
import subprocess
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # locally: final/
sys.path.insert(0, "/root/final")                                                  # in the container
from modal_aipref import HF_DIR, hf_cache, image as base_image, local_snapshot, results_vol, RES_DIR  # noqa: E402

app = modal.App("final-part4-profile")
image = base_image.run_commands("cd /root/final && nvcc --enable-tile -std=c++20 -arch=sm_90 -O2 tests/moe_profile_test.cu "
                                "-o build/moe_profile -lnccl && test -x build/moe_profile")
MODEL, MINUTES = "Qwen/Qwen3-30B-A3B", 15
ARGS = ["256,2048,3072", "1,8,32,64", "256", "16"]
USD_PER_GPU_HOUR = {"h200": 4.54, "h100": 3.95}


def run(world: int) -> int:
    os.chdir("/root/final")
    snap = local_snapshot(MODEL)
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout, flush=True)
    env = dict(os.environ)
    if world > 1:
        code = ("import ctypes, sys; lib = ctypes.CDLL(sys.argv[1]); n = lib.qe_nccl_id_bytes(); "
                "b = ctypes.create_string_buffer(n); assert lib.qe_nccl_unique_id(b) == 1; print(b.raw[:n].hex(), flush=True); "
                "sys.stdin.read()")   # stays alive: the NCCL bootstrap root lives in this process
        holder = subprocess.Popen([sys.executable, "-c", code, "build/libqwen3engine.so"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, env={**env, "CUDA_VISIBLE_DEVICES": ""})
        env["QWEN3_NCCL_ID"] = holder.stdout.readline().strip()
    procs = [subprocess.Popen(["build/moe_profile", snap] + ARGS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              env={**env, "CUDA_VISIBLE_DEVICES": str(r), "QWEN3_WORLD": str(world), "QWEN3_RANK": str(r)})
             for r in range(world)]
    outs = [p.communicate(timeout=(MINUTES - 2) * 60)[0] for p in procs]
    text = "\n".join(f"===== rank {r} (exit code {p.returncode})\n{o}" for r, (p, o) in enumerate(zip(procs, outs)))
    print(text, flush=True)
    dest = f"{RES_DIR}/profile_30B_w{world}_{int(time.time())}.txt"
    with open(dest, "w") as f:
        f.write(text)
    results_vol.commit()
    print(f"saved to volume final-results:{dest[len(RES_DIR) + 1:]}", flush=True)
    return max(p.returncode for p in procs)


@app.function(image=image, gpu="H200", volumes={HF_DIR: hf_cache, RES_DIR: results_vol}, timeout=MINUTES * 60)
def profile1() -> int:
    return run(1)


@app.function(image=image, gpu="H200:2", volumes={HF_DIR: hf_cache, RES_DIR: results_vol}, timeout=MINUTES * 60)
def profile2() -> int:
    return run(2)


@app.local_entrypoint()
def main(yes: bool = False, world: int = 1):
    if world not in (1, 2):
        raise SystemExit("--world 1 or 2")
    usd = world * USD_PER_GPU_HOUR["h200"] * MINUTES / 60
    print(f"PLAN: per-stage profile of {MODEL} on {world}x H200: prefill {ARGS[0]} tokens, decode batches {ARGS[1]} "
          f"(context {ARGS[2]}, {ARGS[3]} steps); container cap {MINUTES} min -> at most ~${usd:.2f} of shared credit")
    if not yes:
        print("dry run: add --yes to run it")
        return
    raise SystemExit((profile1 if world == 1 else profile2).remote())
