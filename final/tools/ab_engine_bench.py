"""A/B timing of two engine versions in ONE container, so that GPU-to-GPU and run-to-run differences cancel out.

  modal run tools/ab_engine_bench.py                           # plan only (builds the image on a CPU)
  modal run tools/ab_engine_bench.py --yes                     # old = 8bcdbd5 (before paging), new = working tree

tools/ab_decode_bench.cu is compiled against each tree (-I<tree>/final) while the image is built (CPU only); the GPU
run alternates old, new, old, new. Written on 2026-11-18 when the suite after paging was 7-28% slower than the suite
measured before paging in another container.
"""
import os
import subprocess

import modal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # final/
OLD_COMMIT = os.environ.get("AB_OLD", "8bcdbd5")
OLD_DIR = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"ab_old_{OLD_COMMIT}")
if modal.is_local():   # export the old engine (engine, kernels, tests/common.h) from git, unchanged
    if not os.path.isfile(os.path.join(OLD_DIR, "final/engine/qwen3_engine.cuh")):
        os.makedirs(OLD_DIR, exist_ok=True)
        subprocess.run(f"set -o pipefail; git -C {os.path.dirname(HERE)} archive {OLD_COMMIT} final/engine final/kernels "
                       f"final/tests/common.h | tar -x -C {OLD_DIR}", shell=True, check=True, executable="/bin/bash")
    assert os.path.isfile(os.path.join(OLD_DIR, "final/engine/qwen3_engine.cuh")), "old engine not exported"

NVCC = "nvcc --enable-tile -std=c++20 -arch=sm_90 -O2"
app = modal.App("final-ab-engine-bench")
hf_cache = modal.Volume.from_name("qwen3-hf-cache", create_if_missing=False)
image = (
    modal.Image.from_registry("nvidia/cuda:13.5.0-devel-ubuntu24.04", add_python="3.12")
    .add_local_dir(HERE, "/root/final", ignore=[".venv", "__pycache__", "*.pyc", "build", "test_cases.db", "results"], copy=True)
    .add_local_dir(OLD_DIR, "/root/old", copy=True)
    .run_commands(f"cd /root/final && mkdir -p build && "
                  f"({NVCC} -I/root/old/final tools/ab_decode_bench.cu -o build/ab_old -lnccl & "
                  f"{NVCC} -I/root/final tools/ab_decode_bench.cu -o build/ab_new -lnccl & wait) && "
                  f"test -x build/ab_old && test -x build/ab_new")
)
PRICE_H100 = 3.95


@app.function(image=image, gpu="H100", volumes={"/hf_cache": hf_cache}, timeout=15 * 60)   # hard cap
def run() -> int:
    import glob
    snap = glob.glob("/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/*")[0]
    rc = 0
    for label in ("old", "new", "old", "new"):
        r = subprocess.run(["timeout", "240", f"/root/final/build/ab_{label}", snap, label])
        rc |= r.returncode
    return rc


@app.local_entrypoint()
def main(yes: bool = False):
    print(f"PLAN: A/B engine timing, old = {OLD_COMMIT} vs new = working tree, Qwen3-4B, 1x H100, 4 runs alternating, "
          f"hard cap 15 min -> at most ~${15 / 60 * PRICE_H100:.2f} of shared credit")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    raise SystemExit(run.remote())
