"""Attention tile sweep on an H100: which (BR, BC, P-precision) configurations launch, how fast, how accurate.

  modal run attn_sweep_modal.py          # prints the plan only
  modal run attn_sweep_modal.py --yes    # 1x H100, hard cap 6 minutes of run time

The binary is compiled while the image is built (CPU only), so the GPU is only billed for the sweep itself.
Shared workspace credit: nothing starts without --yes.
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("final-attn-sweep")
IMAGE = "nvidia/cuda:13.5.0-devel-ubuntu24.04"
image = (
    modal.Image.from_registry(IMAGE, add_python="3.12")
    .add_local_dir(HERE, "/root/final", ignore=[".venv", "__pycache__", "*.pyc", "build", "test_cases.db"], copy=True)
    .run_commands("cd /root/final && mkdir -p build && nvcc --enable-tile -std=c++20 -arch=sm_90 -O2 "
                  "tests/attn_sweep_test.cu -o build/attn_sweep")
)
PRICE_H100 = 3.95


@app.function(image=image, gpu="H100", timeout=900)
def run(run_minutes: int):
    import subprocess
    r = subprocess.run(["timeout", str(run_minutes * 60), "/root/final/build/attn_sweep"], capture_output=True, text=True)
    print(r.stdout)
    if r.returncode:
        print(f"exit code {r.returncode} (124 = time cap)\n{r.stderr[-1500:]}")
    return r.returncode


@app.local_entrypoint()
def main(run_minutes: int = 6, yes: bool = False):
    print(f"PLAN: attention tile sweep, 1x H100, hard cap {run_minutes} min -> at most ~${(run_minutes + 1) / 60 * PRICE_H100:.2f} of shared credit")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    raise SystemExit(run.remote(run_minutes))
