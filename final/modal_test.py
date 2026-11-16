"""Compile and test the Tile kernels on Modal.

  modal run modal_test.py                         # prints the plan only
  modal run modal_test.py --compile-only --yes    # CPU container: just compile (a few cents), shows compiler errors
  modal run modal_test.py --yes                   # L4 GPU: compile + run all SQLite cases (~5 min cap)

Needs final/test_cases.db (modal run gen_cases.py). L4 is the cheapest GPU that runs Tile (sm_89 runs the sm_80 build).
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("final-tile-test")
IMAGE = "nvidia/cuda:13.5.0-devel-ubuntu24.04"
image = (
    modal.Image.from_registry(IMAGE, add_python="3.12")
    .pip_install("numpy")
    .add_local_dir(HERE, "/root/final", ignore=[".venv", "__pycache__", "*.pyc", "build"])
)
PRICE_L4 = 0.80  # $/hour approx.

TESTS = ["rmsnorm", "rope", "mlp", "embedding", "linear", "argmax"]
BUILD = "cd /root/final && mkdir -p build && " + " && ".join(
    f"nvcc --enable-tile -std=c++20 -arch=sm_80 -O2 tests/{t}_test.cu -o build/{t}_test" for t in TESTS)


def _compile():
    import subprocess
    r = subprocess.run(BUILD, shell=True, capture_output=True, text=True)
    print(f"compile exit code: {r.returncode}")
    print((r.stdout + r.stderr)[-6000:])
    return r.returncode


@app.function(image=image, timeout=900)
def compile_job():
    return _compile()


@app.function(image=image, gpu="L4", timeout=900)
def run_all():
    import subprocess
    subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if _compile() != 0:
        return 1
    r = subprocess.run(["python", "run_tests.py", "test_cases.db", "build"] + TESTS, cwd="/root/final")
    return r.returncode


@app.local_entrypoint()
def main(compile_only: bool = False, yes: bool = False):
    if compile_only:
        print("PLAN: compile only in a CPU container (a few cents)")
    else:
        print(f"PLAN: compile + run all cases on 1 L4, hard cap 15 min -> at most ~${15 / 60 * PRICE_L4:.2f} of shared credit")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    if compile_only:
        raise SystemExit(compile_job.remote())
    raise SystemExit(run_all.remote())
