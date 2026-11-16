"""Run the whole unit-test suite (all SQLite reference cases) on an H100, with the big GEMM tiles.

  modal run modal_test_h100.py          # prints the plan only
  modal run modal_test_h100.py --yes    # 1x H100, hard cap 15 minutes
  modal run modal_test_h100.py --yes --tests attention     # only some modules

The binaries are compiled for sm_90 while the image is built (CPU only), so the GPU is billed only for the tests.
Needs final/test_cases.db (modal run gen_cases.py). Shared workspace credit: nothing starts without --yes.
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE = "nvidia/cuda:13.5.0-devel-ubuntu24.04"
# keep in sync with modal_test.py (this file must not import it: only this file is mounted in the container)
TESTS = ["rmsnorm", "rope", "mlp", "embedding", "linear", "argmax", "attention"]


def build_cmd(arch):
    return "cd /root/final && mkdir -p build && " + " && ".join(
        f"nvcc --enable-tile -std=c++20 -arch={arch} -O2 tests/{t}_test.cu -o build/{t}_test" for t in TESTS)


app = modal.App("final-tile-test-h100")
image = (
    modal.Image.from_registry(IMAGE, add_python="3.12")
    .pip_install("numpy")
    .add_local_dir(HERE, "/root/final", ignore=[".venv", "__pycache__", "*.pyc", "build"], copy=True)
    .run_commands(build_cmd("sm_90"))
)
PRICE_H100 = 3.95


@app.function(image=image, gpu="H100", timeout=1200)
def run_all(tests: list):
    import subprocess
    subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    r = subprocess.run(["python", "run_tests.py", "test_cases.db", "build"] + tests, cwd="/root/final")
    return r.returncode


@app.local_entrypoint()
def main(tests: str = "all", yes: bool = False):
    chosen = TESTS if tests == "all" else tests.split(",")
    print(f"PLAN: unit tests {chosen} on 1x H100, hard cap 15 min -> at most ~${15 / 60 * PRICE_H100:.2f} of shared credit")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    raise SystemExit(run_all.remote(chosen))
