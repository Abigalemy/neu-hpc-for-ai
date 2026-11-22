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


# self-checking binaries without SQLite cases (exit code 0 = pass): `--tests paged` runs the PagedAttention test
MOE = ["moe"]   # Part 1: cases in test_cases_moe.db (modal run part1/gen_moe_cases.py), `--tests moe`
SELF_CHECKING = ["paged_attention", "paged_bench", "moe_tiles"]   # paged_bench: timings only; moe_tiles: Part 4 work list


def build_cmd(arch):
    # compiled in parallel: one after another, the binaries no longer fit the image-build time limit (seen 2026-11-18);
    # xargs exits non-zero if any compile fails, which fails the build before a GPU is started
    names = " ".join(TESTS + MOE + SELF_CHECKING)
    return (f"cd /root/final && mkdir -p build && printf '%s\\n' {names} | xargs -P 8 -I{{}} "
            f"nvcc --enable-tile -std=c++20 -arch={arch} -O2 tests/{{}}_test.cu -o build/{{}}_test -lnccl")


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
    rc = 0
    db_tests = [t for t in tests if t in TESTS]
    if db_tests:
        rc |= subprocess.run(["python", "run_tests.py", "test_cases.db", "build"] + db_tests, cwd="/root/final").returncode
    if "moe_debug" in tests:
        rc |= subprocess.run(["python", "tools/moe_debug.py", "test_cases_moe.db", "build/moe_test"], cwd="/root/final").returncode
    if "moe" in tests:
        rc |= subprocess.run(["python", "run_tests.py", "test_cases_moe.db", "build", "moe"], cwd="/root/final").returncode
    if "paged" in tests:
        rc |= subprocess.run(["build/paged_attention_test"], cwd="/root/final").returncode
    if "paged_bench" in tests:
        rc |= subprocess.run(["build/paged_bench_test"], cwd="/root/final").returncode
    if "moe_tiles" in tests:
        rc |= subprocess.run(["build/moe_tiles_test"], cwd="/root/final").returncode
    unknown = [t for t in tests if t not in TESTS + ["moe", "moe_debug", "paged", "paged_bench", "moe_tiles"]]
    if unknown:                      # a name with no branch above would otherwise "pass" without running anything
        print(f"unknown tests: {unknown}")
        rc |= 1
    return rc


@app.local_entrypoint()
def main(tests: str = "all", yes: bool = False):
    chosen = TESTS + ["paged"] if tests == "all" else tests.split(",")
    print(f"PLAN: unit tests {chosen} on 1x H100, hard cap 15 min -> at most ~${15 / 60 * PRICE_H100:.2f} of shared credit")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    raise SystemExit(run_all.remote(chosen))
