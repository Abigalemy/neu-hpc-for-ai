"""CPU only: compile every CUDA source of final/ for sm_90 (the engine library and every test), so that compile errors
are found before any GPU is paid for. Nothing runs on a GPU.

  modal run compile_check.py
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE = "nvidia/cuda:13.5.0-devel-ubuntu24.04"
NVCC = "nvcc --enable-tile -std=c++20 -arch=sm_90 -O2"

app = modal.App("final-compile-check")
image = (
    modal.Image.from_registry(IMAGE, add_python="3.12")
    .add_local_dir(HERE, "/root/final", ignore=[".venv", "__pycache__", "*.pyc", "build", "test_cases.db", "results"])
)


@app.function(image=image, cpu=8, timeout=30 * 60)   # CPU container: no GPU is attached
def build() -> int:
    import glob
    import subprocess
    os.chdir("/root/final")
    os.makedirs("build", exist_ok=True)
    jobs = [f"{NVCC} -shared -Xcompiler -fPIC engine/qwen3_capi.cu -o build/libqwen3engine.so"]
    for src in sorted(glob.glob("tests/*.cu")):
        jobs.append(f"{NVCC} {src} -o build/{os.path.basename(src)[:-3]}")
    procs = [(j, subprocess.Popen(j, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)) for j in jobs]
    failed = 0
    for j, p in procs:
        out = p.communicate()[0]
        ok = p.returncode == 0
        failed += not ok
        print(("ok    " if ok else "FAIL  ") + j.split()[-1])
        if not ok:
            print(out[-6000:])
    syms = subprocess.run("nm -D --defined-only build/libqwen3engine.so | grep -o ' qe_[a-z_]*$'", shell=True,
                          capture_output=True, text=True).stdout.split()
    print("exported:", " ".join(sorted(syms)))
    print("COMPILE OK" if failed == 0 else f"{failed} FAILED")
    return failed


@app.local_entrypoint()
def main():
    print("CPU only (8 cores, no GPU), a few cents: compile engine + all tests for sm_90")
    raise SystemExit(build.remote())
