"""Part 2 runner: the all-to-all mock (a2a_mock.cu) on two GPUs.

  modal run part2/modal_a2a.py          # CPU only: builds the image = compiles a2a_mock.cu and runs test_plan.cpp
  modal run part2/modal_a2a.py --yes    # 2x H100, about a minute

Both programs are built while the image is built (CPU), so a compile error or a wrong plan fails before any GPU starts.
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("final-part2-a2a")
image = (
    modal.Image.from_registry("nvidia/cuda:13.5.0-devel-ubuntu24.04", add_python="3.12")
    .add_local_dir(HERE, "/root/part2", copy=True)
    .run_commands("cd /root/part2 && g++ -std=c++20 -O2 -Wall test_plan.cpp -o test_plan && ./test_plan && "
                  "nvcc --enable-tile -std=c++20 -arch=sm_90 -O2 a2a_mock.cu -o a2a_mock -lnccl")
)
PRICE_H100 = 3.95


@app.function(image=image, gpu="H100:2", timeout=5 * 60)   # hard cap 5 minutes
def run() -> int:
    import subprocess
    subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"])
    return subprocess.run(["/root/part2/a2a_mock"], env={**os.environ, "NCCL_DEBUG": "WARN"}).returncode


@app.local_entrypoint()
def main(yes: bool = False):
    print(f"PLAN: 2x H100, about 1 minute (hard cap 5 min) -> at most ~${2 * 5 / 60 * PRICE_H100:.2f} of shared credit")
    if not yes:
        print("Image built: plan test passed and a2a_mock compiled (CPU only). Pass --yes to run on the GPUs.")
        return
    raise SystemExit(run.remote())
