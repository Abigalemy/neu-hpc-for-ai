"""Part 3: one real MoE layer with expert parallelism on 2 GPUs (tests/ep_layer_test.cu) against the single-GPU layer
of Part 1 on the same tokens: every case must be bit-exact.

  modal run part3/modal_ep_layer.py          # CPU only: builds the image (compiles the test)
  modal run part3/modal_ep_layer.py --yes    # 2x H100, layers 0 and 47

Only the tested layer's router and experts are read (about 1.2 GB), not the whole model.
"""
import os

import modal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # final/
SNAP = "/hf_cache/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
NVCC = "nvcc --enable-tile -std=c++20 -arch=sm_90 -O2"
app = modal.App("final-ep-layer")
hf_cache = modal.Volume.from_name("qwen3-hf-cache", create_if_missing=False)
image = (
    modal.Image.from_registry("nvidia/cuda:13.5.0-devel-ubuntu24.04", add_python="3.12")
    .add_local_dir(HERE, "/root/final", ignore=[".venv", "__pycache__", "*.pyc", "build", "test_cases*.db", "results"], copy=True)
    .run_commands(f"cd /root/final && mkdir -p build && {NVCC} tests/ep_layer_test.cu -o build/ep_layer_test -lnccl && "
                  "test -x build/ep_layer_test")
)
PRICE_H100 = 3.95


@app.function(image=image, gpu="H100:2", volumes={"/hf_cache": hf_cache.read_only()}, timeout=10 * 60)   # hard cap
def run(layers: list) -> int:
    import subprocess
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"])
    rc = 0
    for layer in layers:
        print(f"=== layer {layer}", flush=True)
        rc |= subprocess.run(["/root/final/build/ep_layer_test", SNAP, str(layer)],
                             env={**os.environ, "NCCL_DEBUG": "WARN"}).returncode
    print("ALL EP LAYER CHECKS PASS" if rc == 0 else "SOME EP LAYER CHECKS FAILED")
    return rc


@app.local_entrypoint()
def main(yes: bool = False, layers: str = "0,47"):
    print(f"PLAN: 2x H100, one real MoE layer with expert parallelism, layers {layers}, hard cap 10 min -> at most "
          f"~${2 * 10 / 60 * PRICE_H100:.2f} of shared credit")
    if not yes:
        print("Dry run only (the image is built: the test compiled on a CPU). Pass --yes to run.")
        return
    raise SystemExit(run.remote([int(v) for v in layers.split(",")]))
