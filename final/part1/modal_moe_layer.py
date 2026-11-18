"""Part 1: one real MoE layer of Qwen3-30B-A3B on one GPU (tests/moe_layer_test.cu), checked against numpy
(tools/moe_layer_check.py). Production shape: 128 experts of width 768, top-8, hidden 2048.

  modal run part1/modal_moe_layer.py          # CPU only: builds the image (compiles the test and the engine library)
  modal run part1/modal_moe_layer.py --yes    # 1x H100: layers 0 and 47, 1 / 33 / 1024 tokens
  modal run part1/modal_moe_layer.py --yes --cases 0:1024              # only some (layer:tokens) cases

Only the router and the 128 experts of the tested layer are read (about 1.2 GB per layer), not the whole model.
"""
import os

import modal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # final/
SNAP = "/hf_cache/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
NVCC = "nvcc --enable-tile -std=c++20 -arch=sm_90 -O2"
app = modal.App("final-moe-layer")
hf_cache = modal.Volume.from_name("qwen3-hf-cache", create_if_missing=False)
image = (
    modal.Image.from_registry("nvidia/cuda:13.5.0-devel-ubuntu24.04", add_python="3.12")
    .pip_install("numpy")
    .add_local_dir(HERE, "/root/final", ignore=[".venv", "__pycache__", "*.pyc", "build", "test_cases*.db", "results"], copy=True)
    .run_commands(f"cd /root/final && mkdir -p build && ({NVCC} tests/moe_layer_test.cu -o build/moe_layer_test & "
                  f"{NVCC} -shared -Xcompiler -fPIC engine/qwen3_capi.cu -o build/libqwen3engine.so & wait) && "
                  f"test -x build/moe_layer_test && test -f build/libqwen3engine.so")
)
PRICE_H100 = 3.95


@app.function(image=image, gpu="H100", volumes={"/hf_cache": hf_cache}, timeout=15 * 60)   # hard cap
def run(cases: list) -> int:
    import subprocess
    os.chdir("/root/final")
    rc = 0
    for layer, T in cases:
        if True:
            d = f"/tmp/moe_l{layer}_t{T}"
            os.makedirs(d, exist_ok=True)
            r = subprocess.run(["build/moe_layer_test", SNAP, str(layer), str(T), d])
            if r.returncode:
                print(f"moe_layer_test failed (layer {layer}, {T} tokens)")
                rc = 1
                continue
            rc |= subprocess.run(["python", "tools/moe_layer_check.py", SNAP, str(layer), str(T), d]).returncode
    print("ALL MOE LAYER CHECKS PASS" if rc == 0 else "SOME MOE LAYER CHECKS FAILED")
    return rc


@app.local_entrypoint()
def main(yes: bool = False, cases: str = "0:1,0:33,0:1024,47:1,47:33,47:1024"):
    chosen = [tuple(int(v) for v in c.split(":")) for c in cases.split(",")]
    print(f"PLAN: 1x H100, one real MoE layer, cases (layer:tokens) {cases}, hard cap 15 min -> at most "
          f"~${15 / 60 * PRICE_H100:.2f} of shared credit")
    if not yes:
        print("Dry run only (the image is built: the test and the engine library compiled on a CPU). Pass --yes to run.")
        return
    raise SystemExit(run.remote(chosen))
