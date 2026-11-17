"""GPU check that batched decode equals decoding each sequence alone (tests/batch_test.cu), on real Qwen3 weights.

  modal run batch_check.py                    # plan only; compiles on CPU while the image is built
  modal run batch_check.py --yes              # Qwen3-4B, 1x H100, 4 sequences (27, 300, 1000, 3000 tokens), 16 steps
  modal run batch_check.py --yes --splits ,4  # once with the engine's split rule, once with 4 splits forced for both runs

Weights come from our read-only volume `qwen3-hf-cache`; nothing is downloaded. Shared credit: --yes required.
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("final-batch-check")
hf_cache = modal.Volume.from_name("qwen3-hf-cache", create_if_missing=False)
try:
    hf_cache = hf_cache.read_only()
except AttributeError:
    pass
IMAGE = "nvidia/cuda:13.5.0-devel-ubuntu24.04"
image = (
    modal.Image.from_registry(IMAGE, add_python="3.12")
    .add_local_dir(HERE, "/root/final", ignore=[".venv", "__pycache__", "*.pyc", "build", "test_cases.db"], copy=True)
    .run_commands("cd /root/final && mkdir -p build && nvcc --enable-tile -std=c++20 -arch=sm_90 -O2 tests/batch_test.cu "
                  "-o build/batch_test")
)
PRICE_H100 = 3.95


@app.function(image=image, gpu="H100", volumes={"/hf_cache": hf_cache}, timeout=10 * 60)   # hard cap 10 min
def run(model: str, steps: int, lens: str, splits: str):
    """Runs the check once per entry of `splits` ("" = the engine's own split rule, "4" = force 4 splits, ...)."""
    import glob
    import subprocess
    snap = glob.glob(f"/hf_cache/hub/models--{model.replace('/', '--')}/snapshots/*")[0]
    rc = 0
    for sp in splits.split(","):
        env = dict(os.environ)
        if sp:
            env["QWEN3_DECODE_SPLITS"] = sp
        r = subprocess.run(["/root/final/build/batch_test", snap, str(steps)] + lens.split(","), capture_output=True,
                           text=True, env=env)
        print(r.stdout)
        if r.returncode:
            print(r.stderr[-2000:])
        rc = rc or r.returncode
    return rc


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-4B", steps: int = 16, lens: str = "27,300,1000,3000", splits: str = "", yes: bool = False):
    print(f"PLAN: batched-vs-alone decode check, {model}, 1x H100, sequences {lens}, {steps} steps, hard cap 10 min "
          f"-> at most ~${10 / 60 * PRICE_H100:.2f} of shared credit")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    raise SystemExit(run.remote(model, steps, lens, splits))
