"""Single-sequence TTFT / ITL of OUR engine, same workload shape as modal_vllm_aipref.py (the vLLM baseline).

  modal run bench_modal.py                       # prints the plan only
  modal run bench_modal.py --yes                 # Qwen3-4B, 1x H100, 3072 in / 1024 out, 3 requests
  modal run bench_modal.py --yes --profile-steps 32 --check-tiles   # also: per-stage profile + big-vs-small tile check
  modal run bench_modal.py --yes --prompt-len 131072 --gen-len 128 --repeats 2 --yarn 4   # 128K input (YaRN factor 4)

The workspace credit is shared by the whole class: nothing starts without --yes and the run has a hard time cap.
Weights come from our own volume `qwen3-hf-cache` (read-only, nothing downloaded). Raw output is saved to our volume
`final-results`. Engine only (no HTTP server / tokenizer), so compare with vLLM's numbers knowing that.
The binary is compiled while the image is built (CPU only), so the GPU is billed only for the benchmark.
"""
import os
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("final-engine-bench")
hf_cache = modal.Volume.from_name("qwen3-hf-cache", create_if_missing=False)
try:
    hf_cache = hf_cache.read_only()
except AttributeError:
    pass
results_vol = modal.Volume.from_name("final-results", create_if_missing=True)   # ours (created 2026-11-16)
IMAGE = "nvidia/cuda:13.5.0-devel-ubuntu24.04"
image = (
    modal.Image.from_registry(IMAGE, add_python="3.12")
    .pip_install("numpy")
    .add_local_dir(HERE, "/root/final", ignore=[".venv", "__pycache__", "*.pyc", "build", "test_cases.db"], copy=True)
    .run_commands("cd /root/final && mkdir -p build && nvcc --enable-tile -std=c++20 -arch=sm_90 -O2 "
                  "tests/engine_bench_test.cu -o build/engine_bench")
)
PRICE_H100 = 3.95


@app.function(image=image, gpu="H100", volumes={"/hf_cache": hf_cache, "/results_store": results_vol}, timeout=1500)
def run(model_name: str, prompt_len: int, gen_len: int, repeats: int, run_minutes: int, profile_steps: int,
        check_tiles: bool, yarn: float):
    import glob
    import subprocess

    import numpy as np

    os.chdir("/root/final")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    snap = glob.glob(f"/hf_cache/hub/models--{model_name.replace('/', '--')}/snapshots/*")[0]
    cmd = ["timeout", str(run_minutes * 60), "build/engine_bench", snap, str(prompt_len), str(gen_len), str(repeats), "16",
           str(profile_steps)]
    env = dict(os.environ)
    if yarn > 0:
        env["QWEN3_YARN"] = str(yarn)   # YaRN rope scaling (original context 32768); needed for 128K on Qwen3-4B
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    text = out.stdout
    print(out.stdout)
    if out.returncode != 0:
        print(f"engine_bench exit code {out.returncode} (124 = time cap)\n{out.stderr[-1500:]}")

    rc = out.returncode
    if check_tiles:
        # same prompt, last-position logits with the big tiles (default on H100) and with the verified small tiles
        def last_logits(env_small, path):
            env = dict(os.environ)
            if env_small:
                env["QWEN3_TILES"] = "small"
            subprocess.run(["timeout", "300", "build/engine_bench", snap, str(prompt_len), "2", "1", "2", "0", path],
                           capture_output=True, text=True, env=env)
            raw = np.frombuffer(open(path, "rb").read(), dtype=np.uint16).astype(np.uint32) << 16
            return raw.view(np.float32)
        big, small = last_logits(False, "/tmp/big.bin"), last_logits(True, "/tmp/small.bin")
        cos = float(big @ small / (np.linalg.norm(big) * np.linalg.norm(small)))
        top2 = np.sort(small)[-2:]
        msg = (f"TILE CHECK (last position of the {prompt_len}-token prompt): top-1 big={int(big.argmax())} small={int(small.argmax())} "
               f"({'same' if big.argmax() == small.argmax() else 'DIFFERENT'}), cosine {cos:.6f}, "
               f"max|logit diff| {np.abs(big - small).max():.3f} (max|logit| {np.abs(small).max():.1f}), "
               f"small-tile top1-top2 margin {top2[1] - top2[0]:.3f}")
        print(msg)
        text += "\n" + msg
        if cos < 0.999:
            rc = rc or 3

    dest = f"/results_store/engine_bench_{model_name.split('/')[-1]}_isl{prompt_len}_osl{gen_len}_{int(time.time())}.txt"
    open(dest, "w").write(text + "\n" + out.stderr)
    results_vol.commit()
    print("raw output saved to volume final-results:", dest)
    return rc


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-4B", prompt_len: int = 3072, gen_len: int = 1024, repeats: int = 3,
         run_minutes: int = 14, profile_steps: int = 0, check_tiles: bool = False, yarn: float = 0.0, yes: bool = False):
    print(f"PLAN: our engine, {model}, 1x H100, single sequence {prompt_len} in / {gen_len} out, {repeats} requests, "
          f"YaRN={yarn or 'off'}, benchmark cap {run_minutes} min (+ ~1 min load) -> at most ~${(run_minutes + 2) / 60 * PRICE_H100:.2f} of shared credit")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    raise SystemExit(run.remote(model, prompt_len, gen_len, repeats, run_minutes, profile_steps, check_tiles, yarn))
