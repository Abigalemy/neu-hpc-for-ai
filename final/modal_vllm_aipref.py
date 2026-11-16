"""vLLM baseline for the Final Project: one command -> AIPerf table for a SINGLE sequence.

The Modal workspace credit is shared by the whole class, so nothing starts without --yes.

  modal run modal_vllm_aipref.py                       # prints the plan only (no GPU, no cost)
  modal run modal_vllm_aipref.py --yes                 # 3072 in / 1024 out, 1 sequence, Qwen3-4B, H100
  modal run modal_vllm_aipref.py --yes --isl 131072 --osl 128 --yarn   # 128K input (needs YaRN for Qwen3-4B)

The result table is printed, and the raw AIPerf output is saved on the Modal volume `final-results`.
"""
import os
import subprocess
import time

import modal

app = modal.App("final-vllm-baseline")
# The workspace is shared by the whole class, so volume names must be ours. "final-hf-cache" belongs to another
# student (sunnypeony); we use the HF cache from our own earlier assignment, which already holds Qwen3-4B / 0.6B.
hf_cache = modal.Volume.from_name("qwen3-hf-cache", create_if_missing=False)
results_vol = modal.Volume.from_name("final-results", create_if_missing=True)  # created by us on 2026-11-16
HF_DIR, RES_DIR, PORT = "/hf_cache", "/results_store", 8000

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "aiperf", "huggingface_hub")
    # slim image has no nvcc; FlashInfer sampler would JIT-compile at warmup and crash
    .env({"HF_HOME": HF_DIR, "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

PRICE_H100 = 3.95  # $/hour, approximate; check modal.com/pricing


def _wait_ready(proc, timeout_s=1200):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError("vllm exited early; see server log above")
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            return
        except Exception:
            time.sleep(3)
    raise TimeoutError("vllm never became ready")


@app.function(image=vllm_image, gpu="H100", volumes={HF_DIR: hf_cache, RES_DIR: results_vol}, timeout=2 * 3600)
def run(model: str, isl: int, osl: int, yarn: bool, requests: int, max_minutes: int):
    max_len = isl + osl + 16
    cmd = ["vllm", "serve", model, "--dtype", "bfloat16", "--max-model-len", str(max_len),
           "--max-num-seqs", "1", "--no-enable-prefix-caching", "--port", str(PORT)]
    if yarn:  # Qwen3 native context is 32K; YaRN extends it (factor 4 -> 128K). Not yet verified on this vLLM version.
        cmd += ["--rope-scaling", '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}']
    server = subprocess.Popen(cmd)
    out = f"/tmp/aiperf_isl{isl}_osl{osl}"
    try:
        _wait_ready(server)
        aiperf = ["aiperf", "profile", "--model", model, "--url", f"http://localhost:{PORT}",
                  "--endpoint-type", "chat", "--streaming", "--tokenizer", model,
                  "--extra-inputs", '{"ignore_eos": true, "temperature": 0}',
                  "--concurrency", "1", "--request-count", str(requests), "--warmup-request-count", "2",
                  "--synthetic-input-tokens-mean", str(isl), "--synthetic-input-tokens-stddev", "0",
                  "--output-tokens-mean", str(osl), "--output-tokens-stddev", "0",
                  "--random-seed", "0", "--artifact-dir", out]
        rc = subprocess.run(["timeout", str(max_minutes * 60)] + aiperf).returncode
        print(f"aiperf exit code: {rc}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
    dest = f"{RES_DIR}/vllm_isl{isl}_osl{osl}_{int(time.time())}"
    subprocess.run(["cp", "-r", out, dest], check=False)
    results_vol.commit()
    print(f"raw results saved to volume final-results:{dest}")


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-4B", isl: int = 3072, osl: int = 1024, yarn: bool = False,
         requests: int = 5, max_minutes: int = 25, yes: bool = False):
    est = max_minutes / 60 * PRICE_H100
    print(f"PLAN: vLLM {model}, 1 H100, single sequence {isl} in / {osl} out, {requests} requests, "
          f"YaRN={yarn}, hard cap {max_minutes} min -> at most ~${est:.2f} of shared credit")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    run.remote(model, isl, osl, yarn, requests, max_minutes)
