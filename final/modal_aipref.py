"""AIPerf against OUR HTTP server, with exactly the same AIPerf settings as modal_vllm_aipref.py (the vLLM baseline).

  modal run modal_aipref.py                                          # plan only; builds the image on CPU
  modal run modal_aipref.py --compat                                 # CPU only: AIPerf vs the fake-engine server
  modal run modal_aipref.py --yes --suite                            # all SUITE workloads, one server instance
  modal run modal_aipref.py --yes                                    # Qwen3-4B, 1x H100, 3072 in / 1024 out
  modal run modal_aipref.py --yes --isl 130816 --osl 128 --yarn --requests 2   # 128K context (YaRN factor 4)

Everything that needs no GPU happens while the image is built on a CPU: the engine's shared library is compiled for
sm_90 and the server's own tests (fake engine) run inside the image, so the GPU is billed only for the benchmark.
Weights come from our read-only volume `qwen3-hf-cache`; raw AIPerf output is saved to our volume `final-results`.
"""
import os
import subprocess
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("final-our-server-aiperf")
# Read-write like in modal_vllm_aipref.py: AIPerf validates --tokenizer as a Hugging Face repo id (a local directory is
# rejected with HFValidationError, found on the first GPU run), so it loads Qwen/Qwen3-4B from the cache by name.
hf_cache = modal.Volume.from_name("qwen3-hf-cache", create_if_missing=False)
results_vol = modal.Volume.from_name("final-results", create_if_missing=True)   # ours (created 2026-11-16)
HF_DIR, RES_DIR, PORT = "/hf_cache", "/results_store", 8000
IMAGE = "nvidia/cuda:13.5.0-devel-ubuntu24.04"
image = (
    modal.Image.from_registry(IMAGE, add_python="3.12")
    .pip_install("fastapi", "uvicorn", "httpx", "transformers", "aiperf", "huggingface_hub")   # no PyTorch
    .env({"HF_HOME": HF_DIR, "HF_HUB_OFFLINE": "1"})   # use the cached files only, never the network
    .add_local_dir(HERE, "/root/final", ignore=[".venv", "__pycache__", "*.pyc", "build", "test_cases.db"], copy=True)
    .run_commands(
        "cd /root/final && mkdir -p build && nvcc --enable-tile -std=c++20 -arch=sm_90 -O2 -shared -Xcompiler -fPIC "
        "engine/qwen3_capi.cu -o build/libqwen3engine.so",
        "cd /root/final && nm -D --defined-only build/libqwen3engine.so "
        "| grep -E ' qe_(create|prefill|decode_batch|vocab|max_len|max_chunk|max_seqs)$' "
        "| tee /dev/stderr | test $(wc -l) -eq 7",                 # the C interface (engine/qwen3_capi.cu) is exported
        "cd /root/final && python server/test_server.py",          # server logic, fake engine (CPU)
    )
)
PRICE_H100 = 3.95

# The workload suite (keep identical in modal_aipref.py and modal_vllm_aipref.py):
#   (name, input tokens, output tokens, concurrency, requests)
SUITE = [
    ("baseline", 128, 128, 1, 8),            # latency floor without queueing
    ("long_out", 3072, 1024, 1, 3),          # the single-sequence workload measured before
    ("balanced_c16", 256, 256, 16, 32),      # batching at moderate concurrency
    ("balanced_c64", 256, 256, 64, 128),     # high concurrency
    ("prefill_heavy", 2048, 32, 8, 16),      # is prefill (TTFT) the bottleneck?
    ("decode_heavy", 64, 1024, 32, 64),      # large decode batches (ITL)
]
SUITE_MAX_SEQS = 64
SUITE_MAX_LEN = max(i + o for _, i, o, _, _ in SUITE) + 128
YARN_CONTEXT = 32768 * 4
CHAT_TEMPLATE_MARGIN = 128


def aiperf_cmd(model: str, url: str, isl: int, osl: int, requests: int, out: str, concurrency: int = 1):
    """The same AIPerf invocation as modal_vllm_aipref.py."""
    return ["aiperf", "profile", "--model", model, "--url", url,
            "--endpoint-type", "chat", "--streaming", "--tokenizer", model,
            "--extra-inputs", '{"ignore_eos": true, "temperature": 0}',
            "--concurrency", str(concurrency), "--request-count", str(requests),
            "--warmup-request-count", "1" if isl > 32768 else str(max(2, concurrency)),
            "--synthetic-input-tokens-mean", str(isl), "--synthetic-input-tokens-stddev", "0",
            "--output-tokens-mean", str(osl), "--output-tokens-stddev", "0",
            "--random-seed", "0", "--artifact-dir", out]


def lengths(isl: int, osl: int, yarn: bool):
    if yarn:
        if isl + osl + CHAT_TEMPLATE_MARGIN > YARN_CONTEXT:
            raise ValueError(f"isl + osl + {CHAT_TEMPLATE_MARGIN} exceeds {YARN_CONTEXT}; use --isl {YARN_CONTEXT - osl - CHAT_TEMPLATE_MARGIN}")
        return YARN_CONTEXT
    return isl + osl + CHAT_TEMPLATE_MARGIN


@app.function(image=image, gpu="H100", volumes={HF_DIR: hf_cache, RES_DIR: results_vol}, timeout=25 * 60)  # hard cap
def run(model: str, isl: int, osl: int, yarn: bool, requests: int, max_minutes: int):
    import urllib.request

    from huggingface_hub import snapshot_download
    os.chdir("/root/final")
    snap = snapshot_download(model, local_files_only=True)
    max_len = lengths(isl, osl, yarn)
    max_chunk = min(isl + CHAT_TEMPLATE_MARGIN, max_len)   # the whole prompt in one chunk, as in bench_modal.py
    cmd = ["python", "server/server.py", "--model-dir", snap, "--served-model-name", model, "--port", str(PORT),
           "--max-len", str(max_len), "--max-chunk", str(max_chunk)] + (["--yarn", "4"] if yarn else [])
    print("starting:", " ".join(cmd), flush=True)
    server = subprocess.Popen(cmd)
    out = f"/tmp/aiperf_ours_isl{isl}_osl{osl}"
    try:
        t0 = time.time()
        while True:
            if server.poll() is not None:
                raise RuntimeError("our server exited early (see its output above)")
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
                break
            except Exception:
                if time.time() - t0 > 600:
                    raise TimeoutError("server never became ready")
                time.sleep(2)
        print(f"server ready after {time.time() - t0:.0f} s", flush=True)
        rc = subprocess.run(["timeout", str(max_minutes * 60)] + aiperf_cmd(model, f"http://localhost:{PORT}", isl, osl,
                                                                             requests, out)).returncode
        print(f"aiperf exit code: {rc}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
    dest = f"{RES_DIR}/ours_isl{isl}_osl{osl}_{int(time.time())}"
    subprocess.run(["cp", "-r", out, dest], check=False)
    results_vol.commit()
    print(f"raw results saved to volume final-results:{dest}")


@app.function(image=image, gpu="H100", volumes={HF_DIR: hf_cache, RES_DIR: results_vol}, timeout=25 * 60)  # hard cap
def run_suite(model: str, max_minutes: int):
    """All SUITE workloads against ONE server instance (64 KV-cache slots), one after another."""
    import urllib.request

    from huggingface_hub import snapshot_download
    os.chdir("/root/final")
    snap = snapshot_download(model, local_files_only=True)
    cmd = ["python", "server/server.py", "--model-dir", snap, "--served-model-name", model, "--port", str(PORT),
           "--max-len", str(SUITE_MAX_LEN), "--max-chunk", str(max(i for _, i, _, _, _ in SUITE) + CHAT_TEMPLATE_MARGIN),
           "--max-seqs", str(SUITE_MAX_SEQS)]
    print("starting:", " ".join(cmd), flush=True)
    server = subprocess.Popen(cmd)
    stamp = int(time.time())
    try:
        t0 = time.time()
        while True:
            if server.poll() is not None:
                raise RuntimeError("our server exited early (see its output above)")
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
                break
            except Exception:
                if time.time() - t0 > 600:
                    raise TimeoutError("server never became ready")
                time.sleep(2)
        deadline = time.time() + max_minutes * 60
        for name, isl, osl, conc, reqs in SUITE:
            left = int(deadline - time.time())
            if left < 30:
                print(f"=== {name}: skipped, benchmark time cap reached")
                continue
            out = f"/tmp/suite_ours/{name}"
            print(f"=== WORKLOAD {name}: {isl} in / {osl} out, concurrency {conc}, {reqs} requests", flush=True)
            rc = subprocess.run(["timeout", str(left)] + aiperf_cmd(model, f"http://localhost:{PORT}", isl, osl, reqs, out,
                                                                     conc)).returncode
            print(f"=== {name}: aiperf exit code {rc}", flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
    subprocess.run(["cp", "-r", "/tmp/suite_ours", f"{RES_DIR}/suite_ours_{stamp}"], check=False)
    results_vol.commit()
    print(f"raw results saved to volume final-results:{RES_DIR}/suite_ours_{stamp}")


@app.function(image=image, volumes={HF_DIR: hf_cache}, timeout=15 * 60)   # CPU only
def compat_check():
    """AIPerf, with the same settings shape, against server.py --fake on a CPU: checks that AIPerf accepts our responses
    (streaming format, usage, fields) before any GPU is used. Token counts are meaningless here (fake tokenizer)."""
    import urllib.request
    os.chdir("/root/final")
    server = subprocess.Popen(["python", "server/server.py", "--fake", "--port", str(PORT), "--max-len", "8192",
                               "--max-seqs", "8"])
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        rcs = []
        for conc, reqs in ((1, 3), (8, 24)):     # one sequence, then a batch of 8 through the continuous batcher
            r = subprocess.run(aiperf_cmd("Qwen/Qwen3-4B", f"http://localhost:{PORT}", 64, 16, reqs, f"/tmp/aiperf_compat_c{conc}", conc))
            print(f"=== concurrency {conc}: aiperf exit code {r.returncode}", flush=True)
            rcs.append(r.returncode)
        return max(rcs)
    finally:
        server.terminate()


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-4B", isl: int = 3072, osl: int = 1024, yarn: bool = False, requests: int = 5,
         max_minutes: int = 18, compat: bool = False, suite: bool = False, yes: bool = False):
    if suite:
        print(f"PLAN: our server, {model}, 1 H100, workload suite {[w[0] for w in SUITE]} (max {SUITE_MAX_SEQS} sequences), "
              f"benchmark cap {max_minutes} min, container hard cap 25 min -> at most ~${25 / 60 * PRICE_H100:.2f} of shared credit")
        if yes:
            run_suite.remote(model, max_minutes)
        else:
            print("Dry run only. Pass --yes to start.")
        return
    if compat:
        print("CPU-only: AIPerf against the server with the fake engine (a few cents)")
        raise SystemExit(compat_check.remote())
    lengths(isl, osl, yarn)   # fails here, locally, before any GPU is started
    print(f"PLAN: our server, {model}, 1 H100, single sequence {isl} in / {osl} out, {requests} requests, YaRN={yarn}, "
          f"benchmark cap {max_minutes} min, container hard cap 25 min -> at most ~${25 / 60 * PRICE_H100:.2f} of shared credit")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    run.remote(model, isl, osl, yarn, requests, max_minutes)
