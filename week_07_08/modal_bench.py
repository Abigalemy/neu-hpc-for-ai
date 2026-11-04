"""Run the Qwen3 serving benchmark on Modal GPUs (our server and vLLM, same AIPerf workloads).

The Modal workspace credit is SHARED by the whole class, so this script is cost-conscious:
  * nothing starts unless you pass --yes (otherwise it only prints the plan);
  * every config has a hard wall-clock cap (--max-minutes);
  * --scale-pct shrinks the number of requests for cheap smoke runs;
  * develop on a cheap GPU (default L4) and only use --gpu h100 for the final run.

Typical use (from week_07_08/):
  modal run modal_bench.py --download-only --model Qwen/Qwen3-0.6B --yes          # cache weights (CPU, cheap)
  modal run modal_bench.py --configs ours_paged --workloads baseline --scale-pct 25 --yes   # smoke test on L4
  modal run modal_bench.py --configs ours_paged,ours_contiguous,ours_nocache,ours_static,vllm \\
        --model Qwen/Qwen3-4B --gpu h100 --kv-gb 8 --max-minutes 40 --yes                   # final run
Results are written to ./results/<config>/ and ./results/summary.md.
Use `modal run --detach ...` for long runs so the app keeps running if this terminal/session disconnects; results are
also saved to the Modal volume `qwen3-results` (fetch with: modal volume get qwen3-results / ./results_modal).
"""
import io
import os
import subprocess
import tarfile
import time

import modal

APP_NAME = "qwen3-week07-bench"
HF_DIR = "/hf_cache"
APP_DIR = "/root/app"
PORT = 8000

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("qwen3-hf-cache", create_if_missing=True)
# Every finished config also saves its result tarball here, so a dropped local client can't lose paid-for runs:
#   modal volume get qwen3-results / ./results_modal
results_vol = modal.Volume.from_name("qwen3-results", create_if_missing=True)
RES_DIR = "/results_store"

_ignore = [".venv", "**/__pycache__", ".pytest_cache", "results", "logs", "artifacts", ".git"]
_here = os.path.dirname(os.path.abspath(__file__))

# Our server: PyTorch ships its own CUDA runtime, so a slim Python image is enough.
ours_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "safetensors", "huggingface_hub", "fastapi", "uvicorn",
                 "pydantic", "aiperf", "numpy")
    .env({"HF_HOME": HF_DIR})
    .add_local_dir(_here, APP_DIR, ignore=_ignore)
)
# vLLM pins its own torch, so it gets a separate image.
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "aiperf", "huggingface_hub")
    # The slim image has no nvcc; vLLM's FlashInfer sampler would JIT-compile at warmup and crash.
    .env({"HF_HOME": HF_DIR, "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir(_here, APP_DIR, ignore=_ignore)
)

# Approximate on-demand prices ($/hour) for the plan printout only. Check modal.com/pricing.
PRICE = {"t4": 0.59, "l4": 0.80, "a10g": 1.10, "a100-80gb": 2.50, "h100": 3.95}
VALID = ["ours_paged", "ours_contiguous", "ours_nocache", "ours_static", "vllm"]

# Final-run preset (--final). Sweeps are trimmed on purpose: concurrency-1 decode-heavy runs alone would take
# ~25 min on our CPU-bound engine, and the no-cache engine re-runs the whole context every step so it only
# gets the short, low-concurrency workloads. "name:a+b+c" = sweep values for that workload.
_FULL = ["baseline", "balanced:4+16+64", "prefill_heavy:1+8+32", "decode_heavy:8+32+64",
         "mixed_poisson:2+8", "pressure"]
FINAL_PLAN = {  # config -> (workloads, max_minutes)
    "vllm": (_FULL, 20),
    "ours_paged": (_FULL, 40),
    "ours_contiguous": (_FULL, 40),
    "ours_nocache": (["baseline", "balanced:4+16", "prefill_heavy:1+8"], 25),
    "ours_static": (["balanced:16+64", "mixed_poisson:2+8"], 20),  # optional batching ablation
}


def _wait_ready(proc, timeout_s=900):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError("server exited early; see server.log in the results tarball")
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError("server never became ready")


def _server_cmd(config, model, kv_gb, max_len):
    if config == "vllm":
        return ["vllm", "serve", model, "--dtype", "bfloat16", "--max-model-len", str(max_len),
                "--max-num-seqs", "64", "--no-enable-prefix-caching", "--port", str(PORT)]
    mode, batching = {"ours_paged": ("paged", "continuous"), "ours_contiguous": ("contiguous", "continuous"),
                      "ours_nocache": ("none", "continuous"), "ours_static": ("paged", "static")}[config]
    return ["python", "server.py", "--model", model, "--cache-mode", mode, "--batching", batching,
            "--max-model-len", str(max_len), "--max-num-seqs", "64", "--kv-cache-gb", str(kv_gb),
            "--port", str(PORT)]


def _sanity(out, model):
    """One real chat request, saved with the results, so garbage output (e.g. a bf16 problem) is visible."""
    import json
    import urllib.request
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "What is the capital of France? Answer briefly."}],
                       "max_tokens": 32}).encode()
    req = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    try:
        with open(f"{out}/sanity.json", "wb") as f:
            f.write(urllib.request.urlopen(req, timeout=120).read())
    except Exception as e:
        open(f"{out}/sanity.json", "w").write(f"sanity request failed: {e}")


def _run(config, model, kv_gb, workloads, scale_pct, max_minutes, max_len):
    os.chdir(APP_DIR)
    out = f"/tmp/results/{config}"
    os.makedirs(out, exist_ok=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
    log = open(f"{out}/server.log", "w")
    server = subprocess.Popen(_server_cmd(config, model, kv_gb, max_len), stdout=log, stderr=subprocess.STDOUT)
    status = "ok"
    try:
        _wait_ready(server)
        _sanity(out, model)
        env = dict(os.environ, OUT_DIR="/tmp/results", SCALE_PCT=str(scale_pct))
        cmd = ["timeout", str(max_minutes * 60), "bash", "bench/run_aiperf.sh", config,
               f"http://localhost:{PORT}", model, model] + workloads
        rc = subprocess.run(cmd, env=env).returncode
        if rc == 124:
            status = f"TIMEOUT after {max_minutes} min (partial results kept)"
        elif rc != 0:
            status = f"aiperf exited with code {rc}"
        if config != "vllm":
            import urllib.request
            with open(f"{out}/engine_stats.json", "wb") as f:
                f.write(urllib.request.urlopen(f"http://localhost:{PORT}/metrics_json").read())
    except Exception as e:  # keep whatever we have; the log explains the failure
        status = f"FAILED: {e}"
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add("/tmp/results", arcname=".")
    print(f"[{config}] {status}")
    if status != "ok" or config == "vllm":  # make failures diagnosable from the app logs alone
        try:
            tail = open(f"{out}/server.log").read().splitlines()[-60:]
            print(f"--- last lines of {config} server.log ---\n" + "\n".join(tail))
        except Exception:
            pass
    blob = buf.getvalue()
    try:
        with open(f"{RES_DIR}/{config}_{model.split('/')[-1]}_{int(time.time())}.tar.gz", "wb") as f:
            f.write(blob)
        results_vol.commit()
    except Exception as e:
        print(f"could not save to results volume: {e}")
    return config, status, blob


# One function per GPU tier (the GPU is fixed at decoration time).
_common = dict(volumes={HF_DIR: hf_cache, RES_DIR: results_vol}, timeout=3 * 3600)


@app.function(image=ours_image, gpu="L4", **_common)
def ours_dev(*a):
    return _run(*a)


@app.function(image=ours_image, gpu="H100", **_common)
def ours_h100(*a):
    return _run(*a)


@app.function(image=vllm_image, gpu="L4", **_common)
def vllm_dev(*a):
    return _run(*a)


@app.function(image=vllm_image, gpu="H100", **_common)
def vllm_h100(*a):
    return _run(*a)


@app.function(image=ours_image, gpu="L4", volumes={HF_DIR: hf_cache}, timeout=1800)
def check_correctness(model: str):
    """Real weights + real tokenizer: our engine (every cache mode, concurrent requests) vs HuggingFace.

    float32 on both sides so that greedy decoding is deterministic enough to compare token by token.
    """
    import sys
    import torch
    sys.path.insert(0, APP_DIR)
    os.chdir(APP_DIR)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine import EngineConfig, LLMEngine, SamplingParams
    from qwen3_model import load_model

    dev, n_new = "cuda", 24
    mine, cfg, d = load_model(model, dev, torch.float32, max_positions=1024)
    tok = AutoTokenizer.from_pretrained(d)
    hf = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).to(dev).eval()
    prompts = ["The capital of France is", "def fibonacci(n):\n    ", "中国的首都是",
               "Once upon a time, in a small village by the sea, there lived an old fisherman who"]
    ids = [tok.encode(p) for p in prompts]

    ref = []
    for p in ids:
        seq = list(p)
        for _ in range(n_new):
            with torch.no_grad():
                seq.append(int(hf(torch.tensor([seq], device=dev)).logits[0, -1].argmax()))
        ref.append(seq[len(p):])

    report, ok = {}, True
    for mode in ["paged", "contiguous", "none"]:
        ecfg = EngineConfig(cache_mode=mode, max_num_seqs=4, max_model_len=512, kv_cache_gb=1.0)
        eng = LLMEngine(mine, cfg, ecfg, eos_token_ids=[], device=dev, dtype=torch.float32)
        outs = [[] for _ in ids]
        for i, p in enumerate(ids):
            eng.add_request(p, SamplingParams(max_tokens=n_new, ignore_eos=True),
                            lambda t, r, i=i: outs[i].append(t) if t is not None else None)
        while eng.has_work():
            eng.step()
        res = []
        for i in range(len(ids)):
            same = outs[i] == ref[i]
            first_bad = next((j for j, (a, b) in enumerate(zip(outs[i], ref[i])) if a != b), None)
            res.append({"prompt": prompts[i], "match": same, "first_mismatch": first_bad,
                        "ours": tok.decode(outs[i])[:120], "hf": tok.decode(ref[i])[:120]})
            ok &= same
        report[mode] = res
    report["all_match"] = ok
    return report


@app.function(image=ours_image, gpu="L4", volumes={HF_DIR: hf_cache}, timeout=1800)
def profile_decode(model: str, gpu_label: str = "L4"):
    """Where does a decode step spend its time? Wall time vs GPU-busy time vs CPU work, per batch size."""
    import sys
    import torch
    sys.path.insert(0, APP_DIR)
    os.chdir(APP_DIR)
    from torch.profiler import ProfilerActivity, profile
    from engine import EngineConfig, LLMEngine, SamplingParams
    from kv_cache import PagedDecodeBatch
    from qwen3_model import load_model

    dev, dtype = "cuda", torch.bfloat16
    m, cfg, _ = load_model(model, dev, dtype, max_positions=1024)

    def sync():
        torch.cuda.synchronize()

    out = {"gpu": torch.cuda.get_device_name(0), "model": model, "batches": {}}
    for B in [1, 8, 32]:
        eng = LLMEngine(m, cfg, EngineConfig(cache_mode="paged", max_num_seqs=B, max_model_len=1024,
                                             kv_cache_gb=2.0), [], dev, dtype)
        for i in range(B):
            eng.add_request([(7 * i + j) % 1000 + 10 for j in range(128)],
                            SamplingParams(max_tokens=400, ignore_eos=True), lambda t, r: None)
        for _ in range(3):  # prefill + warmup decode steps
            eng.step()
        sync()
        n = 30
        t0 = time.perf_counter()
        for _ in range(n):
            eng.step()
        sync()
        wall = (time.perf_counter() - t0) / n * 1e3

        seqs = list(eng.running)
        t0 = time.perf_counter()
        for _ in range(n):
            batch = PagedDecodeBatch(eng.cache, seqs, dev)
        sync()
        build = (time.perf_counter() - t0) / n * 1e3

        ids = torch.tensor([s.output_ids[-1] for s in seqs], device=dev)
        pos = torch.tensor([s.num_tokens - 1 for s in seqs], device=dev)
        for _ in range(3):
            m(ids, pos, batch)
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            logits = m(ids, pos, batch)
        sync()
        fwd = (time.perf_counter() - t0) / n * 1e3  # includes CPU launch overhead, GPU-synchronised

        # CPU-side time to *issue* a forward without waiting for the GPU.
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            m(ids, pos, batch)
        issue = (time.perf_counter() - t0) / n * 1e3
        sync()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                m(ids, pos, batch)
            sync()
        gpu_us, kernels = 0.0, 0
        for ka in prof.key_averages():
            if str(ka.device_type).endswith("CUDA"):
                gpu_us += getattr(ka, "self_device_time_total", None) or getattr(ka, "self_cuda_time_total", 0)
                kernels += ka.count
        out["batches"][B] = {
            "engine_step_ms": round(wall, 2), "build_batch_ms": round(build, 2),
            "forward_ms": round(fwd, 2), "forward_cpu_issue_ms": round(issue, 2),
            "gpu_busy_ms_per_forward": round(gpu_us / 5 / 1e3, 2), "kernels_per_forward": kernels // 5,
        }
    return out


@app.function(image=ours_image, volumes={HF_DIR: hf_cache}, timeout=3600)
def download(models: list):
    from huggingface_hub import snapshot_download
    for m in models:
        print("downloading", m)
        snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
    hf_cache.commit()


@app.local_entrypoint()
def main(configs: str = "ours_paged", model: str = "Qwen/Qwen3-0.6B", gpu: str = "l4",
         workloads: str = "baseline", kv_gb: float = 2.0, scale_pct: int = 100, max_minutes: int = 40,
         max_len: int = 4096, download_only: bool = False, check_only: bool = False,
         profile_only: bool = False, final: bool = False, with_static: bool = False,
         yes: bool = False):
    if profile_only:
        print(f"Plan: decode-step profile of {model} on 1x L4 (a few minutes, ~cents).")
        if not yes:
            raise SystemExit("Add --yes to run.")
        import json
        print(json.dumps(profile_decode.remote(model), indent=2))
        return
    if check_only:
        print(f"Plan: correctness check of {model} vs HuggingFace on 1x L4 (a few minutes, ~cents).")
        if not yes:
            raise SystemExit("Add --yes to run.")
        import json
        rep = check_correctness.remote(model)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        print("ALL MATCH" if rep["all_match"] else "MISMATCH FOUND")
        return
    if final:  # H100 / Qwen3-4B preset; --configs still selects which configs to run
        if configs == "ours_paged":
            configs = "vllm,ours_paged,ours_contiguous,ours_nocache" + (",ours_static" if with_static else "")
        model, gpu = (model if "4B" in model else "Qwen/Qwen3-4B"), "h100"
        kv_gb = kv_gb if kv_gb != 2.0 else 8.0
    cfgs = [c for c in configs.split(",") if c]
    bad = [c for c in cfgs if c not in VALID]
    if bad:
        raise SystemExit(f"unknown config(s) {bad}; choose from {VALID}")
    if gpu not in ("l4", "h100"):
        raise SystemExit("--gpu must be l4 (dev) or h100 (final)")
    wl = [w for w in workloads.split(",") if w]
    plan = {c: (FINAL_PLAN[c][0], FINAL_PLAN[c][1]) if final else (wl, max_minutes) for c in cfgs}

    if download_only:
        print(f"Plan: download {model} into the Modal volume (CPU only, a few minutes).")
        if not yes:
            raise SystemExit("Add --yes to run.")
        download.remote([model, "Qwen/Qwen3-0.6B"] if "4B" in model else [model])
        return

    startup_min = 8  # image start + model load (+ vLLM compile) per container, rough
    worst = sum(plan[c][1] + startup_min for c in cfgs) / 60 * PRICE[gpu]
    print(f"Plan: {len(cfgs)} config(s) on 1x {gpu.upper()} each, model={model}, scale={scale_pct}%, kv_gb={kv_gb}")
    for c in cfgs:
        print(f"  - {c}: cap {plan[c][1]} min; workloads {plan[c][0]}")
    print(f"      worst-case GPU cost ~ ${worst:.2f} (caps + ~{startup_min} min startup each, ~${PRICE[gpu]:.2f}/h). SHARED class credit!")
    if not yes:
        raise SystemExit("Dry run only. Add --yes to start.")

    args = [(c, model, kv_gb, plan[c][0], scale_pct, plan[c][1], max_len) for c in cfgs]
    ours = [a for a in args if a[0] != "vllm"]
    vl = [a for a in args if a[0] == "vllm"]
    ours_fn = ours_h100 if gpu == "h100" else ours_dev
    vllm_fn = vllm_h100 if gpu == "h100" else vllm_dev
    handles = [ours_fn.spawn(*a) for a in ours] + [vllm_fn.spawn(*a) for a in vl]
    os.makedirs("results", exist_ok=True)
    for h in handles:  # configs run in parallel on separate GPUs: same cost, less wall-clock
        try:
            config, status, blob = h.get()
        except Exception as e:  # one failing config must not lose the others
            print(f"a config failed to return: {e}")
            continue
        print(f"{config}: {status}")
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            tar.extractall("results", filter="data")
    subprocess.run(["python3", "bench/summarize.py", "results"], stdout=open("results/summary.md", "w"))
    print("wrote results/summary.md")
