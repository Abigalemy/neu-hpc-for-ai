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
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

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
_repo_dir = os.path.dirname(_here)
_core_dir = os.path.join(_repo_dir, "hpc_core")
_tokenizer_cpp_dir = os.path.join(_repo_dir, "week_04_05", "tokenizer_cpp")
_tokenizer_header = os.path.join(_repo_dir, "week_04_05", "tokenizer.h")

# Our server: PyTorch ships its own CUDA runtime, so a slim Python image is enough.
ours_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "cmake", "curl", "git", "pkg-config", "libssl-dev")
    .pip_install("torch==2.10.0", "transformers", "safetensors", "huggingface_hub", "fastapi", "uvicorn",
                 "pydantic", "aiperf", "numpy", "cuda-tile==1.8.0",
                 "cuda-toolkit[tileiras,nvvm,nvcc]>=13.3,<13.6", "pybind11==2.13.6")
    .env({"HF_HOME": HF_DIR,
          "PATH": "/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
    .add_local_dir(_here, APP_DIR, ignore=_ignore, copy=True)
    .add_local_dir(_core_dir, "/root/hpc_core", copy=True)
    .add_local_dir(_tokenizer_cpp_dir, "/root/tokenizer_cpp", copy=True)
    .add_local_file(_tokenizer_header, "/root/tokenizer.h", copy=True)
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
        "cmake -S /root/tokenizer_cpp -B /tmp/qwen3-tokenizer-build "
        "-DCMAKE_BUILD_TYPE=Release "
        "-Dpybind11_DIR=$(python -m pybind11 --cmakedir) "
        "-DQWEN3_TOKENIZER_PYTHON_OUTPUT_DIR=/root/app && "
        "cmake --build /tmp/qwen3-tokenizer-build --target _qwen3_cpp_tokenizer --parallel 4",
    )
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


def _run(config, model, kv_gb, workloads, scale_pct, max_minutes, max_len, output_group=None):
    os.chdir(APP_DIR)
    tag = f"{output_group}/{config}" if output_group else config
    out = f"/tmp/results/{tag}"
    os.makedirs(out, exist_ok=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
    log = open(f"{out}/server.log", "w")
    server = subprocess.Popen(_server_cmd(config, model, kv_gb, max_len), stdout=log, stderr=subprocess.STDOUT)
    status = "ok"
    try:
        _wait_ready(server)
        _sanity(out, model)
        env = dict(os.environ, OUT_DIR="/tmp/results", SCALE_PCT=str(scale_pct))
        cmd = ["timeout", str(max_minutes * 60), "bash", "bench/run_aiperf.sh", tag,
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


# Supplemental grading ablation: the function-level deadline bounds GPU allocation, including model load.
@app.function(image=ours_image, gpu="H100", volumes={HF_DIR: hf_cache, RES_DIR: results_vol}, timeout=15 * 60)
def ours_h100_capped(*a):
    return _run(*a)


@app.function(image=vllm_image, gpu="L4", **_common)
def vllm_dev(*a):
    return _run(*a)


@app.function(image=vllm_image, gpu="H100", **_common)
def vllm_h100(*a):
    return _run(*a)


@app.function(image=vllm_image, gpu="H100", volumes={HF_DIR: hf_cache, RES_DIR: results_vol}, timeout=15 * 60)
def vllm_h100_capped(*a):
    return _run(*a)


@app.function(image=ours_image, gpu="H100", volumes={HF_DIR: hf_cache, RES_DIR: results_vol}, timeout=30 * 60)
def warmup_sensitivity_h100(model: str):
    """Run baseline and balanced c16 three times per warmup mode in one server process/container."""
    os.chdir(APP_DIR)
    root = "/tmp/results/warmup_sensitivity"
    os.makedirs(root, exist_ok=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
    server_log = open(f"{root}/server.log", "w")
    server = subprocess.Popen(_server_cmd("ours_paged", model, 8.0, 4096),
                              stdout=server_log, stderr=subprocess.STDOUT)
    status = {"gpu": "H100", "model": model, "status": "ok", "runs": []}
    try:
        _wait_ready(server)
        # Keep one serving process alive across all profiles to isolate request warmup from restarts.
        deadline = time.monotonic() + 25 * 60
        for repetition in range(1, 4):
            for warmup in (0, 8):
                tag = f"warmup_sensitivity/repeat{repetition}_warmup{warmup}"
                remaining = max(1, int(deadline - time.monotonic()))
                env = dict(os.environ, OUT_DIR="/tmp/results", SCALE_PCT="100",
                           AIPERF_WARMUP_REQUEST_COUNT=str(warmup))
                cmd = ["timeout", f"{remaining}s", "bash", "bench/run_aiperf.sh", tag,
                       f"http://localhost:{PORT}", model, model, "baseline", "balanced:16"]
                with open(f"{root}/repeat{repetition}_warmup{warmup}.log", "w") as log:
                    completed = subprocess.run(cmd, env=env, stdout=log,
                                               stderr=subprocess.STDOUT).returncode
                status["runs"].append({"repetition": repetition, "warmup_requests": warmup,
                                       "return_code": completed})
                if completed == 124 or time.monotonic() >= deadline:
                    status["status"] = "partial: 25-minute benchmark budget reached"
                    break
            if status["status"] != "ok":
                break
        try:
            import urllib.request
            with open(f"{root}/engine_stats.json", "wb") as f:
                f.write(urllib.request.urlopen(f"http://localhost:{PORT}/metrics_json", timeout=10).read())
        except Exception as e:
            status["engine_stats_error"] = str(e)
    except Exception as e:
        status["status"] = f"FAILED: {e}"
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()
    (Path(root) / "run_manifest.json").write_text(json.dumps(status, indent=2))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(root, arcname="warmup_sensitivity")
    blob = buf.getvalue()
    try:
        with open(f"{RES_DIR}/warmup_sensitivity_{int(time.time())}.tar.gz", "wb") as f:
            f.write(blob)
        results_vol.commit()
    except Exception as e:
        print(f"could not save warmup-sensitivity results to volume: {e}")
    return status, blob


@app.function(image=ours_image, gpu="H100", volumes={HF_DIR: hf_cache}, timeout=900)
def check_correctness(model: str):
    """Real weights + real tokenizer: our engine (every cache mode, concurrent requests) vs HuggingFace.

    float32 on both sides so that greedy decoding is deterministic enough to compare token by token.
    """
    import sys
    import torch
    import transformers
    import importlib.metadata
    sys.path.insert(0, APP_DIR)
    os.chdir(APP_DIR)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine import EngineConfig, LLMEngine, SamplingParams
    from qwen3_model import load_model
    from kv_cache import NoCacheBatch

    dev, n_new = "cuda", 24
    mine, cfg, d = load_model(model, dev, torch.float32, max_positions=1024)
    tok = AutoTokenizer.from_pretrained(d)
    hf = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).to(dev).eval()
    prompts = ["The capital of France is", "def fibonacci(n):\n    ", "中国的首都是",
               "Once upon a time, in a small village by the sea, there lived an old fisherman who"]
    ids = [tok.encode(p) for p in prompts]

    # Persist a real-weight prefill computation as a SQLite test case from the
    # local entrypoint. Keep all prompt-position logits so the CUDA implementation
    # can be compared with the exact Hugging Face reference later.
    case_ids = torch.tensor(ids[0], dtype=torch.long, device=dev)
    case_positions = torch.arange(len(ids[0]), dtype=torch.long, device=dev)
    with torch.no_grad():
        hf_case_logits = hf(case_ids[None, :], use_cache=False).logits[0].float()
        case_batch = NoCacheBatch([0, len(ids[0])], dev)
        case_batch.logit_indices = torch.arange(len(ids[0]), device=dev)
        ours_case_logits = mine(case_ids, case_positions, case_batch).float()
    case_max_diff = (ours_case_logits - hf_case_logits).abs().max().item()
    case_match = torch.allclose(ours_case_logits, hf_case_logits, atol=2e-4, rtol=2e-4)
    import json
    import numpy as np
    e2e_case = {
        "model_id": model,
        "model_revision": os.path.basename(d.rstrip("/")),
        "model_config_json": hf.config.to_json_string(),
        "runtime_json": json.dumps({
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_tile": importlib.metadata.version("cuda-tile"),
        }, sort_keys=True),
        "input_ids": ids[0],
        "positions": list(range(len(ids[0]))),
        "dtype": "float32",
        "causal": True,
        "expected_logits": np.asarray(hf_case_logits.cpu(), dtype=np.float32).tobytes(),
        "logits_rows": int(hf_case_logits.shape[0]),
        "vocab_size": int(hf_case_logits.shape[1]),
        "max_abs_diff": case_max_diff,
        "match": case_match,
    }

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
    report["all_match"] = ok and case_match
    report["e2e_case"] = e2e_case
    return report


@app.function(image=ours_image, gpu="H100", volumes={HF_DIR: hf_cache}, timeout=15 * 60)
def profile_decode(model: str, gpu_label: str = "H100"):
    """Compare the original gather+attention and direct paged-kernel paths on H100."""
    import sys
    import torch
    sys.path.insert(0, APP_DIR)
    os.chdir(APP_DIR)
    from torch.profiler import ProfilerActivity, profile
    from engine import EngineConfig, LLMEngine, SamplingParams
    from kv_cache import PagedDecodeBatch
    from hpc_core.kv_cache import _tile_attention
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

        # Recreate the previous implementation for a same-device, same-input
        # A/B: gather every sequence's KV and call the ordinary Tile attention.
        class GatherReferenceBatch:
            logit_indices = batch.logit_indices

            def attend(self, layer_idx, q, k, v):
                c = eng.cache
                c.k[layer_idx].index_copy_(0, batch.slot_mapping, k)
                c.v[layer_idx].index_copy_(0, batch.slot_mapping, v)
                gathered_k = c.k[layer_idx][batch.gather_idx].transpose(1, 2)
                gathered_v = c.v[layer_idx][batch.gather_idx].transpose(1, 2)
                return torch.cat([
                    _tile_attention(q[i:i + 1, :, None, :],
                                    gathered_k[i:i + 1, :, :ctx, :],
                                    gathered_v[i:i + 1, :, :ctx, :], causal=False)
                    for i, ctx in enumerate(batch.ctx)
                ], dim=0)[:, :, 0, :]

        ref_tables = torch.tensor([s.block_table for s in seqs], device=dev)
        ar_bs = torch.arange(eng.cache.block_size, device=dev)
        gather_batch = GatherReferenceBatch()
        batch.gather_idx = (ref_tables[:, :, None] * eng.cache.block_size + ar_bs).reshape(B, -1)

        ids = torch.tensor([s.output_ids[-1] for s in seqs], device=dev)
        pos = torch.tensor([s.num_tokens - 1 for s in seqs], device=dev)
        for _ in range(3):
            m(ids, pos, batch)
            m(ids, pos, gather_batch)
        sync()
        gather_logits = m(ids, pos, gather_batch)
        paged_logits = m(ids, pos, batch)
        sync()
        max_logit_diff = (gather_logits.float() - paged_logits.float()).abs().max().item()
        greedy_match = bool(torch.equal(gather_logits.argmax(-1), paged_logits.argmax(-1)))
        t0 = time.perf_counter()
        for _ in range(n):
            logits = m(ids, pos, batch)
        sync()
        fwd = (time.perf_counter() - t0) / n * 1e3  # includes CPU launch overhead, GPU-synchronised

        def cuda_event_ms(fn, repeats=10):
            for _ in range(3):
                fn()
            sync()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                fn()
            end.record()
            sync()
            return start.elapsed_time(end) / repeats

        gather_fwd_ms = cuda_event_ms(lambda: m(ids, pos, gather_batch))
        paged_fwd_ms = cuda_event_ms(lambda: m(ids, pos, batch))

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
            "gather_reference_forward_cuda_ms": round(gather_fwd_ms, 3),
            "block_table_forward_cuda_ms": round(paged_fwd_ms, 3),
            "max_abs_logit_diff_vs_gather": max_logit_diff,
            "greedy_match_vs_gather": greedy_match,
            "gpu_busy_ms_per_forward": round(gpu_us / 5 / 1e3, 2), "kernels_per_forward": kernels // 5,
        }
    return out


@app.function(image=ours_image, gpu="H100", volumes={HF_DIR: hf_cache, RES_DIR: results_vol}, timeout=45 * 60)
def gap_study(model: str):
    """Repeat the original latency-floor comparison and isolate paged KV gather cost on H100."""
    import json
    import sys
    import torch
    sys.path.insert(0, APP_DIR)
    os.chdir(APP_DIR)
    from hpc_core.kernels.cutile_attention import paged_decode_attention
    from hpc_core.kv_cache import _tile_attention

    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(20261105)
    micro = {"gpu": torch.cuda.get_device_name(0), "dtype": str(dtype), "block_size": 16, "cases": []}

    def timed_ms(fn, repeats=30):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeats

    # Isolate the former K/V materialization and attention costs from the model
    # forward. Random physical block order exercises block-table indirection.
    for batch_size in (1, 8, 32):
        for seq_len in (128, 1024):
            block_size, kv_heads, q_heads, head_dim = 16, 8, 32, 128
            blocks_per_seq = (seq_len + block_size - 1) // block_size
            nblocks = batch_size * blocks_per_seq
            pool = torch.randn((nblocks, block_size, kv_heads, head_dim), device=device, dtype=dtype)
            k_pool, v_pool = pool, torch.randn_like(pool)
            q = torch.randn((batch_size, q_heads, head_dim), device=device, dtype=dtype)
            tables = torch.empty((batch_size, blocks_per_seq), device=device, dtype=torch.int64)
            for i in range(batch_size):
                ids = torch.arange(i * blocks_per_seq, (i + 1) * blocks_per_seq, device=device)
                tables[i] = ids[torch.randperm(blocks_per_seq, device=device)]
            seq_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
            offsets = torch.arange(block_size, device=device)
            gather_idx = (tables[:, :, None] * block_size + offsets).reshape(batch_size, -1)
            k_flat = k_pool.reshape(-1, kv_heads, head_dim)
            v_flat = v_pool.reshape(-1, kv_heads, head_dim)

            def gather():
                return k_flat[gather_idx].transpose(1, 2), v_flat[gather_idx].transpose(1, 2)

            def attend(gathered):
                keys, values = gathered
                return torch.cat([
                    _tile_attention(q[i:i + 1, :, None, :], keys[i:i + 1, :, :seq_len, :],
                                    values[i:i + 1, :, :seq_len, :], causal=False)
                    for i in range(batch_size)
                ], dim=0)[:, :, 0, :]

            gathered = gather()
            reference = attend(gathered)
            actual = paged_decode_attention(q, k_pool, v_pool, tables, seq_lens)
            torch.cuda.synchronize()
            max_diff = (reference.float() - actual.float()).abs().max().item()
            allclose = torch.allclose(reference.float(), actual.float(), atol=2e-2, rtol=2e-2)
            gather_ms = timed_ms(gather)
            attention_ms = timed_ms(lambda: attend(gathered))
            combined_ms = timed_ms(lambda: attend(gather()))
            direct_ms = timed_ms(lambda: paged_decode_attention(q, k_pool, v_pool, tables, seq_lens))
            micro["cases"].append({
                "batch": batch_size, "seq_len": seq_len,
                "gather_only_ms": round(gather_ms, 4),
                "tile_attention_on_materialized_kv_ms": round(attention_ms, 4),
                "gather_plus_tile_attention_ms": round(combined_ms, 4),
                "direct_block_table_attention_ms": round(direct_ms, 4),
                "max_abs_output_diff": max_diff, "allclose_atol_rtol_0_02": allclose,
            })

    study_status = []
    for repetition in range(1, 4):
        for config in ("ours_contiguous", "ours_nocache", "ours_paged"):
            _, status, _ = _run(config, model, 8.0, ["baseline"], 100, 3, 4096,
                                output_group=f"gap_study/repeat{repetition}")
            study_status.append({"repeat": repetition, "config": config, "status": status})

    root = "/tmp/results/gap_study"
    os.makedirs(root, exist_ok=True)
    with open(f"{root}/h100_gather_microbench.json", "w") as f:
        json.dump(micro, f, indent=2)
    with open(f"{root}/run_status.json", "w") as f:
        json.dump(study_status, f, indent=2)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(root, arcname=".")
    return {"microbench": micro, "status": study_status, "results_tar": buf.getvalue()}


@app.function(image=ours_image, gpu="L4", volumes={HF_DIR: hf_cache}, timeout=300)
def model_performance_matrix(model: str):
    """Compare real Qwen3 HF and CUDA Tile prefill over batch/sequence shapes on L4."""
    import sys
    import time
    import torch
    sys.path.insert(0, APP_DIR)
    os.chdir(APP_DIR)
    from transformers import AutoModelForCausalLM
    from kv_cache import NoCacheBatch
    from qwen3_model import load_model

    dev, dtype = "cuda", torch.bfloat16
    ours, cfg, model_dir = load_model(model, dev, dtype, max_positions=256)
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(dev).eval()
    torch.manual_seed(7411)
    results = []

    def timed(fn, repeats=15):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1e3 / repeats

    for batch in (1, 2, 4):
        for seq in (32, 64, 128):
            ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=dev)
            flat = ids.reshape(-1)
            pos = torch.arange(seq, device=dev).repeat(batch)
            cu = [i * seq for i in range(batch + 1)]
            forward_batch = NoCacheBatch(cu, dev)

            def hf_forward():
                return hf(ids, use_cache=False, logits_to_keep=1).logits[:, -1, :]

            def ours_forward():
                return ours(flat, pos, forward_batch)

            with torch.inference_mode():
                ref = hf_forward().float()
                actual = ours_forward().float()
                diff = (actual - ref).abs().max().item()
                greedy_match = bool(torch.equal(actual.argmax(-1), ref.argmax(-1)))
                hf_trials, tile_trials = [], []
                for trial in range(3):
                    if trial % 2 == 0:
                        hf_trials.append(timed(hf_forward))
                        tile_trials.append(timed(ours_forward))
                    else:
                        tile_trials.append(timed(ours_forward))
                        hf_trials.append(timed(hf_forward))
                ref_ms = sorted(hf_trials)[1]
                tile_ms = sorted(tile_trials)[1]
            results.append({
                "batch": batch, "seq": seq, "hf_ms": ref_ms, "cutile_ms": tile_ms,
                "speedup_hf_over_cutile": ref_ms / tile_ms,
                "max_abs_diff": diff, "greedy_match": greedy_match,
                "hf_ms_trials": hf_trials, "cutile_ms_trials": tile_trials,
            })
    return {"gpu": torch.cuda.get_device_name(0), "model": model, "dtype": str(dtype), "results": results}


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
         profile_only: bool = False, gap_study_only: bool = False, final_kernel_compare: bool = False,
         warmup_sensitivity: bool = False, results_group: str = "", model_matrix: bool = False,
         final: bool = False, with_static: bool = False, ablation_budget_cap: bool = False,
         yes: bool = False):
    output_group = results_group or None
    if final_kernel_compare:
        configs = "ours_paged,vllm"
        model, gpu, kv_gb, scale_pct, max_len = "Qwen/Qwen3-4B", "h100", 8.0, 100, 4096
        workloads = "baseline,balanced:16+64,mixed_poisson:8,pressure"
        max_minutes = 10
        output_group = "final_kernel_compare"
    if ablation_budget_cap:
        configs = "ours_paged,ours_static,ours_nocache"
        model, gpu, kv_gb, scale_pct, max_len = "Qwen/Qwen3-4B", "h100", 8.0, 50, 4096
        workloads = "balanced:16,mixed_poisson:8,decode_heavy:8"
        max_minutes = 7
        output_group = "grade_ablation"
    if model_matrix:
        print(f"Plan: Qwen3 HF vs CUDA Tile prefill matrix on 1x L4, 9 shapes, 5-minute cap; "
              f"GPU charge capped near ${5 / 60 * PRICE['l4']:.2f}.")
        if not yes:
            raise SystemExit("Add --yes to run.")
        import csv
        rep = model_performance_matrix.remote(model)
        output = Path(__file__).resolve().parent.parent / "week_04_05" / "real_qwen3_l4_matrix.csv"
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rep["results"][0].keys())
            writer.writeheader()
            writer.writerows(rep["results"])
        print(json.dumps({k: v for k, v in rep.items() if k != "results"}, indent=2))
        print(f"Saved {len(rep['results'])} rows to {output}")
        print(json.dumps(rep["results"], indent=2))
        return
    if profile_only:
        print(f"Plan: same-H100 gather-reference vs block-table decode profile of {model}; 15-minute hard cap (~$0.99 GPU-only).")
        if not yes:
            raise SystemExit("Add --yes to run.")
        report = profile_decode.remote(model)
        profile_path = Path(__file__).resolve().parent / "results" / "grade_ablation" / "h100_profile.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print(f"Saved {profile_path}")
        return
    if gap_study_only:
        cap = 45 / 60 * PRICE["h100"]
        print(f"Plan: exact baseline repeated 3x for contiguous/no-cache/paged plus an isolated H100 gather microbenchmark; "
              f"one H100 container, 45-minute hard cap (~${cap:.2f} GPU-only; CPU/memory extra).")
        if not yes:
            raise SystemExit("Add --yes to run.")
        report = gap_study.remote(model if "4B" in model else "Qwen/Qwen3-4B")
        result_dir = Path(__file__).resolve().parent / "results" / "gap_study"
        result_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(report.pop("results_tar")), mode="r:gz") as tar:
            tar.extractall(result_dir, filter="data")
        (result_dir / "run_manifest.json").write_text(json.dumps(report["status"], indent=2))
        print(json.dumps({"microbench": report["microbench"], "status": report["status"]}, indent=2))
        print(f"Saved raw profiles and microbenchmark to {result_dir}")
        return
    if warmup_sensitivity:
        cap = 30 / 60 * PRICE["h100"]
        print(f"Plan: one H100 container, one persistent paged server, 3 repeats each of baseline/c16 with "
              f"AIPerf warmup omitted and 8 requests; 30-minute hard cap (~${cap:.2f} GPU-only; CPU/memory extra).")
        if not yes:
            raise SystemExit("Add --yes to run.")
        status, blob = warmup_sensitivity_h100.remote(
            model if "4B" in model else "Qwen/Qwen3-4B")
        result_dir = Path(__file__).resolve().parent / "results" / "warmup_sensitivity"
        result_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            tar.extractall(result_dir.parent, filter="data")
        (result_dir / "remote_status.json").write_text(json.dumps(status, indent=2))
        print(json.dumps(status, indent=2))
        print(f"Saved raw AIPerf profiles to {result_dir}")
        return
    if final_kernel_compare:
        hard_cap_min = 15
        worst_gpu = len(configs.split(",")) * hard_cap_min / 60 * PRICE["h100"]
        print(f"Plan: final paged kernel vs vLLM on baseline, balanced c16/c64, mixed Poisson r8, and pressure; "
              f"each H100 container has a {hard_cap_min}-minute hard cap. "
              f"GPU-only maximum ~${worst_gpu:.2f}; CPU/memory extra.")
    if check_only:
        print(f"Plan: correctness check of {model} vs HuggingFace on 1x H100, max 15 minutes. "
              f"GPU charge is capped at about ${15 / 60 * PRICE['h100']:.2f}; CPU/memory may add a small amount.")
        if not yes:
            raise SystemExit("Add --yes to run.")
        import sqlite3
        rep = check_correctness.remote(model)
        case = rep.pop("e2e_case")
        db_path = Path(__file__).resolve().parent.parent / "week_04_05" / "test_cases.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS qwen3_4b_prefill_cases")
            conn.execute("""CREATE TABLE qwen3_4b_prefill_cases (
                id INTEGER PRIMARY KEY,
                model_id TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                model_config_json TEXT NOT NULL,
                runtime_json TEXT NOT NULL,
                input_ids_json TEXT NOT NULL,
                positions_json TEXT NOT NULL,
                dtype TEXT NOT NULL,
                causal INTEGER NOT NULL,
                expected_logits BLOB NOT NULL,
                logits_rows INTEGER NOT NULL,
                vocab_size INTEGER NOT NULL,
                max_abs_diff REAL NOT NULL,
                matches_huggingface INTEGER NOT NULL
            )""")
            conn.execute(
                "INSERT INTO qwen3_4b_prefill_cases VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (case["model_id"], case["model_revision"], case["model_config_json"],
                 case["runtime_json"], json.dumps(case["input_ids"]),
                 json.dumps(case["positions"]), case["dtype"],
                 int(case["causal"]), case["expected_logits"], case["logits_rows"],
                 case["vocab_size"], case["max_abs_diff"], int(case["match"])),
            )
        print(f"Saved real Qwen3-4B HF logits to {db_path} (revision {case['model_revision']}, "
              f"max diff {case['max_abs_diff']:.3e}, match={case['match']})")
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
    if ablation_budget_cap:
        plan = {
            "ours_paged": (["balanced:16", "mixed_poisson:8", "decode_heavy:8"], max_minutes),
            "ours_static": (["balanced:16", "mixed_poisson:8", "decode_heavy:8"], max_minutes),
            "ours_nocache": (["decode_heavy:8"], max_minutes),
        }
    else:
        plan = {c: (FINAL_PLAN[c][0], FINAL_PLAN[c][1]) if final else (wl, max_minutes) for c in cfgs}

    if download_only:
        print(f"Plan: download {model} into the Modal volume (CPU only, a few minutes).")
        if not yes:
            raise SystemExit("Add --yes to run.")
        download.remote([model, "Qwen/Qwen3-0.6B"] if "4B" in model else [model])
        return

    startup_min = 8  # image start + model load (+ vLLM compile) per container, rough
    if ablation_budget_cap or final_kernel_compare:
        hard_cap_min = 15
        worst = len(cfgs) * hard_cap_min / 60 * PRICE[gpu]
        print(f"Hard cap: {hard_cap_min} minutes per H100 function, {len(cfgs) * hard_cap_min} GPU-minutes total; "
              f"GPU-only upper estimate ~${worst:.2f}. CPU and memory are additional.")
    else:
        worst = sum(plan[c][1] + startup_min for c in cfgs) / 60 * PRICE[gpu]
    print(f"Plan: {len(cfgs)} config(s) on 1x {gpu.upper()} each, model={model}, scale={scale_pct}%, kv_gb={kv_gb}")
    for c in cfgs:
        print(f"  - {c}: cap {plan[c][1]} min; workloads {plan[c][0]}")
    if ablation_budget_cap or final_kernel_compare:
        print(f"      H100 GPU-only hard upper estimate ~${worst:.2f}; CPU/memory are additional.")
    else:
        print(f"      worst-case GPU cost ~ ${worst:.2f} (caps + ~{startup_min} min startup each, ~${PRICE[gpu]:.2f}/h). SHARED class credit!")
    if not yes:
        raise SystemExit("Dry run only. Add --yes to start.")

    args = [((c, model, kv_gb, plan[c][0], scale_pct, plan[c][1], max_len, output_group)
             if output_group else (c, model, kv_gb, plan[c][0], scale_pct, plan[c][1], max_len)) for c in cfgs]
    ours = [a for a in args if a[0] != "vllm"]
    vl = [a for a in args if a[0] == "vllm"]
    ours_fn = ours_h100_capped if (ablation_budget_cap or final_kernel_compare) else (ours_h100 if gpu == "h100" else ours_dev)
    vllm_fn = vllm_h100_capped if final_kernel_compare else (vllm_h100 if gpu == "h100" else vllm_dev)
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
    summary_root = os.path.join("results", output_group) if output_group else "results"
    os.makedirs(summary_root, exist_ok=True)
    summary_path = os.path.join(summary_root, "summary.md")
    subprocess.run(["python3", "bench/summarize.py", summary_root], stdout=open(summary_path, "w"))
    print(f"wrote {summary_path}")
