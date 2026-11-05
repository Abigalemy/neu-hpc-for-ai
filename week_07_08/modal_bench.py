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
         profile_only: bool = False, model_matrix: bool = False,
         final: bool = False, with_static: bool = False, ablation_budget_cap: bool = False,
         yes: bool = False):
    output_group = None
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
        import json
        from pathlib import Path
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
        print(f"Plan: decode-step profile of {model} on 1x L4 (a few minutes, ~cents).")
        if not yes:
            raise SystemExit("Add --yes to run.")
        import json
        print(json.dumps(profile_decode.remote(model), indent=2))
        return
    if check_only:
        print(f"Plan: correctness check of {model} vs HuggingFace on 1x H100, max 15 minutes. "
              f"GPU charge is capped at about ${15 / 60 * PRICE['h100']:.2f}; CPU/memory may add a small amount.")
        if not yes:
            raise SystemExit("Add --yes to run.")
        import json
        import sqlite3
        from pathlib import Path
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
    if ablation_budget_cap:
        hard_cap_min = 15
        worst = len(cfgs) * hard_cap_min / 60 * PRICE[gpu]
        print(f"Hard cap: {hard_cap_min} minutes per H100 function, {len(cfgs) * hard_cap_min} GPU-minutes total; "
              f"GPU-only upper estimate ~${worst:.2f}. CPU and memory are additional.")
    else:
        worst = sum(plan[c][1] + startup_min for c in cfgs) / 60 * PRICE[gpu]
    print(f"Plan: {len(cfgs)} config(s) on 1x {gpu.upper()} each, model={model}, scale={scale_pct}%, kv_gb={kv_gb}")
    for c in cfgs:
        print(f"  - {c}: cap {plan[c][1]} min; workloads {plan[c][0]}")
    if ablation_budget_cap:
        print(f"      H100 GPU-only hard upper estimate ~${worst:.2f}; CPU/memory are additional.")
    else:
        print(f"      worst-case GPU cost ~ ${worst:.2f} (caps + ~{startup_min} min startup each, ~${PRICE[gpu]:.2f}/h). SHARED class credit!")
    if not yes:
        raise SystemExit("Dry run only. Add --yes to start.")

    args = [((c, model, kv_gb, plan[c][0], scale_pct, plan[c][1], max_len, output_group)
             if output_group else (c, model, kv_gb, plan[c][0], scale_pct, plan[c][1], max_len)) for c in cfgs]
    ours = [a for a in args if a[0] != "vllm"]
    vl = [a for a in args if a[0] == "vllm"]
    ours_fn = ours_h100_capped if ablation_budget_cap else (ours_h100 if gpu == "h100" else ours_dev)
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
    summary_root = os.path.join("results", output_group) if output_group else "results"
    os.makedirs(summary_root, exist_ok=True)
    summary_path = os.path.join(summary_root, "summary.md")
    subprocess.run(["python3", "bench/summarize.py", summary_root], stdout=open(summary_path, "w"))
    print(f"wrote {summary_path}")
