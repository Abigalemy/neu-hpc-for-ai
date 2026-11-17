"""vLLM baseline for the Final Project: one command -> AIPerf table for a SINGLE sequence.

The Modal workspace credit is shared by the whole class, so nothing starts without --yes.

  modal run modal_vllm_aipref.py                       # prints the plan only (no GPU, no cost)
  modal run modal_vllm_aipref.py --yes                 # 3072 in / 1024 out, 1 sequence, Qwen3-4B, H100
  modal run modal_vllm_aipref.py --yes --isl 130816 --osl 128 --yarn   # 128K context (YaRN): 130816 in + 128 out + chat template <= 131072

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

YARN_CONTEXT = 32768 * 4         # Qwen3 static YaRN, factor 4: input + output must fit in 131072 tokens
CHAT_TEMPLATE_MARGIN = 128       # the chat endpoint wraps the prompt in role tokens; keep room for them


def check_yarn_lengths(isl: int, osl: int):
    """Raises before any GPU is started if the request cannot fit the YaRN context together with the chat template."""
    if isl + osl + CHAT_TEMPLATE_MARGIN > YARN_CONTEXT:
        raise ValueError(f"isl + osl + {CHAT_TEMPLATE_MARGIN} (chat template margin) = {isl + osl + CHAT_TEMPLATE_MARGIN} "
                         f"exceeds the YaRN context of {YARN_CONTEXT} tokens; e.g. use --isl {YARN_CONTEXT - osl - CHAT_TEMPLATE_MARGIN}")


def make_yarn_model_dir(src: str, dst: str) -> dict:
    """A copy of the model directory `src` in `dst` whose config.json adds static YaRN (Qwen3 model card); every other
    file is a symlink (no weight copy). Returns the new config."""
    import json
    os.makedirs(dst, exist_ok=True)
    cfg = None
    for f in sorted(os.listdir(src)):
        target = os.path.join(dst, f)
        if f == "config.json":
            cfg = json.load(open(os.path.join(src, f)))
            cfg["rope_scaling"] = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
            # transformers (v5) expects max_position_embeddings to be the post-YaRN length (it warns that 40960/32768
            # implies factor 1.25 otherwise), and this vLLM derives its max_model_len from max_position_embeddings only.
            cfg["max_position_embeddings"] = YARN_CONTEXT
            with open(target, "w") as fh:
                json.dump(cfg, fh, indent=2)
        elif not os.path.lexists(target):
            os.symlink(os.path.join(src, f), target)
    if cfg is None:
        raise FileNotFoundError(f"no config.json in {src}")
    return cfg



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


@app.function(image=vllm_image, gpu="H100", volumes={HF_DIR: hf_cache, RES_DIR: results_vol}, timeout=25 * 60)   # hard cap 25 min
def run(model: str, isl: int, osl: int, yarn: bool, requests: int, max_minutes: int):
    # Static YaRN, factor 4, original context 32768 (Qwen3 model card). The context limit is then 32768 * 4 = 131072
    # tokens for input + output together, so a "128K" run is 130944 in + 128 out.
    # Tried 2026-11-16 on the current vLLM, all rejected or not recognised as YaRN:
    #   --rope-scaling '{...}'                                  -> "unrecognized arguments"
    #   --hf-overrides '{"rope_scaling": {...}}'                -> max_model_len still derived from 40960
    #   --hf-overrides '{"rope_parameters": {...}}'             -> same
    #   ... + "max_position_embeddings": 131072                 -> derived exactly 131072 (no YaRN factor applied)
    #   config.json with rope_scaling only                      -> transformers applies YaRN (warns 40960/32768 = 1.25
    #                                                              != 4), vLLM still derives max_model_len = 40960
    # This vLLM derives the length from max_position_embeddings only, so "accepting 131072" no longer proves YaRN.
    # Now: config.json with rope_scaling AND max_position_embeddings = 131072 (the consistent form transformers asks
    # for); evidence = the config as transformers parses it (printed and checked below, before vLLM starts).
    # For timing it hardly matters: RoPE is a lookup in a precomputed cos/sin table either way.
    serve_path = model
    if yarn:
        if os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN"):
            raise RuntimeError("VLLM_ALLOW_LONG_MAX_MODEL_LEN must not be set: it would hide a missing YaRN")
        check_yarn_lengths(isl, osl)
        from huggingface_hub import snapshot_download
        serve_path = "/tmp/qwen3-yarn"
        cfg = make_yarn_model_dir(snapshot_download(model, local_files_only=True), serve_path)
        print("config.json: max_position_embeddings", cfg.get("max_position_embeddings"), "rope_scaling", cfg["rope_scaling"], flush=True)
        # Evidence that the YaRN config is what the model code will see: parse it the way vLLM does (transformers).
        from transformers import AutoConfig
        parsed = AutoConfig.from_pretrained(serve_path)
        rp = getattr(parsed, "rope_parameters", None) or getattr(parsed, "rope_scaling", None)
        print("transformers parses rope parameters as:", rp, "| max_position_embeddings", parsed.max_position_embeddings, flush=True)
        if not rp or rp.get("rope_type", rp.get("type")) != "yarn" or float(rp.get("factor", 0)) != 4.0:
            raise RuntimeError("YaRN not recognised in the parsed config; not starting vLLM")
    max_len = YARN_CONTEXT if yarn else isl + osl + 16
    base = ["vllm", "serve", serve_path, "--served-model-name", model, "--dtype", "bfloat16", "--max-model-len", str(max_len),
            "--max-num-seqs", "1", "--no-enable-prefix-caching", "--port", str(PORT)]
    variants = [base]
    out = f"/tmp/aiperf_isl{isl}_osl{osl}"
    server = None
    for cmd in variants:
        print("starting:", " ".join(cmd), flush=True)
        server = subprocess.Popen(cmd)
        try:
            _wait_ready(server)
            break
        except RuntimeError as e:   # the server exited early: the flag may be unsupported, try the next variant
            print(f"{e}")
            server = None
    if server is None:
        raise RuntimeError("vllm did not start with any of the YaRN flag variants")
    try:
        aiperf = ["aiperf", "profile", "--model", model, "--url", f"http://localhost:{PORT}",
                  "--endpoint-type", "chat", "--streaming", "--tokenizer", model,
                  "--extra-inputs", '{"ignore_eos": true, "temperature": 0}',
                  "--concurrency", "1", "--request-count", str(requests), "--warmup-request-count", "1" if isl > 32768 else "2",
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


@app.function(image=vllm_image, gpu="H100", volumes={HF_DIR: hf_cache, RES_DIR: results_vol}, timeout=25 * 60)   # hard cap
def run_suite(model: str, max_minutes: int):
    """All SUITE workloads against ONE vLLM server (max 64 sequences, prefix caching off), one after another, with the
    same AIPerf settings as modal_aipref.py."""
    cmd = ["vllm", "serve", model, "--dtype", "bfloat16", "--max-model-len", str(SUITE_MAX_LEN),
           "--max-num-seqs", str(SUITE_MAX_SEQS), "--no-enable-prefix-caching", "--port", str(PORT)]
    print("starting:", " ".join(cmd), flush=True)
    server = subprocess.Popen(cmd)
    stamp = int(time.time())
    try:
        _wait_ready(server)
        deadline = time.time() + max_minutes * 60
        for name, isl, osl, conc, reqs in SUITE:
            left = int(deadline - time.time())
            if left < 30:
                print(f"=== {name}: skipped, benchmark time cap reached")
                continue
            out = f"/tmp/suite_vllm/{name}"
            print(f"=== WORKLOAD {name}: {isl} in / {osl} out, concurrency {conc}, {reqs} requests", flush=True)
            aiperf = ["aiperf", "profile", "--model", model, "--url", f"http://localhost:{PORT}",
                      "--endpoint-type", "chat", "--streaming", "--tokenizer", model,
                      "--extra-inputs", '{"ignore_eos": true, "temperature": 0}',
                      "--concurrency", str(conc), "--request-count", str(reqs), "--warmup-request-count", str(max(2, conc)),
                      "--synthetic-input-tokens-mean", str(isl), "--synthetic-input-tokens-stddev", "0",
                      "--output-tokens-mean", str(osl), "--output-tokens-stddev", "0",
                      "--random-seed", "0", "--artifact-dir", out]
            rc = subprocess.run(["timeout", str(left)] + aiperf).returncode
            print(f"=== {name}: aiperf exit code {rc}", flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
    subprocess.run(["cp", "-r", "/tmp/suite_vllm", f"{RES_DIR}/suite_vllm_{stamp}"], check=False)
    results_vol.commit()
    print(f"raw results saved to volume final-results:{RES_DIR}/suite_vllm_{stamp}")


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-4B", isl: int = 3072, osl: int = 1024, yarn: bool = False,
         requests: int = 5, max_minutes: int = 25, suite: bool = False, yes: bool = False):
    if suite:
        print(f"PLAN: vLLM {model}, 1 H100, workload suite {[w[0] for w in SUITE]} (max {SUITE_MAX_SEQS} sequences), "
              f"benchmark cap {min(max_minutes, 18)} min, container hard cap 25 min -> at most ~${25 / 60 * PRICE_H100:.2f} of shared credit")
        if yes:
            run_suite.remote(model, min(max_minutes, 18))
        else:
            print("Dry run only. Pass --yes to start.")
        return
    if yarn:
        check_yarn_lengths(isl, osl)   # fails here, locally, before any GPU is started
    est = 25 / 60 * PRICE_H100
    print(f"PLAN: vLLM {model}, 1 H100, single sequence {isl} in / {osl} out, {requests} requests, "
          f"YaRN={yarn}, benchmark cap {max_minutes} min, container hard cap 25 min -> at most ~${est:.2f} of shared credit")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    run.remote(model, isl, osl, yarn, requests, max_minutes)
