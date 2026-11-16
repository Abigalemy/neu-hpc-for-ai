"""End-to-end check of the Tile inference engine against HF Transformers (CPU, bf16) on a real model.

  modal run e2e_test.py                         # prints the plan only
  modal run e2e_test.py --yes                   # Qwen3-0.6B on 1x L4, hard cap 25 min
  modal run e2e_test.py --yes --model Qwen/Qwen3-4B
  modal run e2e_test.py --yes --model Qwen/Qwen3-4B --gpu h100     # big tiles + bf16-P attention

What is checked:
  * teacher forcing: prefill the prompt in one chunk, then feed HF's own continuation one token at a time
    (this exercises the KV-cache decode path); logits at every position vs HF: top-1 agreement, cosine, max error
  * greedy generation of the engine vs HF's greedy tokens
  * the engine's RoPE tables vs HF's rotary embedding
  * TTFT / ITL of this (unoptimised) engine on the L4 -- NOT comparable with the vLLM H100 baseline.
Weights come from the read-only volume `qwen3-hf-cache`; nothing is downloaded.
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("final-e2e")
hf_cache = modal.Volume.from_name("qwen3-hf-cache", create_if_missing=False)
try:
    hf_cache = hf_cache.read_only()
except AttributeError:
    pass
IMAGE = "nvidia/cuda:13.5.0-devel-ubuntu24.04"
image = (
    modal.Image.from_registry(IMAGE, add_python="3.12")
    .pip_install("numpy", "safetensors", "transformers", "torch", extra_index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(HERE, "/root/final", ignore=[".venv", "__pycache__", "*.pyc", "build", "test_cases.db"])
)
PRICE_L4 = 0.80
PROMPT = ("Explain in two sentences why tiling a matrix multiplication in GPU shared memory makes it faster. "
          "Then name one limit of the technique.")


def _impl(model_name: str, n_gen: int, arch: str, ab: bool = False):
    import glob
    import re
    import subprocess

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.chdir("/root/final")
    r = subprocess.run(f"mkdir -p build && nvcc --enable-tile -std=c++20 -arch={arch} -O2 tests/engine_test.cu -o build/engine_test",
                       shell=True, capture_output=True, text=True)
    print("compile exit code:", r.returncode, "arch", arch)
    if r.returncode:
        print(r.stderr[-3000:])
        return 1

    snap = glob.glob(f"/hf_cache/hub/models--{model_name.replace('/', '--')}/snapshots/*")[0]
    tok = AutoTokenizer.from_pretrained(snap)
    try:
        model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16)
    model.eval()

    prompt_ids = tok(PROMPT, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        full = model.generate(prompt_ids, max_new_tokens=n_gen, do_sample=False)
    full = full[0]
    n_prompt, n_total = prompt_ids.shape[1], full.shape[0]
    n_gen_real = n_total - n_prompt
    print(f"model {model_name}: prompt {n_prompt} tokens, HF generated {n_gen_real}")
    print("HF text:", repr(tok.decode(full[n_prompt:])))
    with torch.no_grad():
        ref = model(full[None]).logits[0].float()                      # [n_total, vocab]
    open("/tmp/tokens.bin", "wb").write(full.to(torch.int32).numpy().tobytes())

    V = ref.shape[1]
    top_ref = ref.argmax(-1)
    top2 = torch.topk(ref, 2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    scale = ref.abs().max().item()
    exp = full[n_prompt:].numpy()
    variants = [("default (what the engine picks on this device)", {})]
    if ab:   # same HF reference, three engine configurations: isolates what changed the numbers
        variants += [("big GEMM tiles + SMALL attention (QWEN3_ATTN=small)", {"QWEN3_ATTN": "small"}),
                     ("all small tiles (QWEN3_TILES=small): the configuration verified on the L4", {"QWEN3_TILES": "small"})]
    summary = []
    for vi, (tag, extra) in enumerate(variants):
        env = dict(os.environ)
        env.update(extra)
        r = subprocess.run(["build/engine_test", snap, "/tmp/tokens.bin", str(n_total), str(n_prompt), "/tmp/logits.bin",
                            "/tmp/rope.bin", "/tmp/gen.bin", str(n_gen_real)], capture_output=True, text=True, env=env)
        if vi == 0:
            print(r.stdout)
        if r.returncode:
            print("engine_test failed:", r.stderr[-2000:])
            return 1
        raw = np.frombuffer(open("/tmp/logits.bin", "rb").read(), dtype=np.uint16).astype(np.uint32) << 16
        mine = torch.from_numpy(raw.view(np.float32).reshape(n_total, V).copy())
        cos = torch.nn.functional.cosine_similarity(ref, mine, dim=-1)
        maxabs = (ref - mine).abs().max(-1).values
        mism = (top_ref != mine.argmax(-1)).nonzero().flatten().tolist()
        pre_cos, dec_cos = cos[:n_prompt].min().item(), (cos[n_prompt:].min().item() if n_total > n_prompt else 1.0)
        gen = np.frombuffer(open("/tmp/gen.bin", "rb").read(), dtype=np.int32)
        first_bad = next((i for i in range(len(exp)) if gen[i] != exp[i]), None)
        print(f"--- engine configuration: {tag}")
        print(f"teacher-forced positions: {n_total} (prefill path 0..{n_prompt - 1}, KV-cache decode path {n_prompt}..{n_total - 1})")
        print(f"  top-1 agreement: {n_total - len(mism)}/{n_total}")
        for i in mism:
            print(f"    mismatch at position {i} ({'prefill' if i < n_prompt else 'decode'}): HF margin top1-top2 = {margin[i]:.3f}")
        worst = int(torch.argmin(cos))
        print(f"  cosine similarity per position: min {cos.min():.6f} (position {worst}, HF margin {margin[worst]:.3f}), mean {cos.mean():.6f}")
        print(f"  max abs logit error {maxabs.max():.3f}  (largest |logit| {scale:.1f}; relative {maxabs.max() / scale:.2e})")
        print(f"  min cosine: prefill {pre_cos:.6f}  decode {dec_cos:.6f}")
        print(f"  greedy generation: engine matches HF for {len(exp) if first_bad is None else first_bad}/{len(exp)} tokens"
              + ("" if first_bad is None else f" (first difference at token {first_bad})"))
        if vi == 0:
            print("  engine text:", repr(tok.decode(gen.tolist())))
            first = (mism, pre_cos, dec_cos)
        summary.append((tag, n_total - len(mism), pre_cos, dec_cos))
    mism, pre_cos, dec_cos = first
    if ab:
        print("\nA/B summary (same HF reference): configuration | top-1 | min cosine prefill | min cosine decode")
        for tag, ok_n, pc, dc in summary:
            print(f"  {tag[:62]:<62} {ok_n}/{n_total}  {pc:.6f}  {dc:.6f}")

    # RoPE tables vs HF
    rot = model.model.rotary_emb
    P = n_total + n_gen_real + 8
    cos_hf, sin_hf = rot(torch.zeros(1, P, model.config.head_dim, dtype=torch.bfloat16), torch.arange(P)[None])
    rt = np.frombuffer(open("/tmp/rope.bin", "rb").read(), dtype=np.uint16).reshape(2, P, 64)
    hf_c = cos_hf[0, :, :64].contiguous().view(torch.int16).numpy().view(np.uint16)
    hf_s = sin_hf[0, :, :64].contiguous().view(torch.int16).numpy().view(np.uint16)
    bad_c, bad_s = int((rt[0] != hf_c).sum()), int((rt[1] != hf_s).sum())
    print(f"RoPE tables (positions 0..{P - 1}): differing bf16 entries cos {bad_c}/{hf_c.size}, sin {bad_s}/{hf_s.size}")

    # verdict: any top-1 mismatch must be a near tie in HF's own logits
    bad_mism = [i for i in mism if margin[i] >= 0.5]
    ok = (not bad_mism) and min(pre_cos, dec_cos) > 0.999
    print("\nPASS" if ok else "\nFAIL (top-1 mismatch with a clear margin, or cosine below 0.999)")
    return 0 if ok else 1


@app.function(image=image, gpu="L4", volumes={"/hf_cache": hf_cache}, timeout=2400)
def run_l4(model_name: str, n_gen: int, ab: bool = False):
    return _impl(model_name, n_gen, "sm_80", ab)


@app.function(image=image, gpu="H100", volumes={"/hf_cache": hf_cache}, timeout=900)   # hard cap 15 min (~$1)
def run_h100(model_name: str, n_gen: int, ab: bool = False):
    return _impl(model_name, n_gen, "sm_90", ab)


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-0.6B", n_gen: int = 16, gpu: str = "l4", ab: bool = False, yes: bool = False):
    price, cap = (3.95, 15) if gpu == "h100" else (PRICE_L4, 40)
    print(f"PLAN: {model} from volume qwen3-hf-cache (read-only, no download), HF on CPU as reference, engine on 1x {gpu.upper()}, "
          f"hard cap {cap} min -> at most ~${cap / 60 * price:.2f} of shared credit")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    raise SystemExit((run_h100 if gpu == "h100" else run_l4).remote(model, n_gen, ab))
