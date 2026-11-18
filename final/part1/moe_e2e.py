"""Part 1: whole-model check of Qwen3-30B-A3B in our engine against HF Transformers, in two steps so that the slow
CPU reference does not keep a GPU waiting:

  modal run part1/moe_e2e.py --ref --yes            # CPU only: HF bf16 reference -> volume final-results:/moe_e2e_ref
  modal run part1/moe_e2e.py --ref --yes --model Qwen/Qwen3-0.6B   # the same on the small model (to test this script)
  modal run part1/moe_e2e.py --yes                  # 1x H100: engine vs the stored reference (+ dense 0.6B regression)

Same checks as e2e_test.py: teacher forcing (prefill the prompt in one chunk, then HF's own continuation one token at
a time through the KV cache), logits of every position vs HF (top-1, cosine, max error), and greedy generation.
Pass: every top-1 mismatch is a near tie in HF's logits (margin < 0.5) and the minimum cosine is > 0.999. For an MoE
model the per-layer expert choices are also compared with HF's (router hooks / QWEN3_MOE_DUMP): if every primary
routing difference (one that no earlier difference can explain) is a tie in HF's bf16 router logits, the minimum
cosine is reported but not required, as long as the mean cosine is > 0.999 and the greedy tokens are identical.
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # final/
app = modal.App("final-moe-e2e")
hf_cache = modal.Volume.from_name("qwen3-hf-cache", create_if_missing=False)
results = modal.Volume.from_name("final-results", create_if_missing=False)   # ours
image = (
    modal.Image.from_registry("nvidia/cuda:13.5.0-devel-ubuntu24.04", add_python="3.12")
    .pip_install("numpy", "transformers", "torch", extra_index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(HERE, "/root/final", ignore=[".venv", "__pycache__", "*.pyc", "build", "test_cases*.db", "results"], copy=True)
    .run_commands("cd /root/final && mkdir -p build && nvcc --enable-tile -std=c++20 -arch=sm_90 -O2 tests/engine_test.cu "
                  "-o build/engine_test && test -x build/engine_test")
)
PROMPT = ("Explain in two sentences why tiling a matrix multiplication in GPU shared memory makes it faster. "
          "Then name one limit of the technique.")
N_GEN = 16
REF_DIR = "/results/moe_e2e_ref"
PRICE_H100 = 3.95


def snapshot(model):
    import glob
    return glob.glob(f"/hf_cache/hub/models--{model.replace('/', '--')}/snapshots/*")[0]


@app.function(image=image, cpu=16, memory=128 * 1024, volumes={"/hf_cache": hf_cache.read_only(), "/results": results},
              timeout=60 * 60)   # CPU only; hard cap 60 min
def reference(model: str):
    import time
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(16)
    t0 = time.time()
    snap = snapshot(model)
    tok = AutoTokenizer.from_pretrained(snap)
    m = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).eval()
    print(f"loaded {model} in {time.time() - t0:.0f} s (torch {torch.__version__}, transformers {transformers.__version__})", flush=True)
    ids = tok(PROMPT, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        full = m.generate(ids, max_new_tokens=N_GEN, do_sample=False)[0]
        print(f"generated in {time.time() - t0:.0f} s: {tok.decode(full[ids.shape[1]:])!r}", flush=True)
        keep = min(full.shape[0], 64)
        # MoE: record every layer's router logits and chosen experts during this teacher-forced pass (hooks on the
        # Qwen3MoeTopKRouter modules, whose output is (router_logits, router_scores, router_indices))
        route_logits, route_idx, hooks = {}, {}, []
        for li, layer in enumerate(m.model.layers):
            gate = getattr(layer.mlp, "gate", None)
            if gate is None or not hasattr(gate, "top_k"):
                continue
            def hook(mod, inp, out, li=li):
                route_logits[li] = out[0].float().reshape(-1, out[0].shape[-1]).clone()
                route_idx[li] = out[2].reshape(-1, out[2].shape[-1]).to(torch.int32).clone()
            hooks.append(gate.register_forward_hook(hook))
        logits = m(full[None], logits_to_keep=keep).logits[0].float()
        for h in hooks:
            h.remove()
    d = f"{REF_DIR}/{model.split('/')[-1]}"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/tokens.bin", "wb").write(full.to(torch.int32).numpy().tobytes())
    open(f"{d}/logits.f32", "wb").write(logits.numpy().tobytes())
    if route_idx:   # [layers, tokens, K] int32 and [layers, tokens, E] fp32 (HF's bf16 logits, widened)
        L = sorted(route_idx)
        open(f"{d}/route_idx.i32", "wb").write(torch.stack([route_idx[l] for l in L]).numpy().tobytes())
        open(f"{d}/route_logits.f32", "wb").write(torch.stack([route_logits[l] for l in L]).numpy().tobytes())
        print(f"router choices of {len(L)} layers recorded", flush=True)
    json.dump({"model": model, "n_prompt": int(ids.shape[1]), "n_total": int(full.shape[0]), "keep": keep,
               "vocab": int(logits.shape[1]), "transformers": transformers.__version__, "torch": torch.__version__,
               "text": tok.decode(full[ids.shape[1]:])}, open(f"{d}/meta.json", "w"))
    results.commit()
    print(f"reference saved to final-results:{d} ({time.time() - t0:.0f} s in all)")


def read_dump(path, K):
    """Engine routing dump (QWEN3_MOE_DUMP): {(layer, position): set of experts}, first occurrence only (the
    teacher-forced pass; engine_test runs a second, generating pass afterwards)."""
    import numpy as np
    raw = np.fromfile(path, np.int32)
    out, i = {}, 0
    while i < raw.size:
        layer, n = int(raw[i]), int(raw[i + 1]); i += 2
        pos = raw[i:i + n]; i += n
        idx = raw[i:i + n * K].reshape(n, K); i += n * K
        for p_, row in zip(pos, idx):
            out.setdefault((layer, int(p_)), set(row.tolist()))
    return out


def routing_report(d, dump, cos, first_keep):
    """Where do the engine's expert choices differ from HF's, and how close to a tie was HF there? Returns lines."""
    import numpy as np
    hidx = np.fromfile(f"{d}/route_idx.i32", np.int32)
    hlog = np.fromfile(f"{d}/route_logits.f32", np.float32)
    meta = json.load(open(f"{d}/meta.json"))
    T, K = meta["n_total"], 8
    L = hidx.size // (T * K)
    E = hlog.size // (L * T)
    hidx, hlog = hidx.reshape(L, T, K), hlog.reshape(L, T, E)
    ulp = lambda v: 2.0 ** (np.floor(np.log2(max(abs(v), 1e-30))) - 7)
    flips = []                                   # (layer, position, swapped-in, swapped-out, margin in ulps)
    compared = 0
    for (l, t), g in dump.items():
        if l >= L or t >= T:
            continue
        compared += 1
        h = set(hidx[l, t].tolist())
        if g != h:
            sw = sorted(g ^ h)
            v = hlog[l, t, sw]
            flips.append((l, t, sorted(g - h), sorted(h - g), float((v.max() - v.min()) / ulp(v.max()))))
    flips.sort()
    lines = [f"routing: {compared} (layer, position) pairs compared, {len(flips)} with a different expert set"]
    for f in flips[:12]:
        lines.append(f"  layer {f[0]:2d} position {f[1]:2d}: engine {f[2]} vs HF {f[3]}, HF logit spread {f[4]:.1f} bf16 ulps")
    if len(flips) > 12:
        lines.append(f"  ... {len(flips) - 12} more")
    low = [first_keep + i for i in np.nonzero(cos < 0.999)[0]]
    for t in low:
        mine = [f for f in flips if f[1] == t]
        lines.append(f"  low cosine at position {t} ({cos[t - first_keep]:.5f}): " +
                     (f"{len(mine)} layers with other experts, first at layer {mine[0][0]} (spread {mine[0][4]:.1f} ulps)"
                      if mine else "no routing difference at this position"))
    # A difference at (layer l, position t) can be a CONSEQUENCE of an earlier one only if some difference exists at a
    # lower layer at a position <= t (causal attention looks backwards). Every other difference is PRIMARY and must be
    # a tie in HF's own bf16 logits; a primary non-tie would point to a bug in the layers below it.
    primary = [f for f in flips if not any(g[0] < f[0] and g[1] <= f[1] for g in flips)]
    bad = [f for f in primary if f[4] > 2]
    lines.append(f"  primary differences: {len(primary)}, ties (<= 2 ulps): {len(primary) - len(bad)}, NOT ties: {len(bad)}"
                 + "".join(f"; layer {f[0]} position {f[1]} ({f[4]:.1f} ulps)" for f in bad))
    return lines, len(bad)


def compare(model):
    """Engine (tests/engine_test.cu) on the stored reference tokens; returns 0 on pass."""
    import subprocess
    import numpy as np
    d = f"{REF_DIR}/{model.split('/')[-1]}"
    meta = json.load(open(f"{d}/meta.json"))
    n_prompt, n_total, keep, V = meta["n_prompt"], meta["n_total"], meta["keep"], meta["vocab"]
    ref = np.fromfile(f"{d}/logits.f32", np.float32).reshape(keep, V)
    toks = np.fromfile(f"{d}/tokens.bin", np.int32)
    n_gen = n_total - n_prompt
    routed = os.path.exists(f"{d}/route_idx.i32")
    env = dict(os.environ)
    if routed:
        env["QWEN3_MOE_DUMP"] = "/tmp/route.bin"
    r = subprocess.run(["/root/final/build/engine_test", snapshot(model), f"{d}/tokens.bin", str(n_total), str(n_prompt),
                        "/tmp/logits.bin", "/tmp/rope.bin", "/tmp/gen.bin", str(n_gen)], capture_output=True, text=True, env=env)
    print(r.stdout.strip())
    if r.returncode:
        print("engine_test failed:", r.stderr[-2000:])
        return 1
    mine = (np.fromfile("/tmp/logits.bin", np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(keep, V)
    cos = (ref * mine).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(mine, axis=1))
    top2 = np.sort(ref, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    mism = np.nonzero(ref.argmax(1) != mine.argmax(1))[0]
    first_keep = n_total - keep
    b = n_prompt - first_keep
    gen = np.fromfile("/tmp/gen.bin", np.int32)
    exp = toks[n_prompt:]
    same = next((i for i in range(len(exp)) if gen[i] != exp[i]), len(exp))
    print(f"{model}: {keep} positions ({b} prefill, {keep - b} KV-cache decode); top-1 {keep - len(mism)}/{keep}"
          + "".join(f"; mismatch at {first_keep + i} (HF margin {margin[i]:.3f})" for i in mism))
    print(f"  min cosine prefill {cos[:b].min():.6f}, decode {cos[b:].min() if keep > b else 1:.6f}, mean {cos.mean():.6f}; "
          f"max |logit error| {np.abs(ref - mine).max():.3f} (max |logit| {np.abs(ref).max():.1f})")
    print(f"  greedy: engine matches HF for {same}/{len(exp)} tokens (HF text {meta['text']!r})")
    top1_ok = all(margin[i] < 0.5 for i in mism)
    ok = top1_ok and cos.min() > 0.999
    if routed:
        lines, primary_non_ties = routing_report(d, read_dump("/tmp/route.bin", 8), cos, first_keep)
        for line in lines:
            print("  " + line)
        # MoE: bf16 router logits are often exactly equal, and which of two tied experts is taken is arbitrary
        # (torch.topk vs our lowest id); a token that takes the other expert then drifts, and so do later tokens
        # through attention. So for an MoE model the per-position minimum cosine is not required to exceed 0.999 when
        # every routing difference that is not a consequence of an earlier one is such a tie; the mean must still.
        if not ok and top1_ok and primary_non_ties == 0 and cos.mean() > 0.999 and same == len(exp):
            print("  PASS (MoE rule: every primary routing difference is a tie in HF's bf16 router logits; mean cosine "
                  f"{cos.mean():.5f} > 0.999; greedy identical; min cosine {cos.min():.5f} reported)")
            return 0
    print("  PASS" if ok else "  FAIL (top-1 mismatch with a clear margin, or cosine <= 0.999)")
    return 0 if ok else 1


@app.function(image=image, gpu="H100", volumes={"/hf_cache": hf_cache.read_only(), "/results": results.read_only()},
              timeout=20 * 60)   # hard cap 20 min
def engine(models: list) -> int:
    import subprocess
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    rc = 0
    for m in models:
        rc |= compare(m)
    return rc


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-30B-A3B", ref: bool = False, dense_check: bool = True, yes: bool = False):
    if ref:
        print(f"PLAN: CPU only (16 cores, 128 GiB, no GPU): HF reference for {model}, prompt {len(PROMPT.split())} words "
              f"+ {N_GEN} greedy tokens, hard cap 60 min")
        if yes:
            reference.remote(model)
        else:
            print("Dry run only. Pass --yes to start.")
        return
    models = [model] + (["Qwen/Qwen3-0.6B"] if dense_check and model != "Qwen/Qwen3-0.6B" else [])
    print(f"PLAN: 1x H100, engine vs the stored references of {models}, hard cap 20 min -> at most "
          f"~${20 / 60 * PRICE_H100:.2f} of shared credit")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    raise SystemExit(engine.remote(models))
