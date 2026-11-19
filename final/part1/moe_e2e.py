"""Part 1: whole-model check of Qwen3-30B-A3B in our engine against HF Transformers, in two steps so that the slow
CPU reference does not keep a GPU waiting:

  modal run part1/moe_e2e.py --ref --yes            # CPU only: HF bf16 reference -> volume final-results:/moe_e2e_ref
  modal run part1/moe_e2e.py --ref --yes --model Qwen/Qwen3-0.6B   # the same on the small model (to test this script)
  modal run part1/moe_e2e.py --yes                  # 1x H100: engine vs the stored reference (+ dense 0.6B regression)
  modal run part1/moe_e2e.py --ref --yes --prompts heldout          # a prompt the MoE rule was NOT written from
  modal run part1/moe_e2e.py --yes --prompts tiling,heldout

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
                  "-o build/engine_test -lnccl && test -x build/engine_test")
)
PROMPTS = {   # "tiling" is the prompt the MoE rule was written from; "heldout" checks the frozen rule on new data
    "tiling": ("Explain in two sentences why tiling a matrix multiplication in GPU shared memory makes it faster. "
               "Then name one limit of the technique."),
    "heldout": ("Rivers shape valleys over thousands of years. Describe in a few sentences how erosion and deposition "
                "work together, and give one way people use rivers today."),
}
N_GEN = 16
REF_DIR = "/results/moe_e2e_ref"
RUN_DIR = "/results/moe_e2e_runs"
PRICE_H100 = 3.95
LEGACY_REFS = {("Qwen/Qwen3-30B-A3B", "tiling"), ("Qwen/Qwen3-0.6B", "tiling")}   # made before meta had "snapshot"
REVISIONS = {"Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",     # README: the weights in use
             "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca"}


def snapshot(model):
    """The pinned snapshot directory (README revision), not whatever happens to be first in the cache."""
    root = f"/hf_cache/hub/models--{model.replace('/', '--')}"
    rev = REVISIONS.get(model) or open(f"{root}/refs/main").read().strip()
    path = f"{root}/snapshots/{rev}"
    assert os.path.isdir(path), f"snapshot {rev} of {model} is not in the cache"
    return path


def ref_dir(model, prompt_id):
    return f"{REF_DIR}/{model.split('/')[-1]}" + ("" if prompt_id == "tiling" else f"_{prompt_id}")


@app.function(image=image, cpu=16, memory=128 * 1024, volumes={"/hf_cache": hf_cache.read_only(), "/results": results},
              timeout=60 * 60)   # CPU only; hard cap 60 min
def reference(model: str, prompt_id: str = "tiling"):
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
    ids = tok(PROMPTS[prompt_id], return_tensors="pt")["input_ids"]
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
    d = ref_dir(model, prompt_id)
    os.makedirs(d, exist_ok=True)
    open(f"{d}/tokens.bin", "wb").write(full.to(torch.int32).numpy().tobytes())
    open(f"{d}/logits.f32", "wb").write(logits.numpy().tobytes())
    if route_idx:   # [layers, tokens, K] int32 and [layers, tokens, E] fp32 (HF's bf16 logits, widened)
        L = sorted(route_idx)
        open(f"{d}/route_idx.i32", "wb").write(torch.stack([route_idx[l] for l in L]).numpy().tobytes())
        open(f"{d}/route_logits.f32", "wb").write(torch.stack([route_logits[l] for l in L]).numpy().tobytes())
        print(f"router choices of {len(L)} layers recorded", flush=True)
    json.dump({"model": model, "prompt_id": prompt_id, "snapshot": os.path.basename(snap), "n_prompt": int(ids.shape[1]), "n_total": int(full.shape[0]), "keep": keep,
               "vocab": int(logits.shape[1]), "transformers": transformers.__version__, "torch": torch.__version__,
               "text": tok.decode(full[ids.shape[1]:])}, open(f"{d}/meta.json", "w"))
    results.commit()
    print(f"reference saved to final-results:{d} ({time.time() - t0:.0f} s in all)")


def read_dump(path, K):
    """Engine routing dump (QWEN3_MOE_DUMP): {(layer, position): list of K experts}, first occurrence only (the
    teacher-forced pass of engine_test, a single sequence; its later generating pass is ignored). Single-sequence use
    only: the key has no sequence id."""
    import numpy as np
    raw = np.fromfile(path, np.int32)
    out, i = {}, 0
    while i < raw.size:
        layer, n = int(raw[i]), int(raw[i + 1]); i += 2
        pos = raw[i:i + n]; i += n
        idx = raw[i:i + n * K].reshape(n, K); i += n * K
        for p_, row in zip(pos, idx):
            out.setdefault((layer, int(p_)), row.tolist())
    return out


def bf16_steps(a, b):
    """Number of representable bf16 values between a and b (both bf16 numbers), via their ordered bit patterns."""
    import numpy as np
    def key(v):
        u = int(np.float32(v).view(np.uint32)) >> 16
        return -(u & 0x7FFF) if u & 0x8000 else u
    return abs(key(a) - key(b))


def routing_analysis(hidx, hlog, dump, E):
    """Compare the engine's expert choices with HF's. hidx [L, T, K], hlog [L, T, E] (HF's bf16 router logits).
    Returns a dict: coverage problems, every difference with its bf16 step distance (max over the swapped experts'
    HF logits), the primary differences (no difference at a lower layer at the same or an earlier position, i.e. not
    within reach of an earlier one through causal attention), and the primary ones that are not within 2 steps."""
    L, T, K = hidx.shape
    missing = [(l, t) for l in range(L) for t in range(T) if (l, t) not in dump]
    invalid = [(l, t) for (l, t), row in dump.items()
               if l < L and t < T and (len(set(row)) != K or not all(0 <= e < E for e in row))]
    flips = []
    for l in range(L):
        for t in range(T):
            row = dump.get((l, t))
            if row is None or (l, t) in invalid:
                continue
            g, h = set(row), set(hidx[l, t].tolist())
            if g != h:
                sw = sorted(g ^ h)
                v = [float(hlog[l, t, e]) for e in sw]
                steps = max(bf16_steps(a, b) for a in v for b in v)
                flips.append({"layer": l, "pos": t, "engine": sorted(g - h), "hf": sorted(h - g), "steps": steps})
    primary = [f for f in flips if not any(g["layer"] < f["layer"] and g["pos"] <= f["pos"] for g in flips)]
    return {"pairs": L * T, "missing": missing, "invalid": invalid, "flips": flips, "primary": primary,
            "primary_far": [f for f in primary if f["steps"] > 2]}


def verdict(ref, mine, n_prompt, first_keep, gen, exp, routing=None):
    """The pass decision, as a pure function of the data (tested locally with synthetic cases).
    Always: every top-1 mismatch must be a near tie FOR THE TOKEN THE ENGINE CHOSE (HF's best score minus HF's score
    of the engine's token < 0.5, not HF's own top-2 gap).
    Dense (routing None): additionally min cosine > 0.999 (as e2e_test.py).
    MoE (routing given): additionally greedy identical, mean cosine > 0.999, full routing coverage (every (layer,
    position) dumped, K distinct in-range experts), every primary routing difference within 2 bf16 steps of HF's
    logits, and every position with cosine <= 0.999 within reach of a routing difference (one at that position or
    an earlier one). The minimum cosine is reported, not required. Returns (ok, lines)."""
    import numpy as np
    cos = (ref * mine).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(mine, axis=1))
    choice = mine.argmax(1)
    mism = np.nonzero(ref.argmax(1) != choice)[0]
    gaps = {int(i): float(ref[i].max() - ref[i, choice[i]]) for i in mism}
    ranks = {int(i): int((ref[i] > ref[i, choice[i]]).sum()) + 1 for i in mism}
    top1_ok = all(g < 0.5 for g in gaps.values())
    same = next((i for i in range(len(exp)) if gen[i] != exp[i]), len(exp))
    b = n_prompt - first_keep
    lines = [f"{len(cos)} positions ({b} prefill, {len(cos) - b} KV-cache decode); top-1 {len(cos) - len(mism)}/{len(cos)}"
             + "".join(f"; mismatch at {first_keep + i}: engine's token is HF's #{ranks[i]}, {gaps[i]:.3f} below HF's best"
                       for i in gaps),
             f"min cosine prefill {cos[:b].min():.6f}, decode {cos[b:].min() if len(cos) > b else 1:.6f}, mean {cos.mean():.6f}",
             f"greedy: engine matches HF for {same}/{len(exp)} tokens"]
    if routing is None:
        ok = top1_ok and cos.min() > 0.999
        return ok, lines + ["PASS (dense rule)" if ok else "FAIL (dense rule: top-1 / min cosine)"]
    r = routing
    cover_ok = not r["missing"] and not r["invalid"]
    lines.append(f"routing: {r['pairs']} (layer, position) pairs, {len(r['missing'])} missing, {len(r['invalid'])} invalid; "
                 f"{len(r['flips'])} with a different expert set")
    exact = sum(1 for f in r["primary"] if f["steps"] == 0)
    lines.append(f"primary differences: {len(r['primary'])}: {exact} exact ties (0 steps), "
                 f"{len(r['primary']) - exact - len(r['primary_far'])} near the top-8 boundary (1-2 steps), "
                 f"{len(r['primary_far'])} further apart")
    for f in r["primary"]:
        lines.append(f"  primary: layer {f['layer']} position {f['pos']}: engine {f['engine']} vs HF {f['hf']}, "
                     f"{f['steps']} bf16 steps apart in HF's logits")
    low = [i for i in np.nonzero(cos <= 0.999)[0]]
    unexplained = [first_keep + i for i in low if not any(f["pos"] <= first_keep + i for f in r["flips"])]
    for i in low:
        lines.append(f"  cosine {cos[i]:.5f} at position {first_keep + i}: "
                     + ("NO routing difference at or before it" if first_keep + i in unexplained
                        else "within reach of a routing difference (at or before this position)"))
    checks = {"top-1": top1_ok, "greedy": same == len(exp), "mean cosine": cos.mean() > 0.999, "coverage": cover_ok,
              "primary within 2 steps": not r["primary_far"], "low cosine reachable": not unexplained}
    ok = all(checks.values())
    lines.append(("PASS" if ok else "FAIL") + " (MoE rule: " + ", ".join(f"{k} {'ok' if v else 'NO'}" for k, v in checks.items())
                 + f"; min cosine {cos.min():.5f} reported)")
    return ok, lines


def compare(model, prompt_id="tiling"):
    """Engine (tests/engine_test.cu) on the stored reference tokens; saves the engine's logits and routing; 0 = pass."""
    import shutil
    import subprocess
    import time
    import numpy as np
    d = ref_dir(model, prompt_id)
    meta = json.load(open(f"{d}/meta.json"))
    snap = snapshot(model)
    if meta.get("snapshot") is None:
        # references made before the snapshot was recorded; the cache held only this one snapshot of each model then
        if (model, prompt_id) not in LEGACY_REFS:
            print("reference has no snapshot recorded: regenerate it")
            return 1
        print(f"legacy reference (snapshot not recorded; the cache held only {os.path.basename(snap)[:8]} then)")
    elif meta["snapshot"] != os.path.basename(snap):
        print(f"reference made with snapshot {meta['snapshot']}, engine would use {os.path.basename(snap)}: stop")
        return 1
    n_prompt, n_total, keep, V = meta["n_prompt"], meta["n_total"], meta["keep"], meta["vocab"]
    ref = np.fromfile(f"{d}/logits.f32", np.float32).reshape(keep, V)
    toks = np.fromfile(f"{d}/tokens.bin", np.int32)
    cfg = json.load(open(f"{snap}/config.json"))
    moe = cfg.get("num_experts", 0) > 0
    if moe and not os.path.exists(f"{d}/route_idx.i32"):
        print("MoE model but the reference has no routing record: regenerate it")
        return 1
    env = dict(os.environ)
    if moe:
        env["QWEN3_MOE_DUMP"] = "/tmp/route.bin"
    r = subprocess.run(["/root/final/build/engine_test", snap, f"{d}/tokens.bin", str(n_total), str(n_prompt),
                        "/tmp/logits.bin", "/tmp/rope.bin", "/tmp/gen.bin", str(n_total - n_prompt)],
                       capture_output=True, text=True, env=env)
    print(r.stdout.strip())
    if r.returncode:
        print("engine_test failed:", r.stderr[-2000:])
        return 1
    out = f"{RUN_DIR}/{os.path.basename(d)}_{int(time.time())}"      # keep the engine's side for later analysis
    os.makedirs(out, exist_ok=True)
    for f in ("logits.bin", "gen.bin") + (("route.bin",) if moe else ()):
        shutil.copy(f"/tmp/{f}", out)
    mine = (np.fromfile("/tmp/logits.bin", np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(keep, V)
    routing = None
    if moe:
        # the expected shape comes from the model's config, not from the reference file's length: a reference that
        # lost a layer would otherwise shrink the coverage check with it
        T, L, E, K = n_total, cfg["num_hidden_layers"], cfg["num_experts"], cfg["num_experts_per_tok"]
        hidx = np.fromfile(f"{d}/route_idx.i32", np.int32)
        hlog = np.fromfile(f"{d}/route_logits.f32", np.float32)
        if hidx.size != L * T * K or hlog.size != L * T * E:
            print(f"reference routing has {hidx.size} / {hlog.size} values, config wants [{L},{T},{K}] / [{L},{T},{E}]")
            return 1
        routing = routing_analysis(hidx.reshape(L, T, K), hlog.reshape(L, T, E), read_dump("/tmp/route.bin", K), E)
    ok, lines = verdict(ref, mine, n_prompt, n_total - keep, np.fromfile("/tmp/gen.bin", np.int32), toks[n_prompt:], routing)
    print(f"{model} [{prompt_id}, snapshot {os.path.basename(snap)[:8]}], engine side saved to final-results:{out}")
    for line in lines:
        print("  " + line)
    results.commit()
    return 0 if ok else 1


@app.function(image=image, gpu="H100", volumes={"/hf_cache": hf_cache.read_only(), "/results": results},
              timeout=20 * 60)   # hard cap 20 min
def engine(jobs: list) -> int:
    import subprocess
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    rc = 0
    for m, pid in jobs:
        rc |= compare(m, pid)
    return rc


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-30B-A3B", ref: bool = False, prompts: str = "tiling", dense_check: bool = True,
         yes: bool = False):
    pids = prompts.split(",")
    if ref:
        print(f"PLAN: CPU only (16 cores, 128 GiB, no GPU): HF reference for {model}, prompts {pids} "
              f"+ {N_GEN} greedy tokens each, hard cap 60 min per prompt")
        if yes:
            for pid in pids:
                reference.remote(model, pid)
        else:
            print("Dry run only. Pass --yes to start.")
        return
    models = [(model, pid) for pid in pids] + ([("Qwen/Qwen3-0.6B", "tiling")] if dense_check and model != "Qwen/Qwen3-0.6B" else [])
    print(f"PLAN: 1x H100, engine vs the stored references of {models}, hard cap 20 min -> at most "
          f"~${20 / 60 * PRICE_H100:.2f} of shared credit")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    raise SystemExit(engine.remote(models))
