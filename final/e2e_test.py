"""End-to-end check of the Tile inference engine against HF Transformers (CPU, bf16) on a real model.

  modal run e2e_test.py                         # prints the plan only
  modal run e2e_test.py --yes                   # Qwen3-0.6B on 1x L4, hard cap 25 min
  modal run e2e_test.py --yes --model Qwen/Qwen3-4B
  modal run e2e_test.py --yes --model Qwen/Qwen3-4B --gpu h100     # big tiles + split-P attention
  modal run e2e_test.py --yes --model Qwen/Qwen3-0.6B --gpu h100 --long 8192 --yarn 4 --ab   # 8K prompt, YaRN factor 4

What is checked:
  * teacher forcing: prefill the prompt in one chunk, then feed HF's own continuation one token at a time
    (this exercises the KV-cache decode path); logits on the last min(total_tokens, 64) positions vs HF: top-1 agreement, cosine, max error
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

NEAR_TIE = 0.5      # logits: a top-1 or greedy difference passes only if HF scores the engine's token < 0.5 below its best
MIN_COS = 0.999


def dense_verdict(ref, mine, gen, exp, gen_row0, rope_bad):
    """The dense pass decision as a pure function of the data (numpy; tested locally by tools/test_verdicts.py).
    ref, mine: [rows, vocab] logits of HF and the engine on the same teacher-forced positions; gen, exp: the engine's
    and HF's greedy tokens; gen_row0: the row of ref whose logits predict exp[0]; rope_bad: differing RoPE table entries.
    Pass needs all of:
      * every top-1 mismatch a near tie FOR THE ENGINE'S TOKEN: HF's best minus HF's score of the engine's choice < 0.5
        (HF's own top-2 gap is not enough: HF tied on A and B, engine choosing a far lower C must fail)
      * min cosine over all rows > 0.999
      * greedy tokens identical, or the first difference at a near tie for the engine's token (same 0.5, on HF's
        logits at that position); after a difference the two continuations have different prefixes, so later tokens
        are not compared
      * RoPE tables bit-identical
    The greedy output must have exactly len(exp) tokens, each a valid id in [0, vocab): checked before anything is
    indexed with it (numpy would read id -2 as the second-to-last column).
    Returns (ok, lines)."""
    import numpy as np
    gen, exp = np.asarray(gen), np.asarray(exp)
    rows = np.arange(len(ref))
    choice = mine.argmax(1)
    best = ref.max(1)
    gap = best - ref[rows, choice]
    mism = np.nonzero(ref.argmax(1) != choice)[0]
    top1_ok = bool(np.all(gap[mism] < NEAR_TIE))
    cos = (ref * mine).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(mine, axis=1))
    cos_ok = bool(cos.min() > MIN_COS)
    V = ref.shape[1]
    first = None
    if len(gen) != len(exp):
        greedy_ok, why = False, f"engine produced {len(gen)} tokens, HF {len(exp)}"
    elif len(gen) and (gen.min() < 0 or gen.max() >= V):
        greedy_ok, why = False, f"engine produced an id outside [0, {V}): {int(gen.min())}..{int(gen.max())}"
    else:
        first = next((i for i in range(len(exp)) if gen[i] != exp[i]), None)
        if first is None:
            greedy_ok, why = True, f"{len(exp)}/{len(exp)} identical"
        else:
            r = gen_row0 + first
            if 0 <= r < len(ref):
                g = float(best[r] - ref[r, gen[first]])
                greedy_ok, why = g < NEAR_TIE, (f"first difference at token {first}: the engine's token {int(gen[first])} "
                                                f"is {g:.3f} below HF's best there; teacher-forced, the engine chose "
                                                f"{int(choice[r])}")
            else:
                greedy_ok, why = False, f"first difference at token {first}, outside the compared rows"
    rope_ok = rope_bad == 0
    lines = [f"  top-1 {len(ref) - len(mism)}/{len(ref)}, every mismatch's engine token < {NEAR_TIE} below HF's best: "
             f"{'yes' if top1_ok else 'NO'}" + "".join(f"; row {int(i)}: {gap[i]:.3f}" for i in mism),
             f"  min cosine {cos.min():.6f} (> {MIN_COS}: {'yes' if cos_ok else 'NO'})",
             f"  greedy: {why} ({'ok' if greedy_ok else 'FAIL'})",
             f"  RoPE tables: {rope_bad} differing entries ({'ok' if rope_ok else 'FAIL'})"]
    return top1_ok and cos_ok and greedy_ok and rope_ok, lines


def judge_variants(ref, variants, exp, gen_row0):
    """variants: (tag, engine logits, engine greedy tokens, differing RoPE entries of THAT run). Pass only if every
    configuration passes every check. Returns (ok, lines)."""
    ok, lines = True, []
    for tag, mine, gen, rope_bad in variants:
        v_ok, v_lines = dense_verdict(ref, mine, gen, exp, gen_row0, rope_bad)
        lines += [f"verdict, {tag}: {'PASS' if v_ok else 'FAIL'}"] + v_lines
        ok &= v_ok
    return ok, lines


# the revisions the reported results used (as PINNED in modal_aipref.py); a model not listed uses refs/main
PINNED = {"Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
          "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c"}


def pinned_snapshot(hub: str, model: str, pinned=PINNED) -> str:
    """The snapshot directory of `model` in the HF cache `hub`: the pinned revision if there is one (error if it is
    not cached), else the one refs/main names. Never "the first directory found": a cache may hold several."""
    base = os.path.join(hub, "models--" + model.replace("/", "--"))
    rev = pinned.get(model)
    if rev is None:
        ref = os.path.join(base, "refs", "main")
        if not os.path.exists(ref):
            raise SystemExit(f"{model}: no refs/main in {base}")
        rev = open(ref).read().strip()
    snap = os.path.join(base, "snapshots", rev)
    if not os.path.isdir(snap):
        raise SystemExit(f"{model}: revision {rev} is not in the cache ({base}/snapshots)")
    return snap


def output_problems(expected):
    """expected: {path: byte count}. The problems with the files a run should have written (missing, wrong size), so
    a run that wrote nothing or too little is a clear FAIL naming the file, not an exception further on."""
    out = []
    for path, nbytes in expected.items():
        if not os.path.exists(path):
            out.append(f"{path} not written")
        elif os.path.getsize(path) != nbytes:
            out.append(f"{path}: {os.path.getsize(path)} bytes, expected {nbytes}")
    return out


MAX_GEN = 63   # engine_test keeps the last 64 logits rows; the first greedy token is predicted by the prompt's last row


def _impl(model_name: str, n_gen: int, arch: str, ab: bool = False, long_n: int = 0, yarn: float = 0.0):
    import glob
    import re
    import subprocess

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.chdir("/root/final")
    r = subprocess.run(f"mkdir -p build && nvcc --enable-tile -std=c++20 -arch={arch} -O2 tests/engine_test.cu -o build/engine_test -lnccl",
                       shell=True, capture_output=True, text=True)
    print("compile exit code:", r.returncode, "arch", arch)
    if r.returncode:
        print(r.stderr[-3000:])
        return 1

    snap = pinned_snapshot("/hf_cache/hub", model_name)
    print(f"snapshot {snap}")
    tok = AutoTokenizer.from_pretrained(snap)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(snap)
    if yarn > 0:   # same override the engine gets through QWEN3_YARN: static YaRN, original context 32768
        rp = dict(getattr(cfg, "rope_parameters", None) or {})
        rp.update({"rope_type": "yarn", "factor": yarn, "original_max_position_embeddings": 32768})
        rp.setdefault("rope_theta", getattr(cfg, "rope_theta", 1000000.0))
        try:
            cfg.rope_parameters = rp
        except Exception:
            cfg.rope_scaling = rp
    try:
        model = AutoModelForCausalLM.from_pretrained(snap, config=cfg, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(snap, config=cfg, torch_dtype=torch.bfloat16)
    model.eval()

    if long_n:   # deterministic pseudo-text: random common words, truncated to exactly long_n tokens
        import random
        words = ("the of and to in is that for it as was with be by on not he this are or his from at which but have an had "
                 "they you were their one all we can her has there been if more when will would who so no she other its may "
                 "these two like him into time has look more write go see number way could people my than first water been "
                 "call who oil now find long down day did get come made may part matrix memory tile kernel thread block warp "
                 "register cache bandwidth latency throughput tensor attention softmax gradient token model layer").split()
        rng = random.Random(0)
        text = " ".join(rng.choice(words) for _ in range(long_n))
        prompt_ids = tok(text, return_tensors="pt")["input_ids"][:, :long_n]
    else:
        prompt_ids = tok(PROMPT, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        full = model.generate(prompt_ids, max_new_tokens=n_gen, do_sample=False)
    full = full[0]
    n_prompt, n_total = prompt_ids.shape[1], full.shape[0]
    n_gen_real = n_total - n_prompt
    print(f"model {model_name}: prompt {n_prompt} tokens, HF generated {n_gen_real}, YaRN {yarn or 'off'}")
    print("HF text:", repr(tok.decode(full[n_prompt:])))
    keep = min(n_total, 64)           # the engine writes only the last `keep` logits rows (see engine_test.cu)
    first_keep = n_total - keep
    with torch.no_grad():
        ref = model(full[None], logits_to_keep=keep).logits[0].float()      # [keep, vocab]
    open("/tmp/tokens.bin", "wb").write(full.to(torch.int32).numpy().tobytes())

    V = ref.shape[1]
    top_ref = ref.argmax(-1)
    top2 = torch.topk(ref, 2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    scale = ref.abs().max().item()
    exp = full[n_prompt:].numpy()
    variants = [("default (big tiles; decode = split-KV kernel on Hopper)", {})]
    if ab:   # same HF reference, several engine configurations: isolates what changed the numbers
        variants += [("big GEMM tiles + SMALL attention (QWEN3_ATTN=small)", {"QWEN3_ATTN": "small"}),
                     ("all small tiles (QWEN3_TILES=small): the configuration verified on the L4", {"QWEN3_TILES": "small"})]
    rot = model.model.rotary_emb
    P = n_total + n_gen_real + 8
    cos_hf, sin_hf = rot(torch.zeros(1, P, model.config.head_dim, dtype=torch.bfloat16), torch.arange(P)[None])
    hf_c = cos_hf[0, :, :64].contiguous().view(torch.int16).numpy().view(np.uint16)
    hf_s = sin_hf[0, :, :64].contiguous().view(torch.int16).numpy().view(np.uint16)
    summary = []
    for vi, (tag, extra) in enumerate(variants):
        for f in ("/tmp/logits.bin", "/tmp/rope.bin", "/tmp/gen.bin"):   # a run that writes nothing must not reuse the
            if os.path.exists(f):                                        # previous variant's files
                os.remove(f)
        print(f"--- running engine configuration: {tag}", flush=True)
        env = dict(os.environ)
        env.update(extra)
        if yarn > 0:
            env["QWEN3_YARN"] = str(yarn)
        r = subprocess.run(["build/engine_test", snap, "/tmp/tokens.bin", str(n_total), str(n_prompt), "/tmp/logits.bin",
                            "/tmp/rope.bin", "/tmp/gen.bin", str(n_gen_real)], capture_output=True, text=True, env=env)
        if vi == 0:
            print(r.stdout)
        if r.returncode:
            print("engine_test failed:", r.stderr[-2000:])
            return 1
        bad_files = output_problems({"/tmp/logits.bin": keep * V * 2, "/tmp/rope.bin": 2 * P * 64 * 2,
                                     "/tmp/gen.bin": n_gen_real * 4})
        if bad_files:
            print(f"FAIL: engine configuration {tag}: " + "; ".join(bad_files))
            return 1
        raw = np.frombuffer(open("/tmp/logits.bin", "rb").read(), dtype=np.uint16).astype(np.uint32) << 16
        mine = torch.from_numpy(raw.view(np.float32).reshape(keep, V).copy())
        cos = torch.nn.functional.cosine_similarity(ref, mine, dim=-1)
        maxabs = (ref - mine).abs().max(-1).values
        mism = (top_ref != mine.argmax(-1)).nonzero().flatten().tolist()
        b = n_prompt - first_keep      # rows before b are prefill positions, rows from b on are KV-cache decode positions
        pre_cos = cos[:b].min().item() if b > 0 else 1.0
        dec_cos = cos[b:].min().item() if n_total > n_prompt else 1.0
        gen = np.frombuffer(open("/tmp/gen.bin", "rb").read(), dtype=np.int32)
        first_bad = next((i for i in range(min(len(gen), len(exp))) if gen[i] != exp[i]), None)
        rt = np.frombuffer(open("/tmp/rope.bin", "rb").read(), dtype=np.uint16).reshape(2, P, 64)
        bad_c, bad_s = int((rt[0] != hf_c).sum()), int((rt[1] != hf_s).sum())
        print(f"  RoPE tables (positions 0..{P - 1}): differing bf16 entries cos {bad_c}/{hf_c.size}, sin {bad_s}/{hf_s.size}")
        print(f"--- engine configuration: {tag}")
        print(f"compared positions: {keep} (the last ones: {first_keep}..{n_total - 1}; prefill path up to {n_prompt - 1} of a "
              f"{n_prompt}-token prompt, KV-cache decode path {n_prompt}..{n_total - 1})")
        print(f"  top-1 agreement: {keep - len(mism)}/{keep}")
        for i in mism:
            print(f"    mismatch at position {first_keep + i} ({'prefill' if first_keep + i < n_prompt else 'decode'}): HF margin top1-top2 = {margin[i]:.3f}")
        worst = int(torch.argmin(cos))
        print(f"  cosine similarity per position: min {cos.min():.6f} (position {first_keep + worst}, HF margin {margin[worst]:.3f}), mean {cos.mean():.6f}")
        print(f"  max abs logit error {maxabs.max():.3f}  (largest |logit| {scale:.1f}; relative {maxabs.max() / scale:.2e})")
        print(f"  min cosine: prefill {pre_cos:.6f}  decode {dec_cos:.6f}")
        print(f"  greedy generation: {len(gen)} tokens; matches HF for {min(len(gen), len(exp)) if first_bad is None else first_bad}/{len(exp)}"
              + ("" if first_bad is None else f" (first difference at token {first_bad})"))
        if vi == 0 and len(gen) and 0 <= gen.min() and gen.max() < V:   # decode only valid ids (the verdict says why not)
            print("  engine text:", repr(tok.decode(gen.tolist())))
        summary.append((tag, keep - len(mism), pre_cos, dec_cos, mine.numpy(), gen.copy(), bad_c + bad_s))
    if ab:
        print("\nA/B summary (same HF reference): configuration | top-1 | min cosine prefill | min cosine decode")
        for tag, ok_n, pc, dc, *_ in summary:
            print(f"  {tag[:62]:<62} {ok_n}/{keep}  {pc:.6f}  {dc:.6f}")

    # verdict: every engine configuration must pass every check, with its own RoPE tables (see judge_variants)
    ok, lines = judge_variants(ref.numpy(), [(t[0], t[4], t[5], t[6]) for t in summary], exp, n_prompt - 1 - first_keep)
    print("\n" + "\n".join(lines))
    print("\nPASS (all configurations)" if ok else "\nFAIL")
    return 0 if ok else 1


@app.function(image=image, gpu="L4", volumes={"/hf_cache": hf_cache}, timeout=2400)
def run_l4(model_name: str, n_gen: int, ab: bool = False, long_n: int = 0, yarn: float = 0.0):
    return _impl(model_name, n_gen, "sm_80", ab, long_n, yarn)


@app.function(image=image, gpu="H100", volumes={"/hf_cache": hf_cache}, timeout=900)   # hard cap 15 min (~$1)
def run_h100(model_name: str, n_gen: int, ab: bool = False, long_n: int = 0, yarn: float = 0.0):
    return _impl(model_name, n_gen, "sm_90", ab, long_n, yarn)


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-0.6B", n_gen: int = 16, gpu: str = "l4", ab: bool = False, long: int = 0, yarn: float = 0.0, yes: bool = False):
    price, cap = (3.95, 15) if gpu == "h100" else (PRICE_L4, 40)
    print(f"PLAN: {model} from volume qwen3-hf-cache (read-only, no download), HF on CPU as reference, engine on 1x {gpu.upper()}, "
          f"prompt {long or 27} tokens, YaRN={yarn or 'off'}, hard cap {cap} min -> at most ~${cap / 60 * price:.2f} of shared credit")
    if not 1 <= n_gen <= MAX_GEN:
        raise SystemExit(f"--n-gen must be 1..{MAX_GEN}: the check keeps the last 64 logits rows and needs the row of "
                         f"every greedy prediction")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    raise SystemExit((run_h100 if gpu == "h100" else run_l4).remote(model, n_gen, ab, long, yarn))
