"""Check the Tile kernels with the REAL Qwen3-4B weights (bf16) against HF modules run on CPU.

  modal run real_weights_test.py            # prints the plan only
  modal run real_weights_test.py --yes      # 1x L4, hard cap 20 min

Weights are read straight from the safetensors files in our own volume `qwen3-hf-cache` (mounted read-only; nothing is
downloaded or written there). Checks: layers 0 and 35 (first and last of 36) RMSNorm + MLP, the q/k head norms,
the embedding table, lm_head (tied to the embedding in Qwen3-4B) with the real 151936 vocabulary, and greedy argmax
over the real logits.
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("final-real-weights")
hf_cache = modal.Volume.from_name("qwen3-hf-cache", create_if_missing=False)
try:
    hf_cache = hf_cache.read_only()
except AttributeError:  # older client: mount normally, the script never writes to it
    pass
IMAGE = "nvidia/cuda:13.5.0-devel-ubuntu24.04"
image = (
    modal.Image.from_registry(IMAGE, add_python="3.12")
    .pip_install("numpy", "safetensors", "transformers", "torch", extra_index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(HERE, "/root/final", ignore=[".venv", "__pycache__", "*.pyc", "build", "test_cases.db"])
)
TESTS = ["rmsnorm", "mlp", "embedding", "linear", "argmax"]
PRICE_L4 = 0.80


@app.function(image=image, gpu="L4", volumes={"/hf_cache": hf_cache}, timeout=1500)
def run():
    import glob
    import json
    import subprocess
    import sys

    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm

    os.chdir("/root/final")
    sys.path.insert(0, "/root/final")
    from run_tests import compare, compare_matmul, load

    build = "mkdir -p build && " + " && ".join(
        f"nvcc --enable-tile -std=c++20 -arch=sm_80 -O2 tests/{t}_test.cu -o build/{t}_test" for t in TESTS)
    r = subprocess.run(build, shell=True, capture_output=True, text=True)
    print("compile exit code:", r.returncode)
    if r.returncode:
        print(r.stderr[-3000:])
        return 1

    snap = glob.glob("/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/*")[0]
    weight_map = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
    cfg_json = json.load(open(f"{snap}/config.json"))
    print("config:", {k: cfg_json[k] for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size",
                                              "tie_word_embeddings", "rms_norm_eps", "torch_dtype") if k in cfg_json})
    HID, INTER, VOCAB = cfg_json["hidden_size"], cfg_json["intermediate_size"], cfg_json["vocab_size"]
    EPS = cfg_json["rms_norm_eps"]
    LAST = cfg_json["num_hidden_layers"] - 1
    bf = torch.bfloat16

    def get(name):
        with safe_open(f"{snap}/{weight_map[name]}", framework="pt") as f:
            return f.get_tensor(name).to(bf).contiguous()

    def raw(t):
        return t.contiguous().view(torch.int16).numpy().tobytes() if t.dtype == bf else t.numpy().tobytes()

    def shape_s(t):
        return "x".join(str(d) for d in t.shape)

    def run_bin(args):
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"{args[0]} failed: {r.stderr.strip()[:300]}")

    rows = []

    def record(name, y_bytes, expected, matmul):
        e = expected.float().numpy()
        y = load(y_bytes, "bfloat16", shape_s(expected))
        ok, strict, maxabs = (compare_matmul if matmul else compare)("bfloat16", y, e)
        rows.append((name, ok, strict, maxabs / max(float(np.abs(e).max()), 1e-30)))

    embed_w = get("model.embed_tokens.weight")
    lm_w = get("lm_head.weight") if "lm_head.weight" in weight_map else embed_w
    print("lm_head tied to embedding:", lm_w is embed_w)
    g = torch.Generator().manual_seed(7)
    ids3 = torch.tensor([785, 6722, VOCAB - 1])
    ids_sets = {"t1": ids3[:1], "t3": ids3}

    def norm_ref(w, x):
        m = Qwen3RMSNorm(w.numel(), eps=EPS)
        m.weight.data = w.clone()
        return m.to(bf)(x).detach()

    # --- embedding -------------------------------------------------------------------------------------------------
    open("/tmp/ew.bin", "wb").write(raw(embed_w))
    for tag, ids in ids_sets.items():
        open("/tmp/ids.bin", "wb").write(ids.to(torch.int32).numpy().tobytes())
        run_bin(["build/embedding_test", "bfloat16", str(HID), str(VOCAB), str(len(ids)), "/tmp/ids.bin", "/tmp/ew.bin", "/tmp/y.bin"])
        record(f"embedding {tag}", open("/tmp/y.bin", "rb").read(), embed_w[ids], matmul=False)

    # --- RMSNorm with real weights -----------------------------------------------------------------------------------
    norm_names = [f"model.layers.{L}.{n}.weight" for L in (0, LAST) for n in ("input_layernorm", "post_attention_layernorm")]
    norm_names.append("model.norm.weight")
    for name in norm_names:
        w = get(name)
        open("/tmp/w.bin", "wb").write(raw(w))
        for tag, ids in ids_sets.items():
            x = embed_w[ids]
            open("/tmp/x.bin", "wb").write(raw(x))
            run_bin(["build/rmsnorm_test", "bfloat16", str(HID), str(len(ids)), "/tmp/x.bin", "/tmp/w.bin", "/tmp/y.bin"])
            record(f"rmsnorm {name.replace('model.', '').replace('.weight', '')} {tag}", open("/tmp/y.bin", "rb").read(), norm_ref(w, x), matmul=False)
    for name in (f"model.layers.0.self_attn.q_norm.weight", f"model.layers.0.self_attn.k_norm.weight"):
        w = get(name)
        x = torch.randn(96, 128, generator=g).to(bf)
        open("/tmp/w.bin", "wb").write(raw(w)); open("/tmp/x.bin", "wb").write(raw(x))
        run_bin(["build/rmsnorm_test", "bfloat16", "128", "96", "/tmp/x.bin", "/tmp/w.bin", "/tmp/y.bin"])
        record(f"rmsnorm {name.replace('model.', '').replace('.weight', '')} (96 rows)", open("/tmp/y.bin", "rb").read(), norm_ref(w, x), matmul=False)

    # --- MLP with real weights (input = real post-attention-norm activations) -------------------------------------------
    cfg = Qwen3Config(hidden_size=HID, intermediate_size=INTER, hidden_act="silu")
    for L in (0, LAST):
        wg, wu, wd = (get(f"model.layers.{L}.mlp.{n}_proj.weight") for n in ("gate", "up", "down"))
        mlp = Qwen3MLP(cfg).to(bf)
        mlp.gate_proj.weight.data, mlp.up_proj.weight.data, mlp.down_proj.weight.data = wg, wu, wd
        pw = get(f"model.layers.{L}.post_attention_layernorm.weight")
        for f, t in (("wg", wg), ("wu", wu), ("wd", wd)):
            open(f"/tmp/{f}.bin", "wb").write(raw(t))
        for tag, ids in ids_sets.items():
            x = norm_ref(pw, embed_w[ids])
            open("/tmp/x.bin", "wb").write(raw(x))
            with torch.no_grad():
                y = mlp(x)
            run_bin(["build/mlp_test", "bfloat16", str(HID), str(INTER), str(len(ids)), "/tmp/x.bin", "/tmp/wg.bin", "/tmp/wu.bin", "/tmp/wd.bin", "/tmp/y.bin"])
            record(f"mlp layer {L} {tag} ({HID}/{INTER})", open("/tmp/y.bin", "rb").read(), y, matmul=True)

    # --- lm_head with the real vocabulary + greedy argmax -------------------------------------------------------------
    open("/tmp/lw.bin", "wb").write(raw(lm_w))
    fw = get("model.norm.weight")
    agree = []
    for tag, ids in ids_sets.items():
        x = norm_ref(fw, embed_w[ids])
        open("/tmp/x.bin", "wb").write(raw(x))
        with torch.no_grad():
            logits = torch.nn.functional.linear(x, lm_w)
        run_bin(["build/linear_test", "bfloat16", str(HID), str(VOCAB), str(len(ids)), "/tmp/x.bin", "/tmp/lw.bin", "/tmp/logits.bin"])
        gpu_logits_bytes = open("/tmp/logits.bin", "rb").read()
        record(f"lm_head {tag} ({HID}->{VOCAB})", gpu_logits_bytes, logits, matmul=True)
        # argmax kernel on the CPU (reference) logits: must match torch exactly
        open("/tmp/l.bin", "wb").write(raw(logits))
        run_bin(["build/argmax_test", "bfloat16", str(VOCAB), str(len(ids)), "/tmp/l.bin", "/tmp/ids_out.bin"])
        got = np.frombuffer(open("/tmp/ids_out.bin", "rb").read(), dtype=np.int32)
        exp = torch.argmax(logits, dim=-1).to(torch.int32).numpy()
        rows.append((f"argmax on reference logits {tag}", bool(np.array_equal(got, exp)), bool(np.array_equal(got, exp)), 0.0))
        # end to end: argmax of the GPU logits vs argmax of the CPU logits (informative: near-ties can flip)
        open("/tmp/l2.bin", "wb").write(gpu_logits_bytes)
        run_bin(["build/argmax_test", "bfloat16", str(VOCAB), str(len(ids)), "/tmp/l2.bin", "/tmp/ids_out2.bin"])
        got2 = np.frombuffer(open("/tmp/ids_out2.bin", "rb").read(), dtype=np.int32)
        top2 = torch.topk(logits.float(), 2, dim=-1).values
        agree.append((tag, got2.tolist(), exp.tolist(), (top2[:, 0] - top2[:, 1]).tolist()))

    print("\nREAL WEIGHTS (Qwen3-4B bf16): check / pass(tolerance) / pass(bit-identical) / worst err rel. to max|e|")
    for name, ok, strict, worst in rows:
        print("  %-52s tol=%-5s strict=%-5s worst=%.3e" % (name, ok, strict, worst))
    print("\nTop-1 token from GPU logits vs CPU reference logits (informative):")
    for tag, got2, exp, margin in agree:
        print(f"  {tag}: gpu={got2} cpu={exp} top1-top2 margin={[round(m, 3) for m in margin]}")
    all_ok = all(r[1] for r in rows)
    print("\nALL PASS (tolerance)" if all_ok else "\nSOME CHECKS FAILED")
    return 0 if all_ok else 1


@app.local_entrypoint()
def main(yes: bool = False):
    print(f"PLAN: real Qwen3-4B weights from volume qwen3-hf-cache (read-only, no download), 1x L4, hard cap 25 min "
          f"-> at most ~${25 / 60 * PRICE_L4:.2f} of shared credit")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    raise SystemExit(run.remote())
