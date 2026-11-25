"""CPU-only check: the engine's RoPE tables (default and YaRN) against HF's Qwen3RotaryEmbedding, bit for bit.

  modal run rope_table_check.py            # CPU container only, a few cents

Compares cos and sin tables for positions 0..n-1 (n = 131080 for YaRN, i.e. a 128K context plus a few tokens).
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("final-rope-table-check")
IMAGE = "nvidia/cuda:13.5.0-devel-ubuntu24.04"
image = (
    modal.Image.from_registry(IMAGE, add_python="3.12")
    .pip_install("numpy", "transformers", "torch", extra_index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(HERE, "/root/final", ignore=[".venv", "__pycache__", "*.pyc", "build", "test_cases.db"])
)


@app.function(image=image, timeout=1200, cpu=4, memory=16384)
def check():
    import subprocess

    import numpy as np
    import torch
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    os.chdir("/root/final")
    r = subprocess.run("mkdir -p build && nvcc -std=c++17 -O2 tests/rope_table_test.cu -o build/rope_table_test",
                       shell=True, capture_output=True, text=True)
    print("compile exit code:", r.returncode)
    if r.returncode:
        print(r.stderr[-2000:])
        return 1

    def hf(yarn, n):
        kw = dict(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8, head_dim=128, max_position_embeddings=40960)
        rp = ({"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768, "rope_theta": 1000000.0}
              if yarn else {"rope_type": "default", "rope_theta": 1000000.0})
        try:
            cfg = Qwen3Config(rope_parameters=rp, **kw)
        except TypeError:
            cfg = Qwen3Config(rope_theta=1000000.0, rope_scaling=(rp if yarn else None), **kw)
        rot = Qwen3RotaryEmbedding(config=cfg)
        cos, sin = rot(torch.zeros(1, 1, 128, dtype=torch.bfloat16), torch.arange(n)[None])
        return (cos[0, :, :64].contiguous().view(torch.int16).numpy().view(np.uint16),
                sin[0, :, :64].contiguous().view(torch.int16).numpy().view(np.uint16), float(rot.attention_scaling))

    ok = True
    for name, yarn, n in (("default RoPE", 0.0, 4104), ("YaRN factor 4, orig 32768", 4.0, 131080)):
        out = subprocess.run(["build/rope_table_test", "1000000", str(yarn), "32768", str(n), "/tmp/t.bin"],
                             capture_output=True, text=True)
        print(out.stdout.strip())
        mine = np.fromfile("/tmp/t.bin", dtype=np.uint16).reshape(2, n, 64)
        hc, hs, att = hf(yarn > 0, n)
        for tag, a, b in (("cos", mine[0], hc), ("sin", mine[1], hs)):
            diff = a != b
            fa = (a.astype(np.uint32) << 16).view(np.float32)
            fb = (b.astype(np.uint32) << 16).view(np.float32)
            print(f"  {name:28s} {tag}: differing bf16 entries {int(diff.sum())}/{diff.size}"
                  + (f", max |diff| {float(np.abs(fa - fb).max()):.3e} (HF attention_scaling {att:.6f})" if diff.any() else f" (HF attention_scaling {att:.6f})"))
            ok &= not bool(diff.any())
    print("\nALL BIT-IDENTICAL" if ok else "\nTABLES DIFFER (see counts above)")
    return 0 if ok else 1


@app.local_entrypoint()
def main():
    print("PLAN: CPU-only job (a few cents): our RoPE tables vs HF, default and YaRN")
    raise SystemExit(check.remote())
