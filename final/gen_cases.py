"""Generate reference test cases (SQLite) with PyTorch + HF Transformers. CPU only, a few cents on Modal.

  modal run gen_cases.py            # writes final/test_cases.db
"""
import os
import sqlite3

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("final-gen-cases")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "transformers", extra_index_url="https://download.pytorch.org/whl/cpu")
    .add_local_file(os.path.join(HERE, "schema.sql"), "/root/schema.sql")
)

EPS = 1e-6  # Qwen3 rms_norm_eps


def _bytes(t):
    import torch
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.numpy().tobytes()


def _shape(t):
    return "x".join(str(d) for d in t.shape)


def _dtype_name(t):
    return str(t.dtype).replace("torch.", "")


def add_module(db, kind, dtype, params):
    cur = db.execute("INSERT INTO module(kind, dtype) VALUES (?, ?)", (kind, dtype))
    mid = cur.lastrowid
    for name, v in params.items():
        db.execute("INSERT INTO module_param VALUES (?, ?, ?, ?)",
                   (mid, name, v if isinstance(v, int) else None, v if isinstance(v, float) else None))
    return mid


def add_tensor(db, mid, name, t):
    db.execute("INSERT INTO module_tensor VALUES (?, ?, ?, ?, ?)", (mid, name, _dtype_name(t), _shape(t), _bytes(t)))


def add_case(db, mid, label, inputs, expected, params=None):
    cid = db.execute("INSERT INTO test_case(module_id, label) VALUES (?, ?)", (mid, label)).lastrowid
    for name, v in (params or {}).items():
        db.execute("INSERT INTO case_param VALUES (?, ?, ?, ?)",
                   (cid, name, v if isinstance(v, int) else None, v if isinstance(v, float) else None))
    for role, tensors in (("input", inputs), ("expected", expected)):
        for name, t in tensors.items():
            db.execute("INSERT INTO case_tensor VALUES (?, ?, ?, ?, ?, ?)",
                       (cid, role, name, _dtype_name(t), _shape(t), _bytes(t)))


def gen_rmsnorm(db):
    """Random-weight RMSNorm cases. Reference = the real HF Qwen3RMSNorm module."""
    import torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    g = torch.Generator().manual_seed(0)
    # 128 = head_dim (q_norm / k_norm); 2560 = Qwen3-4B hidden; 2048 = Qwen3-30B-A3B hidden.
    for dtype_name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        for hidden in (128, 2048, 2560):
            norm = Qwen3RMSNorm(hidden, eps=EPS)
            w = (1.0 + 0.2 * torch.randn(hidden, generator=g)).to(dtype)
            norm.weight.data = w.clone()
            norm.to(dtype)
            mid = add_module(db, "rmsnorm", dtype_name, {"hidden_size": hidden, "eps": EPS})
            add_tensor(db, mid, "weight", w)
            for tokens in (1, 3, 7, 33, 128):
                for label, scale in (("normal", 1.0), ("large", 30.0), ("small", 0.01)):
                    x = (scale * torch.randn(tokens, hidden, generator=g)).to(dtype)
                    with torch.no_grad():
                        y = norm(x)
                    add_case(db, mid, f"{label}_t{tokens}", {"x": x}, {"y": y})


def gen_rope(db):
    """Qwen3 default RoPE (theta 1e6, head_dim 128). Reference = HF Qwen3RotaryEmbedding + apply_rotary_pos_emb.

    The cos/sin tables are saved as module tensors, taken from HF's own output (first 64 of its duplicated 128
    columns), so this kernel test is independent of how the tables get computed (that is tested separately later).
    """
    import torch
    import transformers
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, apply_rotary_pos_emb
    print("transformers", transformers.__version__)
    g = torch.Generator().manual_seed(1)
    HD, MAXP, THETA = 128, 4096, 1000000.0
    pos_cases = [("decode_p0", [0]), ("decode_p5", [5]), ("decode_p4095", [4095]), ("prefill_0_3", [0, 1, 2]),
                 ("prefill_1000_3", [1000, 1001, 1002]), ("prefill_0_17", list(range(17))),
                 ("prefill_4000_17", list(range(4000, 4017)))]
    rnd = sorted(torch.randperm(MAXP, generator=g)[:33].tolist())
    pos_cases.append(("random_33", rnd))
    for dtype_name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        for nh, nkv in ((32, 8), (32, 4)):   # Qwen3-4B: 32/8 ; Qwen3-30B-A3B: 32/4
            cfg = Qwen3Config(hidden_size=nh * HD, num_attention_heads=nh, num_key_value_heads=nkv, head_dim=HD,
                              rope_theta=THETA, max_position_embeddings=MAXP)
            rot = Qwen3RotaryEmbedding(config=cfg)
            probe = torch.zeros(1, 1, HD, dtype=dtype)  # only the dtype is used by HF
            cos_all, sin_all = rot(torch.zeros(1, MAXP, HD, dtype=dtype), torch.arange(MAXP)[None])
            assert torch.equal(cos_all[0, :, :HD // 2], cos_all[0, :, HD // 2:])
            cos_tab = cos_all[0, :, :HD // 2].contiguous()
            sin_tab = sin_all[0, :, :HD // 2].contiguous()
            mid = add_module(db, "rope", dtype_name, {"head_dim": HD, "num_heads": nh, "num_kv_heads": nkv,
                                                      "max_pos": MAXP, "rope_theta": THETA})
            add_tensor(db, mid, "cos", cos_tab)
            add_tensor(db, mid, "sin", sin_tab)
            for label, positions in pos_cases:
                T = len(positions)
                q = torch.randn(T, nh, HD, generator=g).to(dtype)
                k = torch.randn(T, nkv, HD, generator=g).to(dtype)
                pos = torch.tensor(positions, dtype=torch.int32)
                cos, sin = rot(q, pos.long()[None])
                qe, ke = apply_rotary_pos_emb(q.permute(1, 0, 2)[None], k.permute(1, 0, 2)[None], cos, sin)
                q_out = qe[0].permute(1, 0, 2).contiguous()
                k_out = ke[0].permute(1, 0, 2).contiguous()
                add_case(db, mid, label, {"q": q, "k": k, "positions": pos}, {"q_out": q_out, "k_out": k_out})


def gen_mlp(db):
    """Random-weight SwiGLU MLP. Reference = the real HF Qwen3MLP module (default nn.Linear init, no biases).

    Shapes: (256, 512) small; (2048, 768) = one Qwen3-30B-A3B expert; (2560, 1024) = Qwen3-4B hidden size with a
    reduced intermediate size (the full 9728 would make the database huge; the real shape is checked with real
    weights later). fp32 only on the small shape to keep the database small.
    """
    import torch
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
    g = torch.Generator().manual_seed(2)
    torch.manual_seed(2)
    combos = [("float32", torch.float32, 256, 512), ("bfloat16", torch.bfloat16, 256, 512),
              ("bfloat16", torch.bfloat16, 2048, 768), ("bfloat16", torch.bfloat16, 2560, 1024)]
    for dtype_name, dtype, hid, inter in combos:
        cfg = Qwen3Config(hidden_size=hid, intermediate_size=inter, hidden_act="silu")
        mlp = Qwen3MLP(cfg).to(dtype)
        mid = add_module(db, "mlp", dtype_name, {"hidden_size": hid, "intermediate_size": inter})
        for name in ("gate_proj", "up_proj", "down_proj"):
            add_tensor(db, mid, name, getattr(mlp, name).weight.data.contiguous())
        for tokens in (1, 3, 7, 33, 128):
            x = torch.randn(tokens, hid, generator=g).to(dtype)
            with torch.no_grad():
                y = mlp(x)
            add_case(db, mid, f"normal_t{tokens}", {"x": x}, {"y": y})


def gen_embedding(db):
    """nn.Embedding lookup (reference = torch embedding, a pure copy). ids include 0, vocab-1 and repeats."""
    import torch
    g = torch.Generator().manual_seed(3)
    combos = [("float32", torch.float32, 4096, 256), ("bfloat16", torch.bfloat16, 4096, 256),
              ("bfloat16", torch.bfloat16, 4096, 2560), ("bfloat16", torch.bfloat16, 2048, 2048)]
    for dtype_name, dtype, vocab, hid in combos:
        emb = torch.nn.Embedding(vocab, hid).to(dtype)
        emb.weight.data = torch.randn(vocab, hid, generator=g).to(dtype)
        mid = add_module(db, "embedding", dtype_name, {"vocab_size": vocab, "hidden_size": hid})
        add_tensor(db, mid, "weight", emb.weight.data.contiguous())
        for tokens in (1, 3, 33, 128):
            ids = torch.randint(0, vocab, (tokens,), generator=g)
            ids[0] = 0
            if tokens > 1:
                ids[-1] = vocab - 1
            if tokens > 3:
                ids[2] = ids[1]  # repeated token
            with torch.no_grad():
                y = emb(ids)
            add_case(db, mid, f"ids_t{tokens}", {"ids": ids.to(torch.int32)}, {"y": y})


def gen_linear(db):
    """lm_head-style nn.Linear (no bias, default init). Reference = torch Linear in the activation dtype."""
    import torch
    g = torch.Generator().manual_seed(4)
    torch.manual_seed(4)
    combos = [("float32", torch.float32, 256, 4096), ("bfloat16", torch.bfloat16, 256, 4096),
              ("bfloat16", torch.bfloat16, 2560, 2048), ("bfloat16", torch.bfloat16, 2048, 4096)]
    for dtype_name, dtype, nin, nout in combos:
        lin = torch.nn.Linear(nin, nout, bias=False).to(dtype)
        mid = add_module(db, "linear", dtype_name, {"in_features": nin, "out_features": nout})
        add_tensor(db, mid, "weight", lin.weight.data.contiguous())
        for tokens in (1, 3, 7, 33):
            x = torch.randn(tokens, nin, generator=g).to(dtype)
            with torch.no_grad():
                y = lin(x)
            add_case(db, mid, f"normal_t{tokens}", {"x": x}, {"y": y})


def gen_argmax(db):
    """Greedy sampling: torch.argmax over the vocabulary. Includes forced ties (first index must win), ties across
    a 1024-wide chunk boundary, an all-negative row, and a vocabulary that is not a multiple of 1024."""
    import torch
    g = torch.Generator().manual_seed(5)
    for dtype_name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        for vocab in (4096, 5000):
            mid = add_module(db, "argmax", dtype_name, {"vocab_size": vocab})
            for rows in (1, 4, 16):
                logits = (3.0 * torch.randn(rows, vocab, generator=g)).to(dtype)
                add_case(db, mid, f"random_r{rows}", {"logits": logits},
                         {"ids": torch.argmax(logits, dim=-1).to(torch.int32)})
            tie = (torch.randn(4, vocab, generator=g)).to(dtype)
            top = torch.tensor(50.0).to(dtype)
            tie[0, 7] = top; tie[0, 9] = top                    # tie inside one chunk
            tie[1, 5] = top; tie[1, vocab - 3] = top            # tie across chunks
            tie[2, 1030] = top; tie[2, 1030 + 1024] = top       # tie in the second and third chunk
            tie[3] = -5.0 - torch.rand(vocab, generator=g).to(dtype)   # all negative, max somewhere in the middle
            add_case(db, mid, "ties_and_negative", {"logits": tie}, {"ids": torch.argmax(tie, dim=-1).to(torch.int32)})


@app.function(image=image, timeout=900)
def generate() -> bytes:
    path = "/tmp/test_cases.db"
    if os.path.exists(path):
        os.remove(path)
    db = sqlite3.connect(path)
    db.executescript(open("/root/schema.sql").read())
    gen_rmsnorm(db)
    gen_rope(db)
    gen_mlp(db)
    gen_embedding(db)
    gen_linear(db)
    gen_argmax(db)
    db.commit()
    for t in ("module", "test_case", "case_tensor"):
        print(t, db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
    db.close()
    return open(path, "rb").read()


@app.local_entrypoint()
def main():
    print("CPU-only Modal job: generating reference cases")
    data = generate.remote()
    out = os.path.join(HERE, "test_cases.db")
    open(out, "wb").write(data)
    print(f"wrote {out} ({len(data) / 1e6:.2f} MB)")
