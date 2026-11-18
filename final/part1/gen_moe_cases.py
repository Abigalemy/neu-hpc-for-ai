"""Part 1: reference test cases for the MoE layer of Qwen3-30B-A3B (SQLite, same schema as gen_cases.py), made with
PyTorch + HF Transformers on a CPU. Writes final/test_cases_moe.db.

  modal run part1/gen_moe_cases.py

Modules (all bf16, hidden size 2048 as in Qwen3-30B-A3B):
  moe_router  128 experts, top-8, norm_topk_prob (the real router shape; weight 0.5 MB)
  moe_block   router + experts + weighted sum, 16 experts of width 128, top-8 (the full 128 x 3 x 2048 x 768 experts
              would be 1.2 GB; the expert GEMM at the real width 768 is already covered by the mlp cases)
  moe_block_designed  as moe_block, but the router picks known experts: token 0 -> experts 0..7, token 1 -> 4..11
              (overlap 4..7), so a wrong expert, a wrong row or a wrong sum can be told apart

Expected tensors per case: topk_idx (int32), topk_w (bf16, what HF multiplies with), y (HF's bf16 output: HF adds the
8 expert outputs one by one into a bf16 tensor, in expert order), y_f32 (the same maths in fp32 from the same bf16
inputs and weights: the reference for a kernel that sums in fp32), and expert_out (each token's 8 expert outputs,
[tokens, 8, hidden], for debugging). The generator checks its own reference against the HF module before writing.
"""
import os
import sqlite3

import modal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # final/
app = modal.App("final-gen-moe-cases")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "transformers", extra_index_url="https://download.pytorch.org/whl/cpu")
    .add_local_file(os.path.join(HERE, "schema.sql"), "/root/schema.sql")
)
H = 2048


def _bytes(t):
    import torch
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.numpy().tobytes()


def add_module(db, kind, dtype, params):
    mid = db.execute("INSERT INTO module(kind, dtype) VALUES (?, ?)", (kind, dtype)).lastrowid
    for name, v in params.items():
        db.execute("INSERT INTO module_param VALUES (?, ?, ?, ?)",
                   (mid, name, v if isinstance(v, int) else None, v if isinstance(v, float) else None))
    return mid


def add_tensor(db, mid, name, t):
    db.execute("INSERT INTO module_tensor VALUES (?, ?, ?, ?, ?)",
               (mid, name, str(t.dtype).replace("torch.", ""), "x".join(map(str, t.shape)), _bytes(t)))


def add_case(db, mid, label, inputs, expected):
    cid = db.execute("INSERT INTO test_case(module_id, label) VALUES (?, ?)", (mid, label)).lastrowid
    for role, tensors in (("input", inputs), ("expected", expected)):
        for name, t in tensors.items():
            db.execute("INSERT INTO case_tensor VALUES (?, ?, ?, ?, ?, ?)",
                       (cid, role, name, str(t.dtype).replace("torch.", ""), "x".join(map(str, t.shape)), _bytes(t)))


def route(x, wg, k):
    """HF Qwen3MoeTopKRouter: bf16 logits, fp32 softmax, top-k, renormalise, cast to bf16."""
    import torch
    import torch.nn.functional as F
    logits = F.linear(x, wg)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    w, idx = torch.topk(probs, k, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
    return logits, probs, idx, w.to(x.dtype)


def experts_ref(x, idx, w, gate, up, down, dtype):
    """Each token's k expert outputs and their weighted sum, computed in `dtype` from the given weights."""
    import torch
    import torch.nn.functional as F
    T, k = idx.shape
    out = torch.zeros(T, k, x.shape[1], dtype=dtype)
    for t in range(T):
        for j in range(k):
            e = int(idx[t, j])
            xt = x[t:t + 1].to(dtype)
            h = F.silu(F.linear(xt, gate[e].to(dtype))) * F.linear(xt, up[e].to(dtype))
            out[t, j] = F.linear(h, down[e].to(dtype))[0]
    y = (out * w.to(dtype)[:, :, None]).sum(dim=1)
    return out, y


def hf_block(E, I, k):
    """The real HF module with explicit random weights, and its per-expert gate / up / down (checkpoint layout)."""
    import torch
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    cfg = Qwen3MoeConfig(hidden_size=H, moe_intermediate_size=I, num_experts=E, num_experts_per_tok=k,
                         norm_topk_prob=True, hidden_act="silu")
    blk = Qwen3MoeSparseMoeBlock(cfg)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, p in blk.named_parameters():   # newer transformers allocate expert weights with torch.empty
            fan_in = p.shape[-1]
            p.copy_(torch.randn(p.shape, generator=g) / fan_in ** 0.5)
    blk = blk.to(torch.bfloat16).eval()
    ex = blk.experts
    if hasattr(ex, "gate_up_proj"):              # transformers >= 5: fused [E, 2I, H], gate rows first
        gate, up, down = ex.gate_up_proj[:, :I], ex.gate_up_proj[:, I:], ex.down_proj
    else:                                        # transformers 4.x: a list of Qwen3MoeMLP
        gate = torch.stack([m.gate_proj.weight for m in ex]); up = torch.stack([m.up_proj.weight for m in ex])
        down = torch.stack([m.down_proj.weight for m in ex])
    wg = blk.gate.weight
    return blk, wg.detach(), gate.detach().contiguous(), up.detach().contiguous(), down.detach().contiguous()


def gen_router(db):
    import torch
    g = torch.Generator().manual_seed(3)
    E, k = 128, 8
    wg = (torch.randn(E, H, generator=g) / H ** 0.5).to(torch.bfloat16)
    mid = add_module(db, "moe_router", "bfloat16", {"hidden_size": H, "num_experts": E, "top_k": k})
    add_tensor(db, mid, "gate", wg)
    for T in (1, 2, 7, 64):
        x = torch.randn(T, H, generator=g).to(torch.bfloat16)
        logits, probs, idx, w = route(x, wg, k)
        add_case(db, mid, f"t{T}", {"x": x}, {"logits": logits, "probs": probs, "topk_idx": idx.to(torch.int32),
                                              "topk_w": w})


def add_block_cases(db, kind, blk, wg, gate, up, down, cases, E, I, k):
    import torch
    mid = add_module(db, kind, "bfloat16", {"hidden_size": H, "moe_intermediate_size": I, "num_experts": E, "top_k": k})
    add_tensor(db, mid, "router", wg)
    add_tensor(db, mid, "gate_proj", gate)       # [E, I, H]
    add_tensor(db, mid, "up_proj", up)           # [E, I, H]
    add_tensor(db, mid, "down_proj", down)       # [E, H, I]
    for label, x in cases:
        with torch.no_grad():
            y_hf = blk(x[None])[0]
            _, _, idx, w = route(x, wg, k)
            out16, _ = experts_ref(x, idx, w, gate, up, down, torch.bfloat16)
            out32, y32 = experts_ref(x, idx, w, gate, up, down, torch.float32)
            # HF adds the k expert outputs (each already times its bf16 weight) into a bf16 tensor in expert order
            y16 = torch.zeros_like(x)
            for t in range(x.shape[0]):
                for j in sorted(range(k), key=lambda j: int(idx[t, j])):
                    y16[t] += (out16[t, j] * w[t, j]).to(torch.bfloat16)
        scale = y32.abs().max().item()
        d_hf = (y_hf.float() - y16.float()).abs().max().item() / scale
        d_32 = (y_hf.float() - y32).abs().max().item() / scale
        print(f"{kind} {label}: |HF - own bf16 reference| / max = {d_hf:.3g}, |HF - fp32 reference| / max = {d_32:.3g}")
        # not bit-exact: HF multiplies all tokens of an expert in one matmul (and gate / up as one fused matrix),
        # the reference one token at a time, and CPU bf16 matmuls round differently for different shapes
        assert d_hf < 2e-2 and d_32 < 2e-2, "own reference does not reproduce the HF module"
        add_case(db, mid, label, {"x": x}, {"topk_idx": idx.to(torch.int32), "topk_w": w, "y": y_hf.contiguous(),
                                            "y_f32": y32, "expert_out": out16})
    return idx


def gen_blocks(db):
    import torch
    E, I, k = 16, 128, 8
    blk, wg, gate, up, down = hf_block(E, I, k)
    g = torch.Generator().manual_seed(11)
    cases = [(f"t{T}", torch.randn(T, H, generator=g).to(torch.bfloat16)) for T in (1, 2, 5, 33, 128)]
    add_block_cases(db, "moe_block", blk, wg, gate, up, down, cases, E, I, k)

    # designed routing: router row e reads hidden dim e only, so token t's logits are x[t, 0:16]
    blk2, _, gate2, up2, down2 = hf_block(E, I, k)
    wg2 = torch.zeros(E, H, dtype=torch.bfloat16)
    for e in range(E):
        wg2[e, e] = 1.0
    with torch.no_grad():
        blk2.gate.weight.copy_(wg2)
    x = (torch.randn(2, H, generator=g) * 0.1).to(torch.bfloat16)
    for j, e in enumerate(range(0, 8)):          # token 0 prefers experts 0..7 (descending), token 1 4..11
        x[0, e] = 3.0 - 0.25 * j
    for j, e in enumerate(range(4, 12)):
        x[1, e] = 2.5 - 0.25 * j
    idx = add_block_cases(db, "moe_block_designed", blk2, wg2, gate2, up2, down2, [("two_tokens_overlap", x)], E, I, k)
    assert sorted(idx[0].tolist()) == list(range(0, 8)) and sorted(idx[1].tolist()) == list(range(4, 12)), idx


@app.function(image=image, cpu=4, memory=8192, timeout=20 * 60)
def generate() -> bytes:
    import torch
    import transformers
    print(f"torch {torch.__version__}, transformers {transformers.__version__}")
    path = "/tmp/test_cases_moe.db"
    db = sqlite3.connect(path)
    db.executescript(open("/root/schema.sql").read())
    db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
    db.execute("INSERT INTO meta VALUES ('transformers', ?), ('torch', ?)", (transformers.__version__, torch.__version__))
    gen_router(db)
    gen_blocks(db)
    db.commit()
    for t in ("module", "test_case", "case_tensor"):
        print(t, db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
    db.close()
    return open(path, "rb").read()


@app.local_entrypoint()
def main():
    print("CPU-only Modal job (4 cores, no GPU): generating MoE reference cases")
    data = generate.remote()
    out = os.path.join(HERE, "test_cases_moe.db")
    open(out, "wb").write(data)
    print(f"wrote {out} ({len(data) / 1e6:.1f} MB)")
