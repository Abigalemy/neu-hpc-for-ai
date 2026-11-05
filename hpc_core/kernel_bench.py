"""Correctness and latency matrix for the CUDA Tile SwiGLU and FlashAttention kernels."""
from __future__ import annotations


def run_kernel_matrix():
    import torch
    import torch.nn.functional as F

    from .kernels.cutile_attention import flash_attention
    from .kernels.cutile_ops import fused_swiglu

    torch.manual_seed(2025)
    device = "cuda"
    dtype = torch.bfloat16
    batch_sizes = (1, 2, 4)
    seq_lens = (32, 64, 128)
    q_heads, kv_heads, head_dim = 32, 8, 128
    intermediate = 1024
    warmup, repeats = 5, 30

    def latency_ms(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeats

    print("CUDA Tile performance matrix (bf16; CUDA events, 5 warmups and 30 timed repeats)")
    swiglu_tiles = (128, 256, 512, 1024)
    print("batch,seq,swiglu_ref_ms," + ",".join(f"swiglu_{n}_ms" for n in swiglu_tiles) +
          ",swiglu_max_abs,attention_ref_ms,attention_tile_ms,attention_max_abs")

    for batch in batch_sizes:
        for seq in seq_lens:
            rows = batch * seq
            gate = torch.randn(rows, intermediate, device=device, dtype=dtype)
            up = torch.randn_like(gate)
            ref_mlp = F.silu(gate) * up
            mlp_diffs, mlp_times = [], []
            for tile_size in swiglu_tiles:
                tile_mlp = fused_swiglu(gate, up, tile_size=tile_size)
                mlp_diff = (ref_mlp.float() - tile_mlp.float()).abs().max().item()
                if not torch.allclose(ref_mlp.float(), tile_mlp.float(), atol=2e-2, rtol=2e-2):
                    raise AssertionError(
                        f"SwiGLU mismatch at batch={batch}, seq={seq}, tile={tile_size}: {mlp_diff}"
                    )
                mlp_diffs.append(mlp_diff)
                mlp_times.append(latency_ms(lambda t=tile_size: fused_swiglu(gate, up, tile_size=t)))
            mlp_diff = max(mlp_diffs)
            t_mlp_ref = latency_ms(lambda: F.silu(gate) * up)

            q = torch.randn(batch, q_heads, seq, head_dim, device=device, dtype=dtype)
            k = torch.randn(batch, kv_heads, seq, head_dim, device=device, dtype=dtype)
            v = torch.randn_like(k)
            ref_attn = F.scaled_dot_product_attention(
                q, k.repeat_interleave(q_heads // kv_heads, dim=1),
                v.repeat_interleave(q_heads // kv_heads, dim=1), is_causal=True,
            )
            tile_attn = flash_attention(q, k, v, causal=True)
            attn_diff = (ref_attn.float() - tile_attn.float()).abs().max().item()
            if not torch.allclose(ref_attn.float(), tile_attn.float(), atol=3e-2, rtol=3e-2):
                raise AssertionError(f"FlashAttention mismatch at batch={batch}, seq={seq}: {attn_diff}")
            t_attn_ref = latency_ms(lambda: F.scaled_dot_product_attention(
                q, k.repeat_interleave(q_heads // kv_heads, dim=1),
                v.repeat_interleave(q_heads // kv_heads, dim=1), is_causal=True,
            ))
            t_attn_tile = latency_ms(lambda: flash_attention(q, k, v, causal=True))
            print(f"{batch},{seq},{t_mlp_ref:.5f}," + ",".join(f"{t:.5f}" for t in mlp_times) +
                  f",{mlp_diff:.3e},"
                  f"{t_attn_ref:.5f},{t_attn_tile:.5f},{attn_diff:.3e}")
