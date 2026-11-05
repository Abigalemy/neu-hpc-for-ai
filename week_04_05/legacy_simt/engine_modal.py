import modal

app = modal.App("qwen3-inference-engine")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install("build-essential", "ninja-build")
    .pip_install("torch", "numpy", "transformers")
    .add_local_file("mlp_cuda.cu", remote_path="/root/project/mlp_cuda.cu")
    .add_local_file("flash_attn_cuda.cu", remote_path="/root/project/flash_attn_cuda.cu")
    .add_local_file("ops_cuda.cu", remote_path="/root/project/ops_cuda.cu")
    .add_local_file("tokenizer.h", remote_path="/root/project/tokenizer.h")
    .add_local_file("test_cases.db", remote_path="/root/project/test_cases.db")
)

@app.function(image=image, gpu="T4")
def run_engine():
    import os
    import math
    import torch
    import torch.nn as nn
    from torch.utils.cpp_extension import load

    os.chdir("/root/project")

    print("Building inference engine modules...")
    mlp_mod = load(name="mlp_cuda_module", sources=["mlp_cuda.cu"], verbose=False)
    attn_mod = load(name="attn_cuda_module", sources=["flash_attn_cuda.cu"], verbose=False)
    ops_mod = load(name="ops_cuda_module", sources=["ops_cuda.cu"], verbose=False)

    batch_size = 2
    seq_len = 16
    hidden_size = 1024
    num_heads = 16
    head_dim = hidden_size // num_heads
    intermediate_size = 2816
    eps = 1e-6

    torch.manual_seed(42)

    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda")

    w_input_norm = torch.ones(hidden_size, device="cuda")
    w_post_norm = torch.ones(hidden_size, device="cuda")

    w_q = torch.randn(hidden_size, hidden_size, device="cuda") * 0.02
    w_k = torch.randn(hidden_size, hidden_size, device="cuda") * 0.02
    w_v = torch.randn(hidden_size, hidden_size, device="cuda") * 0.02
    w_o = torch.randn(hidden_size, hidden_size, device="cuda") * 0.02

    w_gate = torch.randn(intermediate_size, hidden_size, device="cuda") * 0.02
    w_up = torch.randn(intermediate_size, hidden_size, device="cuda") * 0.02
    w_down = torch.randn(hidden_size, intermediate_size, device="cuda") * 0.02

    # PyTorch Reference Pipeline
    def pytorch_reference(x_in):
        # 1. Pre-attention RMSNorm
        var = torch.mean(x_in ** 2, dim=-1, keepdim=True)
        norm_x = x_in * torch.rsqrt(var + eps) * w_input_norm

        # 2. Attention
        q = torch.matmul(norm_x, w_q.t()).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = torch.matmul(norm_x, w_k.t()).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = torch.matmul(norm_x, w_v.t()).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        attn_out = torch.matmul(attn_out, w_o.t())

        res1 = x_in + attn_out

        # 3. Post-attention RMSNorm
        var2 = torch.mean(res1 ** 2, dim=-1, keepdim=True)
        norm_res1 = res1 * torch.rsqrt(var2 + eps) * w_post_norm

        # 4. SwiGLU MLP
        g = torch.matmul(norm_res1, w_gate.t())
        u = torch.matmul(norm_res1, w_up.t())
        mlp_inter = torch.nn.functional.silu(g) * u
        mlp_out = torch.matmul(mlp_inter, w_down.t())

        return res1 + mlp_out

    # CUDA Batched Engine Pipeline
    def cuda_engine(x_in):
        # 1. CUDA RMSNorm
        norm_x = ops_mod.rms_norm(x_in, w_input_norm, eps)

        # 2. CUDA Fused Attention
        q = torch.matmul(norm_x, w_q.t()).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
        k = torch.matmul(norm_x, w_k.t()).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
        v = torch.matmul(norm_x, w_v.t()).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

        attn_ctx = attn_mod.tiled_attention(q, k, v)
        attn_out = torch.matmul(attn_ctx.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size), w_o.t())

        res1 = x_in + attn_out

        # 3. CUDA Post-attention RMSNorm
        norm_res1 = ops_mod.rms_norm(res1, w_post_norm, eps)

        # 4. CUDA Fused SwiGLU
        g = torch.matmul(norm_res1, w_gate.t())
        u = torch.matmul(norm_res1, w_up.t())
        mlp_inter = mlp_mod.fused_swiglu(g, u)
        mlp_out = torch.matmul(mlp_inter, w_down.t())

        return res1 + mlp_out

    ref_out = pytorch_reference(x)
    engine_out = cuda_engine(x)

    max_diff = torch.max(torch.abs(engine_out - ref_out)).item()
    print("\nEnd-to-End Qwen3 Layer Verification:")
    print("Maximum absolute difference:", max_diff)
    assert max_diff < 1e-4, f"Engine mismatch: {max_diff}"
    print("End-to-end batched inference engine verification PASSED!")

@app.local_entrypoint()
def main():
    run_engine.remote()