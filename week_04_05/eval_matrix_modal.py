import modal

app = modal.App("qwen3-matrix-evaluation")

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
)

@app.function(image=image, gpu="T4")
def run_matrix_benchmark():
    import os
    import math
    import torch
    from torch.utils.cpp_extension import load

    os.chdir("/root/project")

    print("Compiling CUDA extension modules...")
    mlp_mod = load(name="mlp_cuda_module", sources=["mlp_cuda.cu"], verbose=False)
    attn_mod = load(name="attn_cuda_module", sources=["flash_attn_cuda.cu"], verbose=False)
    ops_mod = load(name="ops_cuda_module", sources=["ops_cuda.cu"], verbose=False)

    batch_sizes = [1, 2, 4]
    seq_lens = [32, 64, 128]
    hidden_size = 1024
    num_heads = 16
    head_dim = hidden_size // num_heads
    intermediate_size = 2816

    print("\n" + "=" * 80)
    print(f"{'Batch':<8}{'SeqLen':<10}{'PyTorch MLP(ms)':<18}{'CUDA Fused(ms)':<18}{'Max Diff':<15}{'Verified'}")
    print("=" * 80)

    for b in batch_sizes:
        for s in seq_lens:
            gate = torch.randn(b, s, intermediate_size, device="cuda")
            up = torch.randn(b, s, intermediate_size, device="cuda")

            cuda_out = mlp_mod.fused_swiglu(gate, up)
            ref_out = torch.nn.functional.silu(gate) * up
            max_diff = torch.max(torch.abs(cuda_out - ref_out)).item()
            verified = "PASS" if max_diff < 1e-5 else "FAIL"

            for _ in range(10):
                _ = mlp_mod.fused_swiglu(gate, up)
                _ = torch.nn.functional.silu(gate) * up
            torch.cuda.synchronize()

            start_e = torch.cuda.Event(enable_timing=True)
            end_e = torch.cuda.Event(enable_timing=True)

            start_e.record()
            for _ in range(50):
                _ = torch.nn.functional.silu(gate) * up
            end_e.record()
            torch.cuda.synchronize()
            t_ref = start_e.elapsed_time(end_e) / 50.0

            start_e.record()
            for _ in range(50):
                _ = mlp_mod.fused_swiglu(gate, up)
            end_e.record()
            torch.cuda.synchronize()
            t_cuda = start_e.elapsed_time(end_e) / 50.0

            print(f"{b:<8}{s:<10}{t_ref:<18.4f}{t_cuda:<18.4f}{max_diff:<15.2e}{verified}")

    print("=" * 80)
    print("\nAttention (Fully Fused FlashAttention) Verification across Dimensions:")
    print("=" * 80)
    print(f"{'Batch':<8}{'SeqLen':<10}{'PyTorch Attn(ms)':<18}{'CUDA Attn(ms)':<18}{'Max Diff':<15}{'Verified'}")
    print("=" * 80)

    scale = 1.0 / math.sqrt(head_dim)
    for b in batch_sizes:
        for s in seq_lens:
            q = torch.randn(b, num_heads, s, head_dim, device="cuda")
            k = torch.randn(b, num_heads, s, head_dim, device="cuda")
            v = torch.randn(b, num_heads, s, head_dim, device="cuda")

            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            weights = torch.softmax(scores, dim=-1)
            ref_attn = torch.matmul(weights, v)

            cuda_attn = attn_mod.tiled_attention(q, k, v)
            max_diff = torch.max(torch.abs(cuda_attn - ref_attn)).item()
            verified = "PASS" if max_diff < 1e-4 else "FAIL"

            start_e = torch.cuda.Event(enable_timing=True)
            end_e = torch.cuda.Event(enable_timing=True)

            start_e.record()
            for _ in range(30):
                sc = torch.matmul(q, k.transpose(-2, -1)) * scale
                _ = torch.matmul(torch.softmax(sc, dim=-1), v)
            end_e.record()
            torch.cuda.synchronize()
            t_ref = start_e.elapsed_time(end_e) / 30.0

            start_e.record()
            for _ in range(30):
                _ = attn_mod.tiled_attention(q, k, v)
            end_e.record()
            torch.cuda.synchronize()
            t_cuda = start_e.elapsed_time(end_e) / 30.0

            print(f"{b:<8}{s:<10}{t_ref:<18.4f}{t_cuda:<18.4f}{max_diff:<15.2e}{verified}")

    print("=" * 80)

@app.local_entrypoint()
def main():
    run_matrix_benchmark.remote()