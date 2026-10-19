import modal

app = modal.App("qwen3-kernel-benchmark")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install("build-essential", "ninja-build")
    .pip_install("torch", "numpy")
    .add_local_file("mlp_cuda.cu", remote_path="/root/project/mlp_cuda.cu")
    .add_local_file("flash_attn_cuda.cu", remote_path="/root/project/flash_attn_cuda.cu")
)

@app.function(image=image, gpu="T4")
def run_benchmark():
    import os
    import math
    import torch
    from torch.utils.cpp_extension import load

    os.chdir("/root/project")

    print("Compiling CUDA modules...")
    mlp_module = load(name="mlp_cuda_module", sources=["mlp_cuda.cu"], verbose=False)
    attn_module = load(name="attn_cuda_module", sources=["flash_attn_cuda.cu"], verbose=False)

    batch_size = 4
    seq_len = 128
    intermediate_size = 2816
    hidden_size = 1024
    num_heads = 16
    head_dim = hidden_size // num_heads

    warmup_iters = 20
    bench_iters = 100

    print("\n--- 1. MLP SwiGLU Benchmark ---")
    gate = torch.randn(batch_size, seq_len, intermediate_size, device="cuda")
    up = torch.randn(batch_size, seq_len, intermediate_size, device="cuda")

    for _ in range(warmup_iters):
        _ = torch.nn.functional.silu(gate) * up
        _ = mlp_module.fused_swiglu(gate, up)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(bench_iters):
        _ = torch.nn.functional.silu(gate) * up
    end_event.record()
    torch.cuda.synchronize()
    torch_mlp_time = start_event.elapsed_time(end_event) / bench_iters

    start_event.record()
    for _ in range(bench_iters):
        _ = mlp_module.fused_swiglu(gate, up)
    end_event.record()
    torch.cuda.synchronize()
    cuda_mlp_time = start_event.elapsed_time(end_event) / bench_iters

    print(f"PyTorch MLP latency: {torch_mlp_time:.4f} ms")
    print(f"CUDA Fused MLP latency: {cuda_mlp_time:.4f} ms")
    print(f"MLP Speedup: {torch_mlp_time / cuda_mlp_time:.2f}x")

    print("\n--- 2. FlashAttention Benchmark ---")
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
    scale = 1.0 / math.sqrt(head_dim)

    def pytorch_attention(q_t, k_t, v_t):
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v_t)

    for _ in range(warmup_iters):
        _ = pytorch_attention(q, k, v)
        _ = attn_module.tiled_attention(q, k, v)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(bench_iters):
        _ = pytorch_attention(q, k, v)
    end_event.record()
    torch.cuda.synchronize()
    torch_attn_time = start_event.elapsed_time(end_event) / bench_iters

    start_event.record()
    for _ in range(bench_iters):
        _ = attn_module.tiled_attention(q, k, v)
    end_event.record()
    torch.cuda.synchronize()
    cuda_attn_time = start_event.elapsed_time(end_event) / bench_iters

    print(f"PyTorch Attention latency: {torch_attn_time:.4f} ms")
    print(f"CUDA Tiled Attention latency: {cuda_attn_time:.4f} ms")
    print(f"Attention Speedup: {torch_attn_time / cuda_attn_time:.2f}x")

@app.local_entrypoint()
def main():
    run_benchmark.remote()