import modal

app = modal.App("attn-cuda-benchmark")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install("build-essential", "ninja-build")
    .pip_install("torch", "numpy")
    .add_local_file("flash_attn_cuda.cu", remote_path="/root/project/flash_attn_cuda.cu")
    .add_local_file("test_cases.db", remote_path="/root/project/test_cases.db")
)

@app.function(image=image, gpu="T4")
def run_attention_cuda_test():
    import os
    import sqlite3
    import pickle
    import torch
    import numpy as np
    from torch.utils.cpp_extension import load

    os.chdir("/root/project")

    print("Compiling FlashAttention CUDA extension on remote GPU...")
    attn_module = load(
        name="attn_cuda_module",
        sources=["flash_attn_cuda.cu"],
        verbose=False
    )

    conn = sqlite3.connect("test_cases.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT weights_data, input_data, output_data, batch_size, seq_len, hidden_size, num_heads FROM attention_test_cases LIMIT 1"
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None, "No test cases found in attention_test_cases!"
    weights_bytes, input_bytes, expected_output_bytes, batch_size, seq_len, hidden_size, num_heads = row

    weights_dict = pickle.loads(weights_bytes)
    input_tensor = torch.from_numpy(np.frombuffer(input_bytes, dtype=np.float32).copy().reshape(batch_size, seq_len, hidden_size)).cuda()
    expected_output = torch.from_numpy(np.frombuffer(expected_output_bytes, dtype=np.float32).copy().reshape(batch_size, seq_len, hidden_size)).cuda()

    head_dim = hidden_size // num_heads

    w_q = weights_dict["q_proj.weight"].cuda()
    w_k = weights_dict["k_proj.weight"].cuda()
    w_v = weights_dict["v_proj.weight"].cuda()
    w_o = weights_dict["o_proj.weight"].cuda()

    q = torch.matmul(input_tensor, w_q.t()).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    k = torch.matmul(input_tensor, w_k.t()).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    v = torch.matmul(input_tensor, w_v.t()).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    context = attn_module.tiled_attention(q, k, v)

    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
    actual_output = torch.matmul(context, w_o.t())

    max_diff = torch.max(torch.abs(actual_output - expected_output)).item()
    print("Database Attention verification max difference:", max_diff)
    assert max_diff < 1e-4, f"Verification failed with max diff {max_diff}"
    print("Attention test passed successfully via remote CUDA kernel against test_cases.db!")

@app.local_entrypoint()
def main():
    run_attention_cuda_test.remote()