import modal

app = modal.App("mlp-cuda-benchmark")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install("build-essential", "ninja-build")
    .pip_install("torch", "numpy")
    .add_local_file("mlp_cuda.cu", remote_path="/root/project/mlp_cuda.cu")
    .add_local_file("test_cases.db", remote_path="/root/project/test_cases.db")
)

@app.function(image=image, gpu="T4")
def run_database_cuda_test():
    import os
    import sqlite3
    import pickle
    import torch
    import numpy as np
    from torch.utils.cpp_extension import load

    os.chdir("/root/project")

    print("Compiling CUDA extension on remote GPU...")
    mlp_module = load(
        name="mlp_cuda_module",
        sources=["mlp_cuda.cu"],
        verbose=False
    )

    conn = sqlite3.connect("test_cases.db")
    cursor = conn.cursor()
    cursor.execute("SELECT weights_data, input_data, output_data FROM mlp_test_cases LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    assert row is not None, "No test cases found in test_cases.db!"
    weights_bytes, input_bytes, expected_output_bytes = row

    weights_dict = pickle.loads(weights_bytes)
    input_tensor = torch.from_numpy(np.frombuffer(input_bytes, dtype=np.float32).copy().reshape(2, 4, 256)).cuda()
    expected_output = torch.from_numpy(np.frombuffer(expected_output_bytes, dtype=np.float32).copy().reshape(2, 4, 256)).cuda()

    w_gate = weights_dict["gate_proj.weight"].cuda()
    w_up = weights_dict["up_proj.weight"].cuda()
    w_down = weights_dict["down_proj.weight"].cuda()

    gate_val = torch.matmul(input_tensor, w_gate.t())
    up_val = torch.matmul(input_tensor, w_up.t())

    fused_inter = mlp_module.fused_swiglu(gate_val, up_val)

    actual_output = torch.matmul(fused_inter, w_down.t())

    max_diff = torch.max(torch.abs(actual_output - expected_output)).item()
    print("Database verification max difference:", max_diff)
    assert max_diff < 1e-5, f"Verification failed with max diff {max_diff}"
    print("Test passed successfully via remote CUDA kernel against test_cases.db!")

@app.local_entrypoint()
def main():
    run_database_cuda_test.remote()