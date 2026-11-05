"""Week 04–05 evaluation entry point for the shared Tile kernel matrix.

The benchmark code is shared with benchmark_modal.py so correctness and timing
cannot silently diverge between the two assignment scripts.
"""
import os

import modal

app = modal.App("qwen3-cutile-matrix")
_here = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.dirname(_here)
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.10.0", "numpy", "cuda-tile==1.8.0",
                 "cuda-toolkit[tileiras,nvvm,nvcc]>=13.3,<13.6")
    .add_local_dir(os.path.join(_repo_dir, "hpc_core"), remote_path="/root/hpc_core")
)


@app.function(image=image, gpu="L4", timeout=300)
def run_matrix_benchmark():
    import sys
    sys.path.insert(0, "/root")
    from hpc_core.kernel_bench import run_kernel_matrix
    run_kernel_matrix()


@app.local_entrypoint()
def main():
    run_matrix_benchmark.remote()
