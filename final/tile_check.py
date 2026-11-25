"""CPU-only check: does a public NVIDIA CUDA image compile CUDA Tile C++ (`nvcc --enable-tile`)?

No GPU is requested (only compiling), so the cost is a few cents of CPU time.
  modal run tile_check.py
"""
import modal

app = modal.App("final-tile-check")
IMAGE = "nvidia/cuda:13.5.0-devel-ubuntu24.04"
image = modal.Image.from_registry(IMAGE, add_python="3.12")

ADD_CU = r'''
#include <cstdio>
#include <cuda_runtime.h>
#include "cuda_tile.h"
namespace ct = cuda::tiles;
__tile_global__ void add(const int* a, const int* b, int* c) {
    auto i = ct::bid().x;
    c[i] = ct::add(a[i], b[i]);
}
int main() {
    int h_a[] = {1, 2, 3, 4}, h_b[] = {10, 20, 30, 40}, h_c[4];
    int *a, *b, *c;
    cudaMalloc(&a, sizeof(h_a)); cudaMalloc(&b, sizeof(h_b)); cudaMalloc(&c, sizeof(h_c));
    cudaMemcpy(a, h_a, sizeof(h_a), cudaMemcpyHostToDevice);
    cudaMemcpy(b, h_b, sizeof(h_b), cudaMemcpyHostToDevice);
    add<<<4, 1>>>(a, b, c);
    cudaMemcpy(h_c, c, sizeof(h_c), cudaMemcpyDeviceToHost);
    for (int x : h_c) std::printf("%d ", x);
    std::printf("\n");
}
'''


@app.function(image=image, timeout=600)
def check():
    import subprocess
    open("/tmp/add.cu", "w").write(ADD_CU)
    print(subprocess.run("nvcc --version | tail -2", shell=True, capture_output=True, text=True).stdout)
    print("cuda_tile.h:", subprocess.run("find /usr/local -name cuda_tile.h | head -3", shell=True,
                                         capture_output=True, text=True).stdout or "NOT FOUND")
    for arch in ("sm_80", "sm_90"):
        r = subprocess.run(f"cd /tmp && nvcc --enable-tile -std=c++20 -arch={arch} add.cu -o add_{arch}",
                           shell=True, capture_output=True, text=True)
        print(f"--- compile {arch}: exit {r.returncode}\n{(r.stdout + r.stderr)[-1500:]}")


@app.local_entrypoint()
def main():
    print(f"CPU-only compile check in {IMAGE} (no GPU, a few cents)")
    check.remote()
