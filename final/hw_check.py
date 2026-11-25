"""Can the class Modal workspace give us what Parts 1-3 and 5 need? 2x H200 in one container (Part 1 asks for an
H200; Parts 2, 3, 5 need several GPUs), peer access between them, and NCCL in our CUDA image.

  modal run hw_check.py          # CPU only: builds the image, checks that NCCL is in it
  modal run hw_check.py --yes    # 2x H200 for about a minute: nvidia-smi, topology, peer access

Nothing is computed on the GPUs: this only asks for them and lists what we got.
"""
import modal

IMAGE = "nvidia/cuda:13.5.0-devel-ubuntu24.04"
app = modal.App("final-hw-check")
image = (
    modal.Image.from_registry(IMAGE, add_python="3.12")
    # NCCL (all-to-all for Parts 2/3) must be in the image: fail the build (CPU) if it is not
    .run_commands("ls /usr/lib/x86_64-linux-gnu/libnccl.so* /usr/include/nccl.h && "
                  "grep -m1 NCCL_MAJOR /usr/include/nccl.h && grep -m1 NCCL_MINOR /usr/include/nccl.h")
)
PRICE_H200 = 4.54   # $/hour per GPU, approximate; check modal.com/pricing


@app.function(image=image, gpu="H200:2", timeout=5 * 60)   # hard cap 5 minutes
def check():
    import subprocess
    for cmd in (["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                ["nvidia-smi", "topo", "-m"],
                ["nvidia-smi", "topo", "-p2p", "r"]):
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd)


@app.local_entrypoint()
def main(yes: bool = False):
    print(f"PLAN: 2x H200 for ~1 minute (hard cap 5 min) -> at most ~${2 * 5 / 60 * PRICE_H200:.2f} of shared credit")
    if not yes:
        print("Image built (CPU only), NCCL found in it. Pass --yes to ask for the GPUs.")
        return
    check.remote()
