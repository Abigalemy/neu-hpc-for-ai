import modal

app = modal.App("gpu-connectivity-check")

image = modal.Image.debian_slim(python_version="3.13").pip_install("torch")

@app.function(image=image, gpu="T4")
def verify_remote_gpu():
    import torch
    is_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if is_available else "None"
    cuda_version = torch.version.cuda
    return {
        "cuda_available": is_available,
        "device_name": device_name,
        "cuda_version": cuda_version,
    }

@app.local_entrypoint()
def main():
    result = verify_remote_gpu.remote()
    print("Remote execution result:")
    print("  CUDA available:", result["cuda_available"])
    print("  GPU Model:     ", result["device_name"])
    print("  CUDA Version:  ", result["cuda_version"])