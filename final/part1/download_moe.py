"""Part 1: download Qwen/Qwen3-30B-A3B (bf16, 61.1 GB in 16 safetensors files) into OUR volume `qwen3-hf-cache`, next
to Qwen3-4B / 0.6B, on a CPU container (no GPU). Pinned to one revision so every later run uses the same weights.

  modal run part1/download_moe.py          # plan only
  modal run part1/download_moe.py --yes    # CPU container, hard cap 40 minutes
"""
import modal

REPO, REVISION = "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
app = modal.App("final-download-moe")
hf_cache = modal.Volume.from_name("qwen3-hf-cache", create_if_missing=False)   # ours
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_transfer]").env(
    {"HF_HOME": "/hf_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})


@app.function(image=image, cpu=4, memory=8192, volumes={"/hf_cache": hf_cache}, timeout=40 * 60)
def download():
    import os
    import time
    from huggingface_hub import snapshot_download
    t0 = time.time()
    path = snapshot_download(REPO, revision=REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt"])
    hf_cache.commit()
    files = sorted(os.listdir(path))
    size = sum(os.path.getsize(os.path.join(path, f)) for f in files)
    print(f"{REPO}@{REVISION[:8]}: {len(files)} files, {size / 1e9:.1f} GB in {time.time() - t0:.0f} s -> {path}")
    print(" ".join(files))


@app.local_entrypoint()
def main(yes: bool = False):
    print(f"PLAN: download {REPO}@{REVISION[:8]} (61.1 GB) into volume qwen3-hf-cache on a CPU container "
          f"(4 cores, no GPU), hard cap 40 min")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    download.remote()
