"""Part 1: download Qwen/Qwen3-30B-A3B (bf16, 61.1 GB in 16 safetensors files) into OUR volume `qwen3-hf-cache`, next
to Qwen3-4B / 0.6B, on a CPU container (no GPU). Pinned to one revision so every later run uses the same weights.

  modal run part1/download_moe.py          # plan only
  modal run part1/download_moe.py --yes    # CPU container, hard cap 40 minutes
  modal run part1/download_moe.py --pin    # CPU, seconds: write refs/main -> REVISION in the cache
  modal run part1/download_moe.py --complete   # CPU, seconds: fetch the small non-weight files, then check offline

Downloading a pinned revision does not write the cache's refs/main, and the servers run with HF_HUB_OFFLINE=1 and look
the model up by name, i.e. by refs/main: --pin writes it (pointing at REVISION) so that "Qwen/Qwen3-30B-A3B" resolves
offline to exactly these weights.
"""
import modal

REPO, REVISION = "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
app = modal.App("final-download-moe")
hf_cache = modal.Volume.from_name("qwen3-hf-cache", create_if_missing=False)   # ours
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_transfer]", "transformers").env(
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


@app.function(image=image, cpu=1, volumes={"/hf_cache": hf_cache}, timeout=5 * 60)
def pin():
    import os
    root = "/hf_cache/hub/models--" + REPO.replace("/", "--")
    assert os.path.isdir(f"{root}/snapshots/{REVISION}"), "the pinned snapshot is not in the cache"
    os.makedirs(f"{root}/refs", exist_ok=True)
    open(f"{root}/refs/main", "w").write(REVISION)
    hf_cache.commit()
    print(f"{root}/refs/main -> {REVISION}")


CHECK = """
import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
path = snapshot_download(sys.argv[1], local_files_only=True)
tok = AutoTokenizer.from_pretrained(sys.argv[1])
print("offline OK:", path, "vocab", len(tok))
"""


@app.function(image=image, cpu=1, volumes={"/hf_cache": hf_cache}, timeout=5 * 60)
def complete():
    """The weights download skipped .gitattributes / LICENSE / README.md. Newer huggingface_hub (used by AIPerf to load
    the tokenizer by repo name, with HF_HUB_OFFLINE=1) refuses such a snapshot as incomplete. Fetch those few small
    files at the same REVISION (nothing already present is downloaded again), then check, offline and in a fresh
    process, exactly what AIPerf does: resolve the model by name and load its tokenizer."""
    import os
    import subprocess
    from huggingface_hub import snapshot_download
    path = snapshot_download(REPO, revision=REVISION, ignore_patterns=["*.safetensors"])
    hf_cache.commit()
    assert path.endswith(REVISION), path
    print("files:", " ".join(sorted(os.listdir(path))))
    subprocess.run(["python", "-c", CHECK, REPO], check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})


@app.local_entrypoint()
def main(yes: bool = False, pin: bool = False, complete: bool = False):
    if pin:
        globals()["pin"].remote()
        return
    if complete:
        globals()["complete"].remote()
        return
    print(f"PLAN: download {REPO}@{REVISION[:8]} (61.1 GB) into volume qwen3-hf-cache on a CPU container "
          f"(4 cores, no GPU), hard cap 40 min")
    if not yes:
        print("Dry run only. Pass --yes to start.")
        return
    download.remote()
