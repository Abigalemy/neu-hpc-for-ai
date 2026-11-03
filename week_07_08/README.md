# Week 07 & 08: Qwen3 Inference Server with KV Cache, PagedAttention and Continuous Batching

A from-scratch (pure PyTorch) Qwen3 inference server with an OpenAI-compatible API, benchmarked with
AIPerf against vLLM on Qwen3-4B / H100.

## Layout

| File | What it does |
|---|---|
| `qwen3_model.py` | Qwen3 decoder (GQA, per-head Q/K RMSNorm, RoPE, SwiGLU), HF safetensors loader. Cache-agnostic. |
| `kv_cache.py` | Three KV backends: **none**, **contiguous** (per-sequence slab), **paged** (block pool + block tables). |
| `engine.py` | Scheduler: continuous (or static) batching, prefill + batched decode, preemption by recompute. |
| `server.py` | FastAPI server: `/v1/completions`, `/v1/chat/completions` (SSE streaming), `/health`, `/metrics_json`. |
| `test_correctness.py` | Tiny random Qwen3 vs HuggingFace: logits, greedy generation, preemption, no leaked blocks. |
| `bench/` | AIPerf workload suite, experiment orchestrator, Slurm script, results summarizer. |

## Design

**Model.** Tokens are passed as one flat varlen tensor, so prefill of several sequences and batched
decode use the same code. The model calls `batch.attend(layer, q, k, v)`; the batch object decides
how K/V are stored and read.

**KV cache.** Without a cache each step recomputes the entire context (O(n) work per generated
token). With a cache, prefill writes K/V once and each decode step only processes the newest token.

**PagedAttention.** K/V live in a pool of fixed-size blocks (default 16 tokens). A sequence owns a
*block table* (logical block -> physical block) that grows one block at a time, so waste is at most
one partially filled block per sequence instead of `max_model_len` reserved up front. Decode gathers
K/V through the block tables. If the pool runs out, the most recently admitted sequence is
preempted (blocks freed, re-queued, recomputed later).
*Limitation:* the gather is done with PyTorch indexing, which copies K/V for each decode step, unlike
vLLM's fused kernel that reads blocks in place. Paged vs contiguous therefore differ mainly in
**memory capacity / concurrency**, not raw kernel speed.

**Continuous batching.** Every engine step (1) admits waiting requests if a slot / enough blocks exist,
(2) prefills them, (3) decodes one token for *all* running sequences. Finished sequences free their
cache immediately so waiting requests join at the next step. `--batching static` (admit only when the
batch is empty) is provided as the ablation baseline.

## Run

```bash
pip install -r requirements.txt
python test_correctness.py                       # CPU, no model download needed
python server.py --model Qwen/Qwen3-0.6B --cache-mode paged --batching continuous --port 8000
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

Server flags: `--cache-mode {none,contiguous,paged}`, `--batching {continuous,static}`, `--max-num-seqs`,
`--max-model-len`, `--block-size`, `--kv-cache-gb` (same KV budget for contiguous and paged),
`--dtype`. Extra request fields understood: `ignore_eos`, `min_tokens`, `top_k`, `chat_template_kwargs`.

## Benchmarking

`bench/run_experiments.sh MODEL KV_GB` launches each config, runs `bench/run_aiperf.sh`, and writes
`results/summary.md`. On a Slurm cluster, edit the partition/account in `bench/slurm_run.sbatch` and run
`sbatch bench/slurm_run.sbatch` from this directory.

Workloads (all use `ignore_eos` so output lengths are exact):

| Workload | Shape | Question it answers |
|---|---|---|
| baseline | conc 1, 128 in / 128 out | Latency floor without queueing |
| balanced | conc 1-64, 256 / 256 | Throughput vs latency curve |
| prefill_heavy | conc 1-32, 2048 / 32 | Is TTFT / prefill the bottleneck? |
| decode_heavy | conc 1-64, 64 / 1024 | Decode cost, KV growth, TPOT vs batch size |
| mixed_poisson | Poisson 2-16 req/s, variable lengths | Realistic open-loop traffic and queueing |
| pressure | conc 128, 1024 / 512 | KV memory pressure: contiguous vs paged, preemption |

## Results

_To be filled after running on the H100 (Qwen3-4B, bf16)._

### With vs without KV Cache + PagedAttention
TBD

### Ours vs vLLM
TBD

### Bottleneck analysis
TBD
