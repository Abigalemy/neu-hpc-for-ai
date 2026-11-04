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

The results below were produced on **Modal** (1x H100 80GB per config, Qwen3-4B, bf16) with `modal_bench.py`:

```bash
modal run --detach modal_bench.py --final --yes         # vllm + ours_{paged,contiguous,nocache}, trimmed sweeps
modal run modal_bench.py --check-only --yes             # real-weights correctness vs HuggingFace (L4)
modal run modal_bench.py --profile-only --yes           # decode-step profile (L4)
```

`bench/run_experiments.sh` + `bench/slurm_run.sbatch` are an equivalent path for a Slurm cluster (not used for
the reported numbers). All servers get the same 4096-token context limit and max 64 concurrent sequences; ours also get
an 8 GB KV-cache budget. AIPerf runs with `ignore_eos` (exact output lengths), streaming, chat endpoint, `temperature=0`.

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

Setup: Qwen3-4B, bf16, 1x H100 80GB (Modal), AIPerf 'chat' endpoint with streaming. Raw AIPerf exports are in
`results/<config>/<workload>_<param>/profile_export_aiperf.json`; `results/summary.md` has the full table (p50/p99).
Tables show **output tokens/s** (higher is better) and **p50 inter-token latency, ITL, in ms** (lower is better).
Every run below had 0% request errors.

### 1. With vs without KV cache + PagedAttention (ours)

| Workload | no cache: tok/s / ITL | contiguous KV: tok/s / ITL | **paged KV: tok/s / ITL** |
|---|---|---|---|
| conc 1, 128 in / 128 out | 47 / 21.4 | 39 / 25.7 | 43 / 23.5 |
| conc 4, 256 / 256 | 105 / 38.2 | 148 / 26.7 | **159** / 24.9 |
| conc 16, 256 / 256 | 115 / **139.2** | 335 / 28.1 | **616** / 25.0 |
| conc 8, 2048 in / 32 out | 22 / 355.1 | 202 / 32.9 | **208** / 30.2 |
| conc 64, 256 / 256 | not run | 467 / 27.2 | **2112** / 28.6 |
| conc 64, 64 in / 1024 out | not run | 473 / 26.7 | **1845** / 30.0 |

* **KV cache:** without a cache every step re-runs the whole context of every running sequence. Over its whole run the no-cache
  engine pushed 24.3M tokens through the model for 297 requests, versus 1.4M prefill tokens for the paged engine's 1849
  requests (engine counters in `engine_stats.json`). At conc 16 this shows up as ITL of 139 ms (5.6x worse) and 5.4x lower throughput. At conc 1 the three are equal because the engine is CPU-bound (below), so
  the cache's compute saving does not show up until the GPU has real work. The no-cache engine was only run on short,
  low-concurrency workloads because it would not finish the others in reasonable time.
* **PagedAttention vs contiguous:** with the same 8 GB KV budget, contiguous slabs reserve 4096 tokens per sequence, so only
  **14 sequences** fit (average decode batch 11.3), versus a paged pool of 3640 blocks (58k tokens, average decode batch 22.3).
  At conc 64 the paged server is **4.5x** faster (2112 vs 467 tok/s) and contiguous has TTFT p50 of 22.8 s because requests
  queue for a slot. Under a 128-request flood (`pressure`) paged completes; contiguous did not finish within its 40-minute cap
  (no `pressure` result, other workloads were completed).
* Paged pays for its flexibility under memory pressure: in `pressure_c128` (conc 128, 1024 in / 512 out) the pool ran out and
  **121 sequences were preempted and recomputed**, giving TTFT p50 of 27.8 s.

### 2. Ours (paged, continuous batching) vs vLLM

| Workload | vLLM: tok/s / ITL / TTFT | ours: tok/s / ITL / TTFT | vLLM / ours throughput |
|---|---|---|---|
| conc 1, 128 / 128 | 223 / 4.3 / 27 | 42 / 23.5 / 25 | 5.3x |
| conc 4, 256 / 256 | 870 / 4.4 / 47 | 159 / 24.9 / 74 | 5.5x |
| conc 16, 256 / 256 | 3115 / 4.8 / 95 | 616 / 25.0 / 157 | 5.1x |
| conc 64, 256 / 256 | 8833 / 6.2 / 269 | 2112 / 28.6 / 343 | 4.2x |
| conc 64, 64 / 1024 | 9390 / 6.7 / 126 | 1845 / 30.0 / 458 | 5.1x |
| conc 32, 2048 / 32 | 877 / 22.3 / 460 | 373 / 63.5 / 478 | 2.4x |
| Poisson 8 req/s, variable lengths | 1783 / 5.4 / 34 | 1053 / 38.4 / 74 | 1.7x |
| pressure, conc 128 | 6064 / 10.1 / 5580 | 1354 / 31.3 / 27849 | 4.5x |

(TTFT = p50 time to first token in ms.) vLLM is 4 to 5x faster on decode-dominated workloads and much closer (1.7x to 2.4x)
on prefill-heavy and open-loop variable-length traffic, where prefill compute (not launch overhead) dominates.

### 3. Root cause of the remaining gap: CPU launch overhead

`profile_decode` (L4, Qwen3-0.6B, paged) shows decode is **launch-bound, not compute-bound**:

| decode batch | engine step | GPU actually busy | kernels per forward |
|---|---|---|---|
| 1 | 36.8 ms | 7.8 ms (21%) | 1694 |
| 8 | 47.6 ms | 9.6 ms | 1694 |
| 32 | 52.1 ms | 15.8 ms | 1694 |

Evidence: the time to merely *issue* a forward without waiting (36.4 ms) equals the synchronized forward time (35.1 ms), and
batch-building costs only 0.25 to 0.5 ms. Each step launches about 1700 small kernels from Python; on a fast GPU the GPU idles waiting
for the CPU. Consistent with this, on the H100 our ITL is almost constant (23.5 to 30 ms) from conc 1 to 64 while vLLM,
which captures decode into CUDA graphs, is at 4.3 to 6.7 ms. This is also why batching helps us so much: the fixed per-step
cost is amortised over more sequences (throughput 42 -> 2112 tok/s from conc 1 to 64 at nearly constant ITL).

**Not implemented (would be the next optimisation):** CUDA-graph capture of the decode step (static buffers, batch-size buckets),
fused RMSNorm/RoPE kernels to cut the kernel count, and a block-table-aware attention kernel instead of the PyTorch gather
(which copies K/V each step).

### 4. Other bottlenecks seen

* **Queueing, not compute, dominates TTFT at high load.** Contiguous at conc 64 and paged at conc 128 have 20+ s TTFT because
  admission is limited by KV memory (slots / blocks), not GPU speed.
* **Prefill is serialized with decode.** Each step prefills newly admitted sequences before decoding, so ITL p99 spikes when many
  requests arrive (mixed Poisson 8 req/s: ITL p50 38 ms vs 25 ms at steady state; `prefill_heavy_c32` ITL 63 ms). Chunked prefill
  would smooth this.
* **Preemption by recompute is expensive** (121 in `pressure`); swapping or a smarter victim policy would help.

### Correctness

`test_correctness.py` (tiny random Qwen3 vs HuggingFace, CPU, 9 tests incl. preemption and block-leak checks) and
`test_server.py` (HTTP + streaming) pass. With the real Qwen3-0.6B weights on a GPU, greedy output of 4 concurrent requests is
token-for-token identical to HuggingFace in float32 for all three cache modes (`modal_bench.py --check-only`).

### Caveats (please read before comparing numbers)

* Single run per point, no repeats, so small differences (a few %) are within noise. Each point is 8 to 384 requests.
* `ours_contiguous` hit its 40-minute cap (no `pressure_c128` point) and `ours_nocache` was run only on low-concurrency workloads.
* vLLM runs in a different container with its own CUDA-graphs/torch-compile; it used `--no-enable-prefix-caching`, bf16, and its
  FlashInfer sampler disabled (no `nvcc` in the image). Its chat template had thinking enabled by default, ours disables thinking,
  so prompts differ by a few tokens (outputs are length-forced with `ignore_eos`, so throughput is comparable).
* Our server is far simpler than vLLM: PyTorch ops only, no CUDA graphs, no prefix caching, no chunked prefill.
* `ours_static` (static-batching ablation) is implemented (`--batching static`) and unit-tested but was **not benchmarked**, to save
  shared GPU credit.
