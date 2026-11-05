# Week 07 & 08: Qwen3 Inference Server with KV Cache, PagedAttention and Continuous Batching

A from-scratch Qwen3 inference server with an OpenAI-compatible API, intended for AIPerf comparison
against vLLM on Qwen3-4B / H100. The model, engine and cache implementations are shared from
`../hpc_core/`. CUDA inference uses shared cuTile SwiGLU and causal/GQA FlashAttention kernels; the
remaining model operations use PyTorch.

## Layout

| File | What it does |
|---|---|
| `qwen3_model.py` | Compatibility import for the shared Qwen3 decoder and HF safetensors loader. |
| `kv_cache.py` | Compatibility import for three shared KV backends: **none**, **contiguous**, **paged**. |
| `engine.py` | Compatibility import for the shared scheduler and inference engine. |
| `../hpc_core/kernels/cutile_ops.py` | CUDA Tile fused SwiGLU invoked by the CUDA model path. |
| `../hpc_core/kernels/cutile_attention.py` | CUDA Tile causal/GQA attention used for prefill and cache decode. |
| `_qwen3_cpp_tokenizer` | C++ `tokenizers-cpp` extension used to encode completion prompts and rendered chat prompts. |
| `server.py` | FastAPI server: `/v1/completions`, `/v1/chat/completions` (SSE streaming), `/health`, `/metrics_json`. |
| `test_correctness.py` | Tiny random Qwen3 vs HuggingFace: logits, greedy generation, preemption, no leaked blocks. |
| `bench/` | AIPerf workload suite, experiment orchestrator, Slurm script, results summarizer. |

## Design

**Model.** Tokens are passed as one flat varlen tensor, so prefill of several sequences and batched
decode use the same code. The model calls `batch.attend(layer, q, k, v)`; the batch object decides
how K/V are stored and read.

**CUDA Tile.** CUDA inference uses online-softmax FlashAttention with causal masking and GQA, plus a
fused SwiGLU kernel. L4 checks passed against PyTorch for FP16/BF16 attention and for a small model
using none, contiguous, and paged cache modes. Real Qwen3-4B greedy outputs matched Hugging Face for
four prompts under no-cache, contiguous, and paged modes in a capped H100 correctness run. The original
H100 AIPerf matrix below measures this CUDA Tile path before the direct block-table decode kernel was added;
final-code measurements are reported in the follow-up sections.

The L4 kernel and small-model checks used the shared Modal workspace and are recorded in the app's
run history. Earlier pre-Tile AIPerf results are not included in the current performance tables.

**KV cache.** Without a cache each step recomputes the entire context (O(n) work per generated
token). With a cache, prefill writes K/V once and each decode step only processes the newest token.

**PagedAttention.** K/V live in a pool of fixed-size blocks (default 16 tokens). A sequence owns a
*block table* (logical block -> physical block) that grows one block at a time. Decode now uses a
CUDA Tile kernel that reads K/V directly from the physical blocks listed in that table and applies
online softmax without materializing a gathered K/V tensor. If the pool runs out, the most recently
admitted sequence is preempted (blocks freed, re-queued, recomputed later). This is an educational
single-token decode kernel: one CTA handles one sequence/query head and streams blocks serially; it
does not use split-KV or prefix sharing.

**Continuous batching.** Every engine step (1) admits waiting requests if a slot / enough blocks exist,
(2) prefills them, (3) decodes one token for *all* running sequences. Finished sequences free their
cache immediately so waiting requests join at the next step. `--batching static` (admit only when the
batch is empty) is provided as the ablation baseline.

## Run

```bash
pip install -r requirements.txt
# Build the C++ tokenizer extension into this directory; see ../week_04_05/README.md for prerequisites.
cmake -S ../week_04_05/tokenizer_cpp -B build/tokenizer_cpp -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DQWEN3_TOKENIZER_PYTHON_OUTPUT_DIR="$PWD"
cmake --build build/tokenizer_cpp --target _qwen3_cpp_tokenizer --parallel
python test_correctness.py                       # CPU, no model download needed
python server.py --model Qwen/Qwen3-0.6B --cache-mode paged --batching continuous --port 8000
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

The server uses Hugging Face's Python tokenizer to render the Qwen3 chat template and decode generated tokens. Actual text-to-token encoding for both endpoints goes through the C++ extension. The broad H100 AIPerf matrix below predates both C++ tokenizer integration and the fused block-table decode kernel. The tokenizer separately passed exact ID parity against Hugging Face on five representative probes, including a rendered chat-template string. The targeted H100 ablations later in this README use the final request path and kernel; the broad vLLM matrix was not rerun on that final code.

Server flags: `--cache-mode {none,contiguous,paged}`, `--batching {continuous,static}`, `--max-num-seqs`,
`--max-model-len`, `--block-size`, `--kv-cache-gb` (same KV budget for contiguous and paged),
`--dtype`. Extra request fields understood: `ignore_eos`, `min_tokens`, `top_k`, `chat_template_kwargs`.

## Benchmarking

The results below were produced on **Modal** (1x H100 80GB per config, Qwen3-4B, bf16) with `modal_bench.py`:

```bash
modal run --detach modal_bench.py --final --yes         # vllm + ours_{paged,contiguous,nocache}, trimmed sweeps
modal run modal_bench.py --check-only --model Qwen/Qwen3-4B --yes  # real-weights correctness (H100, 15-min cap)
modal run modal_bench.py --profile-only --yes           # matched gather vs block-table profile (H100)
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

### Corrected Tile path smoke check

A short AIPerf smoke run was completed after the CUDA Tile integration: Qwen3-4B on H100, one
baseline workload (128 input / 128 output tokens, concurrency 1, 8 measured requests), comparing
`ours_paged` with vLLM. Both returned zero request errors. At the time, the Tile service measured 26.7
output tokens/s and vLLM 228.2 output tokens/s in this small run. Raw outputs are in
`results/cutile_smoke/`; this is a pipeline smoke check, not a complete performance sweep or a
stable performance claim. The original full matrix, before the direct block-table decode kernel, is reported in the historical Results section below.
Raw AIPerf exports and this run's two-row summary are in `results/cutile_smoke/summary.md`.

A decode profile of Qwen3-4B on L4 measured about 1,994 CUDA kernels and 35.4 ms of GPU time per
batch-1 forward (36.9 ms synchronized wall time). At batch sizes 8 and 32, GPU time rose to 42.1 ms
and 60.2 ms per forward. This points to the eager per-layer inference path and its many launches as
a performance bottleneck; the profile is diagnostic, not an H100 performance result. Raw measurements
are in `results/cutile_smoke/decode_profile_l4.json`.

## Results: original CUDA Tile H100 matrix (before direct block-table kernel, single run)

This run used Qwen3-4B BF16 on one Modal H100 per configuration, the AIPerf streaming chat endpoint,
and `ignore_eos` with temperature 0. Each listed point is one run (no repeats). The full raw exports
are in [`results/cutile_final/`](results/cutile_final/); the generated full metrics table is
[`results/cutile_final/summary.md`](results/cutile_final/summary.md), and the profile-by-profile
completion/error audit is [`results/cutile_final/validation.md`](results/cutile_final/validation.md).
Modal run:
[ap-v9yhmH32icaaC7X65dKUQj](https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-v9yhmH32icaaC7X65dKUQj).

The completed matrix contains 13 AIPerf profiles for `vllm`, 13 for `ours_paged`, 13 for
`ours_contiguous` (12 in the initial run, plus a targeted `pressure_c128` follow-up), and 5 for
`ours_nocache`. The initial all-config run reached the contiguous 40-minute cap before pressure; the
follow-up completed that missing comparison in 331.67 seconds of profiling time. The no-cache sweep
was intentionally trimmed to baseline, balanced concurrency 4/16, and prefill-heavy concurrency 1/8.
Every profile with an AIPerf JSON export reported 0% request errors. One request in
`ours_paged/decode_heavy_c64` has an AIPerf token-count discrepancy among 256 requests: AIPerf's
text-side count is 898 tokens while server-reported completion usage is 1024. The server counter
indicates that generation reached the requested length; the available evidence points to
detokenization/re-tokenization accounting, not a confirmed early stop. The text-side count is 12.3%
lower for that request, while the profile-wide mean is 0.4% below the server-reported total. All other
exported profiles reported zero AIPerf output-length mismatch warnings.

This is the original one-run matrix, collected before the direct block-table decode kernel. Its paged
column uses the old PyTorch K/V gather path; its vLLM comparison has not been rerun against the final kernel.

Selected metrics below are p50 TTFT and p50 ITL in milliseconds; output throughput is the AIPerf
average in tokens/second. In this run, throughput uses AIPerf's text-side token count; the benchmark
command now asks AIPerf to use server-reported token counts on future runs. Higher throughput and
lower latency are better. A dash means that workload was not run for that configuration.

| Workload | vLLM tok/s / TTFT / ITL | Paged tok/s / TTFT / ITL | Contiguous tok/s / TTFT / ITL | No-cache tok/s / TTFT / ITL |
|---|---:|---:|---:|---:|
| Baseline, conc 1, 128 in / 128 out | 223.9 / 26.4 / 4.3 | 62.1 / 17.3 / 16.1 | 43.1 / 24.1 / 23.2 | 51.3 / 20.4 / 19.7 |
| Balanced, conc 4, 256 / 256 | 868.6 / 48.4 / 4.4 | 202.6 / 56.0 / 19.6 | 176.1 / 60.9 / 22.1 | 102.8 / 53.4 / 39.0 |
| Balanced, conc 16, 256 / 256 | 3131.8 / 93.2 / 4.7 | 494.1 / 140.4 / 31.8 | 425.7 / 84.8 / 32.2 | 113.1 / 130.0 / 141.5 |
| Balanced, conc 64, 256 / 256 | 8868.1 / 263.2 / 6.1 | 817.4 / 290.0 / 77.2 | 373.0 / 32208.6 / 41.1 | — |
| Prefill-heavy, conc 32, 2048 / 32 | 883.3 / 463.1 / 22.1 | 291.2 / 541.4 / 88.7 | 250.6 / 2107.6 / 50.1 | — |
| Decode-heavy, conc 32, 64 / 1024 | 5584.2 / 79.1 / 5.6 | 677.9 / 102.2 / 47.2 | 367.8 / 40690.2 / 32.9 | — |
| Decode-heavy, conc 64, 64 / 1024 | 9399.0 / 142.2 / 6.6 | 781.8 / 336.0 / 78.1* | 418.4 / 102061.6 / 32.5 | — |
| Mixed Poisson, 8 req/s | 1781.2 / 36.5 / 5.4 | 649.8 / 105.1 / 78.7 | 393.2 / 24827.7 / 33.7 | — |
| Pressure, conc 128, 1024 / 512 | 6062.2 / 5534.0 / 10.0 | 675.0 / 56390.8 / 65.0 | 395.2 / 142332.4 / 34.4 | — |

*The paged decode-heavy conc 64 point contains the single 898-vs-1024 token-count discrepancy noted above.*

### Interpretation

* The in-house CUDA Tile server is slower than vLLM on every selected throughput comparison. At
  balanced concurrency 4/16/64, vLLM delivers about 4.4x/6.3x/10.9x the paged-server output
  throughput. This is a historical baseline and does not measure the final fused kernel against vLLM; it
  should not be presented as a current head-to-head comparison or as matching vLLM performance.
* The KV-cache ablation shows its benefit more clearly at higher concurrency: at balanced concurrency
  16, paged KV reaches 494.1 output tokens/s and 31.8 ms p50 ITL, compared with 113.1 tokens/s and
  141.5 ms for no cache. These are single-run measurements.
* The pressure workload completed for all three cache-enabled configurations. Under concurrency 128,
  contiguous KV had 14 slots and no preemptions, but p50 TTFT reached 142.3 seconds. Paged KV reached
  56.4 seconds p50 TTFT and 675 output tokens/s, versus 142.3 seconds and 395.2 tokens/s for
  contiguous. vLLM was much faster in this workload. The queueing result is consistent with the fixed
  contiguous-cache capacity limiting admission.
* The decode profile is diagnostic rather than a benchmark result: the L4 Qwen3-4B profile recorded
  about 1,994 CUDA kernels and 35.4 ms GPU busy time per batch-1 forward. This older L4 profile motivated
  investigating launch overhead and block-table-aware attention. The later H100 matched A/B profile and
  isolated attention microbenchmark below test parts of that hypothesis, but do not separate GEMM, attention,
  and launch time across a full model step.

### Correctness and measurement caveats

* At the time of the original matrix, 10 CPU checks passed, including an HTTP endpoint smoke test. On the
  final code, `pytest week_07_08/test_correctness.py` passed 9 tests (logits, six generation-mode combinations,
  preemption, and slot release). L4 CUDA Tile checks passed against PyTorch; the real Qwen3-4B H100 greedy
  check matched Hugging Face for four prompts in no-cache, contiguous, and paged modes.
* AIPerf marked all exported profiles complete with zero request errors. A single output-length
  warning in paged `decode_heavy_c64` was a discrepancy between text re-tokenization (898) and
  server usage (1024), not evidence that the engine stopped early. The benchmark command now asks
  AIPerf to use server-reported token counts for future runs.
* Optional GPU/server telemetry missed heartbeats during the run, so telemetry summaries are
  incomplete. AIPerf request metrics remain available, but do not make claims based on missing GPU
  telemetry.
* Every point in this historical broad matrix is a single run, so small differences may be noise. The
  targeted final-code comparisons later have three runs each, but do not replace the broad matrix. vLLM and
  the in-house server use different runtime stacks and chat templates; prompt lengths can differ slightly.
  Both use BF16, forced output lengths, and the same AIPerf workload shapes.
* The final implementation uses CUDA Tile for attention/SwiGLU and PyTorch for the remaining model
  operations. Paged decode now uses the direct block-table kernel documented below. CUDA graphs and chunked
  prefill are not implemented. This original vLLM matrix predates the direct kernel; there is no final-code
  broad vLLM comparison.
* `ours_static` was not included in this historical broad matrix; it appears in the repeated final-code
  batching ablation below. These measurements do not prove correctness for every possible workload.

## Final block-table kernel: profile and repeated ablations

The results above are the original CUDA Tile baseline, captured before replacing the paged decode
K/V gather with the direct block-table kernel. The following measurements use the final code in this
submission. H100 same-process profiling compares the prior gather + Tile-attention path and the new
block-table path on identical model weights and inputs; AIPerf ablations compare continuous admission,
static admission, and no-cache decode. Each AIPerf condition was run three times in separate H100
containers, with 8 warmups and identical request shapes. Raw exports and per-run summaries are in
[`results/grade_ablation_repeat1/`](results/grade_ablation_repeat1/),
[`results/grade_ablation_repeat2/`](results/grade_ablation_repeat2/), and
[`results/grade_ablation_repeat3/`](results/grade_ablation_repeat3/). The H100 profile is
[`results/grade_ablation_repeat1/h100_profile.json`](results/grade_ablation_repeat1/h100_profile.json).

The block-table kernel removes the intermediate gathered K/V tensors. On the same H100, its
whole-model forward CUDA-event time versus the gather reference was:

| Batch | Gather reference (ms) | Direct block table (ms) | Change | Greedy logits match |
|---:|---:|---:|---:|:---:|
| 1 | 20.902 | 18.574 | -11% | yes |
| 8 | 35.882 | 21.856 | -39% | yes |
| 32 | 75.333 | 26.110 | -65% | yes |

These are synchronized whole-forward A/B measurements, not isolated attention-kernel latency. The
H100 profiler recorded about 1,959 kernels per forward. CPU issue time and kernel count show that
Python/PyTorch launch overhead remains material; missing optional Modal GPU telemetry means this run
cannot provide a trustworthy GPU-utilization breakdown. The experiment demonstrates that the fused
path improves this matched decode case, but does not prove that K/V gathering was the sole cause of
end-to-end latency.

The repeated AIPerf results below are computed from three independent runs (mean and sample standard
deviation). They cover balanced concurrency 16 (256 input / 256 output), mixed Poisson 8 req/s, and
decode-heavy concurrency 8 (64 input / 1024 output). Each run had zero request errors. The complete
per-run metrics are in the linked raw directories above and in the
[aggregate summary](results/grade_ablation_repeated_summary.md).

| Workload | Policy | Output tok/s (mean ± SD) | TTFT p50 ms (mean ± SD) | ITL p50 ms (mean ± SD) |
|---|---|---:|---:|---:|
| Balanced c16 | Paged + continuous | 716.8 ± 137.0 | 149.2 ± 10.5 | 20.0 ± 4.9 |
| Balanced c16 | Paged + static | 424.8 ± 81.8 | 4,793.6 ± 745.0 | 19.3 ± 4.1 |
| Mixed Poisson 8 req/s | Paged + continuous | 781.3 ± 35.6 | 54.7 ± 10.5 | 38.9 ± 1.8 |
| Mixed Poisson 8 req/s | Paged + static | 343.6 ± 62.3 | 17,481.8 ± 4,149.1 | 19.9 ± 3.7 |
| Decode-heavy c8 | Paged + continuous | 434.5 ± 85.8 | 66.7 ± 15.6 | 18.9 ± 4.1 |
| Decode-heavy c8 | No KV cache | 74.4 ± 0.5 | 55.0 ± 12.7 | 107.5 ± 0.8 |

The repeated runs show the same qualitative pattern: continuous admission reduces queueing delay
for these open-loop and concurrent workloads, while KV caching substantially reduces per-token
decode cost on long outputs. Run-to-run H100 variation is visible, including larger relative spread for
balanced c16 throughput and decode-heavy ITL; treat those means as directional rather than precise rankings.
The table reports mean and sample standard deviation, and individual results remain available. The
implementation is not claimed to match vLLM: the original broad matrix remains useful as a baseline, while the new kernel and these
ablation results document targeted improvements and their limits.

Correctness was checked with the CPU suite (9 tests), and a real Qwen3-4B H100 greedy-generation
comparison against Hugging Face matched for four prompts in no-cache, contiguous, and paged modes.


## Baseline anomaly and gather microbenchmark

To revisit the single-run result where contiguous KV appeared slower than no-cache, the final code
was measured on one H100 using the original baseline shape (concurrency 1, 128 input / 128 output
tokens), with 8 warmups and 16 measured requests per condition. Three fresh server processes per
cache mode were run sequentially; all nine profiles completed with zero request errors. Values are
mean ± sample standard deviation across the three runs.

| Cache mode | Output tok/s | TTFT p50 (ms) | ITL p50 (ms) | E2E p50 (ms) |
|---|---:|---:|---:|---:|
| Contiguous | 45.80 ± 0.47 | 22.75 ± 0.19 | 21.79 ± 0.23 | 2,790.6 ± 29.7 |
| No cache | 49.91 ± 1.76 | 22.01 ± 0.83 | 19.99 ± 0.69 | 2,560.2 ± 89.0 |
| Paged + direct block-table kernel | 49.66 ± 1.11 | 23.08 ± 0.57 | 20.08 ± 0.42 | 2,573.5 ± 54.0 |

The earlier single run showed 43.1 tok/s for contiguous, 51.3 tok/s for no-cache, and 62.1 tok/s for
paged. In the final-code repeat, contiguous was about 8.2% slower than no-cache, while fused paged and
no-cache were nearly tied. This points more specifically to the contiguous path (or measurement variation)
than to a general penalty from cached decode: contiguous still invokes Tile attention per sequence, while
paged uses the fused kernel. The synthetic microbenchmark supports the cost of the old per-sequence Tile
path, but does not directly profile contiguous model steps. The old paged point (62.1 tok/s) and final-code
mean (49.7 tok/s) are not a matched before/after AIPerf experiment: the old result is one run and the new
result is a three-run mean. The same-H100 whole-forward A/B found the fused path faster than its gather
reference, but it does not explain that cross-run difference. The contiguous-path cause remains unproven;
this short-context, single-request result does not show that KV caching generally reduces performance.

A separate H100 attention microbenchmark used synthetic BF16 Qwen3-shaped Q/K/V tensors (32 query
heads, 8 KV heads, head dimension 128, block size 16), shuffled physical block IDs, and contexts of
128 and 1024 tokens. CUDA events measured 30 iterations after 5 warmups. “Gather + Tile” recreates
the old block-table K/V materialization followed by one Tile attention call per sequence; “Direct”
uses the fused block-table kernel. All six cases passed output comparison (`atol=0.02`, `rtol=0.02`).

| Batch | Context | Gather only (µs) | Tile attention (µs) | Gather + Tile (µs) | Direct block table (µs) | Old / direct |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 21.1 | 49.9 | 75.3 | 20.2 | 3.73× |
| 1 | 1024 | 20.0 | 49.8 | 73.8 | 75.8 | 0.97× |
| 8 | 128 | 19.6 | 293.8 | 326.6 | 24.6 | 13.28× |
| 8 | 1024 | 29.8 | 296.2 | 326.2 | 151.7 | 2.15× |
| 32 | 128 | 20.0 | 1,137.3 | 1,165.8 | 89.5 | 13.03× |
| 32 | 1024 | 98.2 | 1,136.7 | 1,168.6 | 594.0 | 1.97× |

At batch 8–32, gather accounts for roughly 2–9% of the old combined time; repeated per-sequence Tile
attention calls cost more. The direct kernel is 2.0–13.3× faster in those tested cases, but is slightly
slower at batch 1 / context 1024. The data points to attention execution and launch structure as a
larger high-batch cost than copying K/V alone. Because fused paged is nearly tied with no-cache while
contiguous is slower, the short-context anomaly is more consistent with a contiguous-path issue or
measurement variation than a general cached-decode utilization problem. The benchmark does not isolate
the contiguous cache path, so this remains a hypothesis; optional GPU telemetry was unavailable.

Raw AIPerf exports, the full microbenchmark data, and the per-run audit are in
[`results/gap_study/`](results/gap_study/); the detailed report is
[`results/gap_study/summary.md`](results/gap_study/summary.md). Modal run:
[ap-hYtLK44pCoN3y5eA8QQEPD](https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-hYtLK44pCoN3y5eA8QQEPD).
