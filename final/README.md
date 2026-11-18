# Final Project: Qwen3 + MoE inference server in CUDA Tile C++

Goal (from the assignment): approach vLLM performance without PyTorch in the serving path.

## Status (2026-11-17)

| Part | What | Status |
|---|---|---|
| Part 0 | Qwen3 dense (4B) server within 10x of vLLM for any workload, incl. 128K input | **done**: every measured workload within 2.51x (below); paged KV cache (Week 7) in, its benchmark still to run |
| Part 2 | Two-GPU all-to-all mock | next |
| Part 1 | Single-GPU Qwen3-30B-A3B (MoE) | not started |
| Part 3, Part 5 | DP + EP over K GPUs; multi-process architecture | not started |
| Part 4 | Within 3x of vLLM; write-up | not started |

Weekly report item (single 128K-input sequence, Qwen3-4B, 1x H100, measured 2026-11-18 with the paged KV cache):
TTFT 23.37 s vs vLLM 10.45 s (2.24x), ITL 24.73 ms vs 10.43 ms (2.37x).

## What is here

- `kernels/`: CUDA Tile kernels (`__tile_global__`): RMSNorm, RoPE, linear (GEMM), fused gate/up + SiLU, embedding,
  argmax, residual add, KV-cache scatter, flash attention (causal GQA over the paged KV cache: prefill, and split-KV
  batched decode; both read the cache blocks in place through the block tables).
- `engine/`: safetensors loader (weights copied to the GPU once), RoPE tables incl. YaRN, the 36-layer forward pass
  with a paged KV cache (block pool, block tables, block allocator; `qwen3_engine.cuh`), and a C API (`qwen3_capi.cu` -> `libqwen3engine.so`).
- `server/`: OpenAI-compatible HTTP server (FastAPI, SSE streaming) with continuous batching; no PyTorch.
- `tests/`, `gen_cases.py`, `run_tests.py`, `schema.sql`: unit tests per module against HF reference outputs stored
  in SQLite; `e2e_test.py` (whole model vs HF), `batch_test.cu` (batched vs alone), `server/test_server.py`.
- `results/`: AIPerf summary exports of every run in the tables below, plus `summary.csv` and `comparison.csv`
  (`python tools/summarize_results.py` rebuilds both).
- `modal_*.py`, `*_check.py`, `*_sweep_modal.py`, `bench_modal.py`: Modal launchers (each prints its plan and cost
  cap and needs `--yes` to use a GPU).

## Final configuration (Part 0)

- Model: `Qwen/Qwen3-4B` bf16, snapshot `1cfa9a7208912126459214e8b04321603b3df60c` (the 8K check uses
  `Qwen/Qwen3-0.6B`, snapshot `c1899de289a04d12100db370d81485cdf75e47ca`).
- GPU: 1x H100. Tiles chosen by device: on Hopper GEMM 128x128x64 (64x128x64 for <= 64 tokens), fused gate/up 128x64x64 (64x64x64 for 17-64 tokens, 16x64x64 below),
  prefill attention 64x128 with split P (P = P_hi + P_lo in bf16, fp32 accumulator), decode attention split-KV
  (16-row query tile, 128 keys per tile, ~512 blocks in total, fp32 combine). On other GPUs the small tiles that were
  verified on the L4.
- 128K: YaRN factor 4 over the original 32768 context = 131072 tokens for input plus output.
- KV cache (PagedAttention, Week 7): per layer a pool of blocks of 128 tokens shared by all sequences; a sequence's
  block table maps its logical blocks to physical ones, and the attention kernels read the blocks in place (block size
  = the decode key tile). The server takes blocks on demand (first come, first served; a request starts when its
  prompt fits) and, when the pool is empty, preempts the newest sequence and recomputes it later (as in vLLM /
  nano-vllm). `--kv-tokens` sets the pool size, `--max-seqs` the most sequences decoded together.
- Server limits used: workload suite 64 sequences, context 4224 (measured before paging, pool = 64 x 4224); Week 7
  lecture workload 128 sequences, context 2048 (the lecture's vLLM limits; measured before paging); 128K: one sequence,
  pool 131072 tokens (measured after paging). Greedy only.
- Environment switches (for experiments only; none is set in the runs above): `QWEN3_TILES=small`,
  `QWEN3_ATTN=small`, `QWEN3_YARN=<factor>`, `QWEN3_DECODE_SPLITS=<n>` (fixed decode split count).

## Results: our server vs vLLM (Part 0)

Qwen3-4B bf16, 1x H100, both measured with AIPerf through their HTTP servers, same AIPerf settings and seed, greedy,
EOS ignored, prefix caching off for vLLM. 0 request errors in every run. Source: `results/comparison.csv`.

| Workload | in / out, concurrency | TTFT vLLM / ours (ms) | ratio | ITL vLLM / ours (ms) | ratio | output tok/s vLLM / ours |
|---|---|---|---|---|---|---|
| baseline | 128 / 128, 1 | 29.21 / 13.60 | 0.47x | 4.25 / 6.70 | 1.58x | 224 / 147 |
| long_out | 3072 / 1024, 1 | 63.73 / 74.78 | 1.17x | 4.44 / 7.76 | 1.75x | 222 / 128 |
| balanced_c16 | 256 / 256, 16 | 107.33 / 99.53 | 0.93x | 4.70 / 8.73 | 1.86x | 3126 / 1750 |
| balanced_c64 | 256 / 256, 64 | 270.13 / 387.44 | 1.43x | 6.35 / 15.01 | 2.36x | 8610 / 3873 |
| prefill_heavy | 2048 / 32, 8 | 204.67 / 209.26 | 1.02x | 6.99 / 14.76 | 2.11x | 601 / 382 |
| decode_heavy | 64 / 1024, 32 | 95.96 / 141.27 | 1.47x | 5.57 / 11.26 | 2.02x | 5639 / 2804 |
| 128K (YaRN x4), paged KV | 130816 / 128, 1 | 10446.84 / 23366.57 | 2.24x | 10.43 / 24.73 | 2.37x | |
| lecture_c100 (Week 7) | 1000 / 500, 100; 1000 requests | 453.13 / 1137.69 | 2.51x | 13.36 / 33.43 | 2.50x | 7015 / 2803 |

Notes: `lecture_c100` is the AIPerf example from the Week 7 lecture ("Performance Evaluation": concurrency 100, 1000
requests, 1000 in / 500 out, warm-up 16 requests at concurrency 8), with the lecture's vLLM limits
`--max-model-len 2048 --max-num-seqs 128 --gpu-memory-utilization 0.90`; our server gets the same limits (128 slots of
2048 tokens). TTFT p99 there: vLLM 1513 ms, ours 2242 ms. It is the baseline before PagedAttention: our 128 slots
reserve 128 x 2048 tokens of KV cache up front (2 x 2 bytes x 36 layers x 8 KV heads x 128 = 144 KiB per token, so
38.7 GB), whatever the requests actually use. The vLLM 128K run is from an earlier session with the same settings. TTFT ratios below 1 at low concurrency
are not a claim of faster prefill (vLLM's server has its own per-request overhead at small sizes). The 128K workload
is 130816 + 128 tokens because the chat template adds tokens and the YaRN context (131072) covers input and output.

## Correctness checks

- Unit tests per module against HF (SQLite cases, `modal_test_h100.py`): RMSNorm, RoPE, linear, MLP, embedding,
  argmax, attention (36 cases, prefill and decode, heads 32/8, 32/4, 16/8; bf16 bound derived from the arithmetic,
  |d| <= 2^-8 (max|V| + 2|e|)). All pass on the H100 with the final tiles; attention was rerun with the paged kernels
  (2026-11-18).
- Paged vs contiguous attention (`tests/paged_attention_test.cu`, H100, 2026-11-18): the same K/V once contiguous and
  once in a shuffled block pool whose other rows hold large stale values; prefill (64x128 split-P and 16x16 fp32-P,
  lengths 1-3000 incl. block edges 127/128/129, full and chunked) and batched decode (up to 100 sequences): 28/28
  outputs bit-identical. Speed (`tests/paged_bench_test.cu`): decode 500 us vs 497 us per layer at 128K, prefill
  16K 7.94 ms vs 7.65 ms, i.e. the block-table lookup costs nothing measurable.
- Whole model vs HF (`e2e_test.py`, rerun with the paged KV cache on 2026-11-18, identical numbers): Qwen3-0.6B with an 8192-token prompt and YaRN x4: top-1 60/64 (every mismatch an
  HF near-tie, margin <= 0.062), min cosine prefill 0.999789 / decode 0.999687 (criterion > 0.999), greedy tokens 16/16,
  RoPE tables bit-identical. Qwen3-4B, 43 positions: criterion met (see the development log).
- Batched decode vs decoding alone (`batch_check.py`, Qwen3-4B, 4 sequences of 27-3000 tokens, 16 steps): with a fixed
  split count, 64/64 logits rows bit-identical (rerun with the paged KV cache on 2026-11-18, same result).
- Server (`server/test_server.py`, 14 tests, fake engine with the engine's block accounting): stream format, exact
  lengths and usage, EOS handling, rejections, batching equals running alone, more requests than slots, disconnects,
  engine errors, UTF-8 split over tokens; paging: four short requests run together in a pool of one max_len sequence,
  preemption + recompute gives exactly the output of running alone with no block leaked, a request larger than the
  pool is rejected. AIPerf compatibility checked on a CPU against the fake-engine server.

## Known limitations

- Paged KV cache: implemented and verified (see Correctness), but the benchmark that shows its capacity gain over
  fixed per-sequence slots has not been run yet; the suite and lecture rows above were measured before paging.
- Stale rows of a reused block are not zeroed: masked keys get P = 0 exactly, which needs finite stale values (the pool
  is zeroed at start). Zeroing them in the attention loop cost 1.9x (decode) / 2.5x (prefill) and was removed
  (`tests/paged_bench_test.cu`).
- A prefill stalls the decode of the other running sequences (no chunked-prefill interleaving).
- Greedy only (`temperature` > 0 is rejected); `min_tokens` does not mask EOS like vLLM.
- Closest to the Part 4 limit (3x): the lecture workload at concurrency 100 (TTFT 2.51x, ITL 2.50x), long-context
  decode (2.37x) and ITL at concurrency 64 (2.36x); decode attention still dominates the 128K step.
- The old 64x128 attention kernel was less accurate for a single query (decode); the default decode path now uses
  the split-KV kernel, which meets the criterion. The root cause in the old kernel was not identified.

## How to reproduce

```bash
python server/test_server.py                       # local, no GPU
modal run modal_aipref.py --compat                 # CPU only: AIPerf vs the fake-engine server
modal run modal_test_h100.py --yes                 # unit tests, 1x H100
modal run e2e_test.py --yes --model Qwen/Qwen3-0.6B --gpu h100 --long 8192 --yarn 4 --ab
modal run batch_check.py --yes --splits ,4
modal run modal_vllm_aipref.py --yes --suite       # vLLM, workload suite
modal run modal_aipref.py --yes --suite            # ours, workload suite
modal run modal_vllm_aipref.py --yes --isl 130816 --osl 128 --yarn
modal run modal_aipref.py --yes --isl 130816 --osl 128 --yarn --requests 2
modal run modal_vllm_aipref.py --yes --lecture     # vLLM, Week 7 lecture workload
modal run modal_aipref.py --yes --lecture          # ours, Week 7 lecture workload
python tools/summarize_results.py                  # rebuild results/*.csv from results/aiperf
```

Weights come from the Modal volume `qwen3-hf-cache`; unit-test cases come from `modal run gen_cases.py`
(`test_cases.db`, not committed). Raw AIPerf output (per-request logs) stays on the volume `final-results`.

## Rules for myself
- Understand every line; Qwen3 only (head_dim 128), no generic code.
- Attention must be fused (FlashAttention) and must use the KV cache; weights loaded to the GPU once.
- Tile kernels (`__tile_global__`), not SIMT `__global__`.
- Correctness check before any benchmark; record the commit with each measurement.

## Development log (history)

Kept as the record of how the engine got here. Numbers in this section are superseded by the tables above, and some
switches mentioned here no longer exist (`QWEN3_DECODE_SPLIT` was removed once the split-KV decode became the only
decode path on Hopper). Engine-only measurements here time forward + argmax without the HTTP server.

Original plan (dates as planned on 2026-11-16):

| Stage | Dates | Part | Outcome |
|---|---|---|---|
| 0. Setup, vLLM baseline | 11/16 | prep | done |
| 1. Dense modules + tests | 11/17-11/18 | Part 0 | done 11/16 |
| 2. End-to-end generation | 11/18-11/19 | Part 0 | done 11/16 |
| 3. Reach 10x of vLLM, 128K YaRN, serving | 11/20-11/22 | Part 0 | done 11/17 |
| 4. Two-GPU all-to-all mock | 11/23 | Part 2 | |
| 5. Single-GPU Qwen3-30B-A3B | 11/24-11/26 | Part 1 | |
| 6. DP + EP, multi-process architecture | 11/27-11/28 | Part 3, Part 5 | |
| 7. Performance to 3x, write-up | 11/29-11/30 | Part 4 + submission | |

Notes from stages 1-2: RMSNorm passes all 90 synthetic cases (bf16: 89/90 bit-identical, 1 differs by one bf16 ulp);
RoPE 32 cases (bf16 bit-identical); MLP 20 cases; embedding bit-exact; lm_head and argmax (first index wins ties) pass;
real Qwen3-4B weights pass for embedding, the RMSNorms of layers 0 and 35, MLP layers 0 and 35, lm_head and argmax
(`real_weights_test.py`). First end-to-end run vs HF: Qwen3-0.6B 43/43 top-1, Qwen3-4B 40/43 (the 3 mismatches HF
near-ties). First L4 timings (27-token prompt): 4B TTFT 44.5 ms / ITL 43.3 ms.

### First baselines (2026-11-16)

vLLM, Qwen3-4B, bf16, 1x H100, single sequence (concurrency 1), 3072 input / 1024 output tokens, 5 requests, prefix caching off, measured with AIPerf on 2026-11-16. Raw output is on Modal volume `final-results` (`/results_store/vllm_isl3072_osl1024_1794855146`).

| Metric | vLLM | Part 0 target (10x) | Part 4 target (3x) |
|---|---|---|---|
| TTFT avg (ms) | 59.09 (p50 57.08) | <= ~591 | <= ~177 |
| ITL avg (ms) | 4.46 | <= ~44.6 | <= ~13.4 |
| Request latency (ms) | 4618.8 | | |

Our engine (Qwen3-4B bf16, 1x H100, same workload: 1 sequence, 3072 in / 1024 out, greedy, EOS ignored; `bench_modal.py`, 3 requests after a warm-up, 2026-11-16; raw output on volume `final-results`):

| Metric | vLLM | ours | ratio (ours / vLLM) | Part 0 limit (10x) | Part 4 limit (3x) |
|---|---|---|---|---|---|
| TTFT avg (ms) | 59.09 | 1042.6 | 17.6x | <= ~591 (not met) | <= ~177 |
| ITL avg (ms) | 4.46 | 21.12 | 4.7x | <= ~44.6 (met) | <= ~13.4 |

Caveats: ours is engine-only (no HTTP server, no tokenizer; timing around forward + argmax with a host sync every step), vLLM was measured with AIPerf through its server. Numbers are very stable across the 3 requests (TTFT 1042.5-1042.7 ms, ITL 21.12 ms). The prefill runs at roughly 24 TFLOP/s effective (2 * 4B * 3072 tokens in 1.04 s), a few percent of the H100's bf16 peak, so there is a lot of room; where the time goes has NOT been profiled yet.

### Profile of our engine (Qwen3-4B, H100, context ~3072; CUDA events around every stage; 2026-11-16)

Prefill of 3072 tokens, 1035 ms in total (stage sum 1035.3 ms):

| Stage | ms | share | achieved |
|---|---|---|---|
| MLP gate/up + SiLU | 282.2 | 27.3% | 39 TFLOP/s |
| MLP down | 239.8 | 23.2% | 23 TFLOP/s |
| attention | 235.5 | 22.8% | 11.8 TFLOP/s |
| q_proj + o_proj + k_proj + v_proj | 263.1 | 25.4% | 22 TFLOP/s |
| everything else (norms, RoPE, adds, KV write, lm_head, embedding) | ~15 | 1.4% | |

Decode step at context ~3072, 20.4 ms (stage sum):

| Stage | ms | share | achieved |
|---|---|---|---|
| attention | 10.12 | 49.6% | 45 GB/s of the cache (the cache is only ~0.46 GB per step, ~0.14 ms at HBM speed) |
| MLP (gate/up + down) | 4.46 | 21.9% | 0.8-1.6 TB/s of weights |
| q/k/v/o projections | 3.28 | 16.1% | k/v only 0.26 TB/s, q/o 0.7-1.0 TB/s |
| small kernels (norms, RoPE, add, KV write) | ~2.0 | 9.8% | ~8 us each, ~15 launches per layer |
| lm_head | 0.58 | 2.9% | 1.3 TB/s |

Reading: the dense GEMMs run at ~2-4% of the H100's bf16 peak because the tiles are tiny (16x32x32, chosen only to get launches to work; 64x64 and 128x128 tiles failed to launch on bf16). Decode attention launches only 32 blocks (one per query head, each walking ~194 key tiles of 16 one after another) on a 132-SM GPU. These are the biggest levers; they are hypotheses to test, not yet changes.

### After the GEMM tile change (2026-11-16)

Cause found: the earlier launch failures of big bf16 tiles were specific to the L4 (99 KB shared memory per block); on the H100 all 27 swept plain-GEMM configurations and all 9 fused gate/up configurations launch and give bit-identical results to the 16x32x32 baseline (`gemm_sweep_modal.py`). Measured on a 3072x2560 @ 2560x9728 bf16 GEMM: 16x32x32 21 TFLOP/s -> 128x128x64 630 TFLOP/s. The fused gate/up kernel (two accumulators) must NOT use 128x128 (95-166 TFLOP/s, registers spill); 128x64x64 gives 601 TFLOP/s. The engine now picks tiles by device (>= 200 KB opt-in shared memory -> big tiles) and by token count; `QWEN3_TILES=small` forces the old tiles.

| Metric | vLLM | ours before | ours now | ratio now | Part 0 (10x) | Part 4 (3x) |
|---|---|---|---|---|---|---|
| TTFT avg (ms) | 59.09 | 1042.6 | **297.3** | 5.0x | met | needs <= ~177 |
| ITL avg (ms) | 4.46 | 21.12 | **17.83** | 4.0x | met | needs <= ~13.4 |

(3 requests of 3072 in / 1024 out after a warm-up, stable to +-0.1 ms; engine only, see the caveats above; raw output on volume `final-results`.) The unit-test suite passes on the H100 with the big tiles; on the 3072-token prompt the last-position logits of the big and small tiles were identical (cosine 1.000000, max diff 0.000, same top-1). Not yet independently confirmed that the "small" run really took the small path (the benchmark did not print the tile profile; it will from now on).

Profile after the change (H100, context ~3072):
- prefill 298 ms: **attention 249.6 ms = 83.6%** (still 11.2 TFLOP/s); all GEMMs together ~35 ms (620-710 TFLOP/s); the rest ~14 ms.
- decode step 17.3 ms: **attention 10.56 ms = 61%** (43 GB/s of the cache); MLP 2.6 ms, projections 2.0 ms, small kernels ~2 ms, lm_head 0.23 ms (3.4 TB/s).
Attention is now the bottleneck of both phases. 128K is still not measured (needs YaRN).

### Attention tile sweep (H100, Hq 32 / Hkv 8, 3072 keys, 2026-11-16; `attn_sweep_modal.py`)

All 13 configurations launch on the H100. What matters is the precision of P in the second matmul: with P kept in fp32 every configuration is slow (2.4-11 TFLOP/s) because the fp32 mma has no tensor-core fast path; with P cast to bf16 (what FlashAttention-2 does) the same kernel runs at up to 260 TFLOP/s.

| Config (BR x BC, P) | prefill 3072 | decode (1 query x 3072 keys), per layer |
|---|---|---|
| 16x16, fp32 (what the engine used so far) | 6.87 ms, 11 TFLOP/s | 283 us (x 36 layers ~ 10.2 ms) |
| 16x64, bf16 | 1.38 ms | 58 us |
| 64x64, bf16 | 0.322 ms | 46 us |
| **64x128, bf16** | **0.297 ms, 260 TFLOP/s** | **44.9 us (x 36 ~ 1.6 ms)** |

Accuracy cost of bf16 P: max difference to the fp32-P kernel ~4.6e-3 of the largest output value (about one bf16 step). `launch_flash_attention` now uses 64x128 / bf16 P on devices with >= 200 KB shared memory (Hopper) and keeps the verified 16x16 / fp32 P elsewhere. Verification after this change (all on the H100):
- Unit tests: every module passes; the attention bf16 cases pass 36/36 against a bound derived from the arithmetic, |d| <= 2^-8 (max|V| + 2|e|) (bf16 unit roundoff is 2^-8; a first version of the bound used 2^-9 by mistake, two cases failed by exactly one bf16 step, the derivation was corrected, not tuned).
- End-to-end vs HF (Qwen3-4B, 43 teacher-forced positions, same HF reference for all rows, `e2e_test.py --gpu h100 --ab`):

| Engine configuration | top-1 | min cosine, prefill | min cosine, decode | max abs logit error |
|---|---|---|---|---|
| default: big GEMM tiles + bf16 P (64x128) | 41/43 | 0.999606 | **0.998993** | 0.781 |
| big GEMM tiles + small attention (fp32 P) | 41/43 | 0.999592 | 0.999399 | 0.531 |
| all small tiles (verified on the L4) | 41/43 | 0.999592 | 0.999399 | 0.531 |

  The pre-set criterion (min cosine > 0.999) is missed by the default configuration by ~7e-6. Rows 2 and 3 are identical, so the big GEMM tiles change nothing numerically; the drop comes from rounding P to bf16 (the worst position has an HF top1-top2 margin of 2.6, so it is not an HF near-tie). Top-1 and the greedy tokens are the same in all three rows. The HF reference itself is noisy across containers (its own greedy text differed between the L4 and H100 runs).
- NOT yet measured with the new attention: the real TTFT/ITL (the ITL of roughly 9 ms is only an estimate from the per-layer numbers).
- Split-P mode (P = P_hi + P_lo, two bf16 mmas, same fp32 accumulator), swept on the H100 (`attn_sweep_modal.py`):

| 64x128 | prefill 3072 | decode per layer | max diff to fp32-P (prefill / decode) |
|---|---|---|---|
| fp32 P (old default 16x16) | 6.92 ms | 285 us | reference |
| bf16 P | 0.299 ms | 44.6 us | 4.59e-3 / 4.39e-3 |
| **split P** | **0.354 ms** | **54.2 us** | **2.29e-3 / 1.10e-3** |

  The split costs ~18-21% over bf16 P and is still ~19x (prefill) / ~5x (decode) faster than fp32 P. It is now the default on Hopper (64x128, mode 2). NOT yet verified: unit tests, the end-to-end A/B with this default (the criterion min cosine > 0.999 must be met, not re-defined), and the real TTFT/ITL.

### After the attention change (2026-11-16): split-P, 64x128 tiles on Hopper

Verification of this default on the H100: attention unit tests 36/36 (worst error back to ~2.5e-3, as with the original fp32-P kernel); 4B end-to-end vs HF (`e2e_test.py --gpu h100 --ab`): top-1 41/43 (both mismatches are exact HF ties, margin 0.000), min cosine prefill 0.999623 / decode 0.999130 (criterion > 0.999 met, not changed), greedy 16/16 identical to HF, RoPE tables bit-identical. Caveat: HF's own CPU bf16 reference moves by ~5e-4 in this metric between containers (the identical all-small-tile engine configuration scored 0.999399 against one HF run and 0.999766 against another), which is as large as the differences between our attention variants, so 0.999 cannot separate bf16-P, split-P and fp32-P; a cleaner reference (HF in fp32) would be needed for that.

| Metric (1 sequence, 3072 in / 1024 out, 3 requests) | vLLM | ours | ratio | Part 0 (10x) | Part 4 (3x) |
|---|---|---|---|---|---|
| TTFT avg (ms) | 59.09 | **64.10** | 1.08x | met | met (<= ~177) |
| ITL avg (ms) | 4.46 | **9.07** | 2.03x | met | met (<= ~13.4) |

History of our TTFT / ITL: 1042.6 / 21.12 ms (16x32x32 GEMM tiles, 16x16 fp32-P attention) -> 297.3 / 17.83 ms (big GEMM tiles) -> **64.1 / 9.07 ms** (+ 64x128 split-P attention). Engine only (no HTTP server / tokenizer, host sync every decode step); vLLM numbers come from AIPerf through its server, so ours is somewhat optimistic. Stable to +-0.6 ms (TTFT) and +-0.01 ms (ITL) over the 3 requests.

Profile now (H100, context ~3072):
- prefill 64.3 ms: MLP gate/up 18.4 (29%) + down 8.8 (14%) at ~600 TFLOP/s; attention 12.4 (19%, 224 TFLOP/s); **per-head q/k RMSNorm 9.0 ms (14%) at only 252 GB/s** (30k tiny blocks, a clear target); q/o/k/v projections ~10.3 ms; rest ~5 ms.
- decode step 10.1 ms (stage sum): MLP 3.2, attention 2.2 (208 GB/s of the cache), projections 2.3 (k/v only 0.39 TB/s), small kernels (RMSNorm, add, q/k norm, RoPE, KV write) ~2.1 ms in ~15 launches per layer, lm_head 0.27 ms. The HBM floor for reading the 8 GB of weights is ~2.4 ms, so decode is ~3.8x above it.

Last-position logits of the 3072-token prompt, default vs all-small tiles: cosine 0.999960, max |logit diff| 0.113 (max |logit| 11.1). The reported "top-1 differs" (55 vs 11) is a tie artefact: the small-tile run itself has a first-place margin of 0.000 on this random-token prompt, so the argmax is arbitrary.

Not covered by these numbers: 128K input (needs YaRN, the weekly report asks for it), concurrency / batching, "any workload".

### 128K input (2026-11-16): YaRN + first measurement

YaRN is implemented following HF `_compute_yarn_parameters` (`engine/qwen3_rope_table.h`; enabled by `rope_scaling` in the config or `QWEN3_YARN=<factor>`, original context 32768, like vLLM's `--rope-scaling` for Qwen3). The tables were compared with HF's `Qwen3RotaryEmbedding` on a CPU container (`rope_table_check.py`, positions 0..131079): HF's attention_scaling 1.138629 matches 0.1 ln 4 + 1; 125 of 8.39M cos entries and 136 of 8.39M sin entries differ, each by exactly one bf16 step (likely fp32 cos/sin implementation differences on values next to a bf16 rounding boundary; that explanation is not separately verified). Default RoPE: 1 of 262,656 entries differs by one bf16 step.

**The 128K workload is 130816 input + 128 output tokens.** With YaRN factor 4 the context is 32768 x 4 = 131072 tokens for input and output together, and the chat endpoint adds role tokens, so 128 tokens are kept free. (A first run used 131072 + 128, which exceeds the context; corrected.)

| Metric (1 sequence, 130816 in / 128 out, YaRN factor 4, 1x H100, 2 requests) | vLLM | ours | ratio | Part 0 (10x) | Part 4 (3x) |
|---|---|---|---|---|---|
| TTFT avg | 10,446.8 ms | **23,338.7 ms** | **2.23x** | met | met |
| ITL avg | 10.43 ms | **28.80 ms** | **2.76x** | met | met (small margin) |

vLLM: `modal_vllm_aipref.py --isl 130816 --osl 128 --yarn`, AIPerf through its server (input length 130,816 for both requests, error rate 0%, output length 127.5 on average). Ours: `bench_modal.py --prompt-len 130816 --gen-len 128 --yarn 4`, engine only. Our earlier ITL at 128K was 79.3 ms with the old decode path; the split-KV decode brought it to 28.8 ms.

How YaRN was enabled in vLLM (this took five attempts; recorded so nobody repeats them): `--rope-scaling` is not a valid flag in the current vLLM; `--hf-overrides` with `rope_scaling` or with `rope_parameters` was accepted but the max length stayed at 40960; a config.json with only `rope_scaling` made transformers apply YaRN but warn that 40960/32768 implies factor 1.25, and vLLM still derived 40960. What works: a model directory (weights symlinked) whose config.json has `rope_scaling` = {yarn, factor 4, original 32768} AND `max_position_embeddings` = 131072. Evidence: inside the container, transformers parses the config as {rope_type: yarn, factor 4.0, original_max_position_embeddings 32768, rope_theta 1e6} with no factor-mismatch warning, and the script refuses to start vLLM otherwise. This vLLM derives its max length from `max_position_embeddings` only, so "vLLM accepted 131072" alone does not prove YaRN; we did not verify which rotary class vLLM instantiated. For timing this barely matters (RoPE is a lookup in a precomputed cos/sin table either way). `VLLM_ALLOW_LONG_MAX_MODEL_LEN` was never set.

Earlier measurement with the old decode path and the over-long workload (131072 in / 128 out), kept for the record:

| Metric | ours (old decode) |
|---|---|
| TTFT avg | 22.99 s |
| ITL avg | 79.3 ms |

Profile at 128K: prefill 23.0 s = **attention 19.8 s (86%) at 256 TFLOP/s** (quadratic in the length), MLP 1.9 s, projections ~0.7 s, q/k norm 0.44 s. Decode step 81.6 ms = **attention 74.0 ms (91%) at 261 GB/s**: the step reads ~19.3 GB of KV cache, which takes ~5.8 ms at HBM speed, so decode attention is ~13x above its floor. Cause: one block per query head (32 blocks on a 132-SM GPU), each walking ~1024 key tiles one after another. Next: split the KV length across blocks (flash-decoding) and combine.

Correctness at long context (128K itself cannot be compared with HF on a CPU): Qwen3-0.6B, 8192-token prompt, YaRN factor 4 on both sides, 16 HF greedy tokens, last 64 positions compared, same HF reference for all rows (`e2e_test.py --gpu h100 --long 8192 --yarn 4 --ab`):

| Engine configuration | top-1 | min cosine, prefill | min cosine, decode | max abs logit error |
|---|---|---|---|---|
| default: big GEMM tiles + 64x128 split-P attention | 60/64 | 0.999789 | **0.998630** | 0.562 |
| big GEMM tiles + small attention (fp32 P) | 60/64 | 0.999559 | 0.999810 | 0.319 |
| all small tiles | 60/64 | 0.999559 | 0.999810 | 0.319 |

- YaRN end to end is correct: RoPE tables identical to HF on all 8232 positions; every top-1 mismatch is an HF near-tie (margin 0.000 or 0.062); greedy tokens 16/16 identical.
- The default configuration FAILS the pre-set criterion on the decode path (0.998630 < 0.999; worst position has an HF margin of 2.6). The same direction showed up in all three A/B runs (decode, default vs small attention in the same run: 0.998993 vs 0.999399, 0.999130 vs 0.999766, 0.998630 vs 0.999810), while the prefill path of the big attention is as good or better. The within-run gaps (6e-4 to 1.2e-3) are larger than the HF noise seen so far (~5e-4), so this is a real accuracy deficit of the 64x128 attention kernel when n_q = 1, and calling these differences "HF noise" earlier was not accurate. Split P did not fix it, so it is not the rounding of P. Cause: unknown so far.
- Split-KV decode (new default decode path on Hopper; `QWEN3_DECODE_SPLIT=0` switches it off): before running it on a GPU, its split rule, index math and combine were simulated in numpy (`tools/split_kv_sim.py`, runs locally: 13 lengths incl. tile boundaries and 131072, exact in float64, worst float32 error 1.3e-6). On the H100: attention unit tests 36/36; the same 8K A/B with four configurations:

| Engine configuration (same HF reference) | top-1 | min cosine, prefill | min cosine, decode | max abs logit error |
|---|---|---|---|---|
| **default: decode = split-KV kernel** | 60/64 | 0.999789 | **0.999687** | **0.312** |
| default but decode with the 64x128 kernel | 60/64 | 0.999789 | 0.998630 | 0.562 |
| big GEMM tiles + small attention | 60/64 | 0.999559 | 0.999810 | 0.319 |
| all small tiles | 60/64 | 0.999559 | 0.999810 | 0.319 |

  The default now passes (criterion unchanged) and its decode accuracy matches the small fp32-P attention (max error 0.312 vs 0.319). The old 64x128 decode reproduced its deficit to the digit (0.998630, 0.562, same as the previous run), confirming it is a property of that kernel with a single query. Root cause still not identified; a hypothesis consistent with the data is the long serial accumulation (64 key tiles into one accumulator inside the mma) that the split kernel avoids (4 tiles per block, fp32 combine). Not verified.
- Decode at 128K with the split-KV kernel: attention 23.8 ms of a 31.5 ms step (810 GB/s of cache; floor ~5.8 ms). Each query head reads its KV head separately (4 query heads per KV head), relying on L2; packing the 4 heads into one block is the next lever (hypothesis).

### HTTP server (OpenAI-compatible, no PyTorch), 2026-11-16

- `engine/qwen3_capi.cu` -> `libqwen3engine.so`: C interface (`qe_create` with a number of KV-cache slots, `qe_prefill`
  of one slot with chunked prefill for long prompts, `qe_decode_batch` of several slots in one step, `qe_vocab`,
  `qe_max_len`, `qe_max_chunk`, `qe_max_seqs`).
- `server/server.py`: FastAPI; `/v1/chat/completions` and `/v1/completions` (both with SSE streaming), `/v1/models`,
  `/health`; `max_tokens` / `max_completion_tokens`, `min_tokens`, `ignore_eos`, `stream_options.include_usage`,
  `chat_template_kwargs`. Greedy only: `temperature` > 0 is rejected with 400 instead of being silently ignored.
  Lengths are checked before the engine is called (the engine exits on invalid arguments). Incremental detokenisation
  with a sliding window, so the per-token cost does not grow with the output length. Tokenizer and chat template from
  HF `tokenizers` / `transformers` (no PyTorch needed for them).
- Continuous batching: a scheduler thread admits waiting requests into free KV-cache slots (prefill one request at a
  time), then decodes every running sequence in one batched step; finished or disconnected requests free their slot at
  once; an engine error fails only the requests involved. Limitation: a prefill stalls the decode of the others (no
  chunked prefill interleaving). `min_tokens` does not mask
  EOS like vLLM does: an early EOS is kept as a (silent) token and generation continues.
- `server/test_server.py`: 11 tests with a fake engine and fake tokenizers, run locally and again inside the Modal image
  at build time (stream format and [DONE], exact lengths and usage, EOS stop vs ignore_eos, max_completion_tokens
  precedence, non-stream responses, completions with text and token ids, four 400 cases, three concurrent requests decoded in one
  batch with output equal to running alone, more requests than slots, a client disconnect frees its slot, an engine
  error fails only its request, a 3-byte UTF-8 character split over tokens). They found real bugs that are fixed: an
  explicit `max_tokens: 0` was replaced by the default; a failing prefill could leak its slot; an exception could stop
  the engine thread silently.
- `modal_aipref.py`: the library is compiled for sm_90 and its exported symbols checked while the image is built on a
  CPU; the GPU runs the server plus AIPerf with the same settings as `modal_vllm_aipref.py` (tokenizer given as the local
  snapshot directory, offline). `--compat` runs AIPerf against the fake-engine server on a CPU first (0 errors at
  concurrency 1 and 8).

### Batched decode correctness (2026-11-17)

`tests/batch_test.cu` (`batch_check.py`, Qwen3-4B, H100): 4 sequences of 27 / 300 / 1000 / 3000 tokens, each decoded
16 steps alone in slot 0, then all four together in slots 0-3 with the same tokens (teacher forcing).

- First run: min cosine 0.99926 against my pre-set 0.9999 -> FAIL (top-1 identical everywhere). Diagnosis with a fixed
  split count (`QWEN3_DECODE_SPLITS=4`): every sequence whose key tiles were grouped the same way in both runs was
  bit-identical; the rest differed only because the tiles-per-split was computed from the longest sequence of the batch,
  i.e. a different fp32 summation order, not wrong data.
- Fix: each sequence now splits its own key tiles (`s_b = min(target, n_tiles_b)`), so its result does not depend on
  the other sequences in the batch. Rerun: with a fixed split count **64/64 rows bit-identical** (new criterion for this
  test, stricter than a cosine bound); with the default rule (split count depends on the batch size) min cosine 0.99926,
  top-1 0 mismatches, reported as informational. Attention unit tests 36/36 and the 8K YaRN A/B against HF (decode min
  cosine 0.999687, unchanged) rerun after the change.

### Workload suite (2026-11-17)

Moved to "Results: our server vs vLLM (Part 0)" at the top; run directories are listed in `results/comparison.csv`.

## Tile toolchain notes
- Compile: `nvcc --enable-tile -std=c++20 -arch=sm_80 file.cu`; include `cuda_tile.h`; kernels use `__tile_global__`; block dim must be 1.
- Image `nvidia/cuda:13.5.0-devel-ubuntu24.04` has `cuda_tile.h`; every kernel here is compiled for sm_90 (H100) at image build time on a CPU (`tile_check.py` was the first check) and runs on the L4 (sm_89, small tiles) and the H100.
- Big tiles need the >= 200 KB opt-in shared memory of Hopper; on the L4 (99 KB) 64x64 and larger bf16 tiles fail to launch with "invalid argument", so tiles are chosen by device.
- Reference samples: NVIDIA/cuda-samples `cpp/9_CUDA_Tile` (tileLayerNorm, tileRope, tileMatmul, tileBmm).
