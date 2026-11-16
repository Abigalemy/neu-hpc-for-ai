# Final Project: Qwen3 + MoE inference server in CUDA Tile C++

Goal (from the assignment): approach vLLM performance without PyTorch in the serving path.

## Plan -> assignment parts

| Stage | Dates | Part | Status |
|---|---|---|---|
| 0. Setup, vLLM baseline | 11/16 | prep (weekly report) | in progress |
| 1. Dense modules + tests | 11/17-11/18 | Part 0 | done (modules except attention; attention + KV cache are stage 2): RMSNorm passes all 90 synthetic cases on L4 (bf16: 89/90 bit-identical, 1 case differs by one bf16 ulp); RoPE (rotate-half, theta 1e6, in place) passes all 32 synthetic cases on L4 (bf16 bit-identical, fp32 worst 4.8e-7; cos/sin tables come from HF, our own table generator is still to do); MLP (gate/up fused with SiLU, then down) passes all 20 synthetic cases on L4 (fp32 256/512; bf16 256/512, 2048/768, 2560/1024) with tiles 16x32x32; 64x64 and 128x128 tiles failed to launch with "invalid argument" on bf16 (suspected: shared-memory footprint of the compiler's pipeline stages, not verified); embedding (bit-exact), lm_head linear layer, and greedy argmax (first index wins ties, incl. forced ties and vocab not a multiple of the 1024 chunk) all pass on L4 with synthetic data; real Qwen3-4B bf16 weights also pass on L4 (`real_weights_test.py`): embedding, all RMSNorms of layers 0 and 35 + final norm + q/k norms (bit-identical), MLP layers 0 and 35 with the real 2560/9728 shapes, lm_head with the real 151936 vocab, argmax. Inputs are embedding rows (not hidden states from deep inside the network); RoPE has no real-weight check (it has no weights, its tables come from HF) |
| 2. End-to-end generation | 11/18-11/19 | Part 0 | done (first H100 numbers below): fused causal GQA flash attention reading the KV cache directly (online softmax, tiles 16x16) passes 36 synthetic cases on L4 (prefill and decode, heads 32/8, 32/4, 16/8, n_kv up to 1100). Whole model now runs end to end without PyTorch (`engine/`: safetensors parser, weights copied to the GPU once, per-layer KV cache, residual add, 36-layer forward, greedy decode) and is checked against HF on CPU with `e2e_test.py`: Qwen3-0.6B 43/43 top-1 positions agree, greedy output identical for 16/16 tokens, RoPE tables bit-identical; Qwen3-4B 40/43 top-1 (the 3 mismatches are HF near-ties, margins 0.0-0.125), cosine >= 0.9994, KV-cache decode path 16/16. Greedy generation of the 4B diverges from HF `generate` at token 2: most likely HF's cached decode vs full forward differing on a near tie (inferred from the logs, not yet confirmed). First timings on an L4 (27-token prompt, NOT comparable to the H100 vLLM baseline): 0.6B TTFT 9.9 ms / ITL 8.0 ms; 4B TTFT 44.5 ms / ITL 43.3 ms. Next: timing on the H100 with the same 3072-in/1024-out workload as the vLLM baseline, then an OpenAI-compatible server for AIPerf |
| 3. Reach 10x of vLLM, 128K YaRN | 11/20-11/22 | Part 0 | 10x reached for 3072 in / 1024 out (TTFT 1.08x, ITL 2.03x of vLLM); 128K / YaRN and other workloads not done |
| 4. Two-GPU all-to-all mock | 11/23 | Part 2 | not started |
| 5. Single-GPU Qwen3-30B-A3B | 11/24-11/26 | Part 1 | not started |
| 6. DP + EP, multi-process architecture | 11/27-11/28 | Part 3, Part 5 | not started |
| 7. Performance to 3x, write-up | 11/29-11/30 | Part 4 + submission | not started |

## Rules for myself
- Understand every line the coding agent writes; Qwen3 only (head_dim 128), no generic code.
- Attention must be fused (FlashAttention) and must use the KV cache; weights loaded to the GPU once.
- Tile kernels (`__tile_global__`), not SIMT `__global__`.
- Correctness check before any benchmark; record commit hash with each measurement.

## Commands
```bash
modal run modal_vllm_aipref.py            # dry run: prints plan and cost cap
modal run modal_vllm_aipref.py --yes      # vLLM baseline, 3072 in / 1024 out, single sequence
```

## Measured baselines

vLLM, Qwen3-4B, bf16, 1x H100, single sequence (concurrency 1), 3072 input / 1024 output tokens, 5 requests, prefix caching off, measured with AIPerf on 2026-11-16. Raw output is on Modal volume `final-results` (`/results_store/vllm_isl3072_osl1024_1794855146`).

| Metric | vLLM | Part 0 target (10x) | Part 4 target (3x) |
|---|---|---|---|
| TTFT avg (ms) | 59.09 (p50 57.08) | <= ~591 | <= ~177 |
| ITL avg (ms) | 4.46 | <= ~44.6 | <= ~13.4 |
| Request latency (ms) | 4618.8 | | |

Our engine (Qwen3-4B bf16, 1x H100, same workload: 1 sequence, 3072 in / 1024 out, greedy, EOS ignored; `bench_modal.py`, 3 requests after a warm-up, 2026-11-17; raw output on volume `final-results`):

| Metric | vLLM | ours | ratio (ours / vLLM) | Part 0 limit (10x) | Part 4 limit (3x) |
|---|---|---|---|---|---|
| TTFT avg (ms) | 59.09 | 1042.6 | 17.6x | <= ~591 (not met) | <= ~177 |
| ITL avg (ms) | 4.46 | 21.12 | 4.7x | <= ~44.6 (met) | <= ~13.4 |

Caveats: ours is engine-only (no HTTP server, no tokenizer; timing around forward + argmax with a host sync every step), vLLM was measured with AIPerf through its server. Numbers are very stable across the 3 requests (TTFT 1042.5-1042.7 ms, ITL 21.12 ms). The prefill runs at roughly 24 TFLOP/s effective (2 * 4B * 3072 tokens in 1.04 s), a few percent of the H100's bf16 peak, so there is a lot of room; where the time goes has NOT been profiled yet.

### Profile of our engine (Qwen3-4B, H100, context ~3072; CUDA events around every stage; 2026-11-17)

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

### After the GEMM tile change (2026-11-17)

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

### Attention tile sweep (H100, Hq 32 / Hkv 8, 3072 keys, 2026-11-17; `attn_sweep_modal.py`)

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

### After the attention change (2026-11-17): split-P, 64x128 tiles on Hopper

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

Not yet measured: 128K input (needs YaRN), and the same workload for the Qwen3-30B-A3B on H200.

## Tile toolchain notes
- Compile: `nvcc --enable-tile -std=c++20 -arch=sm_80 file.cu`; include `cuda_tile.h`; kernels use `__tile_global__`; block dim must be 1.
- Verified (CPU-only Modal build, `tile_check.py`): image `nvidia/cuda:13.5.0-devel-ubuntu24.04` has `cuda_tile.h` and compiles a Tile kernel for sm_80 and sm_90 (H100/H200). Not yet run on a GPU.
- Reference samples: NVIDIA/cuda-samples `cpp/9_CUDA_Tile` (tileLayerNorm, tileRope, tileMatmul, tileBmm).
