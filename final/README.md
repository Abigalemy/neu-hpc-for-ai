# Final Project: Qwen3 + MoE inference server in CUDA Tile C++

Goal (from the assignment): approach vLLM performance without PyTorch in the serving path.

## Plan -> assignment parts

| Stage | Dates | Part | Status |
|---|---|---|---|
| 0. Setup, vLLM baseline | 11/16 | prep (weekly report) | in progress |
| 1. Dense modules + tests | 11/17-11/18 | Part 0 | done (modules except attention; attention + KV cache are stage 2): RMSNorm passes all 90 synthetic cases on L4 (bf16: 89/90 bit-identical, 1 case differs by one bf16 ulp); RoPE (rotate-half, theta 1e6, in place) passes all 32 synthetic cases on L4 (bf16 bit-identical, fp32 worst 4.8e-7; cos/sin tables come from HF, our own table generator is still to do); MLP (gate/up fused with SiLU, then down) passes all 20 synthetic cases on L4 (fp32 256/512; bf16 256/512, 2048/768, 2560/1024) with tiles 16x32x32; 64x64 and 128x128 tiles failed to launch with "invalid argument" on bf16 (suspected: shared-memory footprint of the compiler's pipeline stages, not verified); embedding (bit-exact), lm_head linear layer, and greedy argmax (first index wins ties, incl. forced ties and vocab not a multiple of the 1024 chunk) all pass on L4 with synthetic data; real Qwen3-4B bf16 weights also pass on L4 (`real_weights_test.py`): embedding, all RMSNorms of layers 0 and 35 + final norm + q/k norms (bit-identical), MLP layers 0 and 35 with the real 2560/9728 shapes, lm_head with the real 151936 vocab, argmax. Inputs are embedding rows (not hidden states from deep inside the network); RoPE has no real-weight check (it has no weights, its tables come from HF) |
| 2. End-to-end generation | 11/18-11/19 | Part 0 | not started |
| 3. Reach 10x of vLLM, 128K YaRN | 11/20-11/22 | Part 0 | not started |
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

Not yet measured: 128K input (needs YaRN), and the same workload for the Qwen3-30B-A3B on H200.

## Tile toolchain notes
- Compile: `nvcc --enable-tile -std=c++20 -arch=sm_80 file.cu`; include `cuda_tile.h`; kernels use `__tile_global__`; block dim must be 1.
- Verified (CPU-only Modal build, `tile_check.py`): image `nvidia/cuda:13.5.0-devel-ubuntu24.04` has `cuda_tile.h` and compiles a Tile kernel for sm_80 and sm_90 (H100/H200). Not yet run on a GPU.
- Reference samples: NVIDIA/cuda-samples `cpp/9_CUDA_Tile` (tileLayerNorm, tileRope, tileMatmul, tileBmm).
