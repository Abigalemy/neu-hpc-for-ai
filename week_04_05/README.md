# Week 04-05: High-Performance Batched Inference Engine for Qwen3-4B

## Overview
This repository contains a fully self-contained, high-performance CUDA implementation of the Qwen3-4B transformer layer, featuring:
- **Fully Fused SwiGLU MLP**: Elimination of intermediate DRAM write-backs for activation and Hadamard product.
- **Tiled FlashAttention**: Online Softmax forward implementation avoiding materialized $N \times N$ attention matrices in global memory.
- **RMSNorm and RoPE Kernels**: Fused reduction and position embedding kernels.
- **Self-Contained C++ Tokenizer**: Embedded BPE/byte-level tokenizer header (`tokenizer.h`).
- **Regression Test Harness**: Automated SQLite database verification (`test_cases.db`).

## Repository Structure
- `mlp_cuda.cu`: CUDA implementation of Fused SwiGLU.
- `flash_attn_cuda.cu`: CUDA Tile implementation of FlashAttention.
- `ops_cuda.cu`: CUDA implementations for RMSNorm and Rotary Position Embedding.
- `tokenizer.h`: Standalone C++ tokenizer header.
- `test_cases.db`: SQLite database storing test vectors and PyTorch ground truths.
- `generate_mlp_cases.py` / `verify_mlp_cases.py`: SQLite harness for MLP layer.
- `generate_attention_cases.py` / `verify_attention_cases.py`: SQLite harness for Attention layer.
- `eval_matrix_modal.py`: Multi-dimensional evaluation across batch sizes (1, 2, 4) and sequence lengths (32, 64, 128).
- `engine_modal.py`: End-to-end batched inference engine verified against PyTorch reference.

## Verification & Correctness
All kernels are cross-checked against PyTorch reference implementations using single-precision floating point (`float32`).
- Fused SwiGLU maximum difference: $0.0$
- FlashAttention maximum difference: $< 1.5 \times 10^{-6}$ (Tolerance threshold: $1.0 \times 10^{-4}$)
- Full Layer End-to-End maximum difference: $< 1.0 \times 10^{-5}$

## Performance Matrix (T4 GPU via Modal)
Evaluated across various batch sizes ($B$) and sequence lengths ($S$):
- **MLP Speedup**: Up to 1.64x over PyTorch baseline due to kernel fusion.
- **Attention Verification**: All dimension combinations passed numerical correctness checks.