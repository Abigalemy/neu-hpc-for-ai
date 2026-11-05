# neu-hpc-for-ai

This repository contains coursework, lab assignments, and project implementations for **CSYE 7375 - High Performance Computing** at Northeastern University.

## Assignment work

- [Week 04–05: Qwen3 and CUDA Tile kernels](week_04_05/README.md) — Tile attention and SwiGLU, SQLite correctness cases, L4 and Qwen3-4B validation, and C++ tokenizer.
- [Week 07–08: inference server and performance study](week_07_08/README.md) — KV cache modes, PagedAttention, continuous batching, CUDA Tile integration, correctness tests, and H100 AIPerf results.
- [`hpc_core/`](hpc_core/) contains the shared model, kernels, cache, and scheduler used by both assignments.

The Week 04–05 page records known limitations of the earlier implementation and links to the archived SIMT code. The Week 07–08 page includes the complete collected result tables, raw AIPerf exports, and measurement caveats.
