# Archived Week 04–05 prototypes

These files are the earlier SIMT CUDA C++ and simplified-model attempt. They are retained as a
learning reference only and are not used by the corrected assignment path.

- The `.cu` kernels use `__global__` thread/block indexing and are not CUDA Tile kernels.
- The old FlashAttention kernel handled one query row and did not implement causal masking or GQA.
- The old Qwen3 attention/model wrappers did not implement the full Qwen3-4B computation.
- Their generators and verification scripts use the corresponding legacy SQLite schema.

The current implementation and validation entry points live one directory above this archive.
