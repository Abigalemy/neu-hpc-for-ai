# Week 04–05: Qwen3 Inference Baseline and CUDA Tile Kernels

This directory is corrected against the assignment and grading feedback. The earlier SIMT CUDA C++
and simplified-model attempt is archived under [`legacy_simt/README.md`](legacy_simt/README.md) for
learning reference; the active assignment path uses CUDA Tile.

## Current implementation

- The shared Qwen3 model, KV backends, and serving engine now live in `../hpc_core/` so Week 04–05
  and Week 07–08 can use the same implementation instead of drifting copies.
- `../hpc_core/kernels/cutile_ops.py` contains the cuTile SwiGLU kernel.
- `../hpc_core/kernels/cutile_attention.py` contains tiled online-softmax attention with causal
  masking and GQA. The shared model dispatches to these Tile kernels on CUDA and uses PyTorch as its
  CPU reference.
- `run_cuda_modal.py` checks SwiGLU against the SQLite case, compares Tile attention with PyTorch
  across causal prefill/decode, GQA, partial tiles, FP16/BF16 and non-contiguous inputs, then checks
  a small shared Qwen3 model against its CPU path for all three KV-cache modes. These checks passed
  on L4; see the Modal run history for logs.
- `generate_cutile_cases.py` writes deterministic Q/K/V, shape, dtype, causal-position, GQA, seed,
  and PyTorch reference-output records to the `cutile_attention_cases` SQLite table. The Modal check
  replays those exact stored tensors before running extra low-precision cases.
- `generate_qwen3_cases.py` adds a deterministic `qwen3_e2e_cases` record containing the tiny model's
  full state dict, config, packed inputs, sequence lengths, and CPU reference logits. The L4 check
  reconstructs the model and compares both its CPU result and CUDA Tile result against those stored
  logits before checking greedy decode under all three cache modes.
- `week_07_08/modal_bench.py --check-only --model Qwen/Qwen3-4B --yes` performs the real-weight HF
  comparison and writes a `qwen3_4b_prefill_cases` record with the pinned HF model revision, full
  model config, token/position IDs, dtype, causal setting, and every prompt-position reference logit.
  `verify_qwen3_case_db.py` checks the SQLite integrity, tensor shape, and recorded HF comparison.
- `tokenizer.h` wraps MLC `tokenizers-cpp` and loads Qwen3's actual `tokenizer.json`; the old
  byte-to-ID stub was not a valid Qwen3 tokenizer. `tokenizer_cpp/` has a CMake build pinned to
  upstream commit `acbdc5a2`, a standalone executable, and a pybind11 module. Week 07–08 builds this
  module into its server image and uses it to encode actual completion/chat prompts.
- `eval_matrix_modal.py` and `benchmark_modal.py` use the shared CUDA Tile correctness/latency
  matrix, not the retained legacy `.cu` attention code. The matrix includes causal GQA.
- `l4_kernel_matrix.csv` stores the L4 matrix result for batch sizes 1/2/4 and sequence lengths
  32/64/128 (bf16, 32 Q heads, 8 KV heads, head dimension 128), including four SwiGLU tile widths.
- `real_qwen3_l4_matrix.csv` compares actual Qwen3-4B prefill against Hugging Face for batch sizes
  1/2/4 and sequence lengths 32/64/128; the JSON sidecar records the model revision and timing method.

## Original files and known gaps

- The archived `legacy_simt/` kernels use SIMT CUDA C++ (`__global__`, thread indices, and shared
  memory), not the CUDA Tile programming model. The archived FlashAttention kernel handles one
  query row and omits causal masking/GQA; its Qwen3 attention and engine files are simplified models.
- The C++ tokenizer build fetches external `mlc-ai/tokenizers-cpp` source at a pinned commit and
  needs CMake, Git, Rust/Cargo, and pybind11. The Python inference server uses the C++ extension for
  text-to-token encoding; Hugging Face's Python tokenizer still renders the Qwen3 chat template and
  decodes generated tokens for streaming. The model and serving engine remain Python.
- The original SQLite tables still contain one small synthetic MLP case and one simplified attention
  case. `cutile_attention_cases` adds reproducible kernel cases, `qwen3_e2e_cases` adds a tiny-model
  end-to-end regression, and `qwen3_4b_prefill_cases` stores a real-weight Qwen3-4B reference pass.

These limitations match the issues in the Week 04–05 grading feedback. Do not cite the old CUDA
kernel timings or the synthetic one-layer check as proof that the original assignment requirements
were met. The new tiny-model SQLite case is a reproducible end-to-end regression test; it complements
the separate Qwen3-4B/Hugging Face run rather than replacing it.

## Validation evidence and reproducible build

### C++ tokenizer build and HF parity

The C++ executable and pybind11 extension compiled in a CPU-only Modal job and matched Hugging Face's
tokenizer on vocabulary size and five English, Chinese, punctuation, whitespace, and Qwen3
chat-template probes. It used Qwen3-4B tokenizer
revision `1cfa9a7208912126459214e8b04321603b3df60c`. Upstream's v0.1.1 release has a reported Rust
dependency mismatch; the CMake build uses the upstream fix commit identified in
[issue #94](https://github.com/mlc-ai/tokenizers-cpp/issues/94). The build run is
[ap-deInUEr8CtlI7zHYshYeF8](https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-deInUEr8CtlI7zHYshYeF8).

For a repeatable local build, install CMake, Git, a C++17 compiler, Rust/Cargo, and Python packages
`tokenizers` and `pybind11`. Point the probe at the `tokenizer.json` for the same pinned Qwen3 model
revision used by the assignment. From the repository root:

```bash
python3 -m pip install tokenizers pybind11==2.13.6
cmake -S week_04_05/tokenizer_cpp -B build/tokenizer_cpp -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
  -DQWEN3_TOKENIZER_PYTHON_OUTPUT_DIR="$PWD/week_07_08"
cmake --build build/tokenizer_cpp --parallel
python3 week_04_05/tokenizer_cpp/compare_with_hf.py \
  build/tokenizer_cpp/qwen3_tokenizer_probe /path/to/Qwen3/tokenizer.json week_07_08
```

The build places the importable extension beside `week_07_08/server.py`; build it before starting that server locally. The Modal benchmark image builds the same extension automatically.

The comparison checks vocabulary size and exact token IDs. The C++ executable itself loads
`tokenizer.json` through the C++ adapter and does not call Python during tokenization; Python is used
only by the parity-check script.

The L4 checks establish kernel-level correctness on the listed synthetic shapes and integration
with a small random Qwen3 model, with its expected logits now persisted in SQLite. The H100
Qwen3-4B comparison matched Hugging Face prefill logits (5 prompt positions, max absolute difference
`3.457e-05`) and greedy generation for four prompts in no-cache, contiguous, and paged modes. The
real-weight prefill record is stored in SQLite and passes `verify_qwen3_case_db.py`. The initial L4
microbenchmark matrix is recorded below; the old performance tables remain unrelated to the new
Tile path.

For the real-weight L4 matrix, CUDA Tile greedy tokens matched Hugging Face for all nine batch/length
shapes. The geometric-mean HF/Tile latency ratio was 1.010 (about 1% lower latency for Tile in this
run): Tile was faster in seven shapes and slower in two, with the largest slowdown about 11%. Treat
this as a small, shape-dependent change until repeated runs on the target grading GPU confirm it.

In the recorded L4 run, Tile attention took about 0.050–0.058 ms versus 0.076–0.083 ms for PyTorch
SDPA. SwiGLU took about 0.039–0.045 ms across the tested tile widths, versus 0.020–0.023 ms for
separate PyTorch SiLU and multiply. So the attention kernel was faster in this run, while SwiGLU was
slower and remains an optimization task. These are single-run microbenchmarks, not end-to-end Qwen3
service performance; compare the repeated runs and full H100 service results before making final
performance claims.

Use the CUDA Tile kernel only with a compatible cuTile/TileIR installation and a supported GPU.
The L4 microbenchmark matrix completed and is recorded in `l4_kernel_matrix.csv`. Both Modal
kernel-benchmark entry points cap each function at 300 seconds. Full real-weight correctness was
checked separately on H100; see the Week 07–08 notes. The corrected CUDA Tile service path now also
has a complete AIPerf H100 matrix in the Week 07–08 results, with single-run and measurement caveats
documented there.
