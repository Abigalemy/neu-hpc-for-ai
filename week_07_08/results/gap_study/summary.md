# Baseline anomaly and H100 gather microbenchmark

Qwen3-4B BF16 on one H100 80 GB. Each baseline condition used a fresh server process, 8 warmup requests, and 16 measured requests at concurrency 1 (128 input / 128 output tokens). Three rounds were run sequentially in one H100 function; all 9 profiles returned zero request errors. Reported summaries are mean ± sample standard deviation across the three profiles. Raw AIPerf exports are kept in `repeat1/`, `repeat2/`, and `repeat3/`.

| Cache mode | Output tok/s | TTFT p50 (ms) | ITL p50 (ms) | E2E p50 (ms) | errors |
|---|---:|---:|---:|---:|---:|
| Contiguous | 45.80 ± 0.47 | 22.75 ± 0.19 | 21.79 ± 0.23 | 2790.61 ± 29.74 | 0/16 each |
| No cache | 49.91 ± 1.76 | 22.01 ± 0.83 | 19.99 ± 0.69 | 2560.19 ± 88.96 | 0/16 each |
| Paged + direct block-table | 49.66 ± 1.11 | 23.08 ± 0.57 | 20.08 ± 0.42 | 2573.51 ± 54.04 | 0/16 each |

The earlier single-run baseline reported 43.1 tok/s for contiguous and 51.3 tok/s for no-cache. The difference recurred in these final-code runs: contiguous averaged 45.80 tok/s while no-cache averaged 49.91 tok/s (about 8.2% higher for no-cache). Paged with the fused kernel averaged 49.66 tok/s, close to no-cache in this short, single-request workload. This confirms a small baseline disadvantage for the contiguous path, but does not establish a universal cache penalty.

## Isolated attention microbenchmark

The microbenchmark uses synthetic BF16 Qwen3-shaped tensors (Q heads 32, KV heads 8, head dimension 128, block size 16) with shuffled physical block IDs. CUDA events timed 30 repetitions after 5 warmups. “Gather + Tile” reproduces block-table K/V materialization followed by one Tile attention call per sequence; “Direct” is the fused block-table kernel. Output comparison passed `allclose(atol=0.02, rtol=0.02)` for all six cases. This isolates attention-path costs; it does not include model layers, scheduling, or HTTP overhead.

| Batch | Context | Gather only (µs) | Tile attention on gathered K/V (µs) | Gather + Tile (µs) | Direct block-table (µs) | Old / direct |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 21.1 | 49.9 | 75.3 | 20.2 | 3.73× |
| 1 | 1024 | 20.0 | 49.8 | 73.8 | 75.8 | 0.97× |
| 8 | 128 | 19.6 | 293.8 | 326.6 | 24.6 | 13.28× |
| 8 | 1024 | 29.8 | 296.2 | 326.2 | 151.7 | 2.15× |
| 32 | 128 | 20.0 | 1137.3 | 1165.8 | 89.5 | 13.03× |
| 32 | 1024 | 98.2 | 1136.7 | 1168.6 | 594.0 | 1.97× |

At batch 8–32, the gather itself accounts for roughly 2–9% of the old combined time; the repeated per-sequence Tile attention work is larger. The direct kernel is 2.0–13.3× faster in those tested cases. At batch 1 / context 1024, it is slightly slower (75.8 µs vs 73.8 µs), so the new kernel is not faster for every shape. The data therefore narrows the root cause: copying K/V alone is not the main high-batch cost; combining table lookup and attention into one kernel also avoids repeated per-sequence attention launches. The microbenchmark does not claim a full-model speedup for every sequence length.

## Interpreting the no-cache result

The repeated baseline shows contiguous is about 8% slower than no-cache in this exact short-context, concurrency-1 workload, while fused paged and no-cache are nearly tied. That points more specifically to the contiguous path (or measurement variation) than to a general penalty from cached decode: contiguous still invokes Tile attention per sequence, while paged uses the fused kernel. The attention microbenchmark supports the cost of the old per-sequence Tile path, but it does not directly profile contiguous model steps. The earlier paged result (62.1 tok/s) and final-code mean (49.7 tok/s) are not a matched before/after AIPerf experiment; the old value was one run, whereas the new value is a three-run mean. The matched whole-forward H100 A/B found the fused path faster than its gather reference, but that does not explain the cross-run AIPerf difference. Optional GPU telemetry was unavailable, so the contiguous-path cause remains unproven.

Modal run: [ap-hYtLK44pCoN3y5eA8QQEPD](https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-hYtLK44pCoN3y5eA8QQEPD).
