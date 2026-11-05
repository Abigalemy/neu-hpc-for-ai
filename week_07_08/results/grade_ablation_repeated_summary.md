# Repeated H100 ablation summary

Three separate H100 container runs; values are mean ± sample standard deviation (n=3). See each repeat directory for raw AIPerf exports.

| Workload | Policy | Output tok/s | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---:|---:|---:|
| Balanced c16 | Paged + continuous | 716.8 ± 137.0 | 149.2 ± 10.5 | 20.0 ± 4.9 |
| Balanced c16 | Paged + static | 424.8 ± 81.8 | 4793.6 ± 745.0 | 19.3 ± 4.1 |
| Mixed Poisson 8 req/s | Paged + continuous | 781.3 ± 35.6 | 54.7 ± 10.5 | 38.9 ± 1.8 |
| Mixed Poisson 8 req/s | Paged + static | 343.6 ± 62.3 | 17481.8 ± 4149.1 | 19.9 ± 3.7 |
| Decode-heavy c8 | Paged + continuous | 434.5 ± 85.8 | 66.7 ± 15.6 | 18.9 ± 4.1 |
| Decode-heavy c8 | No KV cache | 74.4 ± 0.5 | 55.0 ± 12.7 | 107.5 ± 0.8 |
