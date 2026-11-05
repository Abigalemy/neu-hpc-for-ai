| config | run | req/s | out tok/s | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | E2E p50 (ms) | E2E p99 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| ours_nocache | decode_heavy_c8 | 0.1 | 74.0 | 71.4 | 177.4 | 108.1 | 108.7 | 110641.2 | 111327.3 |
| ours_paged | balanced_c16 | 1.4 | 360.3 | 161.8 | 181.8 | 43.9 | 44.2 | 11364.0 | 11380.6 |
| ours_paged | decode_heavy_c8 | 0.2 | 236.7 | 84.6 | 85.4 | 33.8 | 33.8 | 34604.8 | 34615.5 |
| ours_paged | mixed_poisson_r8 | 1.7 | 427.1 | 85.0 | 1147.5 | 90.0 | 124.5 | 21137.9 | 31112.0 |
| ours_static | balanced_c16 | 0.9 | 240.9 | 6113.4 | 10905.9 | 42.4 | 43.6 | 16912.8 | 17285.9 |
| ours_static | decode_heavy_c8 | 0.2 | 237.2 | 47.1 | 55.6 | 33.7 | 33.8 | 34521.5 | 34667.4 |
| ours_static | mixed_poisson_r8 | 0.8 | 208.2 | 30418.5 | 56665.9 | 38.2 | 41.5 | 34229.8 | 62774.2 |
