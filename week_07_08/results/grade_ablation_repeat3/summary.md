| config | run | req/s | out tok/s | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | E2E p50 (ms) | E2E p99 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| ours_nocache | decode_heavy_c8 | 0.1 | 74.8 | 49.9 | 134.9 | 106.9 | 107.4 | 109435.0 | 109904.0 |
| ours_paged | balanced_c16 | 3.0 | 763.2 | 144.3 | 708.4 | 18.1 | 25.7 | 4755.6 | 6941.3 |
| ours_paged | decode_heavy_c8 | 0.4 | 459.9 | 58.4 | 67.9 | 17.4 | 17.5 | 17802.8 | 17967.0 |
| ours_paged | mixed_poisson_r8 | 3.2 | 797.3 | 51.9 | 937.2 | 38.3 | 58.0 | 9044.8 | 14869.5 |
| ours_static | balanced_c16 | 1.7 | 436.2 | 4751.6 | 4820.1 | 18.1 | 18.4 | 9319.1 | 9521.4 |
| ours_static | decode_heavy_c8 | 0.2 | 231.4 | 17454.8 | 17915.7 | 17.5 | 17.5 | 35392.3 | 35444.6 |
| ours_static | mixed_poisson_r8 | 1.4 | 355.3 | 16383.6 | 29817.3 | 18.5 | 19.8 | 18660.3 | 33604.7 |
