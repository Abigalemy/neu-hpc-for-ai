| config | run | req/s | out tok/s | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | E2E p50 (ms) | E2E p99 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| ours_nocache | decode_heavy_c8 | 0.1 | 74.5 | 69.5 | 82.5 | 107.3 | 107.7 | 109873.9 | 110265.0 |
| ours_paged | balanced_c16 | 2.2 | 562.6 | 161.3 | 848.0 | 25.6 | 33.5 | 6688.7 | 9001.4 |
| ours_paged | decode_heavy_c8 | 0.3 | 338.8 | 84.7 | 113.3 | 23.5 | 23.9 | 24165.3 | 24478.2 |
| ours_paged | mixed_poisson_r8 | 3.0 | 740.5 | 66.3 | 1063.3 | 40.9 | 62.9 | 9816.8 | 15994.0 |
| ours_static | balanced_c16 | 1.3 | 337.9 | 5558.7 | 7479.3 | 23.8 | 26.6 | 11644.1 | 13079.1 |
| ours_static | decode_heavy_c8 | 0.2 | 164.4 | 24707.2 | 25292.4 | 24.4 | 24.6 | 49819.4 | 50135.0 |
| ours_static | mixed_poisson_r8 | 1.1 | 276.2 | 22069.5 | 40162.6 | 24.1 | 24.9 | 24761.9 | 45390.7 |
