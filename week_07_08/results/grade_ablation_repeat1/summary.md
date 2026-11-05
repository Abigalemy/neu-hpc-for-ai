| config | run | req/s | out tok/s | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | E2E p50 (ms) | E2E p99 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| ours_nocache | decode_heavy_c8 | 0.1 | 73.8 | 45.7 | 53.8 | 108.4 | 108.5 | 110921.0 | 111088.1 |
| ours_paged | balanced_c16 | 3.2 | 824.5 | 142.1 | 655.5 | 16.4 | 23.8 | 4315.6 | 6416.9 |
| ours_paged | decode_heavy_c8 | 0.5 | 504.7 | 57.1 | 61.3 | 15.8 | 15.9 | 16216.9 | 16337.3 |
| ours_paged | mixed_poisson_r8 | 3.3 | 806.1 | 45.8 | 850.5 | 37.4 | 55.2 | 8829.9 | 14675.1 |
| ours_static | balanced_c16 | 2.0 | 500.4 | 4070.5 | 4228.8 | 15.9 | 16.2 | 8129.7 | 8285.4 |
| ours_static | decode_heavy_c8 | 0.3 | 350.7 | 145.4 | 15193.7 | 15.3 | 15.4 | 15885.2 | 30814.7 |
| ours_static | mixed_poisson_r8 | 1.6 | 399.2 | 13992.3 | 25904.0 | 17.2 | 18.5 | 16226.9 | 29106.6 |
