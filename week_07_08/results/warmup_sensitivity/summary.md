# Same-container warmup sensitivity study

One H100 container ran one persistent `ours_paged` server. For each of three sequential repetitions, the same server was profiled with AIPerf warmup omitted (0) and with 8 warmup requests, each for baseline and balanced c16. Order within a repetition was warmup 0 first, then warmup 8. The first no-warmup profile is the only truly first-use observation; later no-warmup profiles inherit JIT/kernel caches from earlier profiles. No preliminary inference request was sent before measurements.

All 12 profiles completed, with 0 request errors and 0 output-length mismatches. Each table cell is mean ± sample standard deviation across the three runs.

| Workload | AIPerf warmup requests | Output tok/s | TTFT p50 (ms) | ITL p50 (ms) | E2E p50 (ms) |
|---|---:|---:|---:|---:|---:|
| Baseline c1 | 0 | 63.7 ± 5.2 | 17.9 ± 0.8 | 15.2 ± 0.5 | 1,946.8 ± 62.3 |
| Baseline c1 | 8 | 66.4 ± 0.5 | 17.9 ± 0.1 | 14.9 ± 0.1 | 1,918.2 ± 11.5 |
| Balanced c16 | 0 | 945.5 ± 64.5 | 128.4 ± 2.3 | 16.1 ± 0.5 | 4,233.1 ± 119.7 |
| Balanced c16 | 8 | 977.6 ± 4.2 | 119.8 ± 20.0 | 16.0 ± 0.0 | 4,192.5 ± 25.8 |

Per-run output throughput (tok/s), to expose the first-use effect:

| Workload | Warmup 0, rounds 1/2/3 | Warmup 8, rounds 1/2/3 |
|---|---|---|
| Baseline c1 | 57.7/66.6/66.7 | 66.9/65.9/66.3 |
| Balanced c16 | 871.1/978.7/986.6 | 980.7/979.4/972.8 |

Interpretation: the first no-warmup round is lower than subsequent runs for both shapes (baseline 57.7 vs 66.6–66.7 tok/s; c16 871.1 vs 978.7–986.6 tok/s), consistent with a first-use or transient startup cost. Warmup-8 repeats are stable in this container. This suggests warmup/first-use explains part of the variation, but it does not explain the earlier baseline 26.4 tok/s or c16 402.4 tok/s; those remain outliers relative to this study and the cause is unverified. The current c16 warmup-8 mean is 977.6 tok/s, also above the separate 716.8 ± 137.0 tok/s three-run result; runs are unpaired and container/run-to-run variability remains a limitation.

Raw AIPerf profiles, per-profile logs, server logs, status manifest, and engine stats are stored beside this report. Modal run: [ap-Z0vxuuHVX0HC0apwTMMeYb](https://modal.com/apps/neu-info5100-oak-spr-2025/main/ap-Z0vxuuHVX0HC0apwTMMeYb).
