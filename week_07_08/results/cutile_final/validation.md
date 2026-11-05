# CUDA Tile H100 matrix validation

Source: AIPerf `profile_export_aiperf.json` files. A profile counts as complete only when `is_complete=true` and `was_cancelled=false`. The contiguous pressure workload was run as a separate, authorized follow-up with a 20-minute cap; its measured profiling phase took 331.67 seconds.

| Config | Complete profiles | Planned | Profiles with request errors | AIPerf OSL mismatch requests |
|---|---:|---:|---:|---:|
| vllm | 13/13 | 13 | 0 | 0 |
| ours_paged | 13/13 | 13 | 0 | 1 |
| ours_contiguous | 13/13 | 13 | 0 | 0 |
| ours_nocache | 5/5 | 5 | 0 | 0 |

All planned profiles now have completed AIPerf JSON exports. No profile had request errors.

## AIPerf token-count discrepancy

- `ours_paged/decode_heavy_c64`: one request has AIPerf text-side output count 898, while server-reported completion usage is 1024. The server usage counter indicates the engine generated the requested count; evidence points to tokenizer round-trip accounting, not a confirmed early stop. For that request the text-side count is 12.30% lower; across the profile the AIPerf text-side count averages 0.4% below server usage. Request error rate is 0%.

That original profile used text-side token counting. `bench/run_aiperf.sh` now passes `--use-server-token-count`; the contiguous pressure follow-up reported exactly 512 tokens for all 256 requests and zero OSL mismatches.

## Pressure comparison

| Config | Output tok/s | TTFT p50 (ms) | ITL p50 (ms) | Completed / errors |
|---|---:|---:|---:|---:|
| vllm | 6062.2 | 5534.0 | 10.0 | 256 / 0% |
| ours_paged | 675.0 | 56390.8 | 65.0 | 256 / 0% |
| ours_contiguous | 395.2 | 142332.4 | 34.4 | 256 / 0% |

- `ours_contiguous` has 14 KV slots, no preemptions, 395.2 output tokens/s, and 142.3 s p50 TTFT under this 128-concurrency workload. This is consistent with severe queueing from its fixed-capacity cache; it is a measured result, not an extrapolation.
- GPU/server telemetry was not collected for the follow-up profile; AIPerf request-level metrics and usage counts are present.
