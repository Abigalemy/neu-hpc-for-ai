"""Builds results/summary.csv and results/comparison.csv from the AIPerf exports in results/aiperf/.

  python tools/summarize_results.py

results/aiperf/<run>/[<workload>/]profile_export_aiperf.json are AIPerf's own summary exports, copied unchanged from
the Modal volume `final-results` (per-request logs and synthetic prompts are left on the volume). Every number in the
README's result tables can be traced to one of these files.
"""
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")

# (workload, our run dir, vLLM run dir): the pairs compared in the README
PAIRS = [
    ("baseline", "suite_ours_1795031469/baseline", "suite_vllm_1794881615/baseline"),
    ("long_out", "suite_ours_1795031469/long_out", "suite_vllm_1794881615/long_out"),
    ("balanced_c16", "suite_ours_1795031469/balanced_c16", "suite_vllm_1794881615/balanced_c16"),
    ("balanced_c64", "suite_ours_1795031469/balanced_c64", "suite_vllm_1794881615/balanced_c64"),
    ("prefill_heavy", "suite_ours_1795031469/prefill_heavy", "suite_vllm_1794881615/prefill_heavy"),
    ("decode_heavy", "suite_ours_1795031469/decode_heavy", "suite_vllm_1794881615/decode_heavy"),
    ("128k_yarn", "ours_isl130816_osl128_1795026977", "vllm_isl130816_osl128_1794875810"),
    # the AIPerf example of the Week 7 lecture: 100 connections, 1000 requests, 1000 in / 500 out, 2048 x 128 limits
    ("lecture_c100", "lecture_ours_1794894311/lecture_c100", "lecture_vllm_1794893920/lecture_c100"),
    # PagedAttention with and without (2026-11-18): one configuration, context 32768, KV pool 262144 tokens for ours;
    # "paged" = 128 sequences share the pool, "slots8" = at most 8 (= 8 contiguous slots of 32768 in the same memory,
    # 200 lecture requests instead of 1000); vLLM with --max-model-len 32768 --max-num-seqs 128
    ("paged/lecture_c100", "paged_ours_1795030271/lecture_c100", "paged_vllm_1795029909/lecture_c100"),
    ("paged/long_c8", "paged_ours_1795030271/long_c8", "paged_vllm_1795029909/long_c8"),
    ("slots8/lecture_c100", "slots8_ours_1795030663/lecture_c100", "paged_vllm_1795029909/lecture_c100"),
    ("slots8/long_c8", "slots8_ours_1795030663/long_c8", "paged_vllm_1795029909/long_c8"),
    # Part 1 (2026-11-18): Qwen3-30B-A3B (MoE), 1x H200, same 6-workload suite, both at revision ad44e777
    ("30B/baseline", "suite_Qwen3-30B-A3B_ours_1795057331/baseline", "suite_Qwen3-30B-A3B_vllm_1795049169/baseline"),
    ("30B/long_out", "suite_Qwen3-30B-A3B_ours_1795057331/long_out", "suite_Qwen3-30B-A3B_vllm_1795049169/long_out"),
    ("30B/balanced_c16", "suite_Qwen3-30B-A3B_ours_1795057331/balanced_c16", "suite_Qwen3-30B-A3B_vllm_1795049169/balanced_c16"),
    ("30B/balanced_c64", "suite_Qwen3-30B-A3B_ours_1795057331/balanced_c64", "suite_Qwen3-30B-A3B_vllm_1795049169/balanced_c64"),
    ("30B/prefill_heavy", "suite_Qwen3-30B-A3B_ours_1795057331/prefill_heavy", "suite_Qwen3-30B-A3B_vllm_1795049169/prefill_heavy"),
    ("30B/decode_heavy", "suite_Qwen3-30B-A3B_ours_1795057331/decode_heavy", "suite_Qwen3-30B-A3B_vllm_1795049169/decode_heavy"),
    # Parts 3 + 5 (2026-11-19): Qwen3-30B-A3B on 2x H200, DP = EP = 2, TP = 1; ours through part5/launch.py, vLLM with
    # --data-parallel-size 2 --enable-expert-parallel
] + [(f"30B_2gpu/{w}", f"suite_Qwen3-30B-A3B_ep2_ours_1795147115/{w}", f"suite_Qwen3-30B-A3B_dp2ep_vllm_1795151843/{w}")
     for w in ("baseline", "long_out", "balanced_c16", "balanced_c64", "prefill_heavy", "decode_heavy")]


def flag(cli, name):
    m = re.search(rf"--{name} (\S+)", cli)
    return m.group(1).strip("'") if m else ""


def load(rel):
    with open(os.path.join(RES, "aiperf", rel, "profile_export_aiperf.json")) as f:
        d = json.load(f)
    cli = d["run_info"]["cli_command"]
    g = lambda k, s="avg": d[k][s]
    return {
        "run": rel, "engine": "vllm" if "vllm" in rel else "ours",
        "isl": flag(cli, "synthetic-input-tokens-mean"), "osl": flag(cli, "output-tokens-mean"),
        "concurrency": flag(cli, "concurrency"), "requests": int(g("request_count")),
        "error_rate_pct": g("request_error_rate"),
        "ttft_avg_ms": g("time_to_first_token"), "ttft_p50_ms": g("time_to_first_token", "p50"),
        "ttft_p99_ms": g("time_to_first_token", "p99"),
        "itl_avg_ms": g("inter_token_latency"), "itl_p50_ms": g("inter_token_latency", "p50"),
        "itl_p99_ms": g("inter_token_latency", "p99"),
        "output_tok_s": g("output_token_throughput"), "request_latency_avg_ms": g("request_latency"),
    }


def main():
    runs = []
    for root, _, files in os.walk(os.path.join(RES, "aiperf")):
        if "profile_export_aiperf.json" in files:
            runs.append(os.path.relpath(root, os.path.join(RES, "aiperf")))
    rows = [load(r) for r in sorted(runs)]
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()} for r in rows)

    with open(os.path.join(RES, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "isl", "osl", "concurrency", "ttft_vllm_ms", "ttft_ours_ms", "ttft_ratio",
                    "itl_vllm_ms", "itl_ours_ms", "itl_ratio", "tok_s_vllm", "tok_s_ours", "errors_ours_pct",
                    "run_ours", "run_vllm"])
        for name, ours, vllm in PAIRS:
            o, v = load(ours), load(vllm)
            assert (o["isl"], o["osl"], o["concurrency"]) == (v["isl"], v["osl"], v["concurrency"]), (name, o, v)
            w.writerow([name, o["isl"], o["osl"], o["concurrency"], round(v["ttft_avg_ms"], 2), round(o["ttft_avg_ms"], 2),
                        round(o["ttft_avg_ms"] / v["ttft_avg_ms"], 2), round(v["itl_avg_ms"], 2),
                        round(o["itl_avg_ms"], 2), round(o["itl_avg_ms"] / v["itl_avg_ms"], 2),
                        round(v["output_tok_s"]), round(o["output_tok_s"]), o["error_rate_pct"], ours, vllm])
            print(f"{name:14s} TTFT {o['ttft_avg_ms'] / v['ttft_avg_ms']:5.2f}x  ITL {o['itl_avg_ms'] / v['itl_avg_ms']:5.2f}x")


if __name__ == "__main__":
    main()
