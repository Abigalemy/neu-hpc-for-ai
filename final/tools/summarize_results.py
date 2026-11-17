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
    ("baseline", "suite_ours_1794881935/baseline", "suite_vllm_1794881615/baseline"),
    ("long_out", "suite_ours_1794881935/long_out", "suite_vllm_1794881615/long_out"),
    ("balanced_c16", "suite_ours_1794881935/balanced_c16", "suite_vllm_1794881615/balanced_c16"),
    ("balanced_c64", "suite_ours_1794881935/balanced_c64", "suite_vllm_1794881615/balanced_c64"),
    ("prefill_heavy", "suite_ours_1794881935/prefill_heavy", "suite_vllm_1794881615/prefill_heavy"),
    ("decode_heavy", "suite_ours_1794881935/decode_heavy", "suite_vllm_1794881615/decode_heavy"),
    ("128k_yarn", "ours_isl130816_osl128_1794882219", "vllm_isl130816_osl128_1794875810"),
]


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
