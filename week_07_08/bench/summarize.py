"""Collect AIPerf results under <results_dir>/<config>/<workload>_<param>/ into one markdown table.

AIPerf writes a JSON export (profile_export_aiperf.json) per run. The exact schema can change between
versions, so this script searches the JSON defensively for the metrics we care about. If a metric is
not found its cell is '-'; check one raw JSON file on the first run and adjust METRICS if needed.
"""
import glob
import json
import os
import sys

# column title -> candidate metric keys in the AIPerf JSON
METRICS = {
    "req/s": ["request_throughput"],
    "out tok/s": ["output_token_throughput"],
    "TTFT p50 (ms)": ["time_to_first_token"],
    "TTFT p99 (ms)": ["time_to_first_token"],
    "ITL p50 (ms)": ["inter_token_latency"],
    "ITL p99 (ms)": ["inter_token_latency"],
    "E2E p50 (ms)": ["request_latency"],
    "E2E p99 (ms)": ["request_latency"],
}
STAT = {"req/s": "avg", "out tok/s": "avg"}


def find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find(v, key)
            if r is not None:
                return r
    return None


def cell(data, col):
    for key in METRICS[col]:
        m = find(data, key)
        if m is None:
            continue
        stat = STAT.get(col, "p50" if "p50" in col else "p99")
        v = m.get(stat) if isinstance(m, dict) else m
        if v is not None:
            return f"{v:.1f}" if isinstance(v, (int, float)) else str(v)
    return "-"


def main(root):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "profile_export_aiperf.json"))):
        parts = path.split(os.sep)
        cfg, run = parts[-3], parts[-2]
        try:
            data = json.load(open(path))
        except Exception:
            continue
        rows.append((cfg, run, [cell(data, c) for c in METRICS]))
    print("| config | run | " + " | ".join(METRICS) + " |")
    print("|---|---|" + "---|" * len(METRICS))
    for cfg, run, cells in rows:
        print(f"| {cfg} | {run} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
