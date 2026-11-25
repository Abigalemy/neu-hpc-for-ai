"""Offline acceptance of a paired AIPerf run (ours vs vLLM), before any number from it is reported. No GPU, no Modal.

  python tools/check_suite.py results/aiperf/suite_Qwen3-30B-A3B_ours_1795057331 \
                              results/aiperf/suite_Qwen3-30B-A3B_vllm_1795049169 [--mode suite]

For every workload of the mode (workloads.py), on BOTH sides: the export exists, the run is complete and not cancelled,
the completed request count is the planned one, no errors, every prompt (AIPerf's count) and every server-reported
completion (usage) is exactly the target length (stddev 0, ignore_eos), every request reports its prompt usage, and the
concurrency is the planned one. Every field a check reads must be present: a missing field fails. Across the two sides:
the same AIPerf version, the same AIPerf input configuration (everything except the artifact directory and the server
URL), and the same server-counted prompt LENGTH per request (e.g. 128 synthetic tokens -> 136 with the chat template, on
both); lengths only, not proof of identical token ids. Needs only the standard library and the result files
(workloads.py is plain data); --sync checks the runners' lists against workloads.py. Exit code 0 only if every check
passes; every failure is printed.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workloads import WORKLOADS  # noqa: E402  (plain data: no Modal needed to check results)


def load(path):
    f = os.path.join(path, "profile_export_aiperf.json")
    return json.load(open(f)) if os.path.exists(f) else None


def side_checks(d, name, isl, osl, conc, reqs):
    """Every field a check relies on must be PRESENT: a missing field is a failure, never a default that passes."""
    bad = []
    if d is None:
        return ["no export (workload skipped or failed)"]
    if d.get("is_complete") is not True or d.get("was_cancelled") is not False:
        bad.append(f"is_complete={d.get('is_complete')} was_cancelled={d.get('was_cancelled')}")
    if not d.get("aiperf_version"):
        bad.append("aiperf_version missing")
    if not d.get("input_config"):
        bad.append("input_config missing")
    done = (d.get("completed_request_count") or {}).get("avg")
    if done != reqs:
        bad.append(f"completed {done} of {reqs} requests")
    rate = (d.get("request_error_rate") or {}).get("avg")
    if rate != 0 or "error_summary" not in d or d["error_summary"]:
        bad.append(f"errors: rate {rate} summary {d.get('error_summary', 'missing')}")
    up = d.get("usage_prompt_tokens") or {}
    if up.get("count") != reqs or up.get("min") is None or up.get("min") != up.get("max"):
        bad.append(f"usage_prompt_tokens: min {up.get('min')} max {up.get('max')} count {up.get('count')}, "
                   f"want one value reported by all {reqs} requests")
    # input: AIPerf's own synthetic prompt length; output: what the SERVER reports generating (usage), which must be
    # exactly the target. AIPerf's output_sequence_length re-tokenises the streamed text and can differ by a few
    # tokens on BOTH servers (token merges at re-tokenisation), so it is not a check, only noted.
    for key, want in (("input_sequence_length", isl), ("usage_completion_tokens", osl)):
        m = d.get(key, {})
        if m.get("min") != want or m.get("max") != want or m.get("count") != reqs:
            bad.append(f"{key}: min {m.get('min')} max {m.get('max')} count {m.get('count')}, want {want} x {reqs}")
    prof = [p for p in d.get("input_config", {}).get("phases", []) if p.get("kind") == "profiling"]
    if not prof or prof[0].get("concurrency") != conc or prof[0].get("requests") != reqs:
        bad.append(f"profiling phase {prof}, want concurrency {conc}, {reqs} requests")
    return bad


def comparable(cfg):
    cfg = json.loads(json.dumps(cfg))
    cfg.pop("artifacts", None)
    cfg.get("endpoint", {}).pop("urls", None)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours", nargs="?")
    ap.add_argument("vllm", nargs="?")
    ap.add_argument("--mode", default="suite")
    ap.add_argument("--sync", action="store_true", help="check the runners' workload lists against workloads.py "
                    "(imports both runners, so it needs the modal package)")
    a = ap.parse_args()
    if a.sync:
        import modal_aipref
        import modal_vllm_aipref
        bad = [f"{runner.__name__} {mode}" for runner in (modal_aipref, modal_vllm_aipref)
               for mode, cfg in runner.CONFIGS.items() if cfg[0] != WORKLOADS.get(mode)]
        print("runner workloads agree with workloads.py" if not bad else f"DIFFER: {bad}")
        sys.exit(1 if bad else 0)
    if not (a.ours and a.vllm):
        ap.error("give the ours and vllm result directories (or --sync)")
    works = WORKLOADS[a.mode]
    failures = 0
    for name, isl, osl, conc, reqs in works:
        osl_note = []
        o, v = load(os.path.join(a.ours, name)), load(os.path.join(a.vllm, name))
        problems = [f"ours: {p}" for p in side_checks(o, name, isl, osl, conc, reqs)]
        problems += [f"vllm: {p}" for p in side_checks(v, name, isl, osl, conc, reqs)]
        if o is not None and v is not None:
            if o.get("aiperf_version") != v.get("aiperf_version"):
                problems.append(f"aiperf version {o.get('aiperf_version')} vs {v.get('aiperf_version')}")
            if comparable(o.get("input_config", {})) != comparable(v.get("input_config", {})):
                problems.append("AIPerf input configuration differs between the two sides")
            up = [(x.get("usage_prompt_tokens") or {}).get(k) for x in (o, v) for k in ("min", "max")]
            if None in up or len(set(up)) != 1:   # the same prompt LENGTH per request (not proof of identical ids)
                problems.append(f"server-counted prompt lengths differ (ours min/max, vllm min/max): {up}")
            osl_note = [x.get("output_sequence_length", {}).get(k) for x in (o, v) for k in ("min", "max")]
        failures += bool(problems)
        note = f"  (AIPerf re-tokenised output min/max: ours {osl_note[:2]}, vllm {osl_note[2:]})" if o and v else ""
        print(f"{'ok  ' if not problems else 'FAIL'} {name}: {isl}/{osl}, c{conc}, {reqs} requests{note}")
        for p in problems:
            print(f"       {p}")
    print("ALL WORKLOADS COMPLETE AND COMPARABLE" if not failures else f"{failures} workload(s) failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
