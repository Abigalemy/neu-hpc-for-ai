"""The AIPerf workloads of every benchmark mode, as plain data (no Modal import), so that tools/check_suite.py runs with
a system Python and only the result files. modal_aipref.py and modal_vllm_aipref.py define the same lists for their
containers; `python tools/check_suite.py --sync` checks that they still agree with this file.

Each workload: (name, input tokens, output tokens, concurrency, requests).
"""
SUITE = [
    ("baseline", 128, 128, 1, 8),            # latency floor without queueing
    ("long_out", 3072, 1024, 1, 3),          # the single-sequence workload measured before
    ("balanced_c16", 256, 256, 16, 32),      # batching at moderate concurrency
    ("balanced_c64", 256, 256, 64, 128),     # high concurrency
    ("prefill_heavy", 2048, 32, 8, 16),      # is prefill (TTFT) the bottleneck?
    ("decode_heavy", 64, 1024, 32, 64),      # large decode batches (ITL)
]
LECTURE = [("lecture_c100", 1000, 500, 100, 1000)]   # the AIPerf example of the Week 7 lecture
LONG = ("long_c8", 16000, 256, 8, 16)                 # long prompts at moderate concurrency (paged configuration)

WORKLOADS = {
    "suite": SUITE,
    "lecture": LECTURE,
    "paged": LECTURE + [LONG],
    "slots8": [("lecture_c100", 1000, 500, 100, 200), LONG],   # ours only: 8 sequences, fewer lecture requests
}
