#!/usr/bin/env bash
# Run the AIPerf workload suite against an already-running OpenAI-compatible server.
#
#   bench/run_aiperf.sh <tag> [url] [model] [tokenizer] [workload ...]
#
# Workloads (each answers a different question about the server):
#   baseline        conc=1, ISL 128 / OSL 128        -> latency floor (no queueing, batch=1)
#   balanced        conc sweep, ISL 256 / OSL 256    -> realistic throughput/latency curve
#   prefill_heavy   conc sweep, ISL 2048 / OSL 32    -> prefill (TTFT) bottleneck
#   decode_heavy    conc sweep, ISL 64 / OSL 1024    -> decode / KV-cache growth, TPOT vs batch size
#   mixed_poisson   Poisson arrivals, variable lengths -> realistic open-loop traffic, queueing delay
#   pressure        conc=128, ISL 1024 / OSL 512     -> KV memory pressure (contiguous vs paged, preemption)
# Output: $OUT_DIR/<tag>/<workload>_<param>/  (AIPerf artifacts)
set -euo pipefail

TAG=${1:?usage: run_aiperf.sh <tag> [url] [model] [tokenizer] [workloads...]}
URL=${2:-http://localhost:8000}
MODEL=${3:-Qwen/Qwen3-4B}
TOK=${4:-$MODEL}
shift $(( $# < 4 ? $# : 4 )) || true
WORKLOADS=("$@")
[ ${#WORKLOADS[@]} -eq 0 ] && WORKLOADS=(baseline balanced prefill_heavy decode_heavy mixed_poisson pressure)
OUT=${OUT_DIR:-results}/$TAG
mkdir -p "$OUT"

COMMON=(--model "$MODEL" --url "$URL" --endpoint-type chat --streaming --tokenizer "$TOK"
        --extra-inputs '{"ignore_eos": true}' --warmup-request-count 8 --random-seed 0)

# SCALE_PCT shrinks every request count (e.g. SCALE_PCT=25 for a cheap smoke run); minimum 8 requests.
scale() { local n=$(( $1 * ${SCALE_PCT:-100} / 100 )); echo $(( n < 8 ? 8 : n )); }

# closed(name conc isl isl_std osl osl_std nreq)
closed() {
  aiperf profile "${COMMON[@]}" --concurrency "$2" --request-count "$(scale "$7")" \
    --synthetic-input-tokens-mean "$3" --synthetic-input-tokens-stddev "$4" \
    --output-tokens-mean "$5" --output-tokens-stddev "$6" --artifact-dir "$OUT/$1_c$2"
}
# open(name rate isl isl_std osl osl_std nreq)
open_loop() {
  aiperf profile "${COMMON[@]}" --request-rate "$2" --arrival-pattern poisson --request-count "$(scale "$7")" \
    --synthetic-input-tokens-mean "$3" --synthetic-input-tokens-stddev "$4" \
    --output-tokens-mean "$5" --output-tokens-stddev "$6" --artifact-dir "$OUT/$1_r$2"
}

# A workload may carry an explicit sweep, e.g. "balanced:4+16+64" (concurrency or request rates).
for w in "${WORKLOADS[@]}"; do
  name=${w%%:*}; LIST=""
  if [[ $w == *:* ]]; then LIST=${w#*:}; LIST=${LIST//+/ }; fi
  echo "=== [$TAG] workload: $name ${LIST:+(sweep: $LIST)} ==="
  case $name in
    baseline)      closed baseline 1 128 0 128 0 16 ;;
    balanced)      for c in ${LIST:-1 4 16 64};   do closed balanced $c 256 0 256 0 $(( c*6 < 48 ? 48 : c*6 )); done ;;
    prefill_heavy) for c in ${LIST:-1 8 32};      do closed prefill_heavy $c 2048 0 32 0 $(( c*6 < 48 ? 48 : c*6 )); done ;;
    decode_heavy)  for c in ${LIST:-1 8 32 64};   do closed decode_heavy $c 64 0 1024 0 $(( c*4 < 32 ? 32 : c*4 )); done ;;
    mixed_poisson) for r in ${LIST:-2 4 8 16};    do open_loop mixed_poisson $r 512 256 256 128 120; done ;;
    pressure)      closed pressure 128 1024 0 512 0 256 ;;
    *) echo "unknown workload $name" >&2; exit 1 ;;
  esac
done
