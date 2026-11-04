#!/usr/bin/env bash
# Launch each server configuration, run the AIPerf suite against it, then shut it down.
#
#   bench/run_experiments.sh [MODEL] [KV_GB] [config ...]
#
# Configs:
#   ours_nocache      continuous batching, NO KV cache (recompute every step)
#   ours_contiguous   continuous batching + KV cache, per-sequence contiguous slabs
#   ours_paged        continuous batching + KV cache + PagedAttention   (the full system)
#   ours_static       static batching + paged KV cache                  (batching ablation)
#   vllm              vLLM reference
# All of "ours_*" get the same KV memory budget (KV_GB) so contiguous vs paged is a fair comparison.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:-Qwen/Qwen3-4B}
KV_GB=${2:-8}
shift $(( $# < 2 ? $# : 2 )) || true
CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(ours_paged ours_contiguous ours_nocache ours_static vllm)
PORT=${PORT:-8000}
MAXLEN=${MAXLEN:-4096}
export OUT_DIR=${OUT_DIR:-results}
mkdir -p "$OUT_DIR" logs

SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; }
trap cleanup EXIT

wait_ready() {
  for _ in $(seq 1 600); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died, see logs/"; return 1; }
    sleep 2
  done
  echo "server did not become ready"; return 1
}

for cfg in "${CONFIGS[@]}"; do
  echo "################ $cfg ################"
  case $cfg in
    ours_paged)      CMD=(python server.py --cache-mode paged      --batching continuous) ;;
    ours_contiguous) CMD=(python server.py --cache-mode contiguous --batching continuous) ;;
    ours_nocache)    CMD=(python server.py --cache-mode none       --batching continuous) ;;
    ours_static)     CMD=(python server.py --cache-mode paged      --batching static) ;;
    vllm)            CMD=(vllm serve "$MODEL" --dtype bfloat16 --max-model-len "$MAXLEN" --max-num-seqs 64
                          --no-enable-prefix-caching --port "$PORT") ;;
    *) echo "unknown config $cfg"; exit 1 ;;
  esac
  if [[ $cfg == ours_* ]]; then
    CMD+=(--model "$MODEL" --max-model-len "$MAXLEN" --max-num-seqs 64 --kv-cache-gb "$KV_GB" --port "$PORT")
  fi
  "${CMD[@]}" > "logs/$cfg.log" 2>&1 &
  SERVER_PID=$!
  wait_ready
  bench/run_aiperf.sh "$cfg" "http://localhost:$PORT" "$MODEL" "$MODEL"
  [[ $cfg == ours_* ]] && curl -s "http://localhost:$PORT/metrics_json" > "$OUT_DIR/$cfg/engine_stats.json" || true
  cleanup
  sleep 5
done
python bench/summarize.py "$OUT_DIR" > "$OUT_DIR/summary.md" && echo "wrote $OUT_DIR/summary.md"
