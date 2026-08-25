#!/bin/bash
# PA1/PA2 러닝 가드: 하드시한 600s, TERM만, 백그라운드 감시.
# usage: pa12_run.sh <python_bin> <plan_json> <logfile> [ENVKEY=VAL ...]
set -uo pipefail
PYBIN="$1"; PLAN="$2"; LOG="$3"; shift 3
for kv in "$@"; do export "$kv"; done
cd /Users/Shared/tp2
echo "[guard] launch $(date) pybin=$PYBIN plan=$PLAN envs=$* " >> "$LOG"
"$PYBIN" e1_prefill_bench.py --model ~/dsv4flash/mlx4bit --require-world 1 --plan "$PLAN" >> "$LOG" 2>&1 &
PID=$!
(
  sleep 600
  if kill -0 "$PID" 2>/dev/null; then
    echo "[guard] 600s hard deadline reached — sending TERM to $PID" >> "$LOG"
    kill -TERM "$PID"
  fi
) &
GUARD=$!
wait "$PID"
RC=$?
kill "$GUARD" 2>/dev/null
wait "$GUARD" 2>/dev/null
echo "[guard] exit rc=$RC" >> "$LOG"
exit $RC
