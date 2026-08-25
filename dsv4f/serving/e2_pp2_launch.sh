#!/bin/bash
# E2 PP2 단독 발사 — 원격 서버(하단 슬라이스) + 로컬 클라이언트(상단 슬라이스).
# 규칙: R4 단독 발사(체인 금지) · R2 TERM-only(KILL 금지) · R5 원시 TCP(=jaccl/RDMA 미사용)
# 사용: e2_pp2_launch.sh <tag> <split> <chunk> <client_deadline_s> [extra client args...]
set -uo pipefail

TAG="$1"; SPLIT="$2"; CHUNK="$3"; DEADLINE="$4"; shift 4

EPS="${BOX_B:?export BOX_B=<user>@10.0.0.2}"        # ssh target of box B
PORT=39931
LOGDIR=${HOME}/dsv4flash/align/logs
CLOG="$LOGDIR/e2_${TAG}_client.log"
SLOG_R="${BOX_B_HOME:?export BOX_B_HOME=/Users/<user>}/dsv4flash/align/logs/e2_${TAG}_server.log"
SLOG_L="$LOGDIR/e2_${TAG}_server.log"
PYL=${HOME}/venv_omlx063/bin/python
PYR=${BOX_B_HOME:?export BOX_B_HOME=/Users/<user>}/venv_omlx063/bin/python
SCRIPT=/Users/Shared/tp2/e2_pp2_prefill.py
PAT="e2_pp2_prefill"

_lp() { pgrep -f "$PAT" 2>/dev/null | tr '\n' ' '; }
_rp() { ssh -o BatchMode=yes $EPS "pgrep -f '$PAT'" 2>/dev/null | tr '\n' ' '; }
_lw() { vm_stat | awk '/wired/ {print int($4*16384/1e9)}'; }
_rw() { ssh -o BatchMode=yes $EPS "vm_stat | awk '/wired/ {print int(\$4*16384/1e9)}'" 2>/dev/null || echo 999; }

say() { echo "[e2-pp2] $*" | tee -a "$CLOG"; }

: > "$CLOG"
say "=== pre-flight $(date) tag=$TAG split=$SPLIT chunk=$CHUNK ==="
say "local pids '$(_lp)' / remote pids '$(_rp)'"
say "local wired $(_lw)G / remote wired $(_rw)G"
if [ -n "$(_lp)$(_rp)" ]; then say "ABORT: 잔존 e2 프로세스"; exit 2; fi
LW=$(_lw); RW=$(_rw)
if [ "${LW:-999}" -ge 50 ] || [ "${RW:-999}" -ge 50 ]; then say "ABORT: wired ≥50G"; exit 2; fi

# --- md5 대조 (랭크 비대칭 함정)
L5=$(md5 -q "$SCRIPT"); R5=$(ssh -o BatchMode=yes $EPS "md5 -q $SCRIPT")
say "md5 local=$L5 remote=$R5"
[ "$L5" = "$R5" ] || { say "ABORT: md5 불일치"; exit 2; }

SRV_STARTED=0
cleanup() {
  local rc=$?
  if [ "$SRV_STARTED" = "1" ]; then
    if [ -n "$(_rp)" ]; then
      say "원격 서버 TERM (KILL 금지)"
      ssh -o BatchMode=yes $EPS "pkill -TERM -f '$PAT'" 2>/dev/null
      for i in $(seq 1 24); do [ -n "$(_rp)" ] || break; sleep 5; done
      [ -n "$(_rp)" ] && say "R2: 원격 TERM-불응 pid '$(_rp)' — KILL 하지 않음, 수동 확인 필요"
    fi
    scp -q $EPS:"$SLOG_R" "$SLOG_L" 2>/dev/null && say "server log -> $SLOG_L"
  fi
  say "=== post-flight: local pids '$(_lp)' remote pids '$(_rp)' ==="
  say "local wired $(_lw)G / remote wired $(_rw)G"
  say "exit rc=$rc $(date)"
}
trap cleanup EXIT

# --- 원격 서버 기동
say "원격 서버 기동 (slice [0,$SPLIT))"
ssh -o BatchMode=yes $EPS "cd ${BOX_B_HOME} && nohup $PYR $SCRIPT --mode server --split $SPLIT --port $PORT --host 0.0.0.0 --model ${BOX_B_HOME:?export BOX_B_HOME=/Users/<user>}/dsv4flash/mlx4bit > $SLOG_R 2>&1 & echo started \$!" | tee -a "$CLOG"
SRV_STARTED=1

for i in $(seq 1 60); do
  if ssh -o BatchMode=yes $EPS "grep -q 'listening on' $SLOG_R" 2>/dev/null; then
    say "원격 서버 준비 완료 (${i}0s 이내)"; break
  fi
  if [ -z "$(_rp)" ]; then say "ABORT: 원격 서버 사망"; ssh -o BatchMode=yes $EPS "tail -30 $SLOG_R" | tee -a "$CLOG"; exit 3; fi
  sleep 10
done
ssh -o BatchMode=yes $EPS "grep -q 'listening on' $SLOG_R" || { say "ABORT: 원격 서버 준비 타임아웃"; exit 3; }

# --- 로컬 클라이언트 (하드 시한)
say "클라이언트 발사 (deadline ${DEADLINE}s)"
"$PYL" "$SCRIPT" --mode client --split "$SPLIT" --chunk "$CHUNK" --host 10.0.0.2 \
  --port $PORT --tag "$TAG" --shutdown-server "$@" >>"$CLOG" 2>&1 &
CPID=$!
W=0
while kill -0 $CPID 2>/dev/null; do
  sleep 5; W=$((W+5))
  if [ $W -ge "$DEADLINE" ]; then
    say "하드 시한 ${DEADLINE}s 초과 — 클라이언트 TERM"
    kill -TERM $CPID 2>/dev/null
    for i in $(seq 1 24); do kill -0 $CPID 2>/dev/null || break; sleep 5; done
    kill -0 $CPID 2>/dev/null && { say "R2: 클라이언트 TERM-불응 — KILL 금지"; exit 124; }
    break
  fi
done
wait $CPID 2>/dev/null; RC=$?
say "client rc=$RC waited=${W}s"
exit $RC
