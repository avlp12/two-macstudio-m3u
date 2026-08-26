#!/bin/bash
# ═══ 웨지 근본원인 감사 · 콜드-스타트 반복 재현 프로토콜 ═══════════════════
# WEDGE_ROOT_CAUSE_2026-08-26.md §5 의 실행체.
#
# 프로토콜: 사이클 = [로드 → 짧은 작업 1개 → 종료]. 매 사이클이 독립 프로세스라
# 매번이 콜드 스타트다 — 웨지 5회가 전부 그 창에서 났다. **두 변종**을 돈다:
#   A) all-topics : --batch 1 --all-topics --max-topics 1 --long-doc 60 (2,120tok, MTP on)
#                   → bs1 웨지 4회와 동일 구성
#   B) batch      : --batch 1 --long-doc 397 --max-tokens 32 (13.9K tok, MTP off)
#                   → 5번째 웨지 b1tp2_on1.log 와 동일 구성 (레드팀 M4)
# 변종 B 가 빠져 있던 초판은 "웨지가 실제로 난 경로"를 검증하지 않았다.
#
# 판정: 전 사이클 클린 + 웨지 시그니처 0건 + WEDGE_ALERT 증가 0건(**양 박스**).
#
# 안전(가드 규약): 러닝당 하드 시한 600s · TERM 만(KILL 금지) · TERM 불응이면
# 즉시 체인 중단·재부팅 권고 · 사이클마다 종료-검증(프로세스 0 + wired<50G 양 박스).
# 웨지 재현 시 스택은 하네스 워치독이 TERM 전에 /usr/bin/sample 로 자동 채취한다.
#
# usage: wedge_verify.sh [N_ALLTOPICS=8] [N_BATCH=4]
set -u
source "$HOME/local-llm-serving/serving/tp2_guard.sh"

N_ALL="${1:-8}"
N_BATCH="${2:-4}"
EPS="m3ms@10.0.0.2"
SSHO="-n -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=4 -o ServerAliveCountMax=3"
LOGDIR="$HOME/dsv4flash/align/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="$LOGDIR/wedge_verify_${STAMP}.log"
DEADLINE=600          # 러닝당 하드 시한
GRACE=150             # pass 감지 후 자연 종료를 기다리는 유예(즉시-TERM 금지)
ALERT="$LOGDIR/WEDGE_ALERT"
mkdir -p "$LOGDIR"

# log 는 **stderr** 로만 나간다. stdout 으로 흘리면 함수를 명령치환으로 받는
# 자리마다 반환값이 오염된다(블랭크-슬레이트 리뷰 MUST-4).
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$SUMMARY" >&2; }

# ── R1: 어떤 실패 경로에서도 정리 블록이 돈다 ─────────────────────────────
FINISHED=0
cleanup() {
  local rc=$?
  if [ "$FINISHED" -ne 1 ]; then
    log "[trap] 비정상 종료(rc=$rc) — 정리 시도(TERM 만)."
    tp2_safe_shutdown || log "[trap] TERM-불응 — KILL 금지. 재부팅 권고."
  fi
}
trap cleanup EXIT

# ── WEDGE_ALERT 계수: **양 박스** ──────────────────────────────────────────
# rank1 은 엡실론에서 돌고 자기 홈의 WEDGE_ALERT 에 쓴다. 게지히트만 세면
# rank1-only 스톨이 통과로 오판된다(블랭크-슬레이트 리뷰 SHOULD).
# grep -cv 는 매치 0건이면 "0" 을 찍고 **exit 1** 이므로, `|| echo 0` 을 쓰면
# "0\n0" 이 되어 산술식이 깨진다(레드팀 M3 — 성공 케이스에서만 터지던 버그).
_alert_local() {
  local n=0
  [ -f "$ALERT" ] && n=$(grep -cv "^#" "$ALERT" 2>/dev/null || true)
  echo "${n:-0}"
}
_alert_remote() {
  local n
  n=$(ssh $SSHO "$EPS" 'f=$HOME/dsv4flash/align/logs/WEDGE_ALERT; [ -f "$f" ] && grep -cv "^#" "$f" || echo 0' 2>/dev/null | tail -1)
  case "$n" in ''|*[!0-9]*) echo 0;; *) echo "$n";; esac
}
alert_total() { echo $(( $(_alert_local) + $(_alert_remote) )); }

# ── 배포 동기화: scp + md5 대조 (랭크 비대칭 사고 2회 전례) ────────────────
deploy() {
  local f a b
  for f in run_tp2_flash.py; do
    cp "$HOME/local-llm-serving/bench/$f" "/Users/Shared/tp2/$f" || return 1
    scp -q "/Users/Shared/tp2/$f" "$EPS:/Users/Shared/tp2/$f" || return 1
    a=$(md5 -q "/Users/Shared/tp2/$f")
    b=$(ssh $SSHO "$EPS" "md5 -q /Users/Shared/tp2/$f" 2>/dev/null)
    [ "$a" = "$b" ] || { log "[deploy] md5 불일치 $f: g=$a e=$b"; return 1; }
    log "[deploy] $f md5=$a (양 랭크 일치)"
  done
  for f in wedge_verify_worker.sh; do
    cp "$HOME/local-llm-serving/bench/$f" "/Users/Shared/tp2/exp_chain/$f" || return 1
    chmod +x "/Users/Shared/tp2/exp_chain/$f"
    scp -q "/Users/Shared/tp2/exp_chain/$f" "$EPS:/Users/Shared/tp2/exp_chain/$f" || return 1
    ssh $SSHO "$EPS" "chmod +x /Users/Shared/tp2/exp_chain/$f" 2>/dev/null
    a=$(md5 -q "/Users/Shared/tp2/exp_chain/$f")
    b=$(ssh $SSHO "$EPS" "md5 -q /Users/Shared/tp2/exp_chain/$f" 2>/dev/null)
    [ "$a" = "$b" ] || { log "[deploy] md5 불일치 $f: g=$a e=$b"; return 1; }
    log "[deploy] $f md5=$a (양 랭크 일치)"
  done
  return 0
}

# ── 하드 시한 러너: TERM 만, KILL 금지 ────────────────────────────────────
# 경과초는 전역 LAST_ELAPSED 로 돌려준다 — 명령치환으로 받으면 안 된다.
LAST_ELAPSED=0
run_cycle() {
  local i="$1" logfile="$2" gate="$3"; shift 3
  # `exec` 가 있어야 $! 가 서브셸이 아니라 mlx.launch 자신을 가리킨다.
  # 없으면 kill -TERM 이 서브셸만 때리고 런처는 살아남는다.
  ( cd /Users/Shared/tp2 && exec "$HOME/venv_omlx063/bin/mlx.launch" \
      --hostfile hostfile_ring2.json \
      /Users/Shared/tp2/exp_chain/wedge_verify_worker.sh "$gate" "$@" ) > "$logfile" 2>&1 &
  local pid=$! waited=0 g=0 hit=""
  while kill -0 "$pid" 2>/dev/null; do
    sleep 5; waited=$((waited+5))
    if [ -z "$hit" ]; then
      grep -qa "tp2-flash-pass" "$logfile" 2>/dev/null && hit=pass
      grep -qa "Traceback" "$logfile" 2>/dev/null && hit=err
    fi
    if [ -n "$hit" ]; then
      # 즉시-TERM 금지: 트레일링 Metal 작업 중 종료는 오염 방아쇠다(가드 R2).
      # 자연 종료를 GRACE 까지 기다리고, 그래도 남으면 tp2_safe_shutdown 이
      # 자체 20s 콰이스를 거쳐 TERM 한다.
      g=$((g+5))
      [ "$g" -ge "$GRACE" ] && break
      continue
    fi
    if [ "$waited" -ge "$DEADLINE" ]; then
      log "  [deadline] 사이클 $i: ${DEADLINE}s 초과 — TERM(KILL 금지). 스택은 하네스 워치독이 이미 채취했어야 한다."
      kill -TERM "$pid" 2>/dev/null
      local w2=0
      while kill -0 "$pid" 2>/dev/null && [ "$w2" -lt 180 ]; do sleep 5; w2=$((w2+5)); done
      break
    fi
  done
  kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  LAST_ELAPSED="$waited"
}

# clean | wedge | error
# 웨지 = "죽지도 끝나지도 않았다". 시그니처를 --all-topics 로그 모양
# (`[p1-topic 0/…]` 에서 끊김)에만 걸어 두면 --batch 변종(로드 후 곧장 무출력,
# 토픽 행 자체가 없음)이 error 로 오분류된다(레드팀 M4). 하드 시한 도달 자체를
# 1차 판별자로 쓰고, 로그 위치는 사후 분류에만 쓴다.
verdict_of() {
  local f="$1" el="$2"
  grep -qa "tp2-flash-pass" "$f" && { echo clean; return; }
  grep -qa "Traceback" "$f" && { echo error; return; }
  [ "$el" -ge "$DEADLINE" ] && { echo wedge; return; }
  grep -qa "p1-topic 0/" "$f" && { echo wedge; return; }
  grep -qa "load+shard" "$f" && { echo wedge; return; }   # 로드는 끝났는데 pass 없음
  echo error
}

# 웨지 시 원격 랭크 스택을 회수해 온다(로컬에만 있으면 rank1 지문을 못 본다).
fetch_remote_stacks() {
  local n
  n=$(ssh $SSHO "$EPS" 'ls -1 $HOME/dsv4flash/align/logs/wedge_sample_r*.txt 2>/dev/null | wc -l' 2>/dev/null | tr -d ' ')
  [ "${n:-0}" -gt 0 ] || { log "  [stack] 원격 스택 없음"; return 0; }
  scp -q "$EPS:\$HOME/dsv4flash/align/logs/wedge_sample_r*.txt" "$LOGDIR/" 2>/dev/null \
    && log "  [stack] 원격 스택 $n건 회수 → $LOGDIR/" \
    || log "  [stack] 원격 스택 회수 실패(수동 확인 필요)"
}

log "═══ 웨지 검증 시작 · all-topics×$N_ALL + batch×$N_BATCH · 하드시한 ${DEADLINE}s ═══"
A0=$(alert_total)
log "WEDGE_ALERT 기준선(양 박스 합계)=$A0"

tp2_chain_ok || { log "[FATAL] 전-발사 게이트 불통과 — 양 박스 재부팅 필요. 중단."; exit 90; }
deploy || { log "[FATAL] 배포 동기화 실패 — 중단."; exit 91; }

# 게이트 스펙 = FS:WARMUP:WTOK:WATCHDOG_S:MTP — argv 로 넘어가므로 양 랭크 동일 보장.
GATE_ALL="0:1:32:300:1"
GATE_BATCH="0:1:32:300:0"

CLEAN=0; WEDGE=0; ERR=0; STOP=0; IDX=0
run_variant() {   # $1=이름 $2=횟수 $3=게이트 나머지=하네스 인자
  local name="$1" n="$2" gate="$3"; shift 3
  local k
  for k in $(seq 1 "$n"); do
    [ "$STOP" -eq 1 ] && return 0
    IDX=$((IDX+1))
    tp2_chain_ok || { log "[FATAL] 사이클 $IDX($name) 전 게이트 불통과 — 체인 중단."; STOP=2; return 0; }
    local LF="$LOGDIR/wedge_cycle${IDX}_${name}_${STAMP}.log"
    log "── 사이클 $IDX [$name $k/$n] → $(basename "$LF")"
    run_cycle "$IDX" "$LF" "$gate" "$@"
    local EL="$LAST_ELAPSED" V BAN WARM
    V=$(verdict_of "$LF" "$EL")
    BAN=$(grep -ao "FAST_SYNCH(요청)=[01]" "$LF" | head -1)
    WARM=$(grep -a "\[tp2-warm\]" "$LF" | head -1)
    log "  판정=$V 경과=${EL}s ${BAN:-배너없음} ${WARM:-}"
    case "$V" in
      clean) CLEAN=$((CLEAN+1));;
      wedge) WEDGE=$((WEDGE+1));;
      *)     ERR=$((ERR+1));;
    esac
    if ! tp2_safe_shutdown; then
      log "[FATAL] 사이클 $IDX 정리 TERM-불응 — R2에 따라 KILL 금지·체인 중단. 재부팅 권고."
      STOP=3; return 0
    fi
    if [ "$V" = "wedge" ]; then
      fetch_remote_stacks
      log "[STOP] 웨지 재현($name) — 수정이 불충분하다. 스택: $LOGDIR/wedge_sample_r*.txt"
      STOP=1; return 0
    fi
  done
}

run_variant alltopics "$N_ALL"  "$GATE_ALL" \
  --batch 1 --all-topics --max-topics 1 --long-doc 60 --max-tokens 256
run_variant batch     "$N_BATCH" "$GATE_BATCH" \
  --batch 1 --long-doc 397 --max-tokens 32

A1=$(alert_total)
DELTA=$((A1-A0))
TOTAL=$((N_ALL+N_BATCH))
log "═══ 결과: clean=$CLEAN/$TOTAL wedge=$WEDGE error=$ERR · WEDGE_ALERT 증가(양 박스)=$DELTA ═══"
FINISHED=1
case "$STOP" in
  2) log "판정: 불통과 — 사이클 전 게이트 실패."; exit 92;;
  3) log "판정: 불통과 — TERM 불응."; exit 93;;
esac
if [ "$CLEAN" -ge "$TOTAL" ] && [ "$DELTA" -eq 0 ]; then
  log "판정: 통과 — $CLEAN/$TOTAL 클린, 웨지 시그니처·워치독 경보 없음(양 박스)."
  exit 0
fi
log "판정: 불통과."
exit 1
