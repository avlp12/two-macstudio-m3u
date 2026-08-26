#!/bin/bash
# TP2 하네스 공유 수명주기 가드 — 모든 스윕/발사 스크립트는 이걸 source 할 것.
# (K3 k3_guard.sh 이식 + 2026-08-24 크래시 2회 교훈 반영)
#  R1. trap EXIT — 어떤 실패 경로에서도 정리 블록 실행
#  R2. TERM-불응 프로세스 KILL 절대 금지(RDMA/collective-갇힘 KILL=커널 오염·wired 누수)
#      → 갇힘 감지 시 즉시 중단+보고, 해당 박스 재부팅 권고
#  R3. 다음 구성 체인은 종료-검증(프로세스 0 + wired<50G 양 박스) 통과 후에만
#  R4. 웨지-위험 구성(그래프 형상 변경: depth/TP/커널 스위치)은 체인 금지 — 단독 발사
#  R5. 위험/미검증 분산 실험은 jaccl(RDMA) 금지 — TCP ring 백엔드 전용
#      (macOS 26 TB-RDMA 커널은 행-중-소멸에 QP/DMA 오염으로 반응 → 시스템 정지)

EPS="m3ms@10.0.0.2"
TP2_PAT="serve_batched_tp2|serve_tp4_dspark|jbench|run_tp2"

tp2_real_pids() {
  pgrep -f "${1:-$TP2_PAT}" 2>/dev/null | while read p; do
    case "$(ps -o comm= -p $p 2>/dev/null)" in *[Pp]ython*) echo $p;; esac
  done
}
_e_real() {
  ssh -o BatchMode=yes $EPS 'pgrep -f "serve_batched_tp2|serve_tp4_dspark|jbench" | while read p; do case "$(ps -o comm= -p $p)" in *[Pp]ython*) echo $p;; esac; done' 2>/dev/null
}
_g_wired() { vm_stat | awk '/wired/ {print int($4*16384/1e9)}'; }
_e_wired() { ssh -o BatchMode=yes $EPS "vm_stat | awk '/wired/ {print int(\$4*16384/1e9)}'" 2>/dev/null || echo 999; }

tp2_safe_shutdown() {   # 0=클린, 1=갇힘(중단·재부팅 권고) — KILL 안 함
  local P0=$(tp2_real_pids | head -1)
  if [ -n "$P0" ]; then
    sleep 20   # quiesce: 직후-TERM은 트레일링 Metal/RDMA 작업 중 종료→오염
    for p in $(tp2_real_pids); do kill -TERM $p 2>/dev/null; done
    ssh -o BatchMode=yes $EPS 'pkill -f "serve_batched_tp2|serve_tp4_dspark|jbench"' 2>/dev/null
    for i in $(seq 1 36); do [ -n "$(tp2_real_pids)" ] || break; sleep 5; done
    if [ -n "$(tp2_real_pids)" ]; then
      echo "[guard] TERM-불응 감지 — R2에 따라 KILL 금지·체인 중단. 이 박스 재부팅 권고." >&2
      return 1
    fi
  fi
  return 0
}

tp2_chain_ok() {   # R3: 체인 전 종료-검증
  [ -z "$(tp2_real_pids)" ] || { echo "[guard] 로컬 프로세스 잔존" >&2; return 1; }
  [ -z "$(_e_real)" ] || { echo "[guard] 엡실론 프로세스 잔존" >&2; return 1; }
  local GW=$(_g_wired) EW=$(_e_wired)
  [ "${GW:-999}" -lt 50 ] || { echo "[guard] 게지히트 wired ${GW}G ≥ 50G" >&2; return 1; }
  [ "${EW:-999}" -lt 50 ] || { echo "[guard] 엡실론 wired ${EW}G ≥ 50G" >&2; return 1; }
  return 0
}

tp2_require_ring() {   # R5: 위험 실험 진입점에서 호출 — jaccl이면 거부
  case "$*" in *jaccl*) echo "[guard] R5 위반: 위험 실험은 ring 전용(jaccl 금지)" >&2; return 1;; esac
  return 0
}
