#!/bin/bash
# serve_b.sh + PP2 프리필 스테이지 (DSV4_PP2_PREFILL=1). mlx.launch 가 양 호스트에서 실행.
# 원본 serve_b.sh 는 무변경 — 이 파일만 바꿔 A/B 한다.
if pgrep -f "serve_batched_tp2|serve_tp4_dspark" >/dev/null 2>&1; then
  echo "[gate] 잔존 서빙 프로세스 — 발사 중단(41). 정리·소멸 검증 후 재시도." >&2; exit 41
fi
WIRED_GB=$(vm_stat | awk '/wired/{gsub("\\.","",$4); print int($4*16384/1073741824)}')
if [ "${WIRED_GB:-0}" -gt 120 ]; then
  echo "[gate] wired ${WIRED_GB}GB > 120GB — 누수/좀비 의심, 발사 중단(42). 재부팅 권고." >&2; exit 42
fi
export TP2_MTP_DEPTH=3
export OMLX_MTP_FIXED_DEPTH=1
export TP2_MTP_CKPT="$HOME/dsv4flash/align/ckpt_r6c_real/step5000.safetensors"
if [ ! -f "$TP2_MTP_CKPT" ]; then
  echo "[gate] TP2_MTP_CKPT 없음: $TP2_MTP_CKPT — 발사 중단(43)." >&2; exit 43
fi
# 재순환(arXiv 2608.17981 착안) 기본 ON — 승격 2026-08-26: 순방향 +1.01%/역순 +0.96%
# tok/cycle, d3 수용 +7pp, mtp/cycle·메모리 비용 0 실측(무대가). 끄려면 두 줄 삭제.
export OMLX_MTP_RECIRC_ALPHA2=0.20
export OMLX_MTP_RECIRC_ALPHA3=0.20
# ── PP2 프리필 스테이지 ──
# 리터럴 고정: 로컬 셸 env는 mlx.launch 원격 랭크로 전파되지 않아
# ${VAR:-기본} 형태는 rank0 만 바뀌는 비대칭 사고를 낸다.
# OFF 대조군은 이 파일이 아니라 **원본 serve_b.sh** 를 그대로 쓴다.
export DSV4_PP2_PREFILL=1
# HOL 인터리브+스냅숏 연동 기본 승격(2026-08-26): 경합 없으면 공짜, 경합 시 단문 TTFT 16.2->2.7s
# (장문이 +5.8% 내고 나머지가 돌려받음). 롤백: 두 줄을 0으로.
export DSV4_PP2_INTERLEAVE=1
export DSV4_PP2_SNAPSTORE=1
export DSV4_PP2_SPLIT=22
export DSV4_PP2_CHUNK=2048
export DSV4_PP2_PORT=39935
export DSV4_PP2_SERVER_IP=10.0.0.2
# 짧은 요청은 기존 경로 유지(BatchGenerator 프리필 = 프라이밍 캡처가 자동으로 붙음).
export DSV4_PP2_MIN_TOKENS=4096
# MTP 프롬프트 프라이밍 복원: rank0 이 청크마다 트렁크 최종 hidden 을 rank1 로 보내고
# **양 랭크가** (hidden, next-token) 페어를 prompt_priming.maybe_capture 로 접는다 →
# take_primed 가 unprimed 폴백 대신 진짜 ctx 를 받는다.
#
# ★구현·검증 완료했으나 **0 유지**(2026-08-26 A/B, prim_ab_{off,on,off2,legacy,legacy_np}):
#   · 기능 정상: 13.9K 프롬프트에서 primed=0 → primed=14045, 비-PP2 네이티브 경로가
#     만드는 ctx 와 동일한 상태(folded=len(pre)-1, expected_offset=len(pre)).
#   · 그러나 이득 없음: PP2 프라이밍 ON vs OFF = tok/cycle −1.1%(512토큰) / −0.7%(128).
#     **비-PP2 네이티브 경로에서도 같다**(−1.2% / +3.9%) → 구현 탓이 아니라 이 모델 탓.
#     3표본 잡음대(±4%) 안이라 어느 쪽도 유의하지 않음.
#   · 구조적 이유: DSv4-Flash 의 MTP 헤드 캐시는 RotatingKVCache(max_size=
#     sliding_window=128)에 window_size=128 마스크 → 프라이밍이 헤드에 줄 수 있는 건
#     **프롬프트 마지막 128토큰뿐**이고, 그마저 생성 128토큰이면 전부 회전 탈락한다.
#     jundot/omlx#3079 의 +19.4% 는 헤드 어텐션 폭이 훨씬 넓은 모델 이야기다.
#   · 비용: TTFT +0.33s(460MB TB5 추가 전송 + 랭크당 MTP 블록 1회 forward), 그리고
#     드래프트 형상이 바뀌어 K분할 GEMM 부동소수점 드리프트로 **출력 텍스트가 달라진다**
#     (OFF 는 재실행 간 바이트 동일 — off vs off2 6/6 일치로 확인).
# 헤드 어텐션 폭이 넓어지는 날 1 로 올릴 것. 양 랭크 동일 리터럴 필수 —
# 어긋나면 fold 횟수 비대칭이라 pp2_prefill_stage 가 (교착 대신) 즉시 죽는다.
export DSV4_PP2_PRIMING=0
# wired 킬스위치 사이드카: PP2 슬라이스 동거로 상주가 ~2배(≈145GB/박스)라 유지 필수.
( while true; do sleep 5
    W=$(vm_stat | awk '/wired/{gsub("\.","",$4); print int($4*16384/1073741824)}')
    [ "${W:-0}" -gt 350 ] && { logger "tp2-killswitch: wired ${W}GB - killing serving"; pkill -9 -f serve_batched_tp2; exit 0; }
    pgrep -f serve_batched_tp2 >/dev/null || exit 0
  done ) &
export MLX_METAL_FAST_SYNCH=1
cd /Users/Shared/tp2
exec "$HOME/venv_omlx063/bin/python" -u /Users/Shared/tp2/serve_batched_tp2.py \
  --model "$HOME/dsv4flash/mlx4bit" --model-name deepseek-v4-flash-tp2 \
  --port 8003 --control-host 10.0.0.1 --control-port 18004 --max-batch 8
