#!/bin/bash
# [웨지 검증] 콜드-스타트 반복 재현 프로토콜의 랭크 워커.
# 웨지 5회가 전부 "로드 직후 첫 대형 프리필"에서 났으므로, 그 창만 반복한다.
#
# ── 랭크 대칭성은 **구조로** 보장한다 ────────────────────────────────────────
# 예전 판은 게이트 값을 셸 환경에서 상속했다. mlx.launch 는 실행 셸의 env 를
# 원격 랭크에 전파하지 않으므로, 조작자가 `TP2_FAST_SYNCH=1 bash wedge_verify.sh`
# 처럼 부르면 rank0=1 / rank1=0 이 되어 **한 랭크만 fast-fence 를 쓰는 비대칭**이
# 만들어졌다. 웜업 배리어에서 그대로 영구 교착이다
# (블랭크-슬레이트 리뷰 MUST-2, 2026-08-26).
# 그래서 게이트 값은 **argv 로만** 받는다 — mlx.launch 는 인자를 양 랭크에
# 글자 그대로 같게 전달하므로 비대칭이 원천적으로 불가능하다. 하네스는 여기에
# 더해 로드 전에 게이트 값을 all_sum 합의로 검사한다(run_tp2_flash.py).
#
# usage: wedge_verify_worker.sh <FS>:<WARMUP>:<WTOK>:<WATCHDOG_S>:<MTP> [하네스 인자...]
set -u
GATE="${1:?게이트 스펙 필요: FS:WARMUP:WTOK:WATCHDOG_S:MTP}"; shift
IFS=: read -r _FS _WARM _WTOK _WD _MTP <<EOF
$GATE
EOF

export TP2_FAST_SYNCH="$_FS"        # MLX_METAL_FAST_SYNCH 명시 지정
export TP2_WARMUP="$_WARM"          # 배리어+웜업 on/off
export TP2_WARMUP_TOKENS="$_WTOK"
export TP2_WATCHDOG_S="$_WD"
export TP2_MTP="$_MTP" OMLX_MTP_FIXED_DEPTH=1 TP2_MTP_DEPTH=3
export OMLX_MTP_RECIRC_ALPHA2=0 OMLX_MTP_RECIRC_ALPHA3=0
export TP2_MTP_CKPT="$HOME/dsv4flash/align/ckpt_r6c_real/step5000.safetensors"

exec "$HOME/venv_omlx063/bin/python" -u /Users/Shared/tp2/run_tp2_flash.py "$@"
