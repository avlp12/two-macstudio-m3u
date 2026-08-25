#!/bin/bash
# serve_b.sh + the PP2 prefill stage (DSV4_PP2_PREFILL=1). mlx.launch runs this
# on BOTH hosts. serve_b.sh itself is left untouched — A/B by swapping which of
# the two launchers you hand to mlx.launch.
if pgrep -f "serve_batched_tp2|serve_tp4_dspark" >/dev/null 2>&1; then
  echo "[gate] serving process still alive — refusing to launch (41). Clean up and verify death first." >&2; exit 41
fi
WIRED_GB=$(vm_stat | awk '/wired/{gsub("\\.","",$4); print int($4*16384/1073741824)}')
if [ "${WIRED_GB:-0}" -gt 120 ]; then
  echo "[gate] wired ${WIRED_GB}GB > 120GB — leak/zombie suspected, refusing to launch (42). Reboot recommended." >&2; exit 42
fi
export TP2_MTP_DEPTH=3
export OMLX_MTP_FIXED_DEPTH=1
export TP2_MTP_CKPT="$HOME/dsv4flash/align/ckpt_r6c_real/step5000.safetensors"
if [ ! -f "$TP2_MTP_CKPT" ]; then
  echo "[gate] TP2_MTP_CKPT missing: $TP2_MTP_CKPT — refusing to launch (43)." >&2; exit 43
fi
# ── PP2 prefill stage ──
# Values are hardcoded literals ON PURPOSE: your local shell env is NOT propagated
# to the remote rank by mlx.launch, so a `${VAR:-default}` here would change rank0
# only and give you a silently asymmetric two-rank config.
# The OFF control arm is the untouched serve_b.sh, not this file with the gate off.
export DSV4_PP2_PREFILL=1
# HOL interleave + snapshot-store integration, promoted to on-by-default 2026-08-26:
# free when nothing else is running; under contention, concurrent short-request TTFT
# drops 16.2s -> 2.7s at a +5.8% cost on the long prefill (refunded to everyone else).
# Roll back by setting both to 0.
export DSV4_PP2_INTERLEAVE=1
export DSV4_PP2_SNAPSTORE=1
export DSV4_PP2_SPLIT=22
export DSV4_PP2_CHUNK=2048
export DSV4_PP2_PORT=39935
export DSV4_PP2_SERVER_IP=10.0.0.2
# Short requests keep the legacy path, which preserves MTP prompt priming
# (the PP2 path inserts unprimed and falls back).
export DSV4_PP2_MIN_TOKENS=4096
# wired killswitch sidecar: mandatory here — the co-resident PP2 slice roughly
# doubles resident memory (~145GB/box).
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
