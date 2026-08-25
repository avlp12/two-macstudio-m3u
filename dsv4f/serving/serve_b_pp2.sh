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
# Short requests keep the legacy path, whose BatchGenerator prefill picks up MTP
# prompt-priming capture for free.
export DSV4_PP2_MIN_TOKENS=4096
# MTP prompt priming on the PP2 path: rank0 ships each prefill chunk's trunk-final
# hidden to rank1 and BOTH ranks fold the (hidden, next-token) pairs through
# prompt_priming.maybe_capture, so take_primed sees a real context instead of the
# unprimed fallback. Implemented and verified (primed=14045 on a 13.9K prompt —
# the same context the non-PP2 path builds natively) but shipped OFF: measured
# 2026-08-26 at -1.1% tok/cycle on a 512-token decode and -0.7% on 128, i.e.
# nothing outside the +/-4% three-sample noise band, and the same neutral result
# reproduces on the non-PP2 path (-1.2% / +3.9%). The reason is structural: this
# model's MTP head is a RotatingKVCache(max_size=sliding_window=128) masked to a
# 128-token window, so priming can hand it at most the last 128 prompt tokens and
# those are fully evicted within 128 generated tokens. It also costs ~0.33s TTFT
# (460MB extra TB5 transfer + one MTP-block forward per rank) and perturbs draft
# shapes enough to change output text through GEMM K-split float drift. Flip to 1
# only if the head ever gets a wider attention span. Both ranks must agree —
# pp2_prefill_stage dies loudly on a mismatch rather than deadlocking.
export DSV4_PP2_PRIMING=0
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
