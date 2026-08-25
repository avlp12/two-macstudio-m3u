#!/bin/bash
# Production launcher. mlx.launch runs this on BOTH hosts.
# (Comments translated for this repo; logic is byte-identical to what runs on
#  our cluster as of 2026-08-25.)
#
# ── launch gates (added after the 2026-08-23 crash) ──
if pgrep -f "serve_batched_tp2|serve_tp4_dspark" >/dev/null 2>&1; then
  echo "[gate] serving process still alive — refusing to launch (41). Clean up and verify death first." >&2; exit 41
fi
WIRED_GB=$(vm_stat | awk '/wired/{gsub("\\.","",$4); print int($4*16384/1073741824)}')
if [ "${WIRED_GB:-0}" -gt 120 ]; then
  echo "[gate] wired ${WIRED_GB}GB > 120GB — leak/zombie suspected, refusing to launch (42). Reboot recommended." >&2; exit 42
fi
# Fixed depth-3 MTP chain (promoted 2026-08-24). The hang mechanism was the
# adaptive depth controller diverging between ranks; OMLX_MTP_FIXED_DEPTH=1
# removes it. Validated ring-then-jaccl (+11%, no hang) before promotion.
# NEVER run depth>1 without FIXED_DEPTH — controller divergence = hang = kernel corruption.
export TP2_MTP_DEPTH=3
export OMLX_MTP_FIXED_DEPTH=1
# Round-6c aligned MTP head (promoted 2026-08-25): 11x corpus + trained directly
# on captured TP2 hidden states. bs1 x 24-topic paired: tok/cycle +3.68%, p=0.0026.
export TP2_MTP_CKPT="$HOME/dsv4flash/align/ckpt_r6c_real/step5000.safetensors"
if [ ! -f "$TP2_MTP_CKPT" ]; then
  echo "[gate] TP2_MTP_CKPT missing: $TP2_MTP_CKPT — refusing to launch (43). If only one box lacks it, the partner rank hangs inside a collective." >&2; exit 43
fi
# Rollback: export TP2_MTP_CKPT="$HOME/dsv4flash/align/ckpt_r2/step1000.safetensors"
#           (round-2 head is also kept on HF: avlp12/dsv4flash-mtp-aligned)
# wired killswitch sidecar: kill serving above 350GB, before the machine locks up.
( while true; do sleep 5
    W=$(vm_stat | awk '/wired/{gsub("\.","",$4); print int($4*16384/1073741824)}')
    # NOTE: this pkill -9 contradicts tp2_guard R2 (never KILL a stuck process). It is a
    # deliberate last resort: wired > 350GB means the system is about to become unusable.
    # After a KILL the partner rank can survive stuck in a collective — clean/verify the other box too.
    [ "${W:-0}" -gt 350 ] && { logger "tp2-killswitch: wired ${W}GB - killing serving"; pkill -9 -f serve_batched_tp2; exit 0; }
    pgrep -f serve_batched_tp2 >/dev/null || exit 0
  done ) &
export MLX_METAL_FAST_SYNCH=1
cd /Users/Shared/tp2
exec "$HOME/venv_omlx063/bin/python" /Users/Shared/tp2/serve_batched_tp2.py \
  --model "$HOME/dsv4flash/mlx4bit" --model-name deepseek-v4-flash-tp2 \
  --port 8003 --control-host 10.0.0.1 --control-port 18004 --max-batch 8
