#!/bin/bash
# Overlay the three patched oMLX modules into an installed venv, on ONE box.
# Run it on BOTH boxes, then verify with `shasum -a 256 -c SHA256SUMS`
# from the installed site-packages parent (or just re-run cluster/check_deploy_sync.sh
# after adding these paths to it).
#
# usage: apply.sh [/path/to/venv]     (default: $HOME/venv_omlx063)
set -euo pipefail
VENV="${1:-$HOME/venv_omlx063}"
SP=$("$VENV/bin/python" -c 'import omlx, os; print(os.path.dirname(os.path.dirname(omlx.__file__)))')
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "site-packages: $SP"
for rel in omlx/patches/deepseek_v4/deepseek_v4_model.py \
           omlx/patches/mlx_lm_mtp/batch_generator.py \
           omlx/patches/mlx_lm_mtp/prompt_priming.py; do
  [ -f "$SP/$rel" ] || { echo "target missing: $SP/$rel — wrong omlx version?" >&2; exit 1; }
  cp -f "$SP/$rel" "$SP/$rel.orig.$(date +%Y%m%d)" 2>/dev/null || true
  cp -f "$HERE/$rel" "$SP/$rel"
  echo "patched $rel"
done
( cd "$SP" && shasum -a 256 -c "$HERE/SHA256SUMS" )
echo "PATCH-OK"
