#!/bin/bash
# sha256-compare the deployed files on both boxes.
#
# Why: `rsync -rt` does NOT detect silent corruption, and a file that is present
# on one rank but stale/missing on the other does not fail loudly — the partner
# rank hangs inside a collective instead, which is the expensive failure mode
# (see cluster/tp2_guard.sh R2). Run this before every launch.
EPS="${BOX_B:?BOX_B not set — export BOX_B=<user>@10.0.0.2 (ssh target for box B)}"
FILES=(
  "/Users/Shared/tp2/serve_b.sh"
  "/Users/Shared/tp2/serve_batched_tp2.py"
  "/Users/Shared/tp2/dspark_tp4_common.py"
  "/Users/Shared/tp2/serve_tp4_dspark.py"
  "/Users/Shared/tp2/pp2_prefill_stage.py"
  "/Users/Shared/tp2/run_tp2_flash.py"
  "\$HOME/dsv4flash/align/ckpt_r6c_real/step5000.safetensors"
)
FAIL=0
for f in "${FILES[@]}"; do
  L=$(eval shasum -a 256 "$f" 2>/dev/null | cut -d' ' -f1)
  R=$(ssh -o BatchMode=yes $EPS "shasum -a 256 $f 2>/dev/null | cut -d' ' -f1")
  if [ -z "$L" ] || [ -z "$R" ]; then echo "MISSING $f (local='${L:0:8}' remote='${R:0:8}')"; FAIL=1
  elif [ "$L" != "$R" ]; then echo "MISMATCH $f"; FAIL=1
  else echo "OK $f ${L:0:12}"; fi
done
[ $FAIL = 0 ] && echo "DEPLOY-SYNC-OK" || echo "DEPLOY-SYNC-FAIL"
exit $FAIL
