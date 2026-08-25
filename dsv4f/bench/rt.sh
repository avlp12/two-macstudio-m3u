#!/bin/bash
export MLX_METAL_FAST_SYNCH=1 OMLX_MTP_FIXED_DEPTH=1 TP2_MTP=1
export TP2_MTP_DEPTH=$(cat /Users/Shared/tp2/exp_chain/depth.cfg 2>/dev/null || echo 1)
exec "$HOME/venv_omlx063/bin/python" -u /Users/Shared/tp2/run_tp2_flash.py --batch 1
