#!/bin/bash
cd ~/dsv4flash/align
~/venv_omlx063/bin/python -u train_align.py --steps 5000 --lr 5e-6 \
  --real-hidden r6c_real_hidden \
  --init-ckpt ckpt_r2/step1000.safetensors \
  --out ckpt_r6c_real > train_run6c_real.log 2>&1
echo "ROUND6C-DONE"
