"""정렬 체크포인트의 라이브 depth-3 수용률 검증 (싱글박스)."""
import os, sys, time, threading, logging
logging.basicConfig(level=logging.INFO)
import mlx.core as mx
DEPTH = int(os.environ.get("DEPTH", "3"))
CKPT = os.environ.get("CKPT", "")
_hb = {"t": time.time()}
def _wd():
    while True:
        time.sleep(5)
        if time.time() - _hb["t"] > 120: os._exit(9)
threading.Thread(target=_wd, daemon=True).start()

from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active, set_mtp_depth
apply_deepseek_v4_patch(); assert apply_mlx_lm_mtp_patch()
set_mtp_active(True); set_mtp_depth(DEPTH)
if os.environ.get("FIXED","1")=="1": os.environ["OMLX_MTP_FIXED_DEPTH"]="1"
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler
sys.path.insert(0, os.path.expanduser("~/dsv4flash/align"))
from train_align import promote_nonexpert, attach_lora_shared_experts
set_mtp_active(True); set_mtp_depth(DEPTH)  # train_align 임포트가 전역을 1로 되돌림 — 재설정

model, tok = load(os.path.expanduser("~/dsv4flash/mlx4bit"), lazy=True)
for layer in model.model.layers:
    mx.eval(layer.parameters()); _hb["t"] = time.time()
mx.eval(model.parameters())
mx.set_wired_limit(min(250 << 30, mx.metal.device_info()["max_recommended_working_set_size"]))
block = model.mtp[0]
if CKPT:
    promote_nonexpert(block)
    if os.environ.get("LORA", "0") == "1":
        n_lora = attach_lora_shared_experts(block, r=int(os.environ.get("LORA_R", "16")),
                                            alpha=float(os.environ.get("LORA_ALPHA", "16")))
        print(f"[ckpt] LoRA 부착 {n_lora}개", flush=True)
    w = mx.load(CKPT)
    block.load_weights(list(w.items()), strict=False)
    mx.eval(block.parameters())
    print(f"[ckpt] {CKPT} 적용", flush=True)

from omlx.patches.mlx_lm_mtp import batch_generator as _bgm
print("[chain-resolve]", _bgm._resolve_mtp_chain_depth(model), "· model._omlx_mtp_depth =", getattr(model, "_omlx_mtp_depth", None), flush=True)
bg = BatchGenerator(model, max_tokens=256, sampler=make_sampler(0.0),
                    completion_batch_size=1, prefill_batch_size=1, prefill_step_size=2048)
msgs = [{"role": "user", "content": "분산 추론 시스템의 병목과 해법을 차분히 서술하라."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
bg.insert([list(ids)], max_tokens=[256])
toks = []; t1 = time.time(); tf = None
while len(toks) < 256:
    _hb["t"] = time.time()
    rs = bg.next_generated()
    for r in rs or []:
        if tf is None: tf = time.time()
        toks.append(int(r.token))
        if r.finish_reason: break
    if rs and any(r.finish_reason for r in rs): break
print(f"[결과] depth={DEPTH} ckpt={'yes' if CKPT else 'no'} · {(len(toks)-1)/(time.time()-tf):.2f} tok/s", flush=True)
os._exit(0)
