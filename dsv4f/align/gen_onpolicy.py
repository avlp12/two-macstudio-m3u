"""on-policy 데이터 생성: 타깃 모델의 그리디 연속 텍스트 수집."""
import os, sys, time, threading
import mlx.core as mx
_hb = {"t": time.time()}
def _wd():
    while True:
        time.sleep(5)
        if time.time() - _hb["t"] > 180: os._exit(9)
threading.Thread(target=_wd, daemon=True).start()
from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active
apply_deepseek_v4_patch(); assert apply_mlx_lm_mtp_patch()
set_mtp_active(False)  # 순수 백본 생성(빠른 편이 아니어도 정확한 정책 분포)
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
model, tok = load(os.path.expanduser("~/dsv4flash/mlx4bit"), lazy=True)
for l in model.model.layers: mx.eval(l.parameters()); _hb["t"] = time.time()
mx.eval(model.parameters())
mx.set_wired_limit(min(250 << 30, mx.metal.device_info()["max_recommended_working_set_size"]))
seeds = [
 "분산 추론 시스템의 병목을 분석하면", "양자화가 모델 품질에 미치는 영향은",
 "The main bottleneck in distributed inference is", "A speculative decoder works by",
 "def batch_scheduler(requests):", "class KVCacheManager:",
 "로컬 LLM 서빙의 미래를 전망하면", "Thunderbolt networking on macOS",
 "The trade-off between latency and throughput", "쿠버네티스 없이 GPU 클러스터를 운영하려면",
 "import mlx.core as mx\n\ndef fused_attention(", "메모리 대역폭이 디코드 속도를 결정하는 이유는",
]
out = open(os.path.expanduser("~/dsv4flash/align/onpolicy.txt"), "w")
t0 = time.time()
for i, s in enumerate(seeds * 5):  # 60 생성 × 400tok
    msgs = [{"role": "user", "content": s}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    txt = ""
    for r in stream_generate(model, tok, ids, max_tokens=400, sampler=make_sampler(0.0)):
        txt += r.text; _hb["t"] = time.time()
    out.write(txt + "\n\n"); out.flush()
    print(f"[{i+1}/60] {time.time()-t0:.0f}s", flush=True)
out.close(); print("GEN-DONE", flush=True); os._exit(0)
