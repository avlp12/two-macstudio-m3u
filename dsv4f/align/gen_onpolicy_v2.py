"""[Round6-v2] on-policy 데이터 확장 생성: 시드 확대(12→24) + 반복 확대(5→10)
= 240 생성 × 400tok ≈ 원본(60생성) 대비 4배 규모. 원본 onpolicy.txt는 보존,
새 파일(onpolicy_v2.txt)에 기록 — 이어서 원본과 concat해 최종 코퍼스 구성."""
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
set_mtp_active(False)
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
 # v2 확장: 주제 다양성 증대(수학/알고리즘, 대화형, 더 긴 코드, 다른 기술영역)
 "MoE 라우팅에서 로드 밸런싱이 중요한 이유는", "Explain how attention mechanisms scale with sequence length",
 "def merge_sort(arr):", "행렬곱 최적화를 위한 타일링 전략을 설명하면",
 "What are the trade-offs between pipeline and tensor parallelism?",
 "class RingAllReduce:", "부동소수점 비결합성이 분산 시스템에 미치는 영향은",
 "Write a function to detect cycles in a directed graph",
 "GPU 메모리 계층 구조와 캐시 활용 전략은", "The future of on-device large language models",
 "async def handle_request(req):", "네트워크 토폴로지 설계 시 고려할 요소들은",
]
out = open(os.path.expanduser("~/dsv4flash/align/onpolicy_v2.txt"), "w")
t0 = time.time()
REPEATS = 10
total = len(seeds) * REPEATS
for i, s in enumerate(seeds * REPEATS):
    msgs = [{"role": "user", "content": s}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    txt = ""
    for r in stream_generate(model, tok, ids, max_tokens=400, sampler=make_sampler(0.0)):
        txt += r.text; _hb["t"] = time.time()
    out.write(txt + "\n\n"); out.flush()
    print(f"[{i+1}/{total}] {time.time()-t0:.0f}s", flush=True)
out.close(); print("GEN-V2-DONE", flush=True); os._exit(0)
