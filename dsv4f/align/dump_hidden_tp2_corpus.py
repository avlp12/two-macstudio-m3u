"""[Round6] TP2 실측 hidden 코퍼스 전량 덤프 — 합성 노이즈 대신 진짜 TP2 오차를
그대로 훈련에 쓰기 위한 페어드 데이터 생성. build_corpus 로직을 train_align.py와
정확히 동일하게 인라인 재현(모듈 임포트로 인한 이중 패치 적용 회피 — 격리 테스트로
확인된 실패 원인). 안전 등급은 dht.sh(검증됨)와 동일 — 순수 추론, 그래디언트 없음.
"""
import os, sys, json, random, time
import mlx.core as mx

random.seed(7)


def build_corpus(tok, seq_len, files):
    windows = []
    for f in files:
        try:
            txt = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        ids = tok.encode(txt)
        for i in range(0, max(0, len(ids) - seq_len - 4), seq_len):
            windows.append(ids[i:i + seq_len + 3])
    random.shuffle(windows)
    return windows


group = mx.distributed.init()
rank, world = group.rank(), group.size()
assert world == 2, f"world={world} != 2"

from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
apply_deepseek_v4_patch()
from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active, set_mtp_depth
assert apply_mlx_lm_mtp_patch()
set_mtp_active(True); set_mtp_depth(1)
from mlx_lm import load

MODEL = os.path.expanduser("~/dsv4flash/mlx4bit")
CORPUS_LIST = os.path.expanduser("~/dsv4flash/align/corpus_onpolicy_c_portable.txt")
SEQ_LEN = 384
OUT_IDS = "/Users/Shared/tp2/exp_chain/r6c_real_hidden_ids.json"
OUT_H = "/Users/Shared/tp2/exp_chain/r6c_real_hidden.safetensors"

t0 = time.monotonic()
model, tok = load(MODEL, lazy=True)
assert hasattr(model, "shard")
model.shard(group)
for l in model.model.layers:
    mx.eval(l.parameters()); mx.synchronize()
mx.eval(model.parameters()); mx.synchronize()
mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])

# 실측 재확인된 취약점: wsdpa 융합 커널이 384토큰 코퍼스 윈도우 forward에서
# GPU Timeout(kIOGPUCommandBufferCallbackErrorTimeout)으로 라이브 행.
# dhl.sh(505토큰 단발, 패치 없이 성공)와 달리 반복 다회 호출에서 재현됨.
# train_align.py와 동일하게 무력화 — 스톡 SDPA 폴백.
_patched = []
for _n in ("mlx_lm.models.deepseek_v4", "mlx_lm.models.deepseek_v4_mtp",
           "omlx.patches.deepseek_v4.wsdpa_attention"):
    _m = sys.modules.get(_n)
    if _m is not None and hasattr(_m, "wsdpa_prefill"):
        _m.wsdpa_prefill = lambda *a, **k: None
        if hasattr(_m, "wsdpa_topk_prefill"):
            _m.wsdpa_topk_prefill = lambda *a, **k: None
        _patched.append(_n)

if rank == 0:
    print(f"[r6-dump] load+shard {time.monotonic()-t0:.1f}s · wsdpa 무력화: {_patched}", flush=True)

files = [os.path.expanduser(l.strip()) for l in open(CORPUS_LIST) if l.strip()]
windows = build_corpus(tok, SEQ_LEN, files)
LIMIT = int(os.environ.get("R6_DUMP_LIMIT", "0")) or len(windows)
windows = windows[:LIMIT]
if rank == 0:
    print(f"[r6-dump] 윈도우 {len(windows)}개 (limit={LIMIT})", flush=True)

tensors = {}
t1 = time.monotonic()
for i, w in enumerate(windows):
    if rank == 0:
        print(f"[r6-dump] 윈도우 {i} 시작", flush=True)
    ids = mx.array([w[:SEQ_LEN]])
    _, h = model.model(ids, None, return_raw_hidden=True)
    mx.eval(h)
    if rank == 0:
        tensors[f"h_{i}"] = h.astype(mx.float32)
        print(f"[r6-dump] 윈도우 {i} 완료 · {(time.monotonic()-t1)/(i+1):.1f}s/win", flush=True)

if rank == 0:
    mx.save_safetensors(OUT_H, tensors)
    with open(OUT_IDS, "w") as f:
        json.dump(windows, f)
    print(f"[r6-dump] 저장 완료: {OUT_H} ({len(tensors)}개) + {OUT_IDS}", flush=True)
    print("[r6-dump-pass]", flush=True)
