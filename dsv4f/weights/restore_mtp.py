"""mtp_raw(체크포인트 원형) → mlx-community 4bit 팩에 mtp 샤드 복원.
표현 미러링: 라우티드 전문가 = mxfp4 g32 바이트-이식, fp8 = 디퀀트→affine 4/64,
norm/sink/hc/gate.bias = 원시. 인덱스·config 양자화 오버라이드 갱신(+백업)."""
import json, os, shutil, struct, sys

import mlx.core as mx

PACK = os.environ.get("DSV4_PACK", os.path.expanduser("~/dsv4flash/mlx4bit"))
RAW = os.environ.get("DSV4_MTP_RAW", os.path.expanduser("~/dsv4flash/mtp_raw.safetensors"))
OUTF = "model-mtp-00001-of-00001.safetensors"

import numpy as np

DT = {"BF16": mx.bfloat16, "F32": mx.float32, "F8_E4M3": "e4m3",
      "F8_E8M0": mx.uint8, "I8": mx.int8}

# e4m3fn(1s·4e·3m·bias7) uint8 → float32 룩업 (256항)
_b = np.arange(256, dtype=np.uint16)
_s = np.where(_b >> 7, -1.0, 1.0)
_e = ((_b >> 3) & 0xF).astype(np.int32)
_m = (_b & 0x7).astype(np.float32)
_val = np.where(_e == 0, _s * (_m / 8.0) * 2.0 ** -6,
                _s * (1.0 + _m / 8.0) * (2.0 ** (_e - 7)))
_val[(_e == 15) & (_b & 0x7 == 7)] = np.nan  # e4m3fn: 0x7F/0xFF = NaN
E4M3_LUT = _val.astype(np.float32)

def load_raw(path):
    out = {}
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n)); base = 8 + n
        for k, m in hdr.items():
            if k == "__metadata__": continue
            fh.seek(base + m["data_offsets"][0])
            b = fh.read(m["data_offsets"][1] - m["data_offsets"][0])
            dt = DT[m["dtype"]]
            if dt == "e4m3":
                arr = E4M3_LUT[np.frombuffer(b, dtype=np.uint8)]
                out[k] = ("f8", mx.array(arr).reshape(m["shape"]))
            else:
                a = mx.array(memoryview(b), dtype=mx.uint8)
                a = a.view(dt) if dt != mx.uint8 else a
                out[k] = (None, a.reshape(m["shape"]))
    return out

raw = load_raw(RAW)
print(f"[1] raw {len(raw)}텐서")

def deq_fp8(w32, s):
    O, I = w32.shape
    s32 = mx.power(mx.array(2.0), s.astype(mx.float32) - 127.0)
    su = mx.repeat(mx.repeat(s32, 128, axis=0)[:O], 128, axis=1)[:, :I]
    return (w32 * su).astype(mx.bfloat16)

out, qcfg = {}, {}
n_x, n_a, n_r = 0, 0, 0
for k in sorted(raw):
    if k.endswith(".scale"):
        continue
    tag, a = raw[k]
    sk = k[:-len(".weight")] + ".scale" if k.endswith(".weight") else None
    sv = raw.get(sk) if sk else None
    s = sv[1] if sv else None
    mod = k[:-len(".weight")] if k.endswith(".weight") else None
    if tag is None and a.dtype == mx.int8 and s is not None:
        # 라우티드 전문가 fp4: 바이트-이식 → mxfp4 g32
        packed = a.view(mx.uint32)
        out[k] = packed
        out[mod + ".scales"] = s if s.dtype == mx.uint8 else s.view(mx.uint8)
        qcfg[mod] = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
        n_x += 1
    elif tag == "f8" and s is not None:
        w16 = deq_fp8(a, s)
        wq, sc, bi = mx.quantize(w16, group_size=64, bits=4)
        out[k] = wq; out[mod + ".scales"] = sc; out[mod + ".biases"] = bi
        qcfg[mod] = {"group_size": 64, "bits": 4, "mode": "affine"}
        n_a += 1
    elif k.endswith("ffn.gate.weight"):
        out[k] = a.astype(mx.bfloat16)  # 백본 미러: gate 는 비양자화
        n_r += 1
    elif k.endswith("ffn.gate.bias"):
        out[k[:-len(".bias")] + ".e_score_correction_bias"] = a  # 팩 명명 미러
        n_r += 1
    else:
        out[k] = a
        n_r += 1
mx.eval(*out.values())
print(f"[2] 변환: mxfp4 {n_x} · affine {n_a} · raw {n_r} → 산출 {len(out)}배열")

mx.save_safetensors(f"{PACK}/{OUTF}", out)
import os
print(f"[3] 샤드 기록 {os.path.getsize(f'{PACK}/{OUTF}')/2**30:.2f} GiB")

# 인덱스 갱신
ip = f"{PACK}/model.safetensors.index.json"
shutil.copy(ip, ip + ".premtp")
idx = json.load(open(ip))
for k in out:
    idx["weight_map"][k] = OUTF
json.dump(idx, open(ip, "w"))
# config 양자화 오버라이드: 후개명 경로(shared w1→gate_proj 등) 포함
cp = f"{PACK}/config.json"
shutil.copy(cp, cp + ".premtp")
cfg = json.load(open(cp))
ren = {".w1": ".gate_proj", ".w3": ".up_proj", ".w2": ".down_proj"}
def rename(mod):
    for o, n in ren.items():
        if mod.endswith(o):
            mod = mod[:-len(o)] + n
    # 라우티드 experts.E.* → switch_mlp.* (스택 후 경로)
    import re
    mod = re.sub(r"\.ffn\.experts\.\d+\.", ".ffn.switch_mlp.", mod)
    return mod
added = set()
for mod, q in qcfg.items():
    added.add((rename(mod), json.dumps(q)))
for mod, qs in sorted(added):
    cfg["quantization"][mod] = json.loads(qs)
    cfg.setdefault("quantization_config", {})[mod] = json.loads(qs) if "quantization_config" in cfg else None
cfg.pop("quantization_config", None) if cfg.get("quantization_config") in ({}, None) else None
json.dump(cfg, open(cp, "w"), indent=1)
print(f"[4] config 오버라이드 +{len(added)}종 · 인덱스 +{len(out)}항목")
print("RESTORE-DONE")
