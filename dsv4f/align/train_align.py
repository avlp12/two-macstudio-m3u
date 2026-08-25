"""DSv4-Flash mtp.0 on-policy 체인 정렬 (K3 기법 이식).

목표: d2 조건부 수용 32% 개선 — mtp가 (자기 출력 g, 실제 t+2) 입력 분포를
본 적이 없어서 생기는 감쇠를 교사-강제 체인 SFT로 교정.
전략: 전문가(ffn) 4bit 동결, 비전문가부(attn/e_proj/h_proj/norm/hc)만 bf16 훈련.
손실: CE_d2 (주) + 0.3·CE_d1 (유지항).
"""
import argparse, glob, json, os, random, time
random.seed(7)
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active, set_mtp_depth
apply_deepseek_v4_patch(); assert apply_mlx_lm_mtp_patch()
set_mtp_active(True); set_mtp_depth(1)
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask


def dequantize_module(qm):
    """QuantizedLinear → bf16 nn.Linear (bias 없음 가정, 있으면 이식)."""
    w = mx.dequantize(qm.weight, qm.scales, getattr(qm, "biases", None),
                      qm.group_size, qm.bits)
    lin = nn.Linear(w.shape[1], w.shape[0], bias=hasattr(qm, "bias"))
    lin.weight = w.astype(mx.bfloat16)
    if hasattr(qm, "bias"):
        lin.bias = qm.bias.astype(mx.bfloat16)
    return lin


def promote_nonexpert(block):
    """mtp 블록의 양자화 비전문가 레이어를 bf16으로 승격(제자리 교체)."""
    n = 0
    # wo_a/wo_b 제외: omlx의 그룹드 출력 프로젝션이 패킹 가중치를 직접 읽음
    for parent, attr in [(block.block.attn, a) for a in
                         ("wq_a", "wq_b", "wkv")] + \
                        [(block, "e_proj"), (block, "h_proj")]:
        m = getattr(parent, attr, None)
        if m is not None and hasattr(m, "scales"):
            setattr(parent, attr, dequantize_module(m)); n += 1
    return n


class LoRALinear(nn.Module):
    """양자화 선형층에 병렬 저랭크 어댑터 부착. base는 완전 동결(순전파는
    stop_gradient), A/B만 학습 — mxfp4/mxfp8 게더의 VJP 부재([CA80] 동류)를
    우회하면서도 shared_experts 표현력을 소폭 확장한다."""

    def __init__(self, base, r=16, alpha=16.0):
        super().__init__()
        self.base = base
        out_dim = base.weight.shape[0]
        in_dim = base.scales.shape[1] * base.group_size
        scale = 1.0 / (r ** 0.5)
        self.lora_a = mx.random.normal((in_dim, r)).astype(mx.bfloat16) * scale
        self.lora_b = mx.zeros((r, out_dim)).astype(mx.bfloat16)  # 0-init: 시작점=base와 동일
        self.alpha_over_r = alpha / r
        self.base.freeze()

    def __call__(self, x):
        y = mx.stop_gradient(self.base(x))
        delta = (x.astype(mx.bfloat16) @ self.lora_a) @ self.lora_b
        return y + self.alpha_over_r * delta


def attach_lora_shared_experts(block, r=16, alpha=16.0):
    """mtp.0 블록 하나의 shared_experts(gate/up/down_proj)에만 LoRA 부착.
    백본(43층)은 이 함수가 절대 건드리지 않음 — 호출부가 model.mtp[0]만 넘긴다.
    라우팅 전문가(switch_mlp)는 동적 게더라 이번 범위 밖 — shared만."""
    se = block.block.ffn.shared_experts
    n = 0
    for attr in ("gate_proj", "up_proj", "down_proj"):
        m = getattr(se, attr, None)
        if m is not None and hasattr(m, "scales"):
            setattr(se, attr, LoRALinear(m, r=r, alpha=alpha)); n += 1
    return n


def merge_lora_into_shared_experts(block, ckpt_weights, alpha_over_r=None,
                                   ckpt_path=None):
    """ckpt의 lora_{a,b} 키를 base(affine 4bit)에 병합해 bf16 Linear로 교체.
    반환: 병합한 proj 수. lora 키가 없으면 0 반환하고 아무것도 안 함.

    관례: y = x @ Wᵀ, LoRA 순전파 y = base(x) + (alpha/r)·(x@lora_a)@lora_b
          → W' = W + (alpha/r)·(lora_a @ lora_b)ᵀ

    부착(LoRALinear) 대신 병합하는 이유: 라이브 TP2의 shard_inplace는 경로 기반
    전수 분할이라 lora_a/lora_b까지 잘라 크래시한다. 평범한 bf16 Linear로 접어
    두면 promote_nonexpert 전례대로 정상 샤딩된다.
    델타는 fp32로 계산해 dequant(fp32)와 합산한 뒤 마지막에만 bf16 캐스트 —
    bf16 직접 누적은 작은 델타가 반올림으로 소실될 수 있다.

    ckpt_path: alpha/r 사이드카(lora_meta.json) 탐색용 체크포인트 파일 경로.
    """
    prefix = "block.ffn.shared_experts."
    if not any(k.startswith(prefix) and k.endswith(".lora_a") for k in ckpt_weights):
        return 0

    if alpha_over_r is not None:
        src = "호출 인자"
    else:
        meta, meta_path = None, ""
        if ckpt_path:
            meta_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)),
                                     "lora_meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"[merge] lora_meta.json 읽기 실패({e}) — 기본값으로", flush=True)
                    meta = None
        if meta and "alpha_over_r" in meta:
            alpha_over_r = float(meta["alpha_over_r"]); src = f"사이드카 {meta_path}"
        elif meta and "alpha" in meta and "r" in meta:
            alpha_over_r = float(meta["alpha"]) / float(meta["r"])
            src = f"사이드카 {meta_path}"
        else:
            alpha_over_r = 1.0; src = "기본값(r16/alpha16)"
    print(f"[merge] alpha/r = {alpha_over_r:.6g} (출처: {src})", flush=True)

    se = block.block.ffn.shared_experts
    n = 0
    for attr in ("gate_proj", "up_proj", "down_proj"):
        ka, kb = prefix + attr + ".lora_a", prefix + attr + ".lora_b"
        if ka not in ckpt_weights or kb not in ckpt_weights:
            continue
        m = getattr(se, attr, None)
        if m is None:
            continue
        base = getattr(m, "base", m)  # 이미 LoRALinear가 붙어 있어도 안전
        if hasattr(base, "scales"):
            w = mx.dequantize(base.weight, base.scales, getattr(base, "biases", None),
                              base.group_size, base.bits,
                              mode=getattr(base, "mode", "affine")).astype(mx.float32)
        else:
            w = base.weight.astype(mx.float32)
        a = ckpt_weights[ka].astype(mx.float32)
        b = ckpt_weights[kb].astype(mx.float32)
        delta = (a @ b).T * alpha_over_r          # [in,r]@[r,out] → [in,out] → [out,in]
        if delta.shape != w.shape:
            raise ValueError(f"{attr} LoRA 형상 불일치: delta {delta.shape} vs W {w.shape}")
        w = w + delta
        lin = nn.Linear(w.shape[1], w.shape[0], bias=hasattr(base, "bias"))
        lin.weight = w.astype(mx.bfloat16)
        if hasattr(base, "bias"):
            lin.bias = base.bias.astype(mx.bfloat16)
        setattr(se, attr, lin)
        mx.eval(lin.parameters())   # fp32 임시항 즉시 해제
        n += 1
    return n


def build_corpus(tok, seq_len, files):
    windows = []
    for f in files:
        try:
            txt = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        ids = tok.encode(txt)
        for i in range(0, max(0, len(ids) - seq_len - 4), seq_len):
            windows.append(ids[i:i + seq_len + 3])  # +3: t+1/t+2/t+3 시프트 여유
    random.shuffle(windows)
    return windows


def load_real_hidden(path):
    """--real-hidden 로더 — 두 형태 모두 지원:
    1) 단일 파일 쌍(base): <base>.safetensors(키 h_0..h_{N-1}) + <base>_ids.json(윈도우 리스트).
       기존 r6/r6b/r6c 캐시 형태 — 전량을 한 번에 메모리에 로드.
    2) [Round6e] 디렉터리(r6e_h류): h_{i}.safetensors(키 "h") + ids_{i}.json 개별 파일.
       ids_{i}.json을 모아 인덱스 오름차순으로 windows를 재구성하고, h는 스텝마다
       mx.load로 lazy 로드(윈도우당 ~24MB라 IO 무시 가능 — 전량을 미리 올리지 않아
       메모리 절약 + 재캡처 도중에도(재개 로직으로 일부만 존재해도) 안전).

    반환: (windows, get_h, file_idx) — file_idx[j]는 windows[j]에 대응하는 실제 파일
    인덱스(디렉터리 모드는 위치==파일인덱스라고 가정하지 않음 — 재캡처 중간에 빈틈이
    있어도 안전하도록 분리). get_h(idx)는 파일 인덱스를 받아 [1,S,4,D] mx.array 반환.
    """
    if os.path.isdir(path):
        ids_files = glob.glob(os.path.join(path, "ids_*.json"))
        idxs = sorted(
            int(os.path.basename(f)[len("ids_"):-len(".json")]) for f in ids_files
        )
        windows = []
        for i in idxs:
            with open(os.path.join(path, f"ids_{i}.json")) as f:
                windows.append(json.load(f))

        def get_h(i):
            return mx.load(os.path.join(path, f"h_{i}.safetensors"))["h"]

        return windows, get_h, idxs
    else:
        with open(path + "_ids.json") as f:
            windows = json.load(f)
        h_all = mx.load(path + ".safetensors")

        def get_h(i):
            return h_all[f"h_{i}"]

        return windows, get_h, list(range(len(windows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/dsv4flash/mlx4bit"))
    ap.add_argument("--corpus-list", default=os.path.expanduser("~/dsv4flash/align/corpus.txt"))
    ap.add_argument("--seq-len", type=int, default=384)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--d1-retain", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--out", default=os.path.expanduser("~/dsv4flash/align/ckpt"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--init-ckpt", default="")
    ap.add_argument("--lora-shared", action="store_true",
                    help="mtp.0의 shared_experts에 LoRA 부착(전문가 표현력 확장)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--noise-std", type=float, default=0.0,
                    help="[I305] TP2 hidden 분포 강건화용 훈련-시-노이즈 (h.std() 대비 배율, 실측 ~0.004)")
    ap.add_argument("--real-hidden", default="",
                    help="[Round6] 실측 TP2 hidden 캐시 base path (합성 노이즈 대신 진짜 "
                         "TP2 forward-only 캡처 사용 — <base>.safetensors + <base>_ids.json). "
                         "디렉터리를 주면 [Round6e] r6e_h류 개별 파일 형태(h_{i}.safetensors + "
                         "ids_{i}.json)로 lazy 로드한다.")
    ap.add_argument("--loss-start-pos", type=int, default=0,
                    help="[Round6e] CE 손실(d1·d2 둘 다)을 이 위치 이상의 시퀀스 위치에서만 "
                         "계산 — chain_logits 자체는 전체 시퀀스로 돌려 어텐션 문맥은 유지하고, "
                         "cross_entropy 직전에 z1/z2·타깃을 [:, N:]로 슬라이스한다. 서빙-충실 "
                         "디코드 구간(예: 320)만 놓고 훈련/평가하고 싶을 때 사용. eval_match도 "
                         "동일 인자로 슬라이스해 훈련-평가 정합을 유지한다.")
    args = ap.parse_args()

    model, tok = load(args.model, lazy=True)
    for layer in model.model.layers:
        mx.eval(layer.parameters())
    mx.eval(model.parameters())
    cap = min(300 * (1 << 30), mx.metal.device_info()["max_recommended_working_set_size"])
    mx.set_wired_limit(cap)

    block = model.mtp[0]
    n_prom = promote_nonexpert(block)
    mx.eval(block.parameters())
    print(f"[init] 비전문가 {n_prom}개 레이어 bf16 승격", flush=True)

    # 학습 대상: 전체 동결 → mtp 블록 해제 → 전문가(ffn)만 재동결
    model.freeze()
    block.unfreeze()
    block.block.ffn.freeze()
    # 양자화 잔존 모듈(wo_a/wo_b 등)은 grad 불가 — 일괄 재동결 스윕
    def _freeze_quantized(m):
        n = 0
        stack = [m]
        while stack:
            cur = stack.pop()
            ch = cur.children()
            items = ch.values() if isinstance(ch, dict) else ch
            for c in items:
                if isinstance(c, nn.Module):
                    if hasattr(c, "scales"):
                        c.freeze(); n += 1
                    else:
                        stack.append(c)
                elif isinstance(c, (list, tuple)):
                    stack.extend(x for x in c if isinstance(x, nn.Module))
        return n
    nq = _freeze_quantized(block)
    print(f"[init] 양자화 잔존 {nq}개 모듈 재동결", flush=True)
    if args.lora_shared:
        n_lora = attach_lora_shared_experts(block, r=args.lora_r, alpha=args.lora_alpha)
        print(f"[init] shared_experts LoRA 부착 {n_lora}개 (r={args.lora_r})", flush=True)
    if args.init_ckpt:
        w = mx.load(args.init_ckpt)
        block.load_weights(list(w.items()), strict=False)
        mx.eval(block.parameters())
        print(f"[init] 체크포인트 이어받기: {args.init_ckpt}", flush=True)
    # mxfp4 라우팅 전문가(switch_mlp) 게더에 VJP 없음(K3 [CA80] 동류) → 그
    # 경로만 stop-grad. shared_experts는 살려둠 — --lora-shared 시 여기로
    # 그래디언트가 흘러야 학습된다(원본 DeepseekV4MoE.__call__ 재구현).
    _FfnCls = type(block.block.ffn)
    def _sg_ffn(self, x, input_ids):
        if self.sharding_group is not None:
            raise RuntimeError("mtp 정렬 훈련은 비분산 전제 — sharding_group 예상 밖")
        inds, scores = self.gate(x, input_ids)
        y = self.switch_mlp(x, inds, scores=scores)
        if y.ndim == scores.ndim + 1:
            y = (y * scores[..., None].astype(y.dtype)).sum(-2)
        y = mx.stop_gradient(y)
        y = y + self.shared_experts(x)
        return y
    _FfnCls.__call__ = _sg_ffn
    # HC 등 커스텀 커널은 training 게이트로 순수-ops 폴백(VJP 가능) — mtp만 train 모드
    block.train()
    # wsdpa 융합 커널(VJP 없음) 무력화 — 호출부의 out=None 폴백이 스톡 SDPA로 전환
    import sys as _sys
    for _n in ("mlx_lm.models.deepseek_v4", "mlx_lm.models.deepseek_v4_mtp",
               "omlx.patches.deepseek_v4.wsdpa_attention"):
        _m = _sys.modules.get(_n)
        if _m is not None and hasattr(_m, "wsdpa_prefill"):
            _m.wsdpa_prefill = lambda *a, **k: None
            if hasattr(_m, "wsdpa_topk_prefill"):
                _m.wsdpa_topk_prefill = lambda *a, **k: None
    n_train = sum(v.size for _, v in tree_flatten(block.trainable_parameters()))
    print(f"[init] 학습 파라미터 {n_train/1e6:.1f}M", flush=True)

    real_h_get = None
    file_idx = None
    if args.real_hidden:
        windows, real_h_get, file_idx = load_real_hidden(args.real_hidden)
        kind = "디렉터리(lazy)" if os.path.isdir(args.real_hidden) else "단일파일"
        print(f"[data] 실측 TP2 hidden 캐시 로드[{kind}]: {len(windows)}개 윈도우 "
              f"({args.real_hidden})", flush=True)
    else:
        files = [l.strip() for l in open(args.corpus_list) if l.strip()]
        windows = build_corpus(tok, args.seq_len, files)
        file_idx = list(range(len(windows)))
    # 소규모 스모크에서 eval이 전량을 삼켜 train이 비는 것 방지(3윈도우 ZeroDivision 전례)
    n_eval = min(max(8, len(windows) // 20), max(1, len(windows) // 2))
    eval_set, train_set = windows[:n_eval], windows[n_eval:]
    eval_idx, train_idx = file_idx[:n_eval], file_idx[n_eval:]
    print(f"[data] train {len(train_set)} · eval {n_eval} 윈도우 · "
          f"loss-start-pos={args.loss_start_pos}", flush=True)

    embed = model.model.embed_tokens

    def chain_logits(h, ids):
        """교사-강제 d1/d2 로짓. h: [B,S,4,D] 백본 hidden(위치 t 정렬), ids: [B, S+3]."""
        S = h.shape[1]
        mask = create_attention_mask(h[:, :, 0, :], None,
                                     window_size=model.args.sliding_window,
                                     return_array=True)
        g1 = block(h, embed, ids[:, 1:S + 1], mask, None)          # d1: (h_t, t+1)
        z1 = model.lm_head(block.norm(block.hc_head(g1)))          # → t+2 예측
        g2 = block(g1, embed, ids[:, 2:S + 2], mask, None)         # d2: (g1, t+2)
        z2 = model.lm_head(block.norm(block.hc_head(g2)))          # → t+3 예측
        return z1, z2

    def loss_fn(_, h, ids):
        z1, z2 = chain_logits(h, ids)
        S = h.shape[1]
        t2 = ids[:, 2:S + 2]
        t3 = ids[:, 3:S + 3]
        n = args.loss_start_pos
        ce1 = nn.losses.cross_entropy(z1[:, n:], t2[:, n:], reduction="mean")
        ce2 = nn.losses.cross_entropy(z2[:, n:], t3[:, n:], reduction="mean")
        return ce2 + args.d1_retain * ce1

    def backbone_hidden(ids_in, noise_std=0.0):
        h_logits, h_raw = model.model(ids_in, None, return_raw_hidden=True)
        h_raw = mx.stop_gradient(h_raw)
        if noise_std > 0:
            # [I305] TP2 백본 hidden은 싱글박스 대비 실측 상대표준편차 ~0.4%
            # (K분할 부분합의 부동소수점 결합법칙 위반 — all_sum 정밀도로는
            # 제거 불가, arXiv 2511.17826과 일치). 싱글박스 훈련만으로는
            # 드래프터가 이 분포를 못 보므로, 배포 시 수용률이 급락한다
            # (train-serve mismatch). 안전한 싱글박스 환경에서 그 통계적
            # 특성(크기 비례 노이즈)을 흉내내 훈련시켜 강건화한다.
            h_raw = h_raw + mx.random.normal(h_raw.shape).astype(h_raw.dtype) * (
                h_raw.std() * noise_std
            )
        return h_raw

    def eval_match():
        m1 = m2 = tot = 0
        n = args.loss_start_pos
        for j, w in enumerate(eval_set[:8]):
            ids = mx.array([w])
            if real_h_get is not None:
                h = real_h_get(eval_idx[j]).astype(mx.bfloat16)
            else:
                h = backbone_hidden(ids[:, :args.seq_len])
            z1, z2 = chain_logits(h, ids)
            S = args.seq_len
            p1 = mx.argmax(z1, axis=-1); p2 = mx.argmax(z2, axis=-1)
            m1 += (p1[:, n:] == ids[:, 2:S + 2][:, n:]).sum().item()
            m2 += (p2[:, n:] == ids[:, 3:S + 3][:, n:]).sum().item()
            tot += (S - n)
        return m1 / tot, m2 / tot

    lvg = nn.value_and_grad(block, loss_fn)
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)

    a1, a2 = eval_match()
    print(f"[eval0] d1일치 {a1:.3f} · d2일치 {a2:.3f}", flush=True)
    if args.smoke:
        w = train_set[0]; ids = mx.array([w])
        h = backbone_hidden(ids[:, :args.seq_len])
        l, g = lvg(block, h, ids)
        mx.eval(l, g)
        gn = sum((v ** 2).sum().item() for _, v in tree_flatten(g))
        print(f"[smoke] loss {l.item():.3f} · grad_norm² {gn:.3e} — PASS", flush=True)
        return

    t0 = time.time()
    for step in range(1, args.steps + 1):
        j = (step - 1) % len(train_set)
        w = train_set[j]
        ids = mx.array([w])
        if real_h_get is not None:
            h = real_h_get(train_idx[j]).astype(mx.bfloat16)
        else:
            h = backbone_hidden(ids[:, :args.seq_len], noise_std=args.noise_std)
        l, grads = lvg(block, h, ids)
        opt.update(block, grads)
        mx.eval(block.parameters(), opt.state, l)
        if step % 20 == 0:
            print(f"[{step}] loss {l.item():.3f} · {(time.time()-t0)/step:.1f}s/step", flush=True)
        if step % args.eval_every == 0:
            a1, a2 = eval_match()
            print(f"[eval@{step}] d1 {a1:.3f} · d2 {a2:.3f}", flush=True)
        if step % args.save_every == 0:
            os.makedirs(args.out, exist_ok=True)
            flat = dict(tree_flatten(block.trainable_parameters()))
            mx.save_safetensors(os.path.join(args.out, f"step{step}.safetensors"),
                                {k: v for k, v in flat.items()})
            if args.lora_shared:
                # 병합 시 alpha/r 를 추정하지 않도록 사이드카로 명시 저장
                # (r/alpha 를 바꾼 뒤 조용한 오-병합 방지)
                with open(os.path.join(args.out, "lora_meta.json"), "w") as f:
                    json.dump({"r": args.lora_r, "alpha": args.lora_alpha,
                               "alpha_over_r": args.lora_alpha / args.lora_r}, f)
            print(f"[ckpt] step{step} 저장", flush=True)


if __name__ == "__main__":
    main()


