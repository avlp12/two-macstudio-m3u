"""DSv4-Flash TP2 스모크: omlx 오버레이 + model.shard(group) + 플레인 greedy.
Pro 레시피(run_tp4_pro0813_dspark.py)의 각색 — DSpark 단언을 선택화."""
import argparse, os, sys, time, logging
logging.basicConfig(level=logging.INFO)

os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
import mlx.core as mx

sys.path.insert(0, "/Users/Shared/tp2")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/dsv4flash/mlx4bit"))
    ap.add_argument("--prompt", default="17 * 23 =")
    ap.add_argument("--long-doc", type=int, default=0, help="0이 아니면 해당 반복수의 장문 문서를 프롬프트 앞에 부착")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--require-world", type=int, default=2)
    ap.add_argument("--all-topics", action="store_true",
                    help="[P1] batch=1 전제, 전체 토픽을 순차 실행(로드 1회 재사용) — bs1 페어드 검증용")
    args = ap.parse_args()

    group = mx.distributed.init()
    rank, world = group.rank(), group.size()
    print(f"[rank {rank}] world={world}", flush=True)
    assert world == args.require_world, f"world {world} != {args.require_world}"

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
    apply_deepseek_v4_patch()
    from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active, set_mtp_depth
    assert apply_mlx_lm_mtp_patch()
    mtp_on = os.environ.get("TP2_MTP", "0") == "1"
    set_mtp_active(mtp_on); set_mtp_depth(int(os.environ.get("TP2_MTP_DEPTH", "1")))
    from mlx_lm import load
    from dspark_tp4_common import shard_mtp

    t0 = time.monotonic()
    model, tok = load(args.model, lazy=True)
    assert hasattr(model, "shard"), "omlx base patch 미적용"
    _ck = os.environ.get("TP2_MTP_CKPT", "")
    if _ck and mtp_on:
        # [I292] 격차 원인 분리 실험: 서빙과 동일하게 샤딩 "전"에 정렬 가중치 적용
        # (서빙 serve_batched_tp2.py와 순서 일치 — 그래야 공정 비교)
        import sys as _s; _s.path.insert(0, os.path.expanduser("~/dsv4flash/align"))
        from train_align import promote_nonexpert, merge_lora_into_shared_experts
        # train_align 모듈 톱레벨이 depth를 1로 되돌림 — 하네스 env 설정 복원
        set_mtp_active(mtp_on); set_mtp_depth(int(os.environ.get("TP2_MTP_DEPTH", "1")))
        promote_nonexpert(model.mtp[0])
        _w = mx.load(_ck)
        # LoRA 체크포인트는 "부착"이 아니라 "병합" — LoRALinear를 남기면
        # shard_inplace(경로 기반 전수 분할)가 lora_a/lora_b까지 잘라 크래시한다.
        # bf16 Linear로 접어 두면 promote_nonexpert 전례대로 정상 샤딩된다.
        _n_lora = merge_lora_into_shared_experts(model.mtp[0], _w, ckpt_path=_ck)
        # 병합분은 이미 가중치에 흡수 — lora 키는 load_weights에 넘기지 않는다
        _rest = [(k, v) for k, v in _w.items()
                 if not (k.endswith(".lora_a") or k.endswith(".lora_b"))]
        model.mtp[0].load_weights(_rest, strict=False)
        if rank == 0:
            print(f"[tp2] 정렬 mtp 적용: {_ck} (lora 병합 {_n_lora}개)", flush=True)
    model.shard(group)
    n_mtp = 0
    try:
        n_mtp = shard_mtp(model, group)
    except Exception as e:
        if rank == 0:
            print(f"[tp2] mtp 샤딩 생략: {e}", flush=True)
    # 대형모델 하네스 규칙: wired limit 필수 + 층별 단계 실체화(메모리 스파이크 방지)
    info = mx.metal.device_info()
    mx.set_wired_limit(info["max_recommended_working_set_size"])
    inner = getattr(model, "model", model)
    heads = [m for m in (getattr(inner, n, None) for n in ("embed_tokens", "norm", "hc_head")) if m]
    if getattr(model, "lm_head", None) is not None:
        heads.append(model.lm_head)
    if heads:
        mx.eval(*[m.parameters() for m in heads])
    for i, layer in enumerate(model.model.layers):
        mx.eval(layer.parameters())
        mx.synchronize()
        if rank == 0 and (i + 1) % 8 == 0:
            print(f"[tp2-load] layer {i+1}/{len(model.model.layers)}", flush=True)
    mx.eval(model.parameters())
    mx.synchronize()
    heads = int(model.model.layers[0].attn.n_heads)
    if rank == 0:
        print(f"[tp2] load+shard {time.monotonic()-t0:.1f}s · layers "
              f"{len(model.model.layers)} · heads/rank {heads} · mtp {n_mtp}", flush=True)

    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    import time as _t
    mx.random.seed(7)  # 전 랭크 동일 시드 (JACCL 요건)
    # [P1] 원본 8개(비교 가능성 유지) + 비-CS 16개 확장 — 검증 도메인 스큐 해소
    TOP = ["MVCC", "B-trees", "ocean currents", "지붕 곡선", "LRU design",
           "raft", "rivers", "compiler IR",
           "조선 후기 상업의 발달", "sourdough fermentation", "고혈압의 관리",
           "contract law basics", "제주 올레길", "romantic era poetry",
           "환율과 물가의 관계", "stoic philosophy", "광합성의 명반응",
           "monsoon climate", "대위법의 기초", "zone defense tactics",
           "장미 전정 시기", "stellar nucleosynthesis", "조기 언어 교육",
           "impressionist painting"]
    B = args.batch
    if args.all_topics:
        assert B == 1, "--all-topics는 bs1 전용(프로덕션 체제 재현)"
        for ti, topic in enumerate(TOP):
            ids1 = [tok.apply_chat_template(
                [{"role": "user", "content": (("참고 문서: " + "분산 텐서 병렬과 압축 어텐션의 상호작용은 위치 기하와 수용률 경제학에 민감하다. " * args.long_doc + " ") if args.long_doc else "") + f"{args.prompt} (topic {topic})"}],
                add_generation_prompt=True)]
            if rank == 0:
                print(f"[p1-topic {ti}/{len(TOP)}] {topic}", flush=True)
            bg1 = BatchGenerator(model, max_tokens=args.max_tokens, sampler=make_sampler(0.0),
                                 completion_batch_size=1, prefill_batch_size=1,
                                 prefill_step_size=2048)
            bg1.insert(ids1, max_tokens=[args.max_tokens])
            done1 = 0
            while done1 < 1:
                rs = bg1.next_generated()
                if not rs: break
                for r in rs:
                    if r.finish_reason: done1 += 1
        if rank == 0:
            print("[tp2-flash-pass]", flush=True)
        return
    idss = [tok.apply_chat_template(
        [{"role": "user", "content": (("참고 문서: " + "분산 텐서 병렬과 압축 어텐션의 상호작용은 위치 기하와 수용률 경제학에 민감하다. " * args.long_doc + " ") if args.long_doc else "") + f"{args.prompt} (topic {TOP[i % 8]})"}],
        add_generation_prompt=True) for i in range(B)]
    bg = BatchGenerator(model, max_tokens=args.max_tokens, sampler=make_sampler(0.0),
                        completion_batch_size=B, prefill_batch_size=B, prefill_step_size=2048)
    bg.insert(idss, max_tokens=[args.max_tokens] * B)
    toks = []; t0 = _t.monotonic(); t1 = None; done = 0
    while done < B:
        rs = bg.next_generated()
        if not rs: break
        if t1 is None: t1 = _t.monotonic()
        for r in rs:
            toks.append(int(r.token))
            if r.finish_reason: done += 1
    dt = _t.monotonic() - t1
    if rank == 0:
        print(f"[tp2] bs={B} · 집계 {(len(toks)-1)/dt:.2f} tok/s · 스트림당 {(len(toks)-1)/dt/B:.1f} · mtp={mtp_on}")
        print("[tp2-flash-pass]", flush=True)

if __name__ == "__main__":
    main()
