"""DSv4-Flash TP2 스모크: omlx 오버레이 + model.shard(group) + 플레인 greedy.
Pro 레시피(run_tp4_pro0813_dspark.py)의 각색 — DSpark 단언을 선택화."""
import argparse, os, sys, time, logging, threading, subprocess
logging.basicConfig(level=logging.INFO)

# ── [웨지 근본치료 · 2026-08-26] fast-fence 정책 ─────────────────────────────
# MLX_METAL_FAST_SYNCH=1 은 스트림-교차 의존을 **무한 스핀 2종**으로 강등한다:
#   · GPU: kernels/fence.metal:40-52 `fence_wait` — 탈출 조건이 값 도착뿐인
#          while(1) 스핀. 타임아웃도, 오류 검사도 없다.
#   · CPU: backend/metal/fence.cpp:59-60 `while (f.cpu_value()[0] < count) {}`
#          — 스트림당 단 하나뿐인 StreamThread(scheduler.h:19-65)를 통째로 점유.
# 느린 경로(=0)는 MTLSharedEvent 를 쓰고, 커맨드버퍼 오류 시 device.cpp:547-553
# 이 이벤트를 수동 시그널·오염시켜 **예외로 죽는다**(event.cpp:28-32). 즉 =0 은
# 같은 정지를 "영구 교착"이 아니라 "시끄러운 실패"로 바꾼다. 상류 #3830 도
# fast synch 를 "not reliable, no way to fix" 로 종결했다(상류 기본은 off — 즉
# `=1` 은 우리가 켠 옵트인이지 상류 기본의 이탈이 아니다).
#
# 정책 [2026-08-26 레드팀 리뷰 M4로 개정]: **전 경로 기본 0**.
#   초판은 `--all-topics` 만 0 으로 내리고 벽시계 tok/s 경로(`--batch N`)는
#   과거 로그 비교가능성을 위해 1 로 남겼다. 그런데 5번째 웨지(b1tp2_on1.log,
#   `--batch 1 --long-doc 397`, 600s 무출력)가 **바로 그 --batch 경로**에서 났다.
#   웨지가 실제로 난 경로에만 완화를 유보하는 것은 뒤집힌 위험 배분이다.
#   비용: 디코드 약 −35%(프리필 0%). 따라서 **과거 --batch tok/s 로그는 이 커밋
#   이후 판과 직접 비교할 수 없고, 재기저(re-baseline)가 필요하다.**
#   옛 수치를 재현하려면 `TP2_FAST_SYNCH=1` 을 명시 지정한다.
_FS = os.environ.get("TP2_FAST_SYNCH")
if _FS is None:
    _FS = "0"
# setdefault 금지: rt.sh:2 / rt_p1_recirc_{on,off}.sh / hostfile envs 가
# MLX_METAL_FAST_SYNCH=1 을 export 하므로 setdefault 로는 효과가 없다.
# 다만 **말없이** 덮어쓰면 그 스크립트들의 A/B 비교성이 조용히 깨진다
# (레드팀 리뷰: 재순환 A/B 대조군이 침묵 피격). 덮어쓸 때는 반드시 말한다.
_FS_PRE = os.environ.get("MLX_METAL_FAST_SYNCH")
if _FS_PRE is not None and _FS_PRE != _FS:
    print(f"[tp2] 주의: 호출 스크립트가 MLX_METAL_FAST_SYNCH={_FS_PRE} 를 export 했으나 "
          f"하네스 정책으로 {_FS} 로 덮어쓴다. 옛 체제로 재현하려면 "
          f"TP2_FAST_SYNCH={_FS_PRE} 를 명시할 것 (과거 로그와 비교 시 재기저 필요).",
          flush=True)
os.environ["MLX_METAL_FAST_SYNCH"] = _FS

# 웜업 게이트도 여기서 확정한다 — 아래 게이트 합의 검사가 이 값을 랭크 간 대조한다.
_WARMUP_ON = 0 if os.environ.get("TP2_WARMUP", "1").strip().lower() in \
    ("0", "", "false", "off") else 1
_WARMUP_TOKENS = int(os.environ.get("TP2_WARMUP_TOKENS", "32"))
import mlx.core as mx

sys.path.insert(0, "/Users/Shared/tp2")

# ── 진행-감시 워치독 (서빙 serve_batched_tp2.py:58-104 이식) ─────────────────
# 유한 시간에 끝나야 하는 구간만 busy 로 마킹하고, 무진행이 시한을 넘기면
# **TERM 전에** /usr/bin/sample 로 스택을 채취한다(가드 R2: 절대 자결·KILL 금지).
_WD = {"busy": 0, "t": time.time(), "tag": "-", "rank": -1, "fired": 0}
_WD_S = float(os.environ.get("TP2_WATCHDOG_S", "300"))
_WD_DIR = os.path.expanduser("~/dsv4flash/align/logs")


def _wd_sample():
    """웨지 스택 채취 — 감사 요건상 TERM 전에 반드시 남긴다."""
    _WD["fired"] += 1
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = f"{_WD_DIR}/wedge_sample_r{_WD['rank']}_{ts}.txt"
    try:
        os.makedirs(_WD_DIR, exist_ok=True)
        subprocess.run(["/usr/bin/sample", str(os.getpid()), "5", "-f", out],
                       timeout=120, check=False)
        print(f"[r{_WD['rank']}] 워치독: 스택 채취 → {out}", flush=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"[r{_WD['rank']}] 워치독: sample 실패 {e}", flush=True)
    try:
        with open(f"{_WD_DIR}/WEDGE_ALERT", "a") as fh:
            fh.write(f"{time.time()} r{_WD['rank']} tag={_WD['tag']} "
                     f"stall>{_WD_S:.0f}s sample={out}\n")
    except Exception:                                        # noqa: BLE001
        pass


def _watchdog():
    while True:
        time.sleep(10)
        if _WD["busy"] and time.time() - _WD["t"] > _WD_S:
            print(f"[r{_WD['rank']}] 워치독 경보: 구간 '{_WD['tag']}' "
                  f"{_WD_S:.0f}s 무진행 — 행 의심(자결 안 함)", flush=True)
            if _WD["fired"] < 3:
                _wd_sample()
            _WD["t"] = time.time()      # 재장전


class _wd_busy:
    """유한 시간 구간 마킹. **반드시 `with` 로 쓸 것** — 수동 __enter__/__exit__ 은
    구간 안에서 예외가 나면 busy 카운트가 새고, 워치독이 죽은 태그로 영원히
    경보한다(블랭크-슬레이트 리뷰 SHOULD, 2026-08-26)."""
    def __init__(self, tag): self.tag = tag
    def __enter__(self):
        self.prev = _WD["tag"]; _WD["busy"] += 1
        _WD["t"] = time.time(); _WD["tag"] = self.tag
        return self
    def __exit__(self, *a):
        _WD["busy"] -= 1; _WD["t"] = time.time(); _WD["tag"] = self.prev
        return False


def _gpu_ping():
    """레지던시 유지용 초소형 GPU 제출(~0.4ms). 상태·캐시·집합연산 미접촉.
    [I358 실측] 145GiB 상주 프로세스는 수 초 유휴 후 첫 제출이 ~0.9s 스톨한다."""
    mx.eval(mx.sum(mx.ones((8, 8))))

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
    ap.add_argument("--max-topics", type=int, default=0,
                    help="0이 아니면 --all-topics 를 앞 N개 토픽으로 자른다 "
                         "(웨지 콜드-스타트 반복 재현 프로토콜용 — 기본 0=전체)")
    args = ap.parse_args()

    group = mx.distributed.init()
    rank, world = group.rank(), group.size()
    _WD["rank"] = rank
    threading.Thread(target=_watchdog, daemon=True).start()
    # 배너는 **요청값**이다(실효값 아님). MLX_METAL_FAST_SYNCH 의 실제 적용 여부는
    # FenceImpl 생성자에서 `supportsFamily(Metal3)` + macOS 15 가용성까지 보고
    # 결정되며 프로세스 밖으로 노출되지 않는다. 그래서 "실측"이라 부르지 않는다.
    # 랭크 비대칭은 배너가 아니라 아래 게이트 합의가 잡는다.
    print(f"[rank {rank}] world={world} · FAST_SYNCH(요청)={_FS} "
          f"· watchdog={_WD_S:.0f}s", flush=True)
    assert world == args.require_world, f"world {world} != {args.require_world}"

    # ── 게이트 합의 검사 = 로드 **전** 랭크 랑데부 ────────────────────────────
    # 두 가지를 한 번에 한다.
    #  ① 랭크 간 게이트 비대칭 조기 사망. mlx.launch 는 실행 셸 env 를 원격에
    #     전파하지 않아, 한쪽만 FAST_SYNCH=1 이거나 한쪽만 웜업을 켜면 배리어에서
    #     그대로 영구 교착이다. 여기서 4바이트 텐서로 먼저 깨뜨린다.
    #  ② 서빙(serve_batched_tp2.py:45-55)의 "로드 전 제어채널 수립" 성질 이식 —
    #     프로세스 최초의 집합연산이 **모델 로드 전, 활성 텐서 ~0** 일 때 일어난다.
    # TP2_GATE_ASSERT=0 으로 끌 수 있다(이 경우 최초 집합연산은 다시 로드 후가 된다).
    if os.environ.get("TP2_GATE_ASSERT", "1").strip().lower() not in ("0", "", "false", "off"):
        # 그래프 형상·동기화 모드를 바꾸는 값은 전부 넣는다. 여기서 한 개라도
        # 어긋나면 첫 집합연산에서 형상 불일치나 영구 교착이 된다.
        _gate_names = ("FAST_SYNCH", "WARMUP", "WARMUP_TOKENS", "WATCHDOG_S",
                       "MTP", "MTP_DEPTH", "MTP_FIXED_DEPTH",
                       "long_doc", "max_tokens", "batch", "all_topics", "max_topics")
        _gate_vals = [float(_FS), float(_WARMUP_ON), float(_WARMUP_TOKENS), _WD_S,
                      float(os.environ.get("TP2_MTP", "0") == "1"),
                      float(os.environ.get("TP2_MTP_DEPTH", "1")),
                      float(os.environ.get("OMLX_MTP_FIXED_DEPTH", "0") == "1"),
                      float(args.long_doc), float(args.max_tokens), float(args.batch),
                      float(args.all_topics), float(args.max_topics)]
        with _wd_busy("gate"):
            _g = mx.array(_gate_vals, dtype=mx.float32)
            _gsum = mx.distributed.all_sum(_g, group=group)
            mx.eval(_gsum)
            _bad = [(n, v, s / world) for n, v, s
                    in zip(_gate_names, _gate_vals, _gsum.tolist())
                    if abs(s - v * world) > 1e-4]
        if _bad:
            for n, mine, avg in _bad:
                print(f"[r{rank}] 게이트 불일치: {n} 내값={mine} 전랭크평균={avg}",
                      flush=True)
            raise SystemExit(f"[r{rank}] 랭크 간 게이트 비대칭 — 교착 전에 중단. "
                             f"워커가 게이트를 argv 로 넘기는지 확인할 것.")
        if rank == 0:
            print(f"[tp2-gate] 전 랭크 합의 OK ({len(_gate_names)}개 값)", flush=True)

    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
    apply_deepseek_v4_patch()
    from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active, set_mtp_depth
    assert apply_mlx_lm_mtp_patch()
    mtp_on = os.environ.get("TP2_MTP", "0") == "1"
    set_mtp_active(mtp_on); set_mtp_depth(int(os.environ.get("TP2_MTP_DEPTH", "1")))
    from mlx_lm import load
    from dspark_tp4_common import shard_mtp

    t0 = time.monotonic()
    with _wd_busy("load"):        # 로드/샤딩도 유한 시간 구간(예외 시 busy 누수 방지)
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
            _WD["t"] = time.time()                       # 층마다 진행 신호
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

    # ── [웨지 근본치료] 랭크 배리어 → 콜드-스타트 웜업 (첫 제출 크기 단조 증가) ──
    # 웨지 5회는 전부 "로드 직후 첫 대형 프리필"에서 났다 — bs1 4회는 2,120토큰
    # (--all-topics 토픽 0), 5번째는 13.9K 토큰(--batch 1 --long-doc 397).
    # 그 프리필이 **프로세스 최초의 대형 집합연산**이라, 로드가 늦은 랭크를
    # 기다리는 동안 상대 랭크의 fast-fence 스핀이 무한정 열린 채로 있게 된다.
    # 서빙(serve_batched_tp2.py)은 ① 로드 전 제어채널 수립 ② 200tok 합성 웜업으로
    # 이 창을 닫아 두었다. 그 두 성질을 이식한다.
    #   1) 1-원소 all_sum 배리어 = 로드 **후**의 가장 작은 교차-랭크 핸드셰이크
    #      (로드 **전** 랑데부는 위 게이트 합의가 이미 해 두었다)
    #   2) 짧은 프롬프트·max_tokens=4 웜업 = 커널 JIT + 레지던시를 미리 태움
    # 측정 오염 방지: max_tokens=4 는 omlx 의 loop-tax 래치 조건(skip2+samples8=
    # 10스텝)에 한참 못 미치고, 그래도 남을 수 있는 래치는 아래에서 명시 제거한다.
    #
    # ⚠ 재기저 주의 [레드팀 S6]: 웜업은 **전 경로**에 적용된다(--batch 포함).
    # --batch 의 tok/s 측정창은 디코드 구간이므로, 웜업이 디코드 커널 JIT 을 미리
    # 태우면 과거 로그보다 초기 스텝이 빨라진다. 즉 이 판의 --batch 수치는
    # 더 정확한 정상상태 측정이지만 **과거 --batch 로그와 직접 비교 불가**다.
    # 어차피 같은 경로에서 FAST_SYNCH 기본값도 1→0 으로 바뀌었으므로 재기저는
    # 한 번에 하면 된다. 옛 체제 재현은 `TP2_FAST_SYNCH=1 TP2_WARMUP=0`.
    if _WARMUP_ON:
        with _wd_busy("barrier"):
            _tb = _t.monotonic()
            mx.eval(mx.distributed.all_sum(mx.ones((1,), dtype=mx.float32), group=group))
            mx.synchronize()
        _dtb = _t.monotonic() - _tb
        _wn = _WARMUP_TOKENS
        _wids = (list(tok.encode("warmup")) * 64)[:_wn]
        with _wd_busy("warmup"):
            _tw = _t.monotonic()
            _bgw = BatchGenerator(model, max_tokens=4, sampler=make_sampler(0.0),
                                  completion_batch_size=1, prefill_batch_size=1,
                                  prefill_step_size=2048)
            _bgw.insert([_wids], max_tokens=[4])     # 서빙 웜업과 동일한 예산
            for _ in range(8):
                _rw = _bgw.next_generated()
                if not _rw:
                    break
                _WD["t"] = _t.time()
                if any(getattr(r, "finish_reason", None) for r in _rw):
                    break
            del _bgw
            # 적응 상태 원복(웜업이 P1 수치에 스며들지 않게).
            # [레드팀 S7] FIXED_DEPTH=1 이면 loop-tax 컨트롤러 자체가 생성되지
            # 않으므로 이 제거는 대개 무동작이다 — 게이트가 바뀌어도 안전하도록
            # 남겨 두는 방어이지, 이것이 오염을 막는 주된 근거는 아니다.
            for _a in ("_omlx_mtp_loop_tax", "_omlx_mtp_loop_tax_ts"):
                try:
                    delattr(model, _a)
                except Exception:                            # noqa: BLE001
                    pass
            _gpu_ping()
        if rank == 0:
            print(f"[tp2-warm] 배리어 {_dtb*1e3:.0f}ms · 웜업 {_wn}tok "
                  f"{(_t.monotonic()-_tw)*1e3:.0f}ms", flush=True)

    mx.random.seed(7)  # 전 랭크 동일 시드 (JACCL 요건) — 웜업 뒤에 설정
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
        if args.max_topics > 0:
            TOP = TOP[:args.max_topics]
        for ti, topic in enumerate(TOP):
            ids1 = [tok.apply_chat_template(
                [{"role": "user", "content": (("참고 문서: " + "분산 텐서 병렬과 압축 어텐션의 상호작용은 위치 기하와 수용률 경제학에 민감하다. " * args.long_doc + " ") if args.long_doc else "") + f"{args.prompt} (topic {topic})"}],
                add_generation_prompt=True)]
            if rank == 0:
                print(f"[p1-topic {ti}/{len(TOP)}] {topic}", flush=True)
            # 토픽 경계는 GPU 유휴 창(이전 bg 파괴·KV GC·토크나이즈·bg 재생성)이다.
            # 유휴 뒤 첫 제출 ~0.9s 스톨이 fence 스핀 창을 넓히므로 핑으로 덮는다.
            # 핑도 집합연산 창에 걸릴 수 있으므로 워치독 안에서 친다(사각 제거).
            with _wd_busy(f"topic{ti}"):
                _gpu_ping()
                bg1 = BatchGenerator(model, max_tokens=args.max_tokens, sampler=make_sampler(0.0),
                                     completion_batch_size=1, prefill_batch_size=1,
                                     prefill_step_size=2048)
                bg1.insert(ids1, max_tokens=[args.max_tokens])
                done1 = 0
                out_toks = []
                while done1 < 1:
                    rs = bg1.next_generated()
                    if not rs: break
                    _WD["t"] = _t.time()          # 스텝마다 진행 신호
                    for r in rs:
                        out_toks.append(int(r.token))
                        if r.finish_reason: done1 += 1
            # 출력 지문(2026-08-26 재순환 A/B): 드래프트가 바뀌면 verify 경로의
            # 부동소수점 비결합성으로 텍스트가 갈릴 수 있다 — 갈림 자체는 결함이
            # 아니고 채록 대상이라, 팔별 비교가 가능하도록 토큰열 해시를 남긴다.
            if rank == 0:
                import hashlib
                h = hashlib.sha1(",".join(map(str, out_toks)).encode()).hexdigest()[:12]
                print(f"[p1-out {ti}] n={len(out_toks)} sha1={h}", flush=True)
        if rank == 0:
            print("[tp2-flash-pass]", flush=True)
        return
    idss = [tok.apply_chat_template(
        [{"role": "user", "content": (("참고 문서: " + "분산 텐서 병렬과 압축 어텐션의 상호작용은 위치 기하와 수용률 경제학에 민감하다. " * args.long_doc + " ") if args.long_doc else "") + f"{args.prompt} (topic {TOP[i % 8]})"}],
        add_generation_prompt=True) for i in range(B)]
    toks = []; t0 = _t.monotonic(); t1 = None; done = 0
    # 5번째 웨지(b1tp2_on1.log)는 정확히 이 경로였다 — 로드 완료 후 13.9K 프리필에서
    # 600s 무출력. 그러므로 여기도 워치독 안에 통째로 들어간다(수동 카운터 금지:
    # 예외 시 busy 가 새면 워치독이 죽은 태그로 영원히 경보한다).
    with _wd_busy(f"bs{B}"):
        _gpu_ping()
        bg = BatchGenerator(model, max_tokens=args.max_tokens, sampler=make_sampler(0.0),
                            completion_batch_size=B, prefill_batch_size=B, prefill_step_size=2048)
        bg.insert(idss, max_tokens=[args.max_tokens] * B)
        while done < B:
            rs = bg.next_generated()
            _WD["t"] = _t.time()
            if not rs: break
            if t1 is None: t1 = _t.monotonic()
            for r in rs:
                toks.append(int(r.token))
                if r.finish_reason: done += 1
    dt = _t.monotonic() - t1
    if rank == 0:
        print(f"[tp2] bs={B} · 집계 {(len(toks)-1)/dt:.2f} tok/s · 스트림당 {(len(toks)-1)/dt/B:.1f} · mtp={mtp_on}")
        # 검증 웨이브(2026-08-25): 체리픽 mlx 휠 vs 스톡 토큰 정합 대조용 — 출력 토큰 id 그대로 노출.
        print(f"[tp2] tokens: {toks}", flush=True)
        try:
            print(f"[tp2] text: {tok.decode(toks)[:300].replace(chr(10), ' ')}", flush=True)
        except Exception as e:
            print(f"[tp2] decode 실패: {e}", flush=True)
        print("[tp2-flash-pass]", flush=True)

if __name__ == "__main__":
    main()
