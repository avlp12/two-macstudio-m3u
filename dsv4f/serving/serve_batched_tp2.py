"""TP2 연속-배칭 OpenAI 서버: 락스텝 BatchGenerator (rank0=HTTP, 워커=제어소켓 미러).
사이클 규약: rank0 이 삽입 목록을 프레임 방송 → 전 랭크 동일 insert → 동일 next_generated.
greedy+동일 시드로 전 랭크 토큰 동일 → 상태 영구 동기."""
import argparse, json, os, queue, select, threading, time, uuid, sys
import logging
logging.basicConfig(level=logging.INFO)
os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_tp4_dspark import (
    RequestError, validate_request, encode_prompt, parse_assistant,
    tokenizer_eos_ids, WorkerControl, connect_worker, send_frame, recv_frame,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/dsv4flash/mlx4bit"))
    ap.add_argument("--model-name", default="deepseek-v4-flash-tp2")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--control-host", default="10.0.0.1")
    ap.add_argument("--control-port", type=int, default=18004)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--max-output-tokens", type=int, default=4096)
    args = ap.parse_args()

    import mlx.core as mx
    group = mx.distributed.init()
    rank, world = group.rank(), group.size()
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
    from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active, set_mtp_depth
    apply_deepseek_v4_patch(); assert apply_mlx_lm_mtp_patch()
    _depth = int(os.environ.get("TP2_MTP_DEPTH", "1"))
    set_mtp_active(True); set_mtp_depth(_depth)
    if rank == 0:
        print(f"[cfg] mtp depth={_depth} (체인={'on' if _depth > 1 else 'off·legacy'})",
              flush=True)
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.cache import LRUPromptCache
    sys.path.insert(0, "/Users/Shared/tp2")
    from dspark_tp4_common import shard_mtp

    # 제어 채널을 로드보다 먼저 수립 — 양 랭크 로드 속도차와 무관해짐
    _control = None
    _wsock = None
    if rank == 0:
        _control = WorkerControl(world, "0.0.0.0", args.control_port)
        _control.listener.settimeout(900)
        _control.accept_all()
        print("[r0] 제어 채널 수립", flush=True)
    else:
        _wsock = connect_worker(args.control_host, args.control_port, rank)
        print(f"[r{rank}] 제어 접속 완료", flush=True)
    mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])

    # ── 집합연산 워치독: bg.next()가 시한 내 안 돌아오면(=분산 교착) 즉시 자결.
    # TERM-불응 좀비가 wired 70GB+를 쥔 채 잔존해 시스템 전체를 죽이는
    # 2026-08-23 크래시의 재발 방지. os._exit는 wired를 즉시 반환한다.
    WATCHDOG_S = float(os.environ.get("TP2_WATCHDOG_S", "300"))
    _wd = {"busy": False, "t": time.time()}
    def _watchdog():
        while True:
            time.sleep(10)
            if _wd["busy"] and time.time() - _wd["t"] > WATCHDOG_S:
                # [I296] 자결(os._exit)이 RDMA 오염→시스템 정지의 방아쇠임이
                # 2회 실증 — 경보만 남기고 유지(정리는 오퍼레이터가 전체-박스
                # 질서 재부팅으로). R2와 동일 원리: 행 중 프로세스는 죽이지 않는다.
                open("/Users/Shared/tp2/WATCHDOG_ALERT", "a").write(
                    f"{time.time()} r{rank} 집합연산 {WATCHDOG_S:.0f}s 무응답\n")
                print(f"[r{rank}] 워치독 경보: 행 감지 — 자결 안 함, 오퍼레이터 개입 필요",
                      flush=True)
                _wd["t"] = time.time()
    threading.Thread(target=_watchdog, daemon=True).start()
    model, tok = load(args.model, lazy=True)
    _ck = os.environ.get("TP2_MTP_CKPT", "")
    if _ck:
        # 정렬 mtp 사이드카: 샤딩 전에 비전문가부 bf16 승격 + 가중치 적용(양 랭크 동일)
        import sys as _s; _s.path.insert(0, os.path.expanduser("~/dsv4flash/align"))
        from train_align import promote_nonexpert, merge_lora_into_shared_experts
        set_mtp_active(True); set_mtp_depth(_depth)   # train_align 임포트 부작용 방어
        promote_nonexpert(model.mtp[0])
        _w = mx.load(_ck)
        # LoRA 체크포인트면 shared_experts에 병합 후 bf16 Linear로 교체.
        # (그냥 load하면 lora 키가 strict=False로 조용히 버려지고, 부착해 두면
        #  shard_inplace가 lora_a/lora_b까지 잘라 크래시한다 — 병합이 정답)
        _n_lora = merge_lora_into_shared_experts(model.mtp[0], _w, ckpt_path=_ck)
        _rest = [(k, v) for k, v in _w.items()
                 if not (k.endswith(".lora_a") or k.endswith(".lora_b"))]
        model.mtp[0].load_weights(_rest, strict=False)
        if rank == 0:
            print(f"[cfg] 정렬 mtp 적용: {_ck} (lora 병합 {_n_lora}개)", flush=True)
    model.shard(group)
    try: shard_mtp(model, group)
    except Exception as e: print(f"[r{rank}] mtp 샤딩 실패: {e}", flush=True)
    for layer in model.model.layers:
        mx.eval(layer.parameters()); mx.synchronize()
    mx.eval(model.parameters()); mx.synchronize()
    mx.random.seed(7)
    eos = set(tokenizer_eos_ids(tok))
    print(f"[r{rank}] 적재·샤딩 완료 (world {world})", flush=True)

    # ── PP2 프리필 스테이지 (DSV4_PP2_PREFILL=1 일 때만; 기본 off = 현행 경로) ──
    # 프리필만 2박스 층-파이프라인으로 교체하고 디코드/배칭/MTP 는 무변경.
    # 양 랭크가 대칭으로 생성·호출해야 락스텝이 유지되므로 setup 실패는 치명 처리.
    import pp2_prefill_stage
    pp2_gpu_ping = pp2_prefill_stage.gpu_ping

    def pp2_keepalive_s():
        return pp2_prefill_stage.KEEPALIVE_S
    _pp2 = pp2_prefill_stage.from_env(
        rank, model, args.model,
        log=(lambda *a: print(*a, flush=True)))
    _PP2_MIN = _pp2.min_tokens if _pp2 is not None else 1 << 30
    if _pp2 is not None:
        print(f"[r{rank}] PP2 프리필 활성 (split={_pp2.split} "
              f"min_tokens={_PP2_MIN} chunk={_pp2.chunk})", flush=True)

    def make_bg():
        return BatchGenerator(model, max_tokens=args.max_output_tokens,
                              sampler=make_sampler(0.0),
                              completion_batch_size=args.max_batch,
                              prefill_batch_size=1, prefill_step_size=2048)

    # ── 구간 트레이스 (DSV4_PP2_TRACE=1) ────────────────────────────────────
    # PP2 통합 오버헤드(스테이지 밖 e2e 손실)를 구간별로 귀속시키기 위한 계측.
    # 양 랭크 대칭으로 같은 자리에만 걸므로 연산 순서·락스텝에 영향 없음.
    TRACE = os.environ.get("DSV4_PP2_TRACE", "0").strip().lower() not in (
        "0", "", "false", "off")
    _TR: dict = {}

    # ── GPU 유휴 킵얼라이브 ──────────────────────────────────────────────
    # 145GiB 상주 프로세스는 GPU 가 수 초 유휴가 되면 **다음 첫 제출이 ~0.9s 스톨**한다
    # (레지던시 재수립. 실측: 유휴 후 mx.eval(mx.sum(mx.ones((8,8)))) 918.6ms → 직후 0.4ms).
    # 유휴 대기 지점(랭크0 gen_loop 폴링 / 랭크1 제어프레임 대기)에서 초소형 op 를 주기적으로
    # 던져 유휴를 없앤다. 상태·캐시·집합연산 미접촉이라 토큰 동일성에 무관.
    _KA = pp2_keepalive_s()

    def _idle_ping(last):
        if _KA <= 0:
            return last
        now = time.monotonic()
        if now - last >= _KA:
            pp2_gpu_ping()
            return time.monotonic()
        return last

    def recv_frame_warm(sock):
        if _KA <= 0:
            return recv_frame(sock)
        while True:
            r, _, _ = select.select([sock], [], [], _KA)
            if r:
                return recv_frame(sock)
            pp2_gpu_ping()

    if TRACE:
        # step0 내부 귀속(트레이스 전용): merge / split-deepcopy / GenerationBatch.__init__.
        # 각 구간 뒤 mx.synchronize() 로 지연 평가를 접어 벽시계에 정직하게 귀속시킨다.
        import importlib
        _mg = importlib.import_module("mlx_lm.generate")  # mlx_lm.generate 는 함수명과 충돌

        def _wrap(name, fn):
            def inner(*a, **k):
                _t = time.perf_counter()
                out = fn(*a, **k)
                mx.synchronize()
                _TR[name] = _TR.get(name, 0.0) + (time.perf_counter() - _t) * 1e3
                return out
            return inner

        _mg._merge_caches = _wrap("merge_ms", _mg._merge_caches)
        _mg.PromptProcessingBatch._copy = _wrap(
            "copy_ms", _mg.PromptProcessingBatch._copy)
        _mg.GenerationBatch.__init__ = _wrap(
            "gbinit_ms", _mg.GenerationBatch.__init__)

    # ── 프리픽스 스냅숏 (양 랭크 대칭: 동일 연산 순서 → 동일 LRU 상태 → 락스텝 유지) ──
    # 캡처 시점 = 프롬프트→생성 전환(end_of_prompt) 순간: 캐시가 정확히 ids[:-1]까지만
    # 처리된 상태라, 멀티턴 재렌더에서 바뀌는 꼬리 토큰(<think> 등)이 자연 배제됨.
    SNAP_MIN_TOKENS = 256           # 이보다 짧은 프롬프트는 스냅숏 비대상
    # 부분 적중 유지/폐기 손익분기(잔여 비율). 544 tok/s(단일-박스) vs 1031(PP2).
    SNAP_KEEP_FRAC = float(os.environ.get("DSV4_SNAP_KEEP_FRAC", "0.53"))
    snapstore = LRUPromptCache(max_size=64, max_bytes=32 << 30)

    def _env_on(name):
        return os.environ.get(name, "0").strip().lower() not in ("0", "", "false", "off")

    # ── 백로그 ①: HOL 인터리브 (DSV4_PP2_INTERLEAVE=1, 기본 0 = 현행 경로) ──
    # PP2 프리필의 2048-청크 이음새마다 라이브 배치 디코드를 1 스텝 끼워 넣어,
    # 13.9K 프리필이 전 트래픽을 13.7s 동결시키던 HOL 을 청크 간격(약 1.9s)으로 쪼갠다.
    # 실행 순서만 바뀌고 각 forward 의 수학은 불변.
    PP2_INTERLEAVE = _env_on("DSV4_PP2_INTERLEAVE")
    # 이음새당 최대 스텝 수. 1 스텝/이음새는 부족했다(실측 on1): BatchGenerator 가
    # prefill_batch_size=1 이라 스텝 1회가 대기 프롬프트 **1건**만 프리필한다 →
    # 7 이음새로는 단문 4건을 못 소화해 TTFT 가 16s 에 머물렀다. 대기 프롬프트가
    # 남아 있는 동안 이음새에서 계속 돌리고, 없으면 디코드 1스텝만 하고 빠진다.
    PP2_SEAM_STEPS = int(os.environ.get("DSV4_PP2_SEAM_STEPS", "12"))
    # ── 백로그 ②: PP2 캐시 스냅숏 등록 (DSV4_PP2_SNAPSTORE=1, 기본 0) ──
    # PP2 로 만든 전층 캐시를 기존 스냅숏 계약(키=ids[:-1])으로 스토어에 등록해
    # 멀티턴 장문 재사용을 PP2 경로에도 회복시킨다.
    PP2_SNAPSTORE = _env_on("DSV4_PP2_SNAPSTORE")
    # 이음새 콜백 홀더 — 랭크별 루프가 자기 콜백을 심는다(OFF 면 None = 무변경).
    _seam = {"cb": None}

    def _snap_canon(full):
        """스토어 저장본을 **와이어 형태로 정규화**한 독립 사본으로 만든다.

        왜 필요한가(리뷰 지적 · 치명):
          `PoolingCache.nbytes` 는 `_pool_buf` 의 **용량**을 센다(논리 길이가 아니라).
          로컬 계산분은 `_grow_pool` 의 기하급수 증가로 슬랙이 남고, 원격 복원분은
          `state` 를 통해 들어와 정확 길이다. 그런데 rank0 의 full 은 [복원|로컬],
          rank1 은 [로컬|복원] 이라 **같은 논리 캐시인데 랭크마다 nbytes 가 다르다**.
          그대로 LRU 에 넣으면 `_n_bytes` 가 랭크마다 어긋나 바이트-축출 시점이 갈리고,
          이후 `fetch_nearest_cache` 결과가 갈려 삽입 형상이 달라진다 = 락스텝 영구 파손.
          (`remainder==0` 일 때 buf_kv 가 로컬엔 있고 state 엔 None 인 차이도 같은 축.)

        `state` 는 논리 뷰만 내놓으므로 state→state 왕복이 곧 원격 복원분이 이미 밟는
        경로이고, 양 랭크 동일 표현·동일 nbytes 를 보장한다. `mx.contiguous` 로
        새 버퍼를 떠서 원본(=bg 에 넘길 full)과의 소유권도 끊는다."""
        arrs = []

        def _cv(x):
            if isinstance(x, mx.array):
                y = mx.contiguous(x)
                arrs.append(y)
                return y
            if isinstance(x, (list, tuple)):
                return type(x)(_cv(v) for v in x)
            return x

        out = model.make_cache()
        for sh, src in zip(out, full):
            sh.meta_state = src.meta_state   # ratio 등 먼저 (state 설정이 참조)
            sh.state = _cv(src.state)
        mx.eval(*arrs)
        return out
    if rank == 0:
        print(f"[cfg] HOL 인터리브={'on' if PP2_INTERLEAVE else 'off'} "
              f"PP2 스냅스토어={'on' if PP2_SNAPSTORE else 'off'}", flush=True)

    def snap_insert(bg, items, pending):
        """items=[(full_ids, n)] → 최근접 프리픽스 복원 삽입. 양 랭크 동일 결정."""
        from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
        uids = []
        for ids, n in items:
            ids = list(ids)
            cache, rest = snapstore.fetch_nearest_cache("m", ids)
            if cache is not None and not rest:
                # 저장분이 ids 전체와 완전 일치(희귀): 삽입엔 최소 1토큰 필요
                if can_trim_prompt_cache(cache):
                    trim_prompt_cache(cache, 1); rest = ids[-1:]
                else:
                    cache = None; rest = ids     # 비트리머블 → 전체 프리필
            if (PP2_SNAPSTORE and cache is not None and _pp2 is not None
                    and len(ids) >= _PP2_MIN
                    and len(rest) > SNAP_KEEP_FRAC * len(ids)):
                # 부분 적중인데 잔여가 크면 스냅숏이 오히려 손해: 잔여는
                # BatchGenerator 단일-박스 프리필(실측 약 544 tok/s)로 도는 반면
                # PP2 전체 재구축은 약 1031 tok/s → 손익분기 rest/544 = ids/1031,
                # 즉 rest > 0.53·ids 일 때만 버리는 게 이득이다. (초기 구현은
                # rest >= _PP2_MIN 절대치라 긴 멀티턴에서 되레 손해였다 — 리뷰 지적)
                # 완전 적중은 rest=1 이라 안 걸림 = 멀티턴 이득 보존.
                # 양 랭크 동일 입력·동일 스토어 → 동일 판정 → 락스텝 유지.
                cache = None; rest = ids
            key = ids[:-1]                       # 저장 키 = 휘발성 꼬리 제외
            if cache is not None:
                pref = len(ids) - len(rest)
                rest = list(rest)
                segs = [rest[:-1], rest[-1:]] if len(rest) >= 2 else [rest]
                new = bg.insert_segments([segs], max_tokens=[n], caches=[cache],
                                         all_tokens=[ids[:pref]])
                if rank == 0:
                    print(f"[snap] 적중 {pref}/{len(ids)} tok 재사용", flush=True)
            elif _pp2 is not None and len(ids) >= _PP2_MIN:
                # PP2 경로: 프리필을 BatchGenerator 밖(2박스 층-파이프라인)에서 끝내고
                # 완성된 전층 캐시를 주입 — 스냅숏 적중과 동일한 삽입 계약.
                # pref=len(key) 로 두어 스냅숏 캡처 대상에서 자연 제외(pending 미등록).
                # ※ MTP 프라이밍은 BatchGenerator 프리필 forward 에 붙으므로 이 경로엔
                #   없다 → take_primed 가 None → unprimed 폴백(설계된 안전 저하).
                pref = len(key)
                _ta = time.perf_counter()
                full = _pp2.build_cache(
                    ids, on_seam=(_seam["cb"] if PP2_INTERLEAVE else None))
                _tb = time.perf_counter()
                if PP2_SNAPSTORE and len(key) >= SNAP_MIN_TOKENS:
                    # 스냅숏 계약과 동일: 키 = ids[:-1], 값 = 그 지점까지의 전층 캐시.
                    # PP2 캐시는 정확히 ids[:-1] 만 처리한 상태라 캡처 경계가 딱 맞는다.
                    # 사본은 **필수**다(선택적 방어가 아님): insert_cache 는 복사를 하지
                    # 않고 객체를 그대로 보유하는데, 같은 full 이 bg 로도 넘어가 꼬리
                    # 토큰 forward 에서 갱신된다. _snap_canon 이 새 버퍼로 뜨면서
                    # 양 랭크 nbytes 대칭까지 함께 해결한다(위 정의 주석 참조).
                    _tsa = time.perf_counter()
                    snapstore.insert_cache("m", key, _snap_canon(full))
                    _TR["pp2snap_ms"] = (time.perf_counter() - _tsa) * 1e3
                    if rank == 0:
                        print(f"[snap] PP2 저장 {len(key)} tok "
                              f"({_TR['pp2snap_ms']:.0f}ms, store={len(snapstore)}, "
                              f"{snapstore.nbytes >> 20}MB)", flush=True)
                _tb2 = time.perf_counter()
                new = bg.insert_segments([[ids[-1:]]], max_tokens=[n],
                                         caches=[full], all_tokens=[key])
                _tc = time.perf_counter()
                _TR["build_ms"] = (_tb - _ta) * 1e3
                _TR["insseg_ms"] = (_tc - _tb2) * 1e3
                if rank == 0:
                    print(f"[pp2] 주입 {len(key)}/{len(ids)} tok", flush=True)
            else:
                pref = 0
                if len(ids) >= SNAP_MIN_TOKENS:
                    # 본문/꼬리 2-세그먼트: 본문 종료 경계(꼬리 미처리)에서 캐시 캡처
                    new = bg.insert_segments([[key, ids[-1:]]], max_tokens=[n])
                else:
                    new = bg.insert([ids], max_tokens=[n])
            u = new[0]; uids.append(u)
            if len(ids) >= SNAP_MIN_TOKENS and len(key) > pref:
                pending[u] = key
        return uids

    def snap_capture(bg, prompt_rs, pending):
        """본문 세그먼트 종료 경계(꼬리 토큰 미처리 상태)에서 캐시를 추출·저장.
        end_of_prompt 시점은 GenerationBatch.__init__의 _step()이 꼬리를 이미
        forward한 뒤라 오프-바이-원 — 반드시 세그먼트 경계에서만 캡처."""
        for r in prompt_rs:
            if not getattr(r, "end_of_segment", False) or getattr(r, "end_of_prompt", False):
                continue
            key = pending.pop(r.uid, None)
            if key is None:
                continue
            try:
                cache, _toks = bg.extract_cache([r.uid])[r.uid]
                snapstore.insert_cache("m", key, cache)
                if rank == 0:
                    print(f"[snap] 저장 {len(key)} tok (store={len(snapstore)}, "
                          f"{snapstore.nbytes >> 20}MB)", flush=True)
            except Exception as e:
                if rank == 0:
                    print(f"[snap] 저장 실패: {e}", flush=True)

    if rank != 0:
        sock = _wsock
        bg = make_bg()
        pending = {}
        _nstep = 0

        def _r1_step():
            prs, rs = bg.next()
            if prs:
                snap_capture(bg, prs, pending)

        def _r1_seam():
            """PP2 청크 이음새(랭크1): r0 이 `seam_end` 를 보낼 때까지 프레임을 소비.
            스텝 수를 r0 이 데이터-주도로 정하므로(대기 프롬프트 유무) 프레임 열
            자체가 계약이다 — 랭크1 은 세지 않고 따라가기만 하면 대칭이 보장된다."""
            while True:
                cmd = recv_frame_warm(sock)
                op = cmd.get("op")
                if op == "insert":
                    # r0 이 PP2-적격 요청을 진행 중엔 절대 안 보내므로 재진입 없음.
                    snap_insert(bg, cmd["items"], pending)
                elif op == "step":
                    _r1_step()
                elif op in ("seam_end", "stop"):
                    return

        if PP2_INTERLEAVE:
            _seam["cb"] = _r1_seam
        while True:
            cmd = recv_frame_warm(sock)
            if cmd.get("op") == "seam_end":
                # 메인 루프에 seam_end 가 오는 건 이음새 계약 위반 = 게이트가 랭크별로
                # 다르게 켜진 상황(랭크0만 INTERLEAVE)뿐이다. 조용히 넘기면 뒤이은
                # step 프레임을 랭크1만 한 번 더 실행해 회복 불가로 어긋난다 —
                # 크게 죽는 편이 낫다(리뷰 지적).
                raise RuntimeError(
                    "[r1] 이음새 계약 위반: 메인 루프에 seam_end 도착 — "
                    "DSV4_PP2_INTERLEAVE 가 랭크마다 다르게 설정됐는지 확인")
            if cmd.get("op") == "insert":
                _t0 = time.perf_counter()
                snap_insert(bg, cmd["items"], pending)
                _t1 = time.perf_counter()
                _nstep = 0
                if TRACE:
                    print(f"[trace-r1] insert 총 {(_t1 - _t0) * 1e3:.1f}ms "
                          f"(build {_TR.get('build_ms', 0):.1f} / "
                          f"insseg {_TR.get('insseg_ms', 0):.1f})", flush=True)
            elif cmd.get("op") == "step":
                _t0 = time.perf_counter()
                prs, rs = bg.next()
                _t1 = time.perf_counter()
                if TRACE and _nstep < 6:
                    print(f"[trace-r1] step{_nstep} bg.next {(_t1 - _t0) * 1e3:.1f}ms"
                          f" rs={len(rs)}", flush=True)
                    _nstep += 1
                if prs:
                    snap_capture(bg, prs, pending)
            elif cmd.get("op") == "stop":
                break
        return

    # ── rank 0 ──
    control = _control
    bg = make_bg()
    inbox: "queue.Queue" = queue.Queue()
    jobs: dict = {}
    lock = threading.Lock()

    def gen_loop():
        uid_by_slot = {}
        pending = {}
        st = {"live": 0, "ka": time.monotonic(), "in_pp2": False}
        deferred: list = []   # PP2 진행 중 보류된 PP2-적격 요청 (FIFO 유지)

        def _emit(rs):
            if not getattr(gen_loop, "_first", False):
                gen_loop._first = True
                print("[gen] 첫 토큰 생성", flush=True)
            for r in rs:
                uid = uid_by_slot.get(getattr(r, "uid", None))
                with lock:
                    j = jobs.get(uid)
                if j is None: continue
                t = int(r.token)
                fin = bool(r.finish_reason) or t in eos
                if not fin:
                    j["tokens"].append(t)
                    if j["sq"] is not None: j["sq"].put(t)
                else:
                    if not j["done"]:
                        j["done"] = True
                        st["live"] -= 1
                        if j["sq"] is not None: j["sq"].put(None)
                        j["ev"].set()

        def _insert_batch(batch, is_pp2=False):
            """batch=[(uid, ids, n, t_enq)] → 제어 프레임 1개 + 대칭 snap_insert.
            is_pp2=True 면 snap_insert 구간에만 in_pp2 를 세운다 — 플래그를 호출
            **밖에서** 세우면 PP2 삽입 자신이 nested 로 분류돼 트레이스(_TR)가
            직전 요청 것으로 잘못 귀속된다(리뷰 지적)."""
            items = [(ids, n) for (_u, ids, n, _t) in batch]
            metas = [u for (u, _i, _n, _t) in batch]
            t_enq = min(b[3] for b in batch)
            t_pick = time.perf_counter()
            print(f"[gen] 삽입 {len(items)}건", flush=True)
            nested = st["in_pp2"]
            if not nested:            # 트레이스 시작점은 최상위 삽입에서만 잡는다
                _TR.clear()
                _TR["t_enq"] = t_enq
                _TR["q_ms"] = (t_pick - t_enq) * 1e3
            control.dispatch({"op": "insert", "items": items})
            t_disp = time.perf_counter()
            if is_pp2:
                st["in_pp2"] = True
            try:
                new_uids = snap_insert(bg, items, pending)
            finally:
                if is_pp2:
                    st["in_pp2"] = False
            t_ins = time.perf_counter()
            if not nested:
                _TR["disp_ms"] = (t_disp - t_pick) * 1e3
                _TR["snapins_ms"] = (t_ins - t_disp) * 1e3
                _TR["steps"] = []
                _TR["armed"] = True
            for u, uid in zip(new_uids, metas):
                uid_by_slot[u] = uid
            st["live"] += len(items)

        def _drain(allow_pp2):
            """인박스(+보류분) 흡수 → 삽입.
            allow_pp2=False(=PP2 진행 중 이음새)면 PP2-적격 요청은 보류만 한다 —
            그래야 랭크1 의 이음새 콜백이 중첩 build_cache 로 재진입하지 않는다."""
            busy = st["live"] + (1 if st["in_pp2"] else 0)
            room = max(0, args.max_batch - busy)
            norm, pp2, keep = [], None, []

            def _take(it):
                """지금 넣을지/보류할지 결정. **보류는 예산(room)을 먹지 않는다** —
                먹게 두면 큐에 쌓인 장문들이 예산을 다 차지해 신규 단문이 인박스에
                갇히고, 이 기능이 없애려던 HOL 이 그대로 재현된다(리뷰 지적)."""
                nonlocal pp2
                if len(norm) + (1 if pp2 is not None else 0) >= room:
                    keep.append(it)
                    return
                if PP2_INTERLEAVE and _pp2 is not None and len(it[1]) >= _PP2_MIN:
                    # PP2 진행 중(allow_pp2=False)엔 보류만 — 랭크1 이음새 콜백의
                    # 중첩 build_cache 재진입을 막는 유일한 장치다.
                    if allow_pp2 and pp2 is None:
                        pp2 = it
                    else:
                        keep.append(it)
                    return
                norm.append(it)

            src = deferred[:]
            deferred.clear()
            for it in src:                       # 보류분이 항상 먼저 = FIFO
                _take(it)
            while len(norm) + (1 if pp2 is not None else 0) < room:
                try:
                    _take(inbox.get_nowait())
                except queue.Empty:
                    break
            deferred.extend(keep)
            if not PP2_INTERLEAVE:
                if norm:
                    _insert_batch(norm)          # 게이트 OFF = 한 배치·원래 순서
                return
            if norm:
                # 장문과 같은 배치로 들어온 단문도 먼저 라이브로 만들어야
                # 뒤이은 PP2 이음새 스텝에서 함께 디코드된다.
                _insert_batch(norm)
            if pp2 is not None:
                if norm:
                    # ★ 장문 프리필을 시작하기 **전에** 같은 배치의 단문을 소진시킨다.
                    # 실측 on2: 이걸 안 하면 5건이 한 drain 에 잡혀 단문 4건이 스텝을
                    # 한 번도 못 받은 채 첫 이음새(약 2.3s)까지 프롬프트 큐에 갇혔고
                    # TTFT 6-11s 로 남았다. 여기서 스텝을 돌리면 0.5-1.5s 로 내려간다.
                    _pump(PP2_SEAM_STEPS, absorb=False)
                _insert_batch([pp2], is_pp2=True)

        def _step():
            t_s0 = time.perf_counter()
            control.dispatch({"op": "step"})
            t_s1 = time.perf_counter()
            prs, rs = bg.next()
            t_s2 = time.perf_counter()
            if _TR.get("armed") and len(_TR.get("steps", [])) < 8:
                _TR["steps"].append(((t_s1 - t_s0) * 1e3, (t_s2 - t_s1) * 1e3))
            if prs:
                snap_capture(bg, prs, pending)
            if not rs:
                return
            if _TR.pop("armed", False) and TRACE:
                _tok = time.perf_counter()
                _st = " ".join(f"[{i}]disp{a:.1f}/next{b:.1f}"
                               for i, (a, b) in enumerate(_TR["steps"]))
                _st += (f" | merge={_TR.get('merge_ms', 0):.1f} "
                        f"copy={_TR.get('copy_ms', 0):.1f} "
                        f"gbinit={_TR.get('gbinit_ms', 0):.1f}")
                print(f"[trace] q={_TR['q_ms']:.1f}ms disp={_TR['disp_ms']:.1f}ms "
                      f"snap_insert={_TR['snapins_ms']:.1f}ms "
                      f"(build={_TR.get('build_ms', 0):.1f} "
                      f"insseg={_TR.get('insseg_ms', 0):.1f}) "
                      f"steps: {_st} | enq→tok0={(_tok - _TR['t_enq']) * 1e3:.1f}ms",
                      flush=True)
            _emit(rs)

        def _prompt_pending():
            """bg 안에 아직 프리필이 안 끝난 프롬프트가 있나(랭크0 로컬 판정).
            판정은 프레임 열로 랭크1 에 전달되므로 랭크1 과의 대칭은 자동."""
            try:
                return bool(bg._unprocessed_sequences) or len(bg._prompt_batch) > 0
            except Exception:
                return False

        def _pump(max_steps, absorb):
            """대기 프롬프트가 소진될 때까지 스텝을 돌린다(최대 max_steps).
            prefill_batch_size=1 이라 스텝 1회당 프롬프트 **1건**만 소화되므로
            단문 여러 건은 여러 스텝이 필요하다. 큐가 빈 뒤에도 1스텝 더 줘서
            갓 프리필된 시퀀스가 첫 토큰을 내도록 한다.
            absorb=True 면 스텝 사이 신규 도착도 흡수(이음새 전용).
            판정은 전부 랭크0 로컬이고 결과는 step 프레임 열로 전달 → 대칭 자동."""
            n, tail = 0, 0
            while n < max_steps and st["live"] > 0:
                _step()
                n += 1
                if absorb:
                    _drain(allow_pp2=False)   # 재귀 없음: allow_pp2=False 면 _pump 미호출
                if not _prompt_pending():
                    if tail:
                        break
                    tail = 1
            return n

        def _r0_seam():
            """PP2 청크 이음새(랭크0): 신규 삽입 + 스텝(들) + `seam_end` 종결 프레임."""
            _drain(allow_pp2=False)
            _pump(PP2_SEAM_STEPS, absorb=True)
            control.dispatch({"op": "seam_end"})

        if PP2_INTERLEAVE:
            _seam["cb"] = _r0_seam
        while True:
            _drain(allow_pp2=True)
            if st["live"] == 0:
                time.sleep(0.004)
                st["ka"] = _idle_ping(st["ka"])
                continue
            _step()

    # jaccl 콜렉티브는 메인 스레드에서 — HTTP 를 스레드로 (레시피 배치 미러)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            if self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": args.model_name, "object": "model", "owned_by": "local"}]})
            else: self._json(404, {"error": "nf"})
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": "nf"})
            t_h0 = time.perf_counter()
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                t_h1 = time.perf_counter()
                req = validate_request(payload, output_cap=args.max_output_tokens)
                ids = encode_prompt(tok, req)
                t_h2 = time.perf_counter()
            except Exception as e:
                return self._json(400, {"error": str(e)})
            print(f"[http] 요청 수신 · max_tokens={req.get('max_tokens')}"
                  + (f" · body={(t_h1 - t_h0) * 1e3:.1f}ms "
                     f"tokenize={(t_h2 - t_h1) * 1e3:.1f}ms n={len(ids)}"
                     if TRACE else ""), flush=True)
            uid = uuid.uuid4().hex
            stream = bool(payload.get("stream"))
            j = {"tokens": [], "done": False, "ev": threading.Event(),
                 "sq": queue.Queue() if stream else None}
            with lock: jobs[uid] = j
            t_enq = time.perf_counter()
            inbox.put((uid, ids, req["max_tokens"], t_enq))
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                _first = True
                while True:
                    t = j["sq"].get()
                    if t is None: break
                    self.wfile.write(b"data: " + json.dumps(
                        {"choices": [{"delta": {"content": tok.decode([t])}, "index": 0}]}
                    ).encode() + b"\n\n")
                    if _first:
                        _first = False
                        if TRACE:
                            print(f"[trace] http enq→SSE-write0 "
                                  f"{(time.perf_counter() - t_enq) * 1e3:.1f}ms",
                                  flush=True)
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                j["ev"].wait()
                text = tok.decode(j["tokens"])
                parsed = parse_assistant(text, thinking_mode="auto", hit_eos=True)
                self._json(200, {"id": "chatcmpl-" + uid[:8], "object": "chat.completion",
                    "model": args.model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant",
                                 "content": parsed.get("content", text)},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(ids),
                              "completion_tokens": len(j["tokens"]),
                              "total_tokens": len(ids) + len(j["tokens"])}})
            with lock: jobs.pop(uid, None)

    # ── 기동 워밍업 ────────────────────────────────────────────────────────
    # 첫 요청의 GenerationBatch.__init__ 이 265ms(2회차부터 50ms) — 프로세스-1회
    # 디코드/MTP 커널 JIT 비용이다(실측: rep0 264.8ms → rep1 49.8 → rep2 49.4).
    # 서빙 시작 전에 합성 시퀀스 1건을 정상 경로로 흘려 미리 태운다. 양 랭크가
    # 같은 제어 프레임을 밟으므로 락스텝 유지, 프롬프트는 SNAP_MIN_TOKENS 미만이라
    # 스냅숏 저장에도 잡히지 않는다.
    if os.environ.get("DSV4_WARMUP", "1").strip().lower() not in ("0", "", "false", "off"):
        _t0 = time.perf_counter()
        # PP2 최소치 미만으로 강제 클램프 — 워밍업이 PP2 경로에 빠지면 랭크0 은
        # 이음새 콜백이 아직 안 심긴 상태(gen_loop 진입 전)인데 랭크1 은 이미 심겨 있어
        # 이음새 호출 횟수가 0 vs N 으로 갈리고 양 박스 영구 교착이 된다(리뷰 지적).
        _wt = min(int(os.environ.get("DSV4_WARMUP_TOKENS", "200")), max(1, _PP2_MIN - 1))
        _wids = (list(tok.encode("warmup")) * 64)[:_wt]
        _wpend: dict = {}
        control.dispatch({"op": "insert", "items": [[_wids, 4]]})
        snap_insert(bg, [(_wids, 4)], _wpend)
        for _ in range(64):
            control.dispatch({"op": "step"})
            _prs, _rs = bg.next()
            if _prs:
                snap_capture(bg, _prs, _wpend)
            if _rs and any(getattr(r, "finish_reason", None) for r in _rs):
                break
        print(f"[warmup] {_wt} tok 합성 시퀀스 완주 "
              f"{(time.perf_counter() - _t0) * 1e3:.0f}ms", flush=True)

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[batched-tp2] :{args.port} 서빙 시작 (생성=메인 스레드)", flush=True)
    gen_loop()

if __name__ == "__main__":
    main()
