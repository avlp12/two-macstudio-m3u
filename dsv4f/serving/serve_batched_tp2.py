"""TP2 연속-배칭 OpenAI 서버: 락스텝 BatchGenerator (rank0=HTTP, 워커=제어소켓 미러).
사이클 규약: rank0 이 삽입 목록을 프레임 방송 → 전 랭크 동일 insert → 동일 next_generated.
greedy+동일 시드로 전 랭크 토큰 동일 → 상태 영구 동기."""
import argparse, json, os, queue, threading, time, uuid, sys
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

    # ── 프리픽스 스냅숏 (양 랭크 대칭: 동일 연산 순서 → 동일 LRU 상태 → 락스텝 유지) ──
    # 캡처 시점 = 프롬프트→생성 전환(end_of_prompt) 순간: 캐시가 정확히 ids[:-1]까지만
    # 처리된 상태라, 멀티턴 재렌더에서 바뀌는 꼬리 토큰(<think> 등)이 자연 배제됨.
    SNAP_MIN_TOKENS = 256           # 이보다 짧은 프롬프트는 스냅숏 비대상
    snapstore = LRUPromptCache(max_size=64, max_bytes=32 << 30)

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
                full = _pp2.build_cache(ids)
                new = bg.insert_segments([[ids[-1:]]], max_tokens=[n],
                                         caches=[full], all_tokens=[key])
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
        while True:
            cmd = recv_frame(sock)
            if cmd.get("op") == "insert":
                snap_insert(bg, cmd["items"], pending)
            elif cmd.get("op") == "step":
                prs, rs = bg.next()
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
        live = 0  # bg 내 활성 시퀀스 수 (finish 시 감소)
        while True:
            items, metas = [], []
            while live + len(items) < args.max_batch:
                try:
                    uid, ids, n = inbox.get_nowait()
                except queue.Empty:
                    break
                items.append((ids, n)); metas.append(uid)
            if items:
                print(f"[gen] 삽입 {len(items)}건", flush=True)
                control.dispatch({"op": "insert", "items": items})
                new_uids = snap_insert(bg, items, pending)
                for u, uid in zip(new_uids, metas):
                    uid_by_slot[u] = uid
                live += len(items)
            if live == 0:
                time.sleep(0.004)
                continue
            control.dispatch({"op": "step"})
            prs, rs = bg.next()
            if prs:
                snap_capture(bg, prs, pending)
            if not rs:
                continue
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
                        live -= 1
                        if j["sq"] is not None: j["sq"].put(None)
                        j["ev"].set()

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
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                req = validate_request(payload, output_cap=args.max_output_tokens)
                ids = encode_prompt(tok, req)
            except Exception as e:
                return self._json(400, {"error": str(e)})
            print(f"[http] 요청 수신 · max_tokens={req.get('max_tokens')}", flush=True)
            uid = uuid.uuid4().hex
            stream = bool(payload.get("stream"))
            j = {"tokens": [], "done": False, "ev": threading.Event(),
                 "sq": queue.Queue() if stream else None}
            with lock: jobs[uid] = j
            inbox.put((uid, ids, req["max_tokens"]))
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                while True:
                    t = j["sq"].get()
                    if t is None: break
                    self.wfile.write(b"data: " + json.dumps(
                        {"choices": [{"delta": {"content": tok.decode([t])}, "index": 0}]}
                    ).encode() + b"\n\n")
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

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[batched-tp2] :{args.port} 서빙 시작 (생성=메인 스레드)", flush=True)
    gen_loop()

if __name__ == "__main__":
    main()
