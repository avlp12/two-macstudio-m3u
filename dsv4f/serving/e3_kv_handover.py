"""E3 — PP2 프리필 → KV 인계(handover) → 1box 디코드 게이트.

배경: E2([I334])에서 PP2 프리필이 13.9K 프롬프트 992-1030 tok/s로 **비트정확** 검증됨
      (~/dsv4flash/align/logs/e2_pp2_s22_c2048_client.log, ref1box 23.0s → pp2 13.8s).
      E2는 최종 로짓까지만 — 프리필 후 KV 는 두 박스에 흩어진 채 버려졌다.

본 게이트가 채우는 구멍:
      server(remote box B) = 하단 슬라이스 layers[0,split) 의 KV
      client(local box A) = 상단 슬라이스 layers[split,43) 의 KV + 최종 로짓
  → server 의 하단 KV 를 client 로 전송하여 client 가 **전층 43개 캐시**를 조립,
    그대로 1box greedy 디코드를 이어받는다("PP2 프리필 + 1box 디코드" 하이브리드).

직렬화 대상(캐시 클래스별 정본 state/meta_state 계약 사용 — mlx_lm
save_prompt_cache 와 동일 경로):
  * RotatingKVCache : state=(keys, values), meta=(keep, max_size, offset, _idx)
      프리필은 전부 S>1 → _update_concat 경로라 keys.shape[2] = max_size-1+S_last,
      offset > keys.shape[2] 이므로 state 게터가 버퍼 전체를 반환(잘림 없음).
      복원 후 첫 디코드 스텝은 _update_in_place 로 진입해 1638행을 trim→128로 접고
      _idx=keep=0 에서 회전 — 1box 가 하는 것과 동일한 상태 전이.
  * PoolingCache   : state=(buf_kv, buf_gate, pooled, prev_win_kv, prev_win_gate),
      meta=ratio. None 슬롯은 매니페스트 플래그로 보존(safetensors 제약 회피 방식과 동일).
      state 세터가 remainder 를 accumulate_windows 로 재생하므로 살아있던 캐시와 무구별.
  * CacheList      : 멤버 재귀.

프로토콜: E2 의 원시 TCP 프레임(MAGIC/T_JSON/T_TENSOR) 재사용. 새 op 는 `fetch_cache`
  → 서버가 {op:cache_manifest, layers:[desc...], n_arrays:N} JSON 1장 + T_TENSOR N장
    (meta.i = 배열 인덱스) + {op:cache_done, 계측} 을 순차 전송. mx.distributed 미사용.

모드:
  --mode server : 하단 슬라이스 서버 (prefill + fetch_cache)
  --mode client : 전층 로드. ref1box(프리필+greedy) → PP2(프리필+인계+greedy) 를
                  **한 프로세스에서** 순차 실행하여 동일 조건 TTFT A/B + 토큰 일치 검증.
"""

import argparse
import json
import os
import signal
import socket
import struct
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "/Users/Shared/tp2")

os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
import mlx.core as mx  # noqa: E402

from e2_pp2_prefill import (  # noqa: E402
    _HDR,
    _NP_OF_TAG,
    MAGIC,
    T_TENSOR,
    PP2Client,
    V4Slice,
    WireError,
    _cache_arrays,
    _do_prefill,
    _npbytes,
    build_prompt_ids,
    load_dsv4,
    log,
    make_schedule,
    recv_msg,
    send_json,
    tune,
)


def to_mx_copy(meta, raw):
    """수신 프레임 → mx.array. E2 의 to_mx 와 달리 **명시적 copy** 로 소스 버퍼와 분리한다.

    인계 배열은 프레임 페이로드(bytearray)보다 오래 살아남아 캐시에 눌러앉으므로,
    zero-copy 참조가 남으면 페이로드 해제 후 조용한 손상이 된다.
    """
    arr = np.frombuffer(raw, dtype=_NP_OF_TAG[meta["d"]]).reshape(meta["s"]).copy()
    out = mx.array(arr)
    if meta["d"] == "bf16":
        out = out.view(mx.bfloat16)
    elif meta["d"] == "f16":
        out = out.view(mx.float16)
    return out


# ------------------------------------------------------- 캐시 직렬화 / 복원
def cache_spec(c, arrays, owners):
    """캐시 객체 → (JSON 기술자). mx.array 는 arrays 에 append, 소유 클래스는 owners 에."""
    tname = type(c).__name__
    if tname == "CacheList":
        return {"cls": "CacheList",
                "members": [cache_spec(m, arrays, owners) for m in c.caches]}
    if tname == "RotatingKVCache":
        slots = []
        if getattr(c, "keys", None) is None:
            slots = [None, None]
        else:
            for x in c.state:
                slots.append(len(arrays))
                arrays.append(x)
                owners.append("rot")
        return {"cls": "RotatingKVCache",
                "meta": [int(c.keep), int(c.max_size), int(c.offset), int(c._idx)],
                "slots": slots}
    if tname == "PoolingCache":
        slots = []
        for x in c.state:
            if x is None:
                slots.append(None)
            else:
                slots.append(len(arrays))
                arrays.append(x)
                owners.append("pool")
        return {"cls": "PoolingCache", "meta": int(c.ratio), "slots": slots}
    raise WireError(f"unsupported cache class {tname}")


def cache_restore(c, desc, arrays):
    tname = type(c).__name__
    if desc["cls"] != tname:
        raise WireError(f"cache class mismatch: wire {desc['cls']} vs local {tname}")
    if tname == "CacheList":
        if len(c.caches) != len(desc["members"]):
            raise WireError("CacheList arity mismatch")
        for m, d in zip(c.caches, desc["members"]):
            cache_restore(m, d, arrays)
        return
    if tname == "RotatingKVCache":
        s = desc["slots"]
        c.keys = arrays[s[0]] if s[0] is not None else None
        c.values = arrays[s[1]] if s[1] is not None else None
        c.keep, c.max_size, c.offset, c._idx = (int(x) for x in desc["meta"])
        return
    if tname == "PoolingCache":
        if int(desc["meta"]) != int(c.ratio):
            raise WireError(f"PoolingCache ratio {desc['meta']} != {c.ratio}")
        c.state = tuple(arrays[i] if i is not None else None for i in desc["slots"])
        return
    raise WireError(f"unsupported cache class {tname}")


def cache_offsets(cache):
    """층별 (RotatingKVCache) offset 목록 — 인계 후 정합성 확인용."""
    out = []
    for lc in cache:
        leaf = lc[0] if hasattr(lc, "caches") else lc
        out.append(getattr(leaf, "offset", None))
    return out


def send_prepped(sock, idx, npbuf, tag, shape):
    hdr = {"n": f"c{idx}", "d": tag, "s": list(shape), "i": idx}
    j = json.dumps(hdr).encode()
    sock.sendall(_HDR.pack(MAGIC, T_TENSOR, 4 + len(j) + npbuf.nbytes))
    sock.sendall(struct.pack("<I", len(j)))
    sock.sendall(j)
    sock.sendall(npbuf)
    return npbuf.nbytes


def _do_fetch_cache(sock, sl):
    t0 = time.perf_counter()
    arrays, owners = [], []
    descs = [cache_spec(c, arrays, owners) for c in sl.cache]
    mx.eval(*arrays)
    preps = []
    for a in arrays:
        n, tag = _npbytes(a)
        preps.append((n, tag, list(a.shape)))
    t_ser = time.perf_counter() - t0

    by_owner = {"rot": 0, "pool": 0}
    for (n, _t, _s), own in zip(preps, owners):
        by_owner[own] += int(n.nbytes)

    send_json(sock, {"op": "cache_manifest", "layers": descs,
                     "n_arrays": len(arrays), "lo": sl.lo, "hi": sl.hi,
                     "offset": sl.offset()})
    ts = time.perf_counter()
    total = 0
    for i, (n, tag, shape) in enumerate(preps):
        total += send_prepped(sock, i, n, tag, shape)
    t_send = time.perf_counter() - ts
    send_json(sock, {"op": "cache_done", "t_serialize": t_ser, "t_send": t_send,
                     "bytes_total": total, "bytes_rot": by_owner["rot"],
                     "bytes_pool": by_owner["pool"], "n_arrays": len(arrays)})
    log(f"fetch_cache: {len(arrays)} arrays {total/1e6:.1f}MB "
        f"(rot {by_owner['rot']/1e6:.1f} / pool {by_owner['pool']/1e6:.1f}) "
        f"ser {t_ser:.3f}s send {t_send:.3f}s")


# ---------------------------------------------------------------------- server
def _handle_conn(sock, sl, model_path):
    tune(sock)
    sock.settimeout(1800)
    while True:
        try:
            kind, *rest = recv_msg(sock)
        except WireError:
            log("client disconnected")
            return False
        if kind != "json":
            log("protocol error: unexpected tensor frame")
            return False
        msg = rest[0]
        op = msg.get("op")
        try:
            if op == "hello":
                send_json(sock, {"op": "hello_ack", "mlx": mx.__version__,
                                 "lo": sl.lo, "hi": sl.hi, "n_layers": sl.n_layers,
                                 "hidden": sl.hidden, "hc_mult": sl.hc_mult,
                                 "pid": os.getpid(), "model": model_path,
                                 "host": socket.gethostname()})
            elif op == "prefill":
                _do_prefill(sock, sl, msg)
            elif op == "fetch_cache":
                _do_fetch_cache(sock, sl)
            elif op == "reset":
                sl.reset()
                mx.clear_cache()
                send_json(sock, {"op": "reset_done"})
            elif op == "quit":
                log("client quit")
                return False
            elif op == "shutdown":
                send_json(sock, {"op": "bye"})
                return True
            else:
                send_json(sock, {"op": "error", "err": f"unknown op {op!r}"})
        except Exception:
            tb = traceback.format_exc()
            log("op failed:\n" + tb)
            try:
                send_json(sock, {"op": "error", "err": tb})
            except Exception:
                return False


def run_server(args):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    model, _tok = load_dsv4(args.model, 0, args.split)
    sl = V4Slice(model, 0, args.split)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    log(f"listening on {args.host}:{args.port} slice [0,{args.split}) [E3 kv-handover]")
    while True:
        conn, addr = srv.accept()
        log(f"client {addr}")
        try:
            shutdown = _handle_conn(conn, sl, args.model)
        finally:
            conn.close()
        sl.reset()
        mx.clear_cache()
        if shutdown:
            log("shutdown requested")
            return


# ---------------------------------------------------------------------- client
class KVClient(PP2Client):
    def fetch_cache(self):
        t0 = time.perf_counter()
        send_json(self.sock, {"op": "fetch_cache"})
        kind, *rest = recv_msg(self.sock)
        if kind != "json" or rest[0].get("op") != "cache_manifest":
            raise WireError(f"expected cache_manifest, got {kind} {rest[0] if rest else ''}")
        man = rest[0]
        n_arr = int(man["n_arrays"])
        slots = [None] * n_arr
        t_recv = 0.0
        nbytes = 0
        for _ in range(n_arr):
            tr = time.perf_counter()
            m = recv_msg(self.sock)
            t_recv += time.perf_counter() - tr
            if m[0] != "tensor":
                raise WireError(f"expected tensor, got {m[0]}")
            meta, raw = m[1], m[2]
            slots[int(meta["i"])] = (meta, raw)
            nbytes += len(raw)
        td = time.perf_counter()
        arrays = [to_mx_copy(meta, raw) for (meta, raw) in slots]
        mx.eval(*arrays)
        t_deser = time.perf_counter() - td
        kind, *rest = recv_msg(self.sock)
        if kind != "json" or rest[0].get("op") != "cache_done":
            raise WireError(f"expected cache_done, got {kind}")
        done = rest[0]
        stats = {"t_wire_wall": time.perf_counter() - t0, "t_recv": t_recv,
                 "t_deser": t_deser, "bytes": nbytes,
                 "server_t_serialize": done["t_serialize"],
                 "server_t_send": done["t_send"],
                 "bytes_rot": done["bytes_rot"], "bytes_pool": done["bytes_pool"],
                 "n_arrays": done["n_arrays"], "server_offset": man["offset"]}
        return man["layers"], arrays, stats

    def handover(self, nc_mode="none"):
        """fetch_cache + 전층 캐시 조립. 반환 (full_cache, stats)."""
        layers, arrays, st = self.fetch_cache()
        tr = time.perf_counter()
        full = self.model.make_cache()
        if len(layers) != self.split:
            raise WireError(f"manifest layers {len(layers)} != split {self.split}")
        if nc_mode == "zero-all":
            arrays = [mx.zeros_like(a) for a in arrays]
            mx.eval(*arrays)
        for i in range(self.split):
            if nc_mode == "skip-restore":
                continue
            cache_restore(full[i], layers[i], arrays)
        full = list(full[:self.split]) + list(self.top.cache)
        mx.eval(*_cache_arrays(full))
        st["t_restore"] = time.perf_counter() - tr
        st["t_total"] = st["t_wire_wall"] + st["t_restore"]
        return full, st


# ------------------------------------------------------------- prefill/decode
def ref_prefill(model, tokens, chunk, k=32):
    """스톡 1box 청크 프리필 — 캐시까지 반환(E2 run_ref1box 와 동일 그래프)."""
    cache = model.make_cache()
    sched = make_schedule(len(tokens), chunk)
    toks = mx.array(np.asarray(tokens, dtype=np.int32))[None]
    pos = 0
    t_chunks = []
    logits = None
    t0 = time.perf_counter()
    for ci, n in enumerate(sched):
        tc = time.perf_counter()
        hn = model.model(toks[:, pos:pos + n], cache=cache)
        if ci == len(sched) - 1:
            logits = model.lm_head(hn[:, -k:, :] if k else hn)
            mx.eval(logits, *_cache_arrays(cache))
        else:
            mx.eval(*_cache_arrays(cache))
        mx.clear_cache()
        t_chunks.append(time.perf_counter() - tc)
        pos += n
    return logits, cache, {"t_total": time.perf_counter() - t0,
                           "t_chunks": t_chunks, "schedule": sched}


def greedy_decode(model, cache, prefill_logits, n_new):
    """프리필 마지막 위치 로짓에서 시작하는 greedy N토큰. 반환 (tokens, t_first, t_steps)."""
    t0 = time.perf_counter()
    y = int(mx.argmax(prefill_logits[0, -1]).item())
    t_first = time.perf_counter() - t0
    toks = [y]
    steps = []
    for _ in range(n_new - 1):
        ts = time.perf_counter()
        x = mx.array(np.array([[y]], dtype=np.int32))
        hn = model.model(x, cache=cache)
        lg = model.lm_head(hn[:, -1:, :])
        y = int(mx.argmax(lg[0, -1]).item())
        mx.eval(*_cache_arrays(cache))
        steps.append(time.perf_counter() - ts)
        toks.append(y)
    return toks, t_first, steps


def emit(obj):
    print(f"[E3-RESULT] {json.dumps(obj, ensure_ascii=False)}", flush=True)


def r4(xs):
    return [round(float(x), 4) for x in xs]


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["server", "client"])
    ap.add_argument("--model", default=os.path.expanduser("~/dsv4flash/mlx4bit"))
    ap.add_argument("--split", type=int, default=22)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--decode-tokens", type=int, default=16)
    ap.add_argument("--warmup-tokens", type=int, default=2048)
    ap.add_argument("--warmup-decode", type=int, default=4)
    ap.add_argument("--host", default="10.0.0.2")
    ap.add_argument("--port", type=int, default=39932)
    ap.add_argument("--source", default="file", choices=["file", "repeat"])
    ap.add_argument("--file",
                    default=os.path.expanduser("~/dsv4flash/align/onpolicy_c.txt"))
    ap.add_argument("--char-start", type=int, default=0)
    ap.add_argument("--char-len", type=int, default=48508)
    ap.add_argument("--n-repeat", type=int, default=397)
    ap.add_argument("--dump-dir", default=os.path.expanduser("~/dsv4flash/e3_artifacts"))
    ap.add_argument("--tag", default="e3kv")
    ap.add_argument("--dump-k", type=int, default=32)
    ap.add_argument("--nc-zero-layer", type=int, default=-1,
                    help="음성 대조(기본 off): 인계·검증 통과 후 해당 층의 Rotating keys 를 "
                         "0으로 덮어써 16토큰 일치 게이트가 실제로 인계 내용에 민감한지 확인")
    ap.add_argument("--nc-mode", default="none",
                    choices=["none", "skip-restore", "zero-all"],
                    help="음성 대조(기본 none). skip-restore=하단층을 인계분 대신 빈 캐시로 "
                         "조립(offset 단언 우회) · zero-all=하단층 전 배열을 0으로 덮어씀. "
                         "둘 다 16토큰이 갈라져야 게이트가 인계 내용에 민감함이 증명된다.")
    ap.add_argument("--skip-ref", action="store_true")
    ap.add_argument("--skip-pp2", action="store_true")
    ap.add_argument("--shutdown-server", action="store_true")
    args = ap.parse_args()

    if args.mode == "server":
        run_server(args)
        return

    os.makedirs(args.dump_dir, exist_ok=True)
    k = args.dump_k
    N = args.decode_tokens

    # 클라이언트는 **전층** 물질화 — 인계 후 1box 디코드를 이어받아야 하므로.
    model, tok = load_dsv4(args.model, 0, None)
    tokens, recipe = build_prompt_ids(tok, args)
    log(f"prompt: n={len(tokens)} recipe={json.dumps(recipe, ensure_ascii=False)}")
    n_prompt = len(tokens)

    ref_tokens = None
    ref_logits_np = None
    ref_rows = []

    # ---------------- A) 1box 스톡 기준선: 프리필 + greedy N
    if not args.skip_ref:
        lg, ca, _ = ref_prefill(model, tokens[:args.warmup_tokens], args.chunk, k)
        greedy_decode(model, ca, lg, args.warmup_decode)
        del ca
        mx.clear_cache()
        log("ref warmup done")
        for rep in range(args.reps):
            mx.reset_peak_memory()
            logits, cache, st = ref_prefill(model, tokens, args.chunk, k)
            toks_out, t_first, steps = greedy_decode(model, cache, logits, N)
            ttft = st["t_total"] + t_first
            row = {"tag": args.tag, "mode": "ref1box", "rep": rep,
                   "n_prompt": n_prompt, "chunk": args.chunk,
                   "t_prefill_s": round(st["t_total"], 4),
                   "prefill_tok_s": round(n_prompt / st["t_total"], 2),
                   "t_first_argmax_s": round(t_first, 4),
                   "ttft_s": round(ttft, 4),
                   "t_decode_steps": r4(steps),
                   "decode_tok_s": round((N - 1) / sum(steps), 2) if steps else None,
                   "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 3),
                   "tokens": toks_out,
                   "cache_offsets_head": cache_offsets(cache)[:3]}
            emit(row)
            ref_rows.append(row)
            if rep == 0:
                ref_tokens = list(toks_out)
                ref_logits_np = np.array(logits.astype(mx.float32))
                np.save(os.path.join(args.dump_dir, f"{args.tag}_ref_logits.npy"),
                        ref_logits_np)
                with open(os.path.join(args.dump_dir, f"{args.tag}_ref_tokens.json"),
                          "w") as f:
                    json.dump({"tokens": toks_out,
                               "text": tok.decode(toks_out)}, f, ensure_ascii=False)
                log(f"ref tokens: {toks_out}")
                log("ref text: " + json.dumps(tok.decode(toks_out), ensure_ascii=False))
            del cache
            mx.clear_cache()

    if args.skip_pp2:
        print("[e3-pass]", flush=True)
        return

    # ---------------- B) PP2 프리필 + KV 인계 + 1box 디코드
    cli = KVClient(model, args.host, args.port, args.split)
    ok = True
    try:
        # 워밍업(짧은 프롬프트로 인계 경로까지 전부 한 번 통과)
        wlg, _ = cli.prefill(tokens[:args.warmup_tokens], args.chunk, k)
        wfull, wst = cli.handover()
        greedy_decode(model, wfull, wlg, args.warmup_decode)
        del wfull
        mx.clear_cache()
        log(f"pp2 warmup done (handover {wst['t_total']:.3f}s "
            f"{wst['bytes']/1e6:.1f}MB)")

        for rep in range(args.reps):
            mx.reset_peak_memory()
            logits, pst = cli.prefill(tokens, args.chunk, k)
            hfull, hst = cli.handover(args.nc_mode)
            offs = cache_offsets(hfull)
            bad = [i for i, o in enumerate(offs) if o != n_prompt]
            if bad and args.nc_mode == "none":
                raise RuntimeError(f"조립 캐시 offset 불일치 층 {bad[:5]} (기대 {n_prompt})")
            if args.nc_mode != "none":
                log(f"[NC] mode={args.nc_mode} offset-불일치 층 {len(bad)}개 (단언 우회)")
            if args.nc_zero_layer >= 0:
                lc = hfull[args.nc_zero_layer]
                leaf = lc[0] if hasattr(lc, "caches") else lc
                leaf.keys = mx.zeros_like(leaf.keys)
                mx.eval(leaf.keys)
                log(f"[NC] layer {args.nc_zero_layer} rotating keys → 0 (음성 대조)")
            toks_out, t_first, steps = greedy_decode(model, hfull, logits, N)
            ttft = pst["t_total"] + hst["t_total"] + t_first
            row = {"tag": args.tag, "mode": "pp2_handover", "rep": rep,
                   "n_prompt": n_prompt, "chunk": args.chunk, "split": args.split,
                   "t_prefill_s": round(pst["t_total"], 4),
                   "prefill_tok_s": round(n_prompt / pst["t_total"], 2),
                   "prefill_wire_mb": round(pst["wire_bytes"] / 1e6, 1),
                   "handover_mb": round(hst["bytes"] / 1e6, 2),
                   "handover_mb_rot": round(hst["bytes_rot"] / 1e6, 2),
                   "handover_mb_pool": round(hst["bytes_pool"] / 1e6, 2),
                   "handover_arrays": hst["n_arrays"],
                   "t_handover_s": round(hst["t_total"], 4),
                   "hv_server_serialize_s": round(hst["server_t_serialize"], 4),
                   "hv_server_send_s": round(hst["server_t_send"], 4),
                   "hv_client_recv_s": round(hst["t_recv"], 4),
                   "hv_client_deser_s": round(hst["t_deser"], 4),
                   "hv_client_restore_s": round(hst["t_restore"], 4),
                   "hv_mb_per_s": round(hst["bytes"] / 1e6 / hst["t_total"], 1),
                   "t_first_argmax_s": round(t_first, 4),
                   "ttft_s": round(ttft, 4),
                   "ttft_overlap_s": round(pst["t_total"] + t_first, 4),
                   "t_decode_steps": r4(steps),
                   "decode_tok_s": round((N - 1) / sum(steps), 2) if steps else None,
                   "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 3),
                   "tokens": toks_out}
            if ref_rows:
                rt = ref_rows[0]["ttft_s"]
                row["ttft_speedup_vs_1box"] = round(rt / ttft, 3)
                row["ttft_speedup_overlap"] = round(rt / row["ttft_overlap_s"], 3)
            emit(row)

            lg_np = np.array(logits.astype(mx.float32))
            if rep == 0:
                np.save(os.path.join(args.dump_dir, f"{args.tag}_pp2_logits.npy"), lg_np)
                log(f"pp2 tokens: {toks_out}")
                log("pp2 text: " + json.dumps(tok.decode(toks_out), ensure_ascii=False))
            if ref_tokens is not None:
                same = list(toks_out) == list(ref_tokens)
                first_bad = next((i for i, (a, b) in
                                  enumerate(zip(toks_out, ref_tokens)) if a != b), None)
                cons = {"label": f"{args.tag}_rep{rep}_tokens",
                        "n_tokens": N, "tokens_match": bool(same),
                        "first_mismatch_idx": first_bad,
                        "ref_tokens": ref_tokens, "pp2_tokens": list(toks_out)}
                if ref_logits_np is not None:
                    cons["prefill_logits_bit_exact"] = bool(
                        ref_logits_np.shape == lg_np.shape
                        and np.array_equal(ref_logits_np, lg_np))
                    cons["prefill_max_abs_diff"] = float(
                        np.abs(ref_logits_np - lg_np).max())
                print(f"[E3-CONSISTENCY] {json.dumps(cons, ensure_ascii=False)}",
                      flush=True)
                ok = ok and same
            del hfull
            mx.clear_cache()
    finally:
        if args.shutdown_server:
            cli.shutdown_server()
        else:
            cli.close()
    print("[e3-pass]" if ok else "[e3-FAIL]", flush=True)


if __name__ == "__main__":
    main()
