"""E2 — DSv4-Flash 2박스 파이프라인 병렬(PP2) 프리필 프로토타입.

설계 근거: ~/dsv4flash/PREFILL_CEILING_INVESTIGATION_2026-08-25.md §E2 / [RA7]
참조 구현: ~/qwen38_alis_mlx/code/prefill_2box/{wire,runner,server,orchestrator}.py
           (qwen38 2박스 층-파이프라인, 8K 1.717× / 32K 1.882× 비트-동일 실측 [I23])

구조(참조구현 그대로):
  - 하단 슬라이스(embed + layers[0,split))  = **server** (remote box, box B)
  - 상단 슬라이스(layers[split,43) + norm+hc_head+lm_head) = **client/orchestrator** (로컬)
  - 원시 TCP 프레임 소켓. mx.distributed 미사용 → jaccl/RDMA 완전 배제(가드 R5 초과 충족),
    MLX 이슈 #3149(형상-변동 send/recv 손상) 직격 회피([RA7]).
  - 청크 마이크로배칭: server 가 청크 i 를 계산하는 동안 client 는 청크 i-1 을 계산.

DSv4 각색 포인트(qwen38 대비 차이):
  1. 경계 활성이 **4D 하이퍼커넥션 hidden [B,S,hc_mult,H]** (qwen38 은 3D [B,S,H]).
     embed 직후 broadcast_to(hc_mult) + contiguous 가 하단 슬라이스에만 존재.
  2. 마스크는 슬라이스 첫 캐시(CacheList 면 [0]=RotatingKVCache)에서 생성.
     양 슬라이스의 rotating 캐시 offset 이 동일하게 진행하므로 마스크가 동일.
  3. 각 층이 `input_ids` 를 받는다(층 0-2 해시 라우팅). client 도 토큰 슬라이스를 직접 보유.
  4. 캐시는 층별 이종(RotatingKVCache / CacheList(Rotating,Pooling[,Pooling])) →
     model.make_cache() 결과를 슬라이스로 잘라 쓴다(구성 동일성 보장).
  5. 본 게이트는 **프리필 최종 로짓까지만** — KV 인계(fetch_cache)는 범위 밖(미구현).

모드:
  --mode ref1box : 스톡 Model.__call__ 청크 프리필(=BatchGenerator 프리필 루프 등가) 기준선
  --mode slice1box: 본 하네스의 슬라이스 코드로 두 반쪽을 **로컬 직렬** 실행(코드 충실도 검증)
  --mode server  : 하단 슬라이스 TCP 서버
  --mode client  : 상단 슬라이스 + 파이프라인 오케스트레이터

로짓 산출물은 --dump 로 float32 npy 저장, --ref 로 비트 비교.
"""

import argparse
import json
import os
import queue
import signal
import socket
import struct
import sys
import threading
import time
import traceback

import numpy as np

os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
import mlx.core as mx  # noqa: E402

# ----------------------------------------------------------------- wire (TCP)
MAGIC = 0x2B0C5EED
_HDR = struct.Struct("<IBQ")
T_JSON = 1
T_TENSOR = 2


class WireError(RuntimeError):
    pass


def tune(sock):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, 16 << 20)
        except OSError:
            pass


def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:n], n - got)
        if r == 0:
            raise WireError("peer closed")
        got += r
    return buf


def send_json(sock, obj):
    b = json.dumps(obj).encode()
    sock.sendall(_HDR.pack(MAGIC, T_JSON, len(b)))
    sock.sendall(b)


_TAG_OF_MX = {
    mx.bfloat16: "bf16",
    mx.float16: "f16",
    mx.float32: "f32",
    mx.uint16: "u16",
    mx.int32: "i32",
}
_NP_OF_TAG = {"bf16": np.uint16, "f16": np.uint16, "f32": np.float32,
              "u16": np.uint16, "i32": np.int32}


def _npbytes(a):
    tag = _TAG_OF_MX.get(a.dtype)
    if tag is None:
        raise WireError(f"unsupported dtype {a.dtype}")
    if a.dtype in (mx.bfloat16, mx.float16):
        a = a.view(mx.uint16)
    try:
        n = np.array(a, copy=False)
    except Exception:
        n = np.array(a)
    if not n.flags["C_CONTIGUOUS"]:
        n = np.ascontiguousarray(n)
    return n, tag


def send_tensor(sock, name, a, **meta):
    n, tag = _npbytes(a)
    hdr = {"n": name, "d": tag, "s": list(a.shape)}
    hdr.update(meta)
    j = json.dumps(hdr).encode()
    sock.sendall(_HDR.pack(MAGIC, T_TENSOR, 4 + len(j) + n.nbytes))
    sock.sendall(struct.pack("<I", len(j)))
    sock.sendall(j)
    sock.sendall(n)
    return n.nbytes


def recv_msg(sock):
    hdr = recv_exact(sock, _HDR.size)
    magic, ftype, plen = _HDR.unpack(bytes(hdr))
    if magic != MAGIC:
        raise WireError(f"bad magic {magic:#x}")
    payload = recv_exact(sock, plen)
    if ftype == T_JSON:
        return ("json", json.loads(bytes(payload).decode()))
    if ftype == T_TENSOR:
        (jlen,) = struct.unpack_from("<I", payload, 0)
        meta = json.loads(bytes(payload[4:4 + jlen]).decode())
        return ("tensor", meta, memoryview(payload)[4 + jlen:])
    raise WireError(f"bad frame type {ftype}")


def to_mx(meta, raw):
    arr = np.frombuffer(raw, dtype=_NP_OF_TAG[meta["d"]]).reshape(meta["s"])
    out = mx.array(arr)
    if meta["d"] == "bf16":
        out = out.view(mx.bfloat16)
    elif meta["d"] == "f16":
        out = out.view(mx.float16)
    return out


# ------------------------------------------------------------------ model bits
def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def set_wired_limit():
    info = mx.device_info()
    lim = info["max_recommended_working_set_size"]
    mx.set_wired_limit(lim)
    return lim


def load_dsv4(path, lo=None, hi=None, group=None):
    """omlx 패치 적용 후 lazy 로드. lo/hi 가 주어지면 해당 슬라이스 가중치만 물질화.

    group 이 주어지면 물질화 **전에** model.shard(group) (=현행 TP2 샤딩)를 적용한다.
    """
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    from mlx_lm import load

    t0 = time.monotonic()
    model, tok = load(path, lazy=True)
    if group is not None and group.size() > 1:
        assert hasattr(model, "shard"), "omlx base patch 미적용"
        model.shard(group)
        try:
            sys.path.insert(0, "/Users/Shared/tp2")
            from dspark_tp4_common import shard_mtp

            shard_mtp(model, group)
        except Exception as e:
            log(f"mtp 샤딩 생략: {e}")
    lim = set_wired_limit()
    core = model.model
    n = len(core.layers)
    lo = 0 if lo is None else lo
    hi = n if hi is None else hi
    if lo == 0:
        mx.eval(core.embed_tokens.parameters())
        mx.synchronize()
    for layer in core.layers[lo:hi]:
        mx.eval(layer.parameters())
        mx.synchronize()
    if hi == n:
        mx.eval(core.norm.parameters())
        mx.eval(core.hc_head.parameters())
        mx.eval(model.lm_head.parameters())
        mx.synchronize()
    log(f"loaded [{lo},{hi}) of {n} in {time.monotonic()-t0:.1f}s "
        f"wired_limit={lim/2**30:.0f}GiB mlx={mx.__version__}")
    return model, tok


class V4Slice:
    """DSv4 디코더 층 [lo, hi) 슬라이스. DeepseekV4Model.__call__ 을 그대로 재현."""

    def __init__(self, model, lo, hi):
        core = model.model
        n = len(core.layers)
        if not (0 <= lo < hi <= n):
            raise ValueError(f"bad slice [{lo},{hi}) of {n}")
        self.model = model
        self.core = core
        self.lo, self.hi, self.n_layers = lo, hi, n
        self.args = core.args
        self.hidden = core.args.hidden_size
        self.hc_mult = core.args.hc_mult
        self.embed = core.embed_tokens if lo == 0 else None
        self.layers = core.layers[lo:hi]
        self.owns_tail = hi == n
        self.reset()

    def reset(self):
        # model.make_cache() 로 만든 전체 리스트를 슬라이스 — 층별 캐시 구성 동일성 보장
        self.cache = self.model.make_cache()[self.lo:self.hi]

    def cache_arrays(self):
        out = []
        for lc in self.cache:
            leaves = getattr(lc, "caches", None) or (lc,)
            for leaf in leaves:
                if leaf is None:
                    continue
                for v in vars(leaf).values():
                    if isinstance(v, mx.array):
                        out.append(v)
        return out

    def offset(self):
        lc = self.cache[0]
        leaf = lc[0] if hasattr(lc, "caches") else lc
        return getattr(leaf, "offset", None)

    def forward(self, x, input_ids, is_tokens=False):
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.cache import CacheList

        if is_tokens:
            h = self.embed(x)
            h = mx.broadcast_to(
                h[:, :, None, :], (h.shape[0], h.shape[1], self.hc_mult, h.shape[2])
            )
            h = mx.contiguous(h)
        else:
            h = x
        first = self.cache[0]
        mask_cache = first[0] if isinstance(first, CacheList) else first
        mask = create_attention_mask(
            h[:, :, 0, :], mask_cache,
            window_size=self.args.sliding_window, return_array=True,
        )
        for layer, lc in zip(self.layers, self.cache):
            h = layer(h, mask, lc, input_ids, _standard_mask=True)
        return h

    def tail_logits(self, h, k=32):
        """norm(hc_head(h)) 후 **마지막 k 위치에만** lm_head.

        스톡 `DeepseekV4Model.__call__` 이 반환하는 값이 norm(hc_head(h)) 이므로,
        슬라이싱 지점을 정확히 그 뒤에 두어야 ref1box 와 그래프가 동일해진다.
        """
        if not self.owns_tail:
            raise RuntimeError("slice does not own the tail")
        hn = self.core.norm(self.core.hc_head(h))
        if k:
            hn = hn[:, -k:, :]
        return self.model.lm_head(hn)


def make_schedule(n, chunk):
    out = []
    while n > 0:
        out.append(min(chunk, n))
        n -= out[-1]
    return out


# ------------------------------------------------------------------- prompting
FILLER = "분산 텐서 병렬과 압축 어텐션의 상호작용은 위치 기하와 수용률 경제학에 민감하다. "


def build_prompt_ids(tok, args):
    """E1/PA2 레시피 재현: file-content 프롬프트 (기본 onpolicy_c.txt 접두 48508자)."""
    tail = "17 * 23 = (topic MVCC)"
    if args.source == "file":
        with open(args.file, "r", encoding="utf-8") as f:
            full = f.read()
        body = full[args.char_start:args.char_start + args.char_len]
        recipe = {"file": args.file, "char_start": args.char_start,
                  "char_len": args.char_len, "body_tokens": len(tok.encode(body)),
                  "search": "fixed"}
        content = "참고 문서: " + body + " " + tail
    else:
        content = ("참고 문서: " + FILLER * args.n_repeat + " " + tail)
        recipe = {"source": "repeat", "n_repeat": args.n_repeat}
    ids = tok.apply_chat_template([{"role": "user", "content": content}],
                                  add_generation_prompt=True)
    return [int(t) for t in ids], recipe


# ------------------------------------------------------------------ ref (1box)
def run_ref1box(model, tokens, chunk, k=32):
    """스톡 청크 프리필 (BatchGenerator 프리필 루프 등가, [I10]).

    `model.model(...)` = norm(hc_head(h)) 까지. lm_head 는 마지막 청크의 마지막 k
    위치에만 적용 — BatchGenerator 가 프리필 로짓을 참조하지 않아 지연평가로
    소거되는 동작([RA1a])과 같은 비용 구조를 유지한다.
    """
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
    t_total = time.perf_counter() - t0
    return logits, {"t_total": t_total, "t_chunks": t_chunks, "schedule": sched}


def run_slice1box(model, tokens, chunk, split, k=32):
    """두 슬라이스를 **로컬 직렬** 실행 — 슬라이스 코드가 스톡과 동치인지 검증용."""
    n_layers = len(model.model.layers)
    bot = V4Slice(model, 0, split)
    top = V4Slice(model, split, n_layers)
    sched = make_schedule(len(tokens), chunk)
    toks = mx.array(np.asarray(tokens, dtype=np.int32))[None]
    pos = 0
    t_bot, t_top = [], []
    logits = None
    t0 = time.perf_counter()
    for ci, n in enumerate(sched):
        ids = toks[:, pos:pos + n]
        tb = time.perf_counter()
        h = bot.forward(ids, ids, is_tokens=True)
        mx.eval(h, *bot.cache_arrays())
        t_bot.append(time.perf_counter() - tb)
        tt = time.perf_counter()
        h2 = top.forward(h, ids, is_tokens=False)
        if ci == len(sched) - 1:
            logits = top.tail_logits(h2, k=k)
            mx.eval(logits, *top.cache_arrays())
        else:
            mx.eval(h2, *top.cache_arrays())
        t_top.append(time.perf_counter() - tt)
        mx.clear_cache()
        pos += n
    t_total = time.perf_counter() - t0
    return logits, {"t_total": t_total, "t_bot": t_bot, "t_top": t_top,
                    "schedule": sched}


def _cache_arrays(cache):
    out = []
    for lc in cache:
        if lc is None:
            continue
        leaves = getattr(lc, "caches", None) or (lc,)
        for leaf in leaves:
            if leaf is None:
                continue
            for v in vars(leaf).values():
                if isinstance(v, mx.array):
                    out.append(v)
    return out


# ---------------------------------------------------------------------- server
def _do_prefill(sock, sl, msg):
    tokens = [int(t) for t in msg["tokens"]]
    sched = msg["schedule"]
    if sum(sched) != len(tokens):
        raise ValueError(f"schedule {sum(sched)} != tokens {len(tokens)}")
    sl.reset()
    mx.clear_cache()

    sendq = queue.Queue(maxsize=4)
    send_err = []

    def sender():
        try:
            while True:
                item = sendq.get()
                if item is None:
                    return
                name, arr, meta = item
                send_tensor(sock, name, arr, **meta)
        except Exception as e:
            send_err.append(e)

    st = threading.Thread(target=sender, daemon=True)
    st.start()

    toks = mx.array(np.asarray(tokens, dtype=np.int32))[None]
    pos = 0
    t_chunks, t_idle = [], []
    t0 = time.perf_counter()
    try:
        for ci, n in enumerate(sched):
            tc = time.perf_counter()
            ids = toks[:, pos:pos + n]
            h = sl.forward(ids, ids, is_tokens=True)
            # sender 스레드로 넘기기 전에 반드시 완전 평가 (스레드-로컬 스트림 함정)
            mx.eval(h, *sl.cache_arrays())
            mx.clear_cache()
            t_chunks.append(time.perf_counter() - tc)
            if send_err:
                raise send_err[0]
            tq = time.perf_counter()
            sendq.put(("act", h, {"c": ci}))   # 큐가 차 있으면 여기서 블록 = 배압
            t_idle.append(time.perf_counter() - tq)
            pos += n
    finally:
        sendq.put(None)
        st.join()
    if send_err:
        raise send_err[0]
    send_json(sock, {
        "op": "prefill_done",
        "t_compute": time.perf_counter() - t0,
        "t_chunks": t_chunks,
        "t_backpressure": t_idle,
        "offset": sl.offset(),
    })
    log(f"prefill: {len(tokens)} tok, sched {sched}, "
        f"compute {time.perf_counter()-t0:.3f}s, offset {sl.offset()}")


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
    log(f"listening on {args.host}:{args.port} slice [0,{args.split})")
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
class PP2Client:
    def __init__(self, model, host, port, split, connect_timeout=30):
        self.model = model
        self.n_layers = len(model.model.layers)
        self.split = split
        self.top = V4Slice(model, split, self.n_layers)
        self.hidden = self.top.hidden
        self.hc_mult = self.top.hc_mult
        self.sock = socket.create_connection((host, port), timeout=connect_timeout)
        tune(self.sock)
        self.sock.settimeout(1800)
        send_json(self.sock, {"op": "hello"})
        kind, ack = recv_msg(self.sock)
        if kind != "json" or ack.get("op") != "hello_ack":
            raise RuntimeError(f"bad hello ack: {ack}")
        if ack["lo"] != 0 or ack["hi"] != split or ack["n_layers"] != self.n_layers:
            raise RuntimeError(
                f"server slice [{ack['lo']},{ack['hi']}) of {ack['n_layers']} "
                f"!= expected [0,{split}) of {self.n_layers}")
        if ack["mlx"] != mx.__version__:
            raise RuntimeError(f"mlx mismatch {mx.__version__} vs {ack['mlx']}")
        self.remote = ack
        log(f"server ok: {ack['host']} pid={ack['pid']} mlx={ack['mlx']} "
            f"slice [0,{ack['hi']}) hidden={ack['hidden']} hc={ack['hc_mult']}")

    def close(self):
        try:
            send_json(self.sock, {"op": "quit"})
        except Exception:
            pass
        self.sock.close()

    def shutdown_server(self):
        try:
            send_json(self.sock, {"op": "shutdown"})
            while True:
                kind, *rest = recv_msg(self.sock)
                if kind == "json" and rest[0].get("op") == "bye":
                    break
        except Exception:
            pass
        self.sock.close()

    def prefill(self, tokens, chunk, k=32):
        tokens = [int(t) for t in tokens]
        n_pre = len(tokens)
        sched = make_schedule(n_pre, chunk)
        self.top.reset()
        mx.clear_cache()

        acts_q = queue.Queue(maxsize=4)
        ctrl, rx_err = {}, []
        wire_bytes = [0]

        def rx():
            try:
                while True:
                    m = recv_msg(self.sock)
                    if m[0] == "json":
                        msg = m[1]
                        op = msg.get("op")
                        if op == "error":
                            raise RuntimeError(f"server error: {msg['err']}")
                        ctrl[op] = msg
                        if op == "prefill_done":
                            return
                    else:
                        meta, raw = m[1], m[2]
                        wire_bytes[0] += len(raw)
                        acts_q.put((meta, raw, time.perf_counter()))
            except Exception as e:
                rx_err.append(e)
                acts_q.put(None)

        rx_thread = threading.Thread(target=rx, daemon=True)
        toks = mx.array(np.asarray(tokens, dtype=np.int32))[None]

        t0 = time.perf_counter()
        send_json(self.sock, {"op": "prefill", "tokens": tokens, "schedule": sched})
        rx_thread.start()

        waits, t_local, t_deser = [], [], []
        logits = None
        pos = 0
        for ci, n in enumerate(sched):
            tw = time.perf_counter()
            item = acts_q.get()
            if item is None:
                raise (rx_err[0] if rx_err else RuntimeError("rx died"))
            waits.append(time.perf_counter() - tw)
            meta, raw, _ = item
            if meta.get("c") != ci or list(meta["s"]) != [1, n, self.hc_mult, self.hidden]:
                raise RuntimeError(f"act mismatch chunk {ci} n {n}: {meta}")
            td = time.perf_counter()
            h = to_mx(meta, raw)
            mx.eval(h)
            t_deser.append(time.perf_counter() - td)
            tc = time.perf_counter()
            ids = toks[:, pos:pos + n]
            h2 = self.top.forward(h, ids, is_tokens=False)
            if ci == len(sched) - 1:
                logits = self.top.tail_logits(h2, k=k)
                mx.eval(logits, *self.top.cache_arrays())
            else:
                mx.eval(h2, *self.top.cache_arrays())
            mx.clear_cache()
            t_local.append(time.perf_counter() - tc)
            pos += n
        t_pipeline = time.perf_counter() - t0
        rx_thread.join(timeout=60)
        if rx_err:
            raise rx_err[0]
        srv = ctrl.get("prefill_done", {})
        if srv.get("offset") != n_pre:
            raise RuntimeError(f"server offset {srv.get('offset')} != {n_pre}")
        if self.top.offset() != n_pre:
            raise RuntimeError(f"local offset {self.top.offset()} != {n_pre}")
        stats = {
            "n_tokens": n_pre, "schedule": sched, "t_total": t_pipeline,
            "t_act_waits": waits, "t_local_chunks": t_local, "t_deser": t_deser,
            "server_t_chunks": srv.get("t_chunks"),
            "server_t_compute": srv.get("t_compute"),
            "server_t_backpressure": srv.get("t_backpressure"),
            "wire_bytes": wire_bytes[0],
        }
        return logits, stats


# ------------------------------------------------------------------------ main
def logits_to_np(logits):
    return np.array(logits.astype(mx.float32))


def compare(dump, ref, label):
    a = np.load(dump)
    b = np.load(ref)
    same_shape = a.shape == b.shape
    exact = same_shape and np.array_equal(a, b)
    am_a = int(a.reshape(-1, a.shape[-1])[-1].argmax())
    am_b = int(b.reshape(-1, b.shape[-1])[-1].argmax())
    d = np.abs(a - b) if same_shape else None
    out = {
        "label": label, "shape_a": list(a.shape), "shape_b": list(b.shape),
        "bit_exact": bool(exact), "argmax_a": am_a, "argmax_b": am_b,
        "argmax_match": am_a == am_b,
        "max_abs_diff": float(d.max()) if d is not None else None,
        "n_mismatch": int((a != b).sum()) if same_shape else None,
        "n_elem": int(a.size),
    }
    print(f"[E2-CONSISTENCY] {json.dumps(out, ensure_ascii=False)}", flush=True)
    return out


def summarize(tag, mode, stats, n_prompt, extra=None):
    t = stats["t_total"]
    out = {"tag": tag, "mode": mode, "n_prompt": n_prompt,
           "t_prefill_s": round(t, 4), "prefill_tok_s": round(n_prompt / t, 2),
           "schedule": stats.get("schedule"),
           "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 3)}
    if extra:
        out.update(extra)
    print(f"[E2-RESULT] {json.dumps(out, ensure_ascii=False)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["ref1box", "slice1box", "local", "tp2ref", "server",
                             "client", "compare"])
    ap.add_argument("--model", default=os.path.expanduser("~/dsv4flash/mlx4bit"))
    ap.add_argument("--split", type=int, default=22)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--warmup-tokens", type=int, default=2048)
    ap.add_argument("--host", default="10.0.0.2")
    ap.add_argument("--port", type=int, default=39931)
    ap.add_argument("--source", default="file", choices=["file", "repeat"])
    ap.add_argument("--file",
                    default=os.path.expanduser("~/dsv4flash/align/onpolicy_c.txt"))
    ap.add_argument("--char-start", type=int, default=0)
    ap.add_argument("--char-len", type=int, default=48508)
    ap.add_argument("--n-repeat", type=int, default=397)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--tag", default="e2")
    ap.add_argument("--dump-k", type=int, default=32,
                    help="정합성 비교용으로 마지막 k 위치의 로짓을 산출(0=전 위치)")
    ap.add_argument("--shutdown-server", action="store_true")
    args = ap.parse_args()

    if args.mode == "compare":
        compare(args.dump, args.ref, args.tag)
        return

    if args.mode == "server":
        run_server(args)
        return

    k = args.dump_k
    rank = 0

    if args.mode == "client":
        model, tok = load_dsv4(args.model, args.split, None)
    elif args.mode == "tp2ref":
        group = mx.distributed.init()
        rank, world = group.rank(), group.size()
        print(f"[rank {rank}] world={world}", flush=True)
        assert world == 2, f"world {world} != 2"
        mx.random.seed(7)  # 전 랭크 동일 시드
        model, tok = load_dsv4(args.model, group=group)
    else:
        model, tok = load_dsv4(args.model)

    tokens, recipe = build_prompt_ids(tok, args)
    log(f"prompt: n={len(tokens)} recipe={json.dumps(recipe, ensure_ascii=False)}")

    if args.mode == "local":
        # 한 프로세스에서 ref1box → slice1box 순차 (로컬 단독, 분산 아님 → 체인 무해)
        run_ref1box(model, tokens[:args.warmup_tokens], args.chunk)
        mx.clear_cache()
        log("warmup done")
        ref_np = None
        for rep in range(args.reps):
            mx.reset_peak_memory()
            logits, st = run_ref1box(model, tokens, args.chunk, k)
            a = logits_to_np(logits)
            summarize(args.tag, "ref1box", st, len(tokens),
                      {"rep": rep, "chunk": args.chunk,
                       "t_chunks": [round(x, 4) for x in st["t_chunks"]],
                       "c_max": round(max(st["t_chunks"]), 4),
                       "bubble_law_T": round((st["t_total"] + max(st["t_chunks"])) / 2, 4),
                       "argmax": int(a.reshape(-1, a.shape[-1])[-1].argmax())})
            if rep == 0:
                ref_np = a
                if args.dump:
                    np.save(args.dump, a)
                    log(f"dumped ref logits -> {args.dump}")
        run_slice1box(model, tokens[:args.warmup_tokens], args.chunk, args.split)
        mx.clear_cache()
        for rep in range(args.reps):
            mx.reset_peak_memory()
            logits, st = run_slice1box(model, tokens, args.chunk, args.split, k)
            b = logits_to_np(logits)
            per_chunk = [x + y for x, y in zip(st["t_bot"], st["t_top"])]
            summarize(args.tag, "slice1box", st, len(tokens),
                      {"rep": rep, "chunk": args.chunk, "split": args.split,
                       "t_bot": [round(x, 4) for x in st["t_bot"]],
                       "t_top": [round(x, 4) for x in st["t_top"]],
                       "sum_bot": round(sum(st["t_bot"]), 4),
                       "sum_top": round(sum(st["t_top"]), 4),
                       "bot_frac": round(sum(st["t_bot"]) / st["t_total"], 4),
                       "c_max": round(max(per_chunk), 4),
                       "bot_c_first": round(st["t_bot"][0], 4),
                       "pp2_pred_T": round(st["t_bot"][0] + sum(st["t_top"]), 4),
                       "argmax": int(b.reshape(-1, b.shape[-1])[-1].argmax())})
            if rep == 0:
                same = ref_np.shape == b.shape and np.array_equal(ref_np, b)
                out = {"label": "slice1box_vs_ref1box", "bit_exact": bool(same),
                       "argmax_a": int(ref_np.reshape(-1, ref_np.shape[-1])[-1].argmax()),
                       "argmax_b": int(b.reshape(-1, b.shape[-1])[-1].argmax()),
                       "max_abs_diff": float(np.abs(ref_np - b).max()),
                       "n_mismatch": int((ref_np != b).sum()), "n_elem": int(b.size)}
                out["argmax_match"] = out["argmax_a"] == out["argmax_b"]
                print(f"[E2-CONSISTENCY] {json.dumps(out, ensure_ascii=False)}", flush=True)
        print("[e2-pass]", flush=True)
        return

    if args.mode == "tp2ref":
        # 현행 TP2 샤딩을 **본 하네스와 동일한 청크 프리필 루프**로 측정 (raw TP2).
        if args.warmup_tokens:
            run_ref1box(model, tokens[:args.warmup_tokens], args.chunk)
            mx.clear_cache()
            if rank == 0:
                log("warmup done")
        for rep in range(args.reps):
            mx.reset_peak_memory()
            logits, st = run_ref1box(model, tokens, args.chunk, k)
            a = logits_to_np(logits)
            if rank == 0:
                summarize(args.tag, "tp2ref", st, len(tokens),
                          {"rep": rep, "chunk": args.chunk,
                           "t_chunks": [round(x, 4) for x in st["t_chunks"]],
                           "argmax": int(a.reshape(-1, a.shape[-1])[-1].argmax())})
                if args.dump and rep == 0:
                    np.save(args.dump, a)
                    log(f"dumped tp2 logits -> {args.dump}")
        if rank == 0 and args.ref and args.dump:
            compare(args.dump, args.ref, f"{args.tag}_vs_ref")
            print("[e2-pass]", flush=True)
        return

    if args.mode == "ref1box":
        if args.warmup_tokens:
            run_ref1box(model, tokens[:args.warmup_tokens], args.chunk)
            mx.clear_cache()
            log("warmup done")
        for rep in range(args.reps):
            mx.reset_peak_memory()
            logits, st = run_ref1box(model, tokens, args.chunk, k)
            summarize(args.tag, "ref1box", st, len(tokens),
                      {"rep": rep, "chunk": args.chunk,
                       "t_chunks": [round(x, 4) for x in st["t_chunks"]],
                       "argmax": int(logits_to_np(logits).reshape(-1, logits.shape[-1])[-1].argmax())})
            if args.dump and rep == 0:
                np.save(args.dump, logits_to_np(logits))
                log(f"dumped logits -> {args.dump}")
        return

    if args.mode == "slice1box":
        if args.warmup_tokens:
            run_slice1box(model, tokens[:args.warmup_tokens], args.chunk, args.split)
            mx.clear_cache()
            log("warmup done")
        for rep in range(args.reps):
            mx.reset_peak_memory()
            logits, st = run_slice1box(model, tokens, args.chunk, args.split, k)
            C = st["t_total"]
            per_chunk = [b + t for b, t in zip(st["t_bot"], st["t_top"])]
            summarize(args.tag, "slice1box", st, len(tokens),
                      {"rep": rep, "chunk": args.chunk, "split": args.split,
                       "t_bot": [round(x, 4) for x in st["t_bot"]],
                       "t_top": [round(x, 4) for x in st["t_top"]],
                       "sum_bot": round(sum(st["t_bot"]), 4),
                       "sum_top": round(sum(st["t_top"]), 4),
                       "c_max": round(max(per_chunk), 4),
                       "bubble_law_T": round((C + max(per_chunk)) / 2, 4),
                       "argmax": int(logits_to_np(logits).reshape(-1, logits.shape[-1])[-1].argmax())})
            if args.dump and rep == 0:
                np.save(args.dump, logits_to_np(logits))
                log(f"dumped logits -> {args.dump}")
        return

    # ---- client
    cli = PP2Client(model, args.host, args.port, args.split)
    try:
        if args.warmup_tokens:
            cli.prefill(tokens[:args.warmup_tokens], args.chunk)
            mx.clear_cache()
            log("warmup done")
        for rep in range(args.reps):
            mx.reset_peak_memory()
            logits, st = cli.prefill(tokens, args.chunk, k)
            summarize(args.tag, "pp2", st, len(tokens), {
                "rep": rep, "chunk": args.chunk, "split": args.split,
                "t_act_waits": [round(x, 4) for x in st["t_act_waits"]],
                "t_local_chunks": [round(x, 4) for x in st["t_local_chunks"]],
                "t_deser": [round(x, 4) for x in st["t_deser"]],
                "server_t_chunks": [round(x, 4) for x in st["server_t_chunks"]],
                "server_t_backpressure": [round(x, 4) for x in st["server_t_backpressure"]],
                "server_t_compute": round(st["server_t_compute"], 4),
                "sum_local": round(sum(st["t_local_chunks"]), 4),
                "sum_wait_bubble": round(sum(st["t_act_waits"]), 4),
                "sum_deser": round(sum(st["t_deser"]), 4),
                "wire_mb": round(st["wire_bytes"] / 1e6, 1),
                "argmax": int(logits_to_np(logits).reshape(-1, logits.shape[-1])[-1].argmax()),
            })
            if args.dump and rep == 0:
                np.save(args.dump, logits_to_np(logits))
                log(f"dumped logits -> {args.dump}")
        if args.ref and args.dump:
            compare(args.dump, args.ref, f"{args.tag}_vs_ref")
    finally:
        if args.shutdown_server:
            cli.shutdown_server()
        else:
            cli.close()
    print("[e2-pass]", flush=True)


if __name__ == "__main__":
    main()
