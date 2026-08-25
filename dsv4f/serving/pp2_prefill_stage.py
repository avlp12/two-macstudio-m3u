"""PP2 프리필 스테이지 — 프로덕션 TP2 서빙(serve_batched_tp2.py)용 프리필 교체 모듈.

설계(옵션A): **프리필만 PP2 로 교체하고 디코드는 검증된 TP2 경로를 그대로 둔다.**
  · 각 랭크는 TP2 샤딩 모델(디코드용)에 더해 **비샤딩 층 슬라이스**(프리필용)를 함께 적재.
      rank0 = box A : 상단 layers[split, n) + TCP 클라이언트/오케스트레이터
      rank1 = box B : 하단 layers[0, split) + TCP 서버
  · PP2 청크 파이프라인으로 ids[:-1] 을 프리필 → 각 랭크가 자기 층들의 KV 만 보유.
  · **리샤드 = 양방향 교환**: DSv4 의 KV 경로(attn.wkv/kv_norm/compressor/indexer)는
    `Model.shard()` 가 건드리지 않고 kv 는 단일 헤드(B,1,L,head_dim) 이므로 캐시는
    **복제형** — 두 랭크가 전층 43개 캐시를 모두 가져야 한다(E4 게이트에서 실측 확인).
  · 조립된 캐시를 BatchGenerator 에 `insert_segments(caches=..., all_tokens=...)` 로
    주입 — 스냅숏 적중 경로와 동일한 계약이라 배칭/MTP/디코드는 무변경.

PP2 소켓은 **원시 TCP** (mx.distributed 미사용) 라 서빙의 jaccl 집합연산과 공존한다.
양 랭크가 완전히 대칭적으로 동일 op 순서를 밟으므로 락스텝이 유지된다.

MTP 프롬프트 프라이밍: 프라이밍은 BatchGenerator 프리필 forward 에 올라타 캡처되므로
  PP2 경로에서는 캡처가 일어나지 않는다 → `take_primed` 가 None 을 돌려주고
  **unprimed 폴백**으로 조용히 degrade 한다(설계된 안전 경로). 그래서 본 모듈은
  `min_tokens` 이상의 장문에만 적용되고, 짧은 요청은 기존 경로(프라이밍 유지)로 간다.

근거 로그: ~/dsv4flash/align/logs/pp2int_{gate1,diag1,diag2,needle}.log
"""

import json
import os
import select
import socket
import struct
import sys
import time

sys.path.insert(0, "/Users/Shared/tp2")

import mlx.core as mx  # noqa: E402

from e2_pp2_prefill import (  # noqa: E402
    _HDR,
    MAGIC,
    T_TENSOR,
    PP2Client,
    V4Slice,
    WireError,
    _cache_arrays,
    _do_prefill,
    _npbytes,
    load_dsv4,
    recv_msg,
    send_json,
    tune,
)
from e3_kv_handover import cache_offsets, cache_restore, cache_spec, to_mx_copy  # noqa: E402


# --------------------------------------------------------------- GPU 킵얼라이브
# [발견 2026-08-25] 145GiB 상주 프로세스의 GPU 가 수 초 유휴가 되면 **다음 첫 제출이
# ~0.9s 스톨**한다(레지던시 재수립). 실측: r1 이 자기 청크를 먼저 끝내고 r0 을 기다린 뒤
# `mx.eval(mx.sum(mx.ones((8,8))))` 단독으로 918.6ms, 바로 다음 동일 op 는 0.4ms.
# 이 스톨이 PP2 통합 오버헤드 1.25s 중 0.91s 였다. 수치·캐시·집합연산과 무관한
# 초소형 op 를 유휴 중 주기적으로 던져 유휴 자체를 없앤다.
KEEPALIVE_S = max(0.0, float(os.environ.get("DSV4_GPU_KEEPALIVE_MS", "50")) / 1000.0)


def gpu_ping():
    """레지던시 유지용 초소형 GPU 제출 (약 0.4ms). 상태를 건드리지 않는다."""
    mx.eval(mx.sum(mx.ones((8, 8))))


def recv_msg_warm(sock, ping_s=None):
    """소켓 대기 중 GPU 를 깨워 두는 recv_msg. 데이터가 오면 즉시 원본 경로."""
    ping_s = KEEPALIVE_S if ping_s is None else ping_s
    if ping_s <= 0:
        return recv_msg(sock)
    while True:
        r, _, _ = select.select([sock], [], [], ping_s)
        if r:
            return recv_msg(sock)
        gpu_ping()


# --------------------------------------------------------------- 캐시 와이어
# (E4 게이트 하네스 e4_pp2_tp2_bridge.py 에서 검증된 것과 동일한 프레이밍)
def send_cache(sock, caches, op):
    arrays, owners = [], []
    descs = [cache_spec(c, arrays, owners) for c in caches]
    mx.eval(*arrays)
    preps = []
    for a in arrays:
        n, tag = _npbytes(a)
        preps.append((n, tag, list(a.shape)))
    send_json(sock, {"op": op, "layers": descs, "n_arrays": len(arrays)})
    total = 0
    for i, (n, tag, shape) in enumerate(preps):
        hdr = {"n": f"c{i}", "d": tag, "s": shape, "i": i}
        j = json.dumps(hdr).encode()
        sock.sendall(_HDR.pack(MAGIC, T_TENSOR, 4 + len(j) + n.nbytes))
        sock.sendall(struct.pack("<I", len(j)))
        sock.sendall(j)
        sock.sendall(n)
        total += n.nbytes
    return total


def recv_cache(sock, expect_op):
    kind, *rest = recv_msg(sock)
    if kind != "json" or rest[0].get("op") != expect_op:
        raise WireError(f"expected {expect_op}, got {kind}")
    man = rest[0]
    n_arr = int(man["n_arrays"])
    slots = [None] * n_arr
    nbytes = 0
    for _ in range(n_arr):
        m = recv_msg(sock)
        if m[0] != "tensor":
            raise WireError(f"expected tensor, got {m[0]}")
        meta, raw = m[1], m[2]
        slots[int(meta["i"])] = (meta, raw)
        nbytes += len(raw)
    arrays = [to_mx_copy(meta, raw) for (meta, raw) in slots]
    mx.eval(*arrays)
    return man["layers"], arrays, nbytes


def assemble_full(model_tp, wire_layers, wire_arrays, local_cache, lo_is_local, split,
                  tr=None):
    """전층 캐시 조립 — 원격분은 model_tp.make_cache() 껍데기에 복원, 로컬분은 객체 그대로."""
    t0 = time.perf_counter()
    shell = model_tp.make_cache()
    t1 = time.perf_counter()
    n = len(shell)
    if lo_is_local:                      # rank1: 하단 로컬 + 상단 원격
        if len(wire_layers) != n - split:
            raise WireError(f"manifest {len(wire_layers)} != {n - split}")
        for j in range(split, n):
            cache_restore(shell[j], wire_layers[j - split], wire_arrays)
        full = list(local_cache) + list(shell[split:])
    else:                                # rank0: 하단 원격 + 상단 로컬
        if len(wire_layers) != split:
            raise WireError(f"manifest {len(wire_layers)} != {split}")
        for i in range(split):
            cache_restore(shell[i], wire_layers[i], wire_arrays)
        full = list(shell[:split]) + list(local_cache)
    t2 = time.perf_counter()
    arrs = _cache_arrays(full)
    t3 = time.perf_counter()
    if tr is None:
        mx.eval(*arrs)
        return full
    # 트레이스: 층별로 나눠 평가해 지연이 어느 층에 있는지 귀속
    per = []
    for i, lc in enumerate(full):
        ta = time.perf_counter()
        mx.eval(*_cache_arrays([lc]))
        per.append((time.perf_counter() - ta) * 1e3)
    t4 = time.perf_counter()
    hot = sorted(range(len(per)), key=lambda i: -per[i])[:6]
    tr(f"[trace] asm: make_cache {(t1 - t0) * 1e3:.1f}ms restore "
       f"{(t2 - t1) * 1e3:.1f}ms collect {(t3 - t2) * 1e3:.1f}ms "
       f"eval {(t4 - t3) * 1e3:.1f}ms n_arr={len(arrs)} "
       f"act={mx.get_active_memory() / 2**30:.1f}GiB")
    tr("[trace] asm per-layer top: "
       + " ".join(f"L{i}={per[i]:.1f}" for i in hot)
       + f" | local={'[0,%d)' % split if lo_is_local else '[%d,%d)' % (split, n)}")
    h0 = hot[0]
    tr(f"[trace] asm hot L{h0} arrays: "
       + " ".join(f"{tuple(a.shape)}/{a.dtype}" for a in _cache_arrays([full[h0]])))
    return full


class _Client(PP2Client):
    """PP2Client 의 청크 파이프라인만 재사용(소켓/모델은 외부 주입)."""

    def __init__(self, model_pp, sock, split, n_layers):
        self.model = model_pp
        self.n_layers = n_layers
        self.split = split
        self.top = V4Slice(model_pp, split, n_layers)
        self.hidden = self.top.hidden
        self.hc_mult = self.top.hc_mult
        self.sock = sock
        self.remote = None


class Pp2Stage:
    """양 랭크에서 **대칭적으로** 생성·호출되는 PP2 프리필 스테이지."""

    def __init__(self, rank, model_tp, model_path, *, split=22, chunk=2048,
                 port=39935, server_ip="10.0.0.2", min_tokens=4096, log=print):
        self.rank = rank
        self.model_tp = model_tp
        self.model_path = model_path
        self.split = split
        self.chunk = chunk
        self.port = port
        self.server_ip = server_ip
        self.min_tokens = min_tokens
        self.log = log
        self.n_layers = len(model_tp.model.layers)
        self.model_pp = None
        self.sock = None
        self.n_calls = 0
        self.trace = os.environ.get("DSV4_PP2_TRACE", "0").strip().lower() not in (
            "0", "", "false", "off")

    # ---------------------------------------------------------------- setup
    def setup(self, connect_timeout=600):
        """슬라이스 적재 + TCP 수립. 실패는 **치명적**(양 랭크 비대칭 = 락스텝 파손)."""
        t0 = time.monotonic()
        if self.rank == 0:
            self.model_pp, _ = load_dsv4(self.model_path, self.split, None)
        else:
            self.model_pp, _ = load_dsv4(self.model_path, 0, self.split)
        self.log(f"[pp2] r{self.rank} 슬라이스 적재 {time.monotonic() - t0:.1f}s "
                 f"mem={mx.get_active_memory() / 2**30:.1f}GiB")

        if self.rank == 1:
            srv = socket.socket()
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", self.port))
            srv.listen(1)
            srv.settimeout(connect_timeout)
            self.log(f"[pp2] r1 listening :{self.port}")
            self.sock, addr = srv.accept()
            self.log(f"[pp2] r1 client {addr}")
            srv.close()
        else:
            deadline = time.monotonic() + connect_timeout
            last = None
            while time.monotonic() < deadline:
                try:
                    self.sock = socket.create_connection(
                        (self.server_ip, self.port), timeout=30)
                    break
                except OSError as e:
                    last = e
                    time.sleep(2)
            if self.sock is None:
                raise RuntimeError(f"[pp2] r0 접속 실패 {self.server_ip}:{self.port} — {last}")
            self.log(f"[pp2] r0 connected {self.server_ip}:{self.port}")
        tune(self.sock)
        self.sock.settimeout(1800)
        return self

    def eligible(self, ids):
        return self.sock is not None and len(ids) >= self.min_tokens

    # ------------------------------------------------------------- 프리필
    def build_cache(self, ids, on_seam=None):
        """ids[:-1] 을 PP2 로 프리필하고 **전층 캐시**를 반환. 양 랭크 대칭 호출 필수.

        `on_seam`(HOL 인터리브): 2048-청크 이음새마다 호출되는 콜백. 양 랭크가
        **같은 횟수**(=청크 수 N) 호출하도록 e2_pp2_prefill 쪽에서 보정돼 있다.
        콜백 안에서 라이브 배치 디코드 1 스텝(집합연산)을 돌려도 안전한 이유:
          · PP2 경로는 원시 TCP 만 쓰고 mx.distributed 집합연산을 **전혀** 안 쓴다
          · 각 forward 의 수학은 불변(청크 스케줄·순서 무변경, 실행 시점만 이동)
          · 디코드는 별도 캐시/그래프(TP2 샤딩 모델) — PP2 슬라이스와 상태 비접촉
        """
        ids = list(ids)
        pre = ids[:-1]
        t0 = time.perf_counter()
        if self.rank == 0:
            cli = _Client(self.model_pp, self.sock, self.split, self.n_layers)
            _lg, pst = cli.prefill(pre, self.chunk, k=1, on_seam=on_seam)
            t_pf = time.perf_counter() - t0
            th = time.perf_counter()
            send_json(self.sock, {"op": "fetch_cache"})
            layers, arrays, rx = recv_cache(self.sock, "cache_manifest")
            kind, *rest = recv_msg(self.sock)
            if kind != "json" or rest[0].get("op") != "cache_done":
                raise WireError("expected cache_done")
            t_rx = time.perf_counter()
            send_json(self.sock, {"op": "push_cache_hdr"})
            k2, *r2 = recv_msg(self.sock)
            if k2 != "json" or r2[0].get("op") != "push_ready":
                raise WireError("expected push_ready")
            tx = send_cache(self.sock, cli.top.cache, "push_manifest")
            k3, *r3 = recv_msg(self.sock)
            if k3 != "json" or r3[0].get("op") != "push_ack":
                raise WireError("expected push_ack")
            t_tx = time.perf_counter()
            if self.trace:
                _s0 = time.perf_counter(); mx.synchronize()
                _s1 = time.perf_counter(); mx.eval(mx.sum(mx.ones((8, 8))))
                self.log(f"[trace-r0] pre-asm sync {(_s1 - _s0) * 1e3:.1f}ms "
                         f"tiny-op {(time.perf_counter() - _s1) * 1e3:.1f}ms")
            full = assemble_full(self.model_tp, layers, arrays, cli.top.cache,
                                 False, self.split,
                                 tr=(self.log if self.trace else None))
            t_hv = time.perf_counter() - th
            t_asm = time.perf_counter()
            send_json(self.sock, {"op": "phase_done"})
            k4, *r4 = recv_msg(self.sock)
            if k4 != "json" or r4[0].get("op") != "phase_ack":
                raise WireError("expected phase_ack")
            t_ph = time.perf_counter()
            self.log(f"[pp2] {len(pre)} tok 프리필 {t_pf:.2f}s "
                     f"({len(pre) / t_pf:.0f} tok/s) · 인계 {t_hv * 1e3:.0f}ms "
                     f"(rx {rx / 1e6:.0f}MB / tx {tx / 1e6:.0f}MB)")
            if self.trace:
                self.log(f"[trace-r0] hv: rx {(t_rx - th) * 1e3:.1f}ms "
                         f"tx {(t_tx - t_rx) * 1e3:.1f}ms "
                         f"asm {(t_asm - t_tx) * 1e3:.1f}ms "
                         f"phase {(t_ph - t_asm) * 1e3:.1f}ms")
        else:
            t_r1a = time.perf_counter()
            sl = V4Slice(self.model_pp, 0, self.split)
            pushed = {}

            def on_push():
                layers, arrays, _n = recv_cache(self.sock, "push_manifest")
                pushed["v"] = (layers, arrays)
                send_json(self.sock, {"op": "push_ack"})

            t_r1b = time.perf_counter()
            self._serve_ops(sl, on_push, on_seam=on_seam)
            t_r1c = time.perf_counter()
            if self.trace:
                _s0 = time.perf_counter(); mx.synchronize()
                _s1 = time.perf_counter(); mx.eval(mx.sum(mx.ones((8, 8))))
                _s2 = time.perf_counter(); mx.eval(mx.sum(mx.ones((8, 8))))
                self.log(f"[trace-r1] pre-asm sync {(_s1 - _s0) * 1e3:.1f}ms "
                         f"tiny-op1 {(_s2 - _s1) * 1e3:.1f}ms "
                         f"tiny-op2 {(time.perf_counter() - _s2) * 1e3:.1f}ms")
            layers, arrays = pushed["v"]
            full = assemble_full(self.model_tp, layers, arrays, sl.cache,
                                 True, self.split,
                                 tr=(self.log if self.trace else None))
            t_r1d = time.perf_counter()
            if self.trace:
                self.log(f"[trace-r1] slice {(t_r1b - t_r1a) * 1e3:.1f}ms "
                         f"serve_ops {(t_r1c - t_r1b) * 1e3:.1f}ms "
                         f"**asm-after-ack {(t_r1d - t_r1c) * 1e3:.1f}ms**")

        offs = cache_offsets(full)
        bad = [i for i, o in enumerate(offs) if o != len(pre)]
        if bad:
            raise RuntimeError(f"[pp2] 조립 캐시 offset 불일치 층 {bad[:5]} "
                               f"(기대 {len(pre)})")
        self.n_calls += 1
        return full

    def _serve_ops(self, sl, on_push, on_seam=None):
        while True:
            kind, *rest = recv_msg_warm(self.sock)
            if kind != "json":
                raise WireError("unexpected tensor frame")
            msg = rest[0]
            op = msg.get("op")
            if op == "prefill":
                _do_prefill(self.sock, sl, msg, on_seam=on_seam)
            elif op == "fetch_cache":
                n = send_cache(self.sock, sl.cache, "cache_manifest")
                send_json(self.sock, {"op": "cache_done", "bytes_total": n})
            elif op == "push_cache_hdr":
                send_json(self.sock, {"op": "push_ready"})
                on_push()
            elif op == "phase_done":
                send_json(self.sock, {"op": "phase_ack"})
                return
            else:
                send_json(self.sock, {"op": "error", "err": f"unknown op {op!r}"})

    def close(self):
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        self.sock = None


def from_env(rank, model_tp, model_path, log=print):
    """`DSV4_PP2_PREFILL=1` 일 때만 스테이지를 만든다(기본 off = 현행 경로 유지)."""
    if os.environ.get("DSV4_PP2_PREFILL", "0").strip().lower() in ("0", "", "false", "off"):
        return None
    return Pp2Stage(
        rank, model_tp, model_path,
        split=int(os.environ.get("DSV4_PP2_SPLIT", "22")),
        chunk=int(os.environ.get("DSV4_PP2_CHUNK", "2048")),
        port=int(os.environ.get("DSV4_PP2_PORT", "39935")),
        server_ip=os.environ.get("DSV4_PP2_SERVER_IP", "10.0.0.2"),
        min_tokens=int(os.environ.get("DSV4_PP2_MIN_TOKENS", "4096")),
        log=log,
    ).setup()
