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


def assemble_full(model_tp, wire_layers, wire_arrays, local_cache, lo_is_local, split):
    """전층 캐시 조립 — 원격분은 model_tp.make_cache() 껍데기에 복원, 로컬분은 객체 그대로."""
    shell = model_tp.make_cache()
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
    mx.eval(*_cache_arrays(full))
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
    def build_cache(self, ids):
        """ids[:-1] 을 PP2 로 프리필하고 **전층 캐시**를 반환. 양 랭크 대칭 호출 필수."""
        ids = list(ids)
        pre = ids[:-1]
        t0 = time.perf_counter()
        if self.rank == 0:
            cli = _Client(self.model_pp, self.sock, self.split, self.n_layers)
            _lg, pst = cli.prefill(pre, self.chunk, k=1)
            t_pf = time.perf_counter() - t0
            th = time.perf_counter()
            send_json(self.sock, {"op": "fetch_cache"})
            layers, arrays, rx = recv_cache(self.sock, "cache_manifest")
            kind, *rest = recv_msg(self.sock)
            if kind != "json" or rest[0].get("op") != "cache_done":
                raise WireError("expected cache_done")
            send_json(self.sock, {"op": "push_cache_hdr"})
            k2, *r2 = recv_msg(self.sock)
            if k2 != "json" or r2[0].get("op") != "push_ready":
                raise WireError("expected push_ready")
            tx = send_cache(self.sock, cli.top.cache, "push_manifest")
            k3, *r3 = recv_msg(self.sock)
            if k3 != "json" or r3[0].get("op") != "push_ack":
                raise WireError("expected push_ack")
            full = assemble_full(self.model_tp, layers, arrays, cli.top.cache,
                                 False, self.split)
            t_hv = time.perf_counter() - th
            send_json(self.sock, {"op": "phase_done"})
            k4, *r4 = recv_msg(self.sock)
            if k4 != "json" or r4[0].get("op") != "phase_ack":
                raise WireError("expected phase_ack")
            self.log(f"[pp2] {len(pre)} tok 프리필 {t_pf:.2f}s "
                     f"({len(pre) / t_pf:.0f} tok/s) · 인계 {t_hv * 1e3:.0f}ms "
                     f"(rx {rx / 1e6:.0f}MB / tx {tx / 1e6:.0f}MB)")
        else:
            sl = V4Slice(self.model_pp, 0, self.split)
            pushed = {}

            def on_push():
                layers, arrays, _n = recv_cache(self.sock, "push_manifest")
                pushed["v"] = (layers, arrays)
                send_json(self.sock, {"op": "push_ack"})

            self._serve_ops(sl, on_push)
            layers, arrays = pushed["v"]
            full = assemble_full(self.model_tp, layers, arrays, sl.cache,
                                 True, self.split)

        offs = cache_offsets(full)
        bad = [i for i, o in enumerate(offs) if o != len(pre)]
        if bad:
            raise RuntimeError(f"[pp2] 조립 캐시 offset 불일치 층 {bad[:5]} "
                               f"(기대 {len(pre)})")
        self.n_calls += 1
        return full

    def _serve_ops(self, sl, on_push):
        while True:
            kind, *rest = recv_msg(self.sock)
            if kind != "json":
                raise WireError("unexpected tensor frame")
            msg = rest[0]
            op = msg.get("op")
            if op == "prefill":
                _do_prefill(self.sock, sl, msg)
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
