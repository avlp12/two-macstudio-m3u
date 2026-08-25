"""PP2 프리필 스테이지 — 프로덕션 TP2 서빙(serve_batched_tp2.py)용 프리필 교체 모듈.

설계(옵션A): **프리필만 PP2 로 교체하고 디코드는 검증된 TP2 경로를 그대로 둔다.**
  · 각 랭크는 TP2 샤딩 모델(디코드용)에 더해 **비샤딩 층 슬라이스**(프리필용)를 함께 적재.
      rank0 = box A   : 상단 layers[split, n) + TCP 클라이언트/오케스트레이터
      rank1 = box B   : 하단 layers[0, split) + TCP 서버
  · PP2 청크 파이프라인으로 ids[:-1] 을 프리필 → 각 랭크가 자기 층들의 KV 만 보유.
  · **리샤드 = 양방향 교환**: DSv4 의 KV 경로(attn.wkv/kv_norm/compressor/indexer)는
    `Model.shard()` 가 건드리지 않고 kv 는 단일 헤드(B,1,L,head_dim) 이므로 캐시는
    **복제형** — 두 랭크가 전층 43개 캐시를 모두 가져야 한다(E4 게이트에서 실측 확인).
  · 조립된 캐시를 BatchGenerator 에 `insert_segments(caches=..., all_tokens=...)` 로
    주입 — 스냅숏 적중 경로와 동일한 계약이라 배칭/MTP/디코드는 무변경.

PP2 소켓은 **원시 TCP** (mx.distributed 미사용) 라 서빙의 jaccl 집합연산과 공존한다.
양 랭크가 완전히 대칭적으로 동일 op 순서를 밟으므로 락스텝이 유지된다.

MTP 프롬프트 프라이밍(`DSV4_PP2_PRIMING=1`): 프라이밍 캡처는 BatchGenerator 프리필
  forward 에 올라타므로 PP2 경로에서는 원래 한 번도 안 불렸다 → `take_primed` 가
  None → **unprimed 폴백**. 아래 `_PrimeFold` 가 프리필 중 이미 갖고 있는
  (hidden, next-token) 페어를 `prompt_priming.maybe_capture` 로 직접 접어 되살린다.
  게이트 OFF 면 종전과 정확히 동일한 경로(unprimed 폴백)다.

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
import numpy as np  # noqa: E402

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
    make_schedule,
    prepare_tensor,
    recv_msg,
    send_json,
    send_prepared,
    to_mx,
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


# ------------------------------------------------------- MTP 프롬프트 프라이밍
# 설계 근거와 계약(정본: omlx/patches/mlx_lm_mtp/prompt_priming.py 모듈 독스트링):
#
#  ① fold 대상 hidden = **트렁크 최종 raw 4D Hyper-stream hidden**.
#     DSv4 캡처 사이트(`deepseek_v4_model.Model.__call__`)는 `return_raw_hidden=True`
#     로 `norm(hc_head(h))` **이전**의 `h` 를 넘긴다("the head input variant; no
#     trunk norm"). PP2 상단 슬라이스(rank0)의 `V4Slice.forward()` 반환값이 정확히
#     그 텐서다 — 서빙 모델은 TP 샤딩이라 `pipeline_size==1`, 스톡 경로의
#     recv/send/all_gather 가 모두 no-op 이므로 두 경로의 `h` 정의가 일치한다.
#
#  ② fold 는 `maybe_capture` 를 **그대로 호출**한다(재구현 아님). 청크마다
#     (ids_chunk, hidden_chunk) 를 순서대로 넘기면 pending_hidden 이음새,
#     offset-연속성 가드, "never a wrong history" 불변이 모듈 자신의 코드로 검증된다.
#     결과 상태는 스톡 경로가 `key=ids[:-1]` 세그먼트를 프리필한 직후와 **동일**
#     (folded == len(pre)-1, expected_offset == len(pre))이라, 뒤이은 꼬리 토큰
#     forward → `_post_init_mtp` → `take_primed` 이음새가 무변경으로 이어진다.
#
#  ③ 앵커는 프리필 중 **실측한 슬라이스 캐시 offset** 을 되감아 쓴다(`_OffsetStub`).
#     fold 시점의 라이브 캐시는 이미 len(pre) 에 멈춰 있어 앵커로 쓸 수 없다.
#     스텁 값은 스케줄 누적합과 대조해 검증하므로 가드가 무력화되지 않는다.
#
#  ④ ctx 슬롯은 **대역 호스트**(`_PrimeHost`)에 둔다. HOL 이음새(DSV4_PP2_INTERLEAVE)
#     가 돌리는 다른 요청의 프리필이 같은 model 슬롯에 ctx 를 쓰므로, 실모델 슬롯에
#     바로 쓰면 서로 덮어써 **잘못된 히스토리**(접두가 빠진 ctx)가 만들어진다.
#     완성·검증 직후에만 model 로 옮긴다.
#
#  ⑤ MTP 헤드는 TP2 샤딩(`dspark_tp4_common.shard_mtp`)이라 `mtp_forward` 는
#     집합연산을 품을 수 있다 → **양 랭크가 같은 횟수·같은 형상으로** 접어야 한다.
#     그래서 rank0 이 청크 hidden 을 rank1 로 보내고 둘 다 접는다. (rank1 이 안 접고
#     완성 캐시만 받아오는 안은 집합연산 비대칭 = 교착 위험이라 기각. lazy 평가가
#     로짓 꼬리를 지워 실제로는 집합연산이 안 뜰 수도 있으나, 그 가정에 서빙
#     락스텝을 걸 수 없다.)
_PRIME_ATTR_FALLBACK = "_omlx_mtp_prime_ctx"


class _OffsetStub:
    """`maybe_capture` 의 offset-연속성 가드에 실제 프리필 타임라인을 되먹이는 앵커.

    `prompt_priming._anchor()` 는 plain-int `offset` 을 가진 첫 항목을 쓰므로
    리스트에 이 객체 하나만 담아 넘긴다."""

    __slots__ = ("offset",)

    def __init__(self, offset=0):
        self.offset = int(offset)


class _PrimeHost:
    """`maybe_capture` 계약은 그대로 두고 **ctx 슬롯만 분리**하는 얇은 대역 호스트.

    `_host_candidates` 가 보는 건 자기 자신과 `language_model`/`_language_model`
    뿐이라, 이 객체를 host 로 넘기면 ctx 가 여기에만 달린다(실모델 슬롯 무접촉).
    `_host_eligible` 이 요구하는 3개 속성과 `make_mtp_cache`/`mtp_forward` 만
    실모델로 위임한다."""

    def __init__(self, model):
        self._model = model
        self._omlx_mtp_decode_enabled = getattr(
            model, "_omlx_mtp_decode_enabled", False)
        self._omlx_mtp_chain = getattr(model, "_omlx_mtp_chain", False)
        self.mtp = getattr(model, "mtp", None)

    def make_mtp_cache(self):
        return self._model.make_mtp_cache()

    def mtp_forward(self, *a, **kw):
        return self._model.mtp_forward(*a, **kw)


class _PrimeFold:
    """양 랭크 대칭 MTP 프라이밍 fold. rank0 이 hidden 을 보내고 둘 다 접는다."""

    def __init__(self, stage):
        from omlx.patches.mlx_lm_mtp import prompt_priming

        self.pp = prompt_priming
        self.attr = getattr(prompt_priming, "_CTX_ATTR", _PRIME_ATTR_FALLBACK)
        self.stage = stage
        self.model_tp = stage.model_tp
        self.host = _PrimeHost(stage.model_tp)
        self.stub = _OffsetStub(0)
        self.chunks = []          # rank0 전용: [(ids, hidden, offset_after)]
        self.n_bytes = 0
        self.n_fold = 0
        # 헤드가 없거나 체인이 꺼져 있으면 maybe_capture 가 전부 조용히 무시하므로
        # 457MB 전송·fold 를 아예 시작하지 않는다(판정은 rank0 이 프레임으로 통보).
        self.eligible = bool(
            getattr(self.host, "_omlx_mtp_decode_enabled", False)
            and getattr(self.host, "_omlx_mtp_chain", False)
            and self.host.mtp is not None
            and prompt_priming.priming_enabled())

    # -- rank0: 프리필 루프 탭 -------------------------------------------
    def tap(self, ci, ids, hidden, offset_after):
        self.chunks.append((ids, hidden, int(offset_after)))
        self.n_bytes += int(hidden.size) * hidden.dtype.size

    # -- 공통 fold 1스텝 --------------------------------------------------
    def _fold(self, ids, hidden, offset_after):
        # maybe_capture 는 "forward 가 이미 돌았고 offset 에 S 가 포함된" 상태를
        # 가정하므로, 앵커를 그 시점 값으로 올려놓은 뒤 호출한다.
        self.stub.offset = int(offset_after)
        self.pp.maybe_capture(self.host, ids, hidden, [self.stub])
        self.n_fold += 1

    # -- 완성 ctx 검증 + 실모델 슬롯 설치 --------------------------------
    def install(self, n_pre, log):
        ctx = getattr(self.host, self.attr, None)
        # 우리 요청이 곧 활성화되므로 다른 요청이 남긴 잔여 ctx 는 무조건 무효화한다
        # (우연히 offset 이 맞아떨어져 **남의 히스토리**가 우리 것으로 소비되는 걸 차단.
        #  그 요청은 어차피 배치가 다행이 되는 순간 patched_extend 가 폐기한다).
        self.pp.drop_ctx(self.model_tp)
        if self.n_fold == 0:
            return 0                      # 이번 요청은 fold 생략 — 조용히 unprimed
        if ctx is None or not getattr(ctx, "valid", False):
            log("[prime] ctx 없음/무효 — unprimed 폴백")
            return 0
        if ctx.expected_offset != n_pre or ctx.folded != n_pre - 1:
            log(f"[prime] 불변 위반 expected_offset={ctx.expected_offset} "
                f"folded={ctx.folded} (기대 {n_pre}/{n_pre - 1}) — 폐기")
            return 0
        if ctx.pending_hidden is None:
            log("[prime] pending_hidden 없음 — 폐기")
            return 0
        setattr(self.model_tp, self.attr, ctx)
        return int(ctx.folded)

    # -- rank0 드라이브 ---------------------------------------------------
    def run_rank0(self, sock, sched, n_pre, log):
        cum, pos = [], 0
        for s in sched:
            pos += s
            cum.append(pos)
        ok = (self.eligible
              and len(self.chunks) == len(sched)
              and [c[2] for c in self.chunks] == cum
              and pos == n_pre)
        if not ok:
            log(f"[prime] 생략: eligible={self.eligible} chunks={len(self.chunks)} "
                f"sched={len(sched)}")
        cap = self.stage.prime_max_bytes
        if ok and cap and self.n_bytes > cap:
            log(f"[prime] hidden {self.n_bytes / 2**30:.2f}GiB > 상한 "
                f"{cap / 2**30:.2f}GiB — 이번 요청 프라이밍 생략")
            ok = False
        # 접을지 말지는 **rank0 이 정하고 프레임으로 통보**한다 — 양 랭크가 각자
        # 술어를 평가하면 드리프트 한 번에 fold 횟수가 갈려 영구 교착이 된다.
        send_json(sock, {"op": "prime", "fold": bool(ok), "n": len(sched),
                         "offsets": cum})
        if not ok:
            self.chunks.clear()
            self._expect_ack(sock)
            return 0
        t0 = time.perf_counter()
        tx = 0
        for i, (ids, h, off) in enumerate(self.chunks):
            tx += send_prepared(sock, "ph", prepare_tensor(h), i=i)
            self._fold(ids, h, off)
        self.chunks.clear()          # hidden 즉시 해제
        mx.clear_cache()
        self._expect_ack(sock)
        log(f"[prime] fold {self.n_fold}청크 "
            f"{(time.perf_counter() - t0) * 1e3:.0f}ms (tx {tx / 1e6:.0f}MB)")
        return self.n_fold

    def _expect_ack(self, sock):
        kind, *rest = recv_msg(sock)
        if kind != "json" or rest[0].get("op") != "prime_ack":
            raise WireError(f"expected prime_ack, got {kind}/{rest[0] if rest else ''}")

    # -- rank1 종속 ------------------------------------------------------
    def run_rank1(self, sock, msg, tokens, sched, n_pre, log):
        cum, pos = [], 0
        for s in sched:
            pos += s
            cum.append(pos)
        if int(msg.get("n", -1)) != len(sched) or msg.get("offsets") != cum:
            raise WireError(
                f"prime 매니페스트 불일치: n={msg.get('n')} vs {len(sched)}")
        if not msg.get("fold"):
            send_json(sock, {"op": "prime_ack", "folded": 0})
            return 0
        toks = mx.array(np.asarray(tokens, dtype=np.int32))[None]
        pos = 0
        for i, n in enumerate(sched):
            m = recv_msg(sock)
            if m[0] != "tensor" or int(m[1].get("i", -1)) != i:
                raise WireError(f"prime hidden {i} 프레임 이상: {m[0]}/{m[1]}")
            meta, raw = m[1], m[2]
            h = to_mx(meta, raw)
            mx.eval(h)
            self._fold(toks[:, pos:pos + n], h, cum[i])
            h = None
            pos += n
        mx.clear_cache()
        send_json(sock, {"op": "prime_ack", "folded": self.n_fold})
        return self.n_fold


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
                 port=39935, server_ip="10.0.0.2", min_tokens=4096,
                 priming=False, prime_max_bytes=4 << 30, log=print):
        self.rank = rank
        self.model_tp = model_tp
        self.model_path = model_path
        self.split = split
        self.chunk = chunk
        self.port = port
        self.server_ip = server_ip
        self.min_tokens = min_tokens
        self.priming = priming
        self.prime_max_bytes = prime_max_bytes
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
        sched = make_schedule(len(pre), self.chunk)
        fold = _PrimeFold(self) if self.priming else None
        t0 = time.perf_counter()
        if self.rank == 0:
            cli = _Client(self.model_pp, self.sock, self.split, self.n_layers)
            _lg, pst = cli.prefill(pre, self.chunk, k=1, on_seam=on_seam,
                                   tap=(fold.tap if fold is not None else None))
            t_pf = time.perf_counter() - t0
            if fold is not None:
                # 이음새 콜백이 **모두 끝난 뒤** 접는다 — 이음새가 돌리는 다른 요청의
                # 프리필이 우리 fold 사이에 끼면 청크 순서·ctx 가 뒤엉킨다.
                fold.run_rank0(self.sock, sched, len(pre), self.log)
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
            self._serve_ops(sl, on_push, on_seam=on_seam, fold=fold)
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
        if fold is not None:
            # 설치는 **맨 마지막**에 — 여기부터 활성화(`take_primed`)까지는 forward 가
            # 없어 다른 요청의 캡처가 슬롯을 덮을 창이 없다. 양 랭크 동일 시점.
            n = fold.install(len(pre), self.log)
            if self.rank == 0:
                self.log(f"[prime] 설치 {n}페어 (프롬프트 {len(ids)} tok)")
        self.n_calls += 1
        return full

    def _serve_ops(self, sl, on_push, on_seam=None, fold=None):
        last = {}
        while True:
            kind, *rest = recv_msg_warm(self.sock)
            if kind != "json":
                raise WireError("unexpected tensor frame")
            msg = rest[0]
            op = msg.get("op")
            if op == "prefill":
                last = {"tokens": [int(t) for t in msg["tokens"]],
                        "schedule": list(msg["schedule"])}
                _do_prefill(self.sock, sl, msg, on_seam=on_seam)
            elif op == "prime":
                if fold is None:
                    # 게이트가 랭크마다 다르게 켜졌다 = fold 횟수 비대칭 =
                    # 집합연산 영구 교착. 조용히 넘기지 말고 크게 죽는다.
                    raise WireError(
                        "prime 프레임 도착했으나 랭크1 프라이밍 게이트가 꺼짐 — "
                        "DSV4_PP2_PRIMING 이 양 랭크에 같게 걸렸는지 확인")
                if not last:
                    raise WireError("prime 이 prefill 보다 먼저 도착 — 프로토콜 파손")
                fold.run_rank1(self.sock, msg, last["tokens"], last["schedule"],
                               sum(last["schedule"]), self.log)
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
        # MTP 프롬프트 프라이밍 복원(기본 off = 종전 unprimed 폴백 경로 그대로).
        # 양 랭크에 **반드시 같은 값**으로 걸려야 한다 — serve_b_pp2.sh 는 리터럴
        # export 라 자동 보장되고, 어긋나면 `_serve_ops` 가 즉시 죽는다.
        priming=os.environ.get("DSV4_PP2_PRIMING", "0").strip().lower()
        not in ("0", "", "false", "off"),
        prime_max_bytes=int(
            float(os.environ.get("DSV4_PP2_PRIME_MAX_GIB", "8")) * (1 << 30)),
        log=log,
    ).setup()
