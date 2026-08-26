"""실시간 프리필/디코드 감시 대시보드 — 이벤트 링버퍼 + /metrics + /dash.

설계 원칙(측정 경로 무해성):
  · 계측은 **append 수준**만 한다. `mx` 연산·동기화·집합연산을 일절 추가하지 않는다.
    (mx.synchronize 하나만 끼워도 lazy 그래프가 접혀 벽시계가 바뀐다 — 트레이스
     경로 DSV4_PP2_TRACE 가 그래서 기본 off 인 것과 같은 이유.)
  · 랭크0 이벤트만 기록한다. 서빙 HTTP 가 rank0 에만 있으므로 충분하고, 랭크1 에
    핸들을 심으면 락스텝 대칭성 검토가 늘어나기만 한다.
  · 락은 잡되 **보유 구간을 리스트 복사까지**로 묶는다. 직렬화는 락 밖에서 한다
    (gen_loop = 메인 스레드 = 집합연산 스레드라 HTTP 스레드가 붙잡으면 안 됨).

이벤트 스키마(모두 `i`=단조 증가 id, `t0`/`t1`=대시보드 기동 이후 상대 초):
  {k:"pf",  n, tps, src}      프리필 청크. src="pp2"(2박스 층-파이프라인 이음새)
                              | "bg"(BatchGenerator 프리필 스텝) | "pp2w"(이음새 없는
                              PP2 전체 빌드 = 요청 단위)
  {k:"dec", n, tps, act}      디코드 1초 버킷. n=그 초에 클라이언트로 나간 토큰 수,
                              act=활성 세션 수
  {k:"req", ttft, n_prompt}   요청 첫 토큰(TTFT, 초)
  {k:"fin", n, tps}           요청 완료(총 생성 토큰·요청 내 평균 tok/s)
  {k:"mtp", tpc, d}           MTP 텔레메트리(시퀀스 종료마다 1건). d=[d1,d2,d3] 수락률
  {k:"snap", kind, n, tot}    스냅숏 이벤트. kind="hit"|"pp2store"|"store"

프리필 tok/s 의 귀속(중요):
  · PP2 경로: 청크 i 의 소요 = (이음새 i 진입 시각) − (이음새 i-1 이탈 시각).
    이음새 **본문**(HOL 인터리브 디코드 스텝)은 빼고 잰다 — 그 시간은 디코드
    버킷이 이미 세고 있으므로 넣으면 이중 계상이다.
  · bg 경로: `BatchGenerator._prompt_tokens_counter` / `._prompt_time_counter` 의
    스텝 간 델타. mlx_lm 이 프롬프트 forward 구간만 재놓은 값이라 큐/디스패치
    오버헤드가 섞이지 않는다.
  · 랭크1 의 하단-슬라이스 프리필은 별도 계측하지 않고 **랭크0 이음새 시점으로
    근사**한다(PP2 는 청크마다 랭크1→랭크0 활성값 전송으로 동기되므로 이음새
    간격이 곧 양 랭크 합산 청크 시간이다). 대시보드 각주에 명시.

영속 축적(SQLite · stdlib 전용, 신규 의존성 없음):
  · 라이브 뷰는 **인메모리 링버퍼 그대로**(성능), 회고 뷰만 DB 를 읽는다.
  · rank0 전용. 파일 `~/dsv4flash/metrics/serving_metrics.sqlite`(DSV4_DASH_DB).
    WAL 모드라 서빙이 쓰는 동안 외부 프로세스가 그대로 읽을 수 있다.
  · 이벤트는 상대시각(t0/t1, 세션 기준)과 **절대시각(w0/w1, unix epoch)** 을 함께
    저장한다 → 재기동으로 상대시각이 0 으로 리셋돼도 세션 간 타임라인이 이어진다.
  · 쓰기는 gen_loop **밖** 사이드 스레드가 2초 주기로 배치 flush(자체 커넥션).
    sqlite 는 순수 CPU 라 mx 제약과 무관하고, 측정 경로는 리스트 append 하나만 는다.
  · 읽기(HTTP 스레드)는 요청마다 짧은 읽기 전용 커넥션을 연다 — sqlite 커넥션을
    스레드 간 공유하지 않는다는 규칙을 그대로 지킨다.
  · **SIGTERM 최종 flush 는 일부러 안 건다**: 핸들러를 걸면 TERM 이 '즉시 종료'에서
    '메인 스레드가 바이트코드 경계에 닿을 때 종료'로 바뀐다. 집합연산에 갇힌
    랭크에는 TERM 이 아예 안 먹게 되어 R2 의 'TERM-불응 → 재부팅 권고' 를 유발한다.
    운영 안전(R2)이 꼬리 2초보다 중요하므로, TERM 시 최대 DSV4_DASH_FLUSH_S 초 분량만
    유실하도록 두고 정상 종료 경로는 atexit 로 덮는다.
"""

import atexit
import json
import os
import re
import socket
import sqlite3
import threading
import time
from collections import deque

_ENV = os.environ.get("DSV4_DASH", "1").strip().lower()
ENABLED = _ENV not in ("0", "", "false", "off")


def _env_on(name, default="1"):
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "off")


WINDOW_S = float(os.environ.get("DSV4_DASH_WINDOW_S", "3600"))
MAXLEN = int(os.environ.get("DSV4_DASH_MAXLEN", "20000"))
# 이보다 작은 프리필 조각은 바로 그리지 않는다(PP2 주입 뒤 꼬리 토큰 1개 forward
# 같은 것들 — 20 tok/s 짜리 노이즈 바가 타임라인을 더럽힌다).
PF_MIN_TOKENS = int(os.environ.get("DSV4_DASH_PF_MIN", "16"))

# ── 영속화 ────────────────────────────────────────────────────────────────
PERSIST = _env_on("DSV4_DASH_PERSIST")
DB_PATH = os.environ.get(
    "DSV4_DASH_DB", os.path.expanduser("~/dsv4flash/metrics/serving_metrics.sqlite"))
RETENTION_DAYS = float(os.environ.get("DSV4_DASH_RETENTION_DAYS", "30"))
FLUSH_S = float(os.environ.get("DSV4_DASH_FLUSH_S", "2"))
# 회고 쿼리 1회 상한(브라우저·직렬화 보호). 초과 시 truncated=true 로 알린다.
READ_LIMIT = int(os.environ.get("DSV4_DASH_READ_LIMIT", "120000"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
  boot_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  started_wall REAL NOT NULL,
  ended_wall   REAL,
  host         TEXT,
  pid          INTEGER,
  model        TEXT,
  cfg          TEXT
);
CREATE TABLE IF NOT EXISTS events(
  id      INTEGER PRIMARY KEY AUTOINCREMENT,
  boot_id INTEGER NOT NULL,
  k       TEXT NOT NULL,
  w0      REAL NOT NULL,   -- 절대 epoch 시작 (세션 간 연결용)
  w1      REAL NOT NULL,   -- 절대 epoch 끝
  t0      REAL NOT NULL,   -- 세션 상대 초 (원본 스키마 보존)
  t1      REAL NOT NULL,
  n       INTEGER,
  tps     REAL,
  src     TEXT,
  act     INTEGER,
  extra   TEXT             -- 나머지 필드 JSON (ttft·d·kind·tot·n_prompt…)
);
CREATE INDEX IF NOT EXISTS ix_events_w0   ON events(w0);
CREATE INDEX IF NOT EXISTS ix_events_boot ON events(boot_id, w0);
"""

# 정규화 컬럼으로 뽑는 필드(나머지는 extra JSON)
_COLS = ("n", "tps", "src", "act")
_SKIP = {"i", "t0", "t1", "k"} | set(_COLS)


class Store:
    """이벤트 SQLite 영속화. 실패는 전부 삼키고 서빙에 영향 주지 않는다."""

    def __init__(self, path=DB_PATH, cfg=None, retention_days=RETENTION_DAYS,
                 started_wall=None):
        self.path = path
        self.ok = False
        self.boot_id = None
        self.err = None
        self._stop = threading.Event()
        self._thr = None
        self._n_written = 0
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            conn = self._connect()
            conn.executescript(_SCHEMA)
            # 보존 정책: 기동 시 1회 정리
            if retention_days > 0:
                cut = time.time() - retention_days * 86400.0
                conn.execute("DELETE FROM events WHERE w0 < ?", (cut,))
                conn.execute(
                    "DELETE FROM sessions WHERE ended_wall IS NOT NULL AND ended_wall < ? "
                    "AND boot_id NOT IN (SELECT DISTINCT boot_id FROM events)", (cut,))
            cur = conn.execute(
                "INSERT INTO sessions(started_wall, host, pid, model, cfg) "
                "VALUES(?,?,?,?,?)",
                # Dash 의 벽시계 기준과 **같은 값**을 쓴다 — 그래야 회고 뷰의 재기동
                # 경계선이 그 세션 첫 이벤트와 정확히 겹친다.
                (float(started_wall) if started_wall else time.time(),
                 socket.gethostname(), os.getpid(),
                 (cfg or {}).get("model"), json.dumps(cfg or {}, ensure_ascii=False)))
            self.boot_id = cur.lastrowid
            conn.commit()
            conn.close()
            self.ok = True
        except Exception as e:                       # noqa: BLE001
            self.err = f"{type(e).__name__}: {e}"

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")      # 서빙이 쓰는 동안 외부 읽기 가능
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # -- 쓰기 -------------------------------------------------------------
    def start(self, drain):
        """`drain()` = 대기 이벤트 리스트를 뽑아오는 콜백(Dash 가 제공)."""
        if not self.ok:
            return

        def run():
            conn = None
            try:
                conn = self._connect()
                while not self._stop.is_set():
                    self._stop.wait(FLUSH_S)
                    self._write(conn, drain())
                self._write(conn, drain())           # 최종 flush
                conn.execute("UPDATE sessions SET ended_wall=? WHERE boot_id=?",
                             (time.time(), self.boot_id))
                conn.commit()
            except Exception as e:                   # noqa: BLE001
                self.err = f"flusher {type(e).__name__}: {e}"
            finally:
                if conn is not None:
                    try: conn.close()
                    except Exception: pass

        self._thr = threading.Thread(target=run, daemon=True, name="dash-flush")
        self._thr.start()
        atexit.register(self.close)

    def _write(self, conn, rows):
        if not rows:
            return
        try:
            conn.executemany(
                "INSERT INTO events(boot_id,k,w0,w1,t0,t1,n,tps,src,act,extra) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)
            conn.commit()
            self._n_written += len(rows)
        except Exception as e:                       # noqa: BLE001
            self.err = f"write {type(e).__name__}: {e}"

    def close(self):
        if self._thr is not None and not self._stop.is_set():
            self._stop.set()
            self._thr.join(timeout=10)

    # -- 읽기(HTTP 스레드: 요청마다 짧은 커넥션) --------------------------
    def sessions(self, limit=100):
        if not self.ok:
            return []
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT s.boot_id, s.started_wall, s.ended_wall, s.host, s.model, "
                "       (SELECT COUNT(*) FROM events e WHERE e.boot_id=s.boot_id), "
                "       (SELECT MIN(w0) FROM events e WHERE e.boot_id=s.boot_id), "
                "       (SELECT MAX(w1) FROM events e WHERE e.boot_id=s.boot_id) "
                "FROM sessions s ORDER BY s.boot_id DESC LIMIT ?", (limit,)).fetchall()
            conn.close()
            return [{"boot_id": r[0], "started_wall": r[1], "ended_wall": r[2],
                     "host": r[3], "model": r[4], "n_events": r[5],
                     "first_w": r[6], "last_w": r[7]} for r in rows]
        except Exception as e:                       # noqa: BLE001
            self.err = f"sessions {type(e).__name__}: {e}"
            return []

    def read_range(self, w_from, w_to, limit=READ_LIMIT):
        if not self.ok:
            return [], False
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT boot_id,k,w0,w1,t0,t1,n,tps,src,act,extra FROM events "
                "WHERE w1>=? AND w0<=? ORDER BY w0 LIMIT ?",
                (w_from, w_to, limit + 1)).fetchall()
            conn.close()
        except Exception as e:                       # noqa: BLE001
            self.err = f"read {type(e).__name__}: {e}"
            return [], False
        truncated = len(rows) > limit
        out = []
        for r in rows[:limit]:
            e = {"boot": r[0], "k": r[1], "w0": r[2], "w1": r[3],
                 "t0": r[4], "t1": r[5]}
            if r[6] is not None: e["n"] = r[6]
            if r[7] is not None: e["tps"] = r[7]
            if r[8] is not None: e["src"] = r[8]
            if r[9] is not None: e["act"] = r[9]
            if r[10]:
                try: e.update(json.loads(r[10]))
                except Exception: pass
            out.append(e)
        return out, truncated

    def stats(self):
        return {"enabled": bool(self.ok), "path": self.path,
                "boot_id": self.boot_id, "written": self._n_written,
                "retention_days": RETENTION_DAYS, "err": self.err}

_MTP_RE = re.compile(r"tok/cycle=([0-9.]+)")
_DEPTH_RE = re.compile(r"depth\[([^\]]*)\]")
_DPAIR_RE = re.compile(r"d(\d+)=(\d+)/(\d+)")


class Dash:
    def __init__(self, enabled=ENABLED, cfg=None):
        self.enabled = bool(enabled)
        self.cfg = dict(cfg or {})
        self._lock = threading.Lock()
        self._ev = deque(maxlen=MAXLEN)
        self._seq = 0
        self._t0_mono = time.monotonic()
        self._t0_wall = time.time()
        self._cur = {"b": -1, "n": 0, "act": 0}
        self._pf_open = None          # 진행 중인 프리필 청크의 시작 시각
        self._n_req = 0
        self._n_fin = 0
        self._snap_hits = 0
        self._pp2_builds = 0
        self._last_ttft = None
        self._last_mtp = None
        self._live = 0
        # 영속화: 측정 경로에서는 이 리스트에 **append 한 번**만 한다.
        # 행 튜플 변환·INSERT 는 전부 플러셔 스레드에서(=gen_loop 밖).
        self.store = None
        self._persist = False
        self._pend = []
        self._pend_dropped = 0

    # ------------------------------------------------------------- 내부
    def _now(self):
        return time.monotonic() - self._t0_mono

    def _put(self, t0, t1, **kw):
        """이벤트 1건 적재. 호출자는 락을 잡지 않는다(여기서 잡는다)."""
        with self._lock:
            self._seq += 1
            kw["i"] = self._seq
            kw["t0"] = round(t0, 3)
            kw["t1"] = round(t1, 3)
            self._ev.append(kw)
            self._stage_locked(kw)

    def _stage_locked(self, ev):
        """영속화 대기열 적재. 플러셔가 죽어도 메모리를 무한히 먹지 않게 상한을 둔다."""
        if not self._persist:
            return
        if len(self._pend) < 500000:
            self._pend.append(ev)
        else:
            self._pend_dropped += 1

    def _flush_bucket_locked(self, upto_b):
        cur = self._cur
        if cur["b"] >= 0 and cur["b"] < upto_b:
            self._seq += 1
            ev = {
                "i": self._seq, "k": "dec",
                "t0": float(cur["b"]), "t1": float(cur["b"] + 1),
                "n": cur["n"], "tps": float(cur["n"]), "act": cur["act"],
            }
            self._ev.append(ev)
            self._stage_locked(ev)
            cur["b"] = -1
            cur["n"] = 0

    # --------------------------------------------------------- 기록 API
    def pf_open(self):
        """프리필 청크 구간 시작(이음새 이탈 / 빌드 시작)."""
        if self.enabled:
            self._pf_open = self._now()

    def pf_close(self, n, src="pp2"):
        """직전 pf_open 부터 지금까지를 n 토큰 프리필로 확정."""
        if not self.enabled or self._pf_open is None:
            return
        t0, t1 = self._pf_open, self._now()
        self._pf_open = None
        d = t1 - t0
        if n < PF_MIN_TOKENS or d <= 0:
            return
        self._put(t0, t1, k="pf", n=int(n), tps=round(n / d, 1), src=src)

    def pf_span(self, n, dur_s, src="bg"):
        """이미 측정된 구간을 그대로 적재(bg 카운터 델타용)."""
        if not self.enabled or n < PF_MIN_TOKENS or dur_s <= 0:
            return
        t1 = self._now()
        self._put(t1 - dur_s, t1, k="pf", n=int(n), tps=round(n / dur_s, 1), src=src)

    def dec_tick(self, n, active):
        """디코드 스텝에서 emit 된 토큰 수 누적(1초 버킷)."""
        if not self.enabled or n <= 0:
            return
        b = int(self._now())
        with self._lock:
            self._flush_bucket_locked(b)
            cur = self._cur
            if cur["b"] < 0:
                cur["b"] = b
                cur["n"] = 0
            cur["n"] += n
            cur["act"] = active

    def first_token(self, ttft_s, n_prompt):
        if not self.enabled:
            return
        self._n_req += 1
        self._last_ttft = ttft_s
        t = self._now()
        self._put(t, t, k="req", ttft=round(ttft_s, 3), n_prompt=int(n_prompt))

    def req_done(self, n_tokens, dur_s):
        if not self.enabled:
            return
        self._n_fin += 1
        t = self._now()
        tps = round(n_tokens / dur_s, 2) if dur_s > 0 else 0.0
        self._put(t, t, k="fin", n=int(n_tokens), tps=tps)

    def snap(self, kind, n=0, tot=0):
        if not self.enabled:
            return
        if kind == "hit":
            self._snap_hits += 1
        elif kind.startswith("pp2"):
            # 스냅스토어 on/off 와 무관하게 PP2 빌드는 모두 센다
            # (kind="pp2store" | "pp2build").
            self._pp2_builds += 1
        t = self._now()
        self._put(t, t, k="snap", kind=kind, n=int(n), tot=int(tot))

    def mtp_line(self, msg):
        """omlx MTP 텔레메트리 한 줄 파싱(시퀀스 종료마다 1건 — 핫패스 아님)."""
        if not self.enabled:
            return
        m = _MTP_RE.search(msg)
        if not m:
            return
        tpc = float(m.group(1))
        d = []
        dm = _DEPTH_RE.search(msg)
        if dm:
            for _i, a, tot in _DPAIR_RE.findall(dm.group(1)):
                a, tot = int(a), int(tot)
                d.append(round(a / tot, 3) if tot else 0.0)
        self._last_mtp = {"tpc": tpc, "d": d}
        t = self._now()
        self._put(t, t, k="mtp", tpc=tpc, d=d)

    def attach_mtp_logger(self):
        """`omlx.patches.mlx_lm_mtp.batch_generator` 로거에 탭을 건다.

        emit 은 시퀀스 **종료 시 1회**뿐이라 디코드 루프에 비용이 없다."""
        if not self.enabled:
            return
        import logging

        dash = self

        class _H(logging.Handler):
            def emit(self, record):
                try:
                    msg = record.getMessage()
                    if msg.startswith("MTP["):
                        dash.mtp_line(msg)
                except Exception:
                    pass

        lg = logging.getLogger("omlx.patches.mlx_lm_mtp.batch_generator")
        h = _H()
        h.setLevel(logging.INFO)
        lg.addHandler(h)

    # ------------------------------------------------------------ 읽기
    def metrics(self, since=0):
        now = self._now()
        b = int(now)
        # 락 보유 구간 = 버킷 플러시 + **리스트 복사**까지. 직렬화·집계는 밖에서.
        # (gen_loop 가 메인 스레드 = 집합연산 스레드라 HTTP 가 오래 잡으면 안 된다.)
        with self._lock:
            self._flush_bucket_locked(b)
            snap = list(self._ev)
            seq = self._seq
            cur = dict(self._cur)
        oldest = snap[0]["i"] if snap else 0
        # since 가 버퍼 앞을 벗어났으면 클라이언트에 전체 재적재를 지시
        reset = bool(since and oldest and since + 1 < oldest)
        evs = snap if reset else [e for e in snap if e["i"] > since]

        # 5초 이동 요약(뒤에서부터, 오래된 것 만나면 중단)
        w0 = now - 5.0
        pf_n = pf_d = 0.0
        dec_n = 0.0
        for e in reversed(snap):
            if e["t1"] < w0 - 120:
                break
            if e["t1"] < w0:
                continue
            if e["k"] == "pf":
                # 비율이라 창 경계에 걸친 청크도 n·duration 을 함께 세면 정확
                pf_n += e["n"]
                pf_d += e["t1"] - e["t0"]
            elif e["k"] == "dec" and e["t0"] >= w0:
                # 디코드는 고정 1초 버킷 → **창 안에 온전히 든 것만** 센다.
                # (t1>=w0 으로 받으면 걸친 버킷까지 통째로 들어와 5초 창에 6버킷이
                #  잡히며 tok/s 가 약 20% 부풀었다 — 합성 하네스에서 실측 89.4 vs 70.)
                dec_n += e["n"]
        dec_n += cur["n"] if cur["b"] >= 0 else 0
        span = max(1e-6, min(5.0, now))
        out = {
            "t_now": round(now, 3),
            "t_start_wall": self._t0_wall,
            "since": since, "next": seq, "reset": reset,
            "win_s": WINDOW_S,
            "cfg": self.cfg,
            "store": (self.store.stats() if self.store is not None
                      else {"enabled": False}),
            "now": {
                "prefill_tps": round(pf_n / pf_d, 1) if pf_d > 0 else 0.0,
                "decode_tps": round(dec_n / span, 1),
                "active": self._live,
                "ttft_ms": round(self._last_ttft * 1000, 0) if self._last_ttft else None,
                "mtp": self._last_mtp,
                "n_req": self._n_req, "n_fin": self._n_fin,
                "snap_hits": self._snap_hits, "pp2_builds": self._pp2_builds,
                "uptime_s": round(now, 1),
            },
            "events": evs,
        }
        return out

    def set_live(self, n):
        """활성 세션 수를 gen_loop 가 직접 알려준다(디코드 정지 중에도 정확)."""
        self._live = int(n)

    # ------------------------------------------------------------ 영속화
    def attach_store(self, path=DB_PATH, retention_days=RETENTION_DAYS):
        """SQLite 영속화 시작. 실패해도 라이브 대시보드는 그대로 동작한다."""
        if not self.enabled or not PERSIST:
            return None
        st = Store(path, cfg=self.cfg, retention_days=retention_days,
                   started_wall=self._t0_wall)
        if not st.ok:
            return st
        self.store = st
        self._persist = True
        st.start(self._drain_pending)
        return st

    def _drain_pending(self):
        """플러셔 스레드 전용: 대기 이벤트를 INSERT 행 튜플로 변환해 반환.

        락은 **리스트 교체까지만** 잡고 변환은 밖에서 한다(측정 스레드 차단 최소화)."""
        with self._lock:
            if not self._pend:
                return []
            batch, self._pend = self._pend, []
        boot, base = self.store.boot_id, self._t0_wall
        rows = []
        for e in batch:
            extra = {k: v for k, v in e.items() if k not in _SKIP}
            rows.append((
                boot, e["k"], base + e["t0"], base + e["t1"], e["t0"], e["t1"],
                e.get("n"), e.get("tps"), e.get("src"), e.get("act"),
                json.dumps(extra, ensure_ascii=False) if extra else None,
            ))
        return rows

    def close(self):
        if self.store is not None:
            self.store.close()

    # ------------------------------------------------------- 회고(DB) 조회
    def history(self, w_from, w_to):
        """절대 epoch 구간 조회 — 라이브 링버퍼를 건드리지 않고 DB 에서만 읽는다."""
        if self.store is None or not self.store.ok:
            return {"error": "persist off", "events": [], "sessions": []}
        evs, truncated = self.store.read_range(w_from, w_to)
        sess = [s for s in self.store.sessions()
                if (s["last_w"] or s["started_wall"]) >= w_from
                and s["started_wall"] <= w_to]
        pf_n = pf_d = dec_n = 0.0
        ttfts, tpcs = [], []
        for e in evs:
            if e["k"] == "pf":
                pf_n += e.get("n", 0); pf_d += e["w1"] - e["w0"]
            elif e["k"] == "dec":
                dec_n += e.get("n", 0)
            elif e["k"] == "req" and e.get("ttft") is not None:
                ttfts.append(e["ttft"])
            elif e["k"] == "mtp" and e.get("tpc") is not None:
                tpcs.append(e["tpc"])
        span = max(1e-6, w_to - w_from)
        return {
            "mode": "history", "from": w_from, "to": w_to,
            "truncated": truncated,
            "sessions": sess,
            "summary": {
                "n_events": len(evs),
                "prefill_tps_avg": round(pf_n / pf_d, 1) if pf_d > 0 else 0.0,
                "prefill_tokens": int(pf_n),
                "decode_tokens": int(dec_n),
                "decode_tps_mean_active": round(dec_n / span, 2),
                "n_req": len(ttfts),
                "ttft_p50": round(sorted(ttfts)[len(ttfts) // 2], 3) if ttfts else None,
                "ttft_max": round(max(ttfts), 3) if ttfts else None,
                "mtp_tpc_mean": round(sum(tpcs) / len(tpcs), 3) if tpcs else None,
            },
            "events": evs,
        }


HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DSv4-Flash TP2 — 프리필/디코드 실시간 감시</title>
<style>
:root{
  --bg:#f6f7f9; --card:#ffffff; --ink:#111827; --muted:#6b7280; --line:#e5e7eb;
  --pf:#3b82f6; --pfw:#93c5fd; --dec:#f59e0b; --both:rgba(15,23,42,.075);
  --mtp:#8b5cf6; --ok:#059669;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0f1216; --card:#161a20; --ink:#e6e8eb; --muted:#9aa3af; --line:#262c35;
  --pf:#60a5fa; --pfw:#1e4b8f; --dec:#fbbf24; --both:rgba(255,255,255,.07);
  --mtp:#a78bfa; --ok:#34d399;
}}
*{box-sizing:border-box}
/* .seg/.hbar 이 display:inline-flex|flex 를 갖고 있어 UA 의 [hidden]{display:none}
   을 특이도로 이긴다 — 명시적으로 눌러야 모드 전환 시 컨트롤이 겹치지 않는다. */
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);
  font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",'Helvetica Neue',
  'Apple SD Gothic Neo',sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1360px;margin:0 auto;padding:18px 20px 32px}
header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:14px}
h1{font-size:16px;font-weight:650;margin:0;letter-spacing:-.01em}
.badges{display:flex;gap:6px;flex-wrap:wrap}
.badge{font-size:11px;padding:2px 7px;border:1px solid var(--line);border-radius:999px;
  color:var(--muted);background:var(--card);font-variant-numeric:tabular-nums}
.badge.on{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 40%,var(--line))}
.spacer{flex:1}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--card)}
.seg button{border:0;background:transparent;color:var(--muted);font:inherit;font-size:12px;
  padding:5px 11px;cursor:pointer}
.seg button+button{border-left:1px solid var(--line)}
.seg button[aria-pressed=true]{background:var(--pf);color:#fff}
.hbar{display:flex;align-items:center;gap:10px;margin:0 0 12px;font-size:12px;
  color:var(--muted);flex-wrap:wrap}
.hbar select{font:inherit;font-size:12px;padding:4px 8px;border:1px solid var(--line);
  border-radius:7px;background:var(--card);color:var(--ink);max-width:520px}
.hbar #hinfo{font-variant-numeric:tabular-nums}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}
@media(max-width:760px){.tiles{grid-template-columns:repeat(2,1fr)}}
.tile{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:12px 14px}
.tile .k{font-size:11px;color:var(--muted);letter-spacing:.02em}
.tile .v{font-size:26px;font-weight:640;font-variant-numeric:tabular-nums;
  letter-spacing:-.02em;margin-top:3px;line-height:1.1}
.tile .u{font-size:12px;font-weight:450;color:var(--muted);margin-left:4px}
.tile .s{font-size:11px;color:var(--muted);margin-top:2px;font-variant-numeric:tabular-nums}
.card{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:12px 14px 8px}
.card+.card{margin-top:12px}
.legend{display:flex;gap:16px;align-items:center;flex-wrap:wrap;font-size:11.5px;
  color:var(--muted);margin-bottom:8px}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:-1px;margin-right:5px}
.note{font-size:11.5px;color:var(--muted);margin-top:6px;line-height:1.5}
.note b{color:var(--ink);font-weight:600}
canvas{display:block;width:100%}
footer{margin-top:14px;font-size:11px;color:var(--muted);line-height:1.6}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}
</style></head><body><div class="wrap">

<header>
  <h1>DSv4-Flash TP2 — 프리필/디코드 실시간 감시</h1>
  <div class="badges" id="badges"></div>
  <div class="spacer"></div>
  <div class="seg" id="mode">
    <button data-m="live" aria-pressed="true">라이브</button>
    <button data-m="history">누적</button>
  </div>
  <div class="seg" id="win">
    <button data-w="300">5분</button>
    <button data-w="900">15분</button>
    <button data-w="3600" aria-pressed="true">60분</button>
  </div>
  <div class="seg" id="hwin" hidden>
    <button data-h="3600">1시간</button>
    <button data-h="21600">6시간</button>
    <button data-h="86400" aria-pressed="true">24시간</button>
    <button data-h="604800">7일</button>
  </div>
</header>
<div id="hbar" class="hbar" hidden>
  <label>세션</label>
  <select id="sess"><option value="">전체 구간</option></select>
  <span id="hinfo"></span>
</div>

<div class="tiles">
  <div class="tile"><div class="k">프리필 tok/s</div>
    <div class="v" id="t_pf">–</div><div class="s" id="s_pf">최근 5초 이동</div></div>
  <div class="tile"><div class="k">디코드 tok/s</div>
    <div class="v" id="t_dec">–</div><div class="s" id="s_dec">최근 5초 이동</div></div>
  <div class="tile"><div class="k">활성 세션</div>
    <div class="v" id="t_act">–</div><div class="s" id="t_reqs">–</div></div>
  <div class="tile"><div class="k">최근 TTFT</div>
    <div class="v" id="t_ttft">–</div><div class="s" id="t_snap">–</div></div>
</div>

<div class="card">
  <div class="legend">
    <span><i class="sw" style="background:var(--pf)"></i>프리필 tok/s</span>
    <span><i class="sw" style="background:var(--dec)"></i>생성(디코드) tok/s</span>
    <span><i class="sw" style="background:var(--both);border:1px solid var(--line)"></i>동시 구간</span>
    <span class="spacer"></span>
    <span id="scales"></span>
  </div>
  <canvas id="tl" height="330"></canvas>
  <div class="note"><b>두 지표는 축이 다르므로 위아래 높이를 서로 견주지 말 것.</b>
    상단 트랙(프리필)과 하단 트랙(생성)은 각각 자기 최대치로 정규화된다.
    회색 음영은 프리필과 디코드가 <b>실제로 겹쳐 돈</b> 구간 — 우리 스택은 HOL
    인터리브(<code>DSV4_PP2_INTERLEAVE=1</code>)로 2048-청크 이음새마다 라이브 배치
    디코드를 끼워 넣으므로 이 겹침이 정상이다.</div>
</div>

<div class="card">
  <div class="legend"><span><i class="sw" style="background:var(--mtp)"></i>MTP tok/cycle
    <span id="mtpnow" style="font-variant-numeric:tabular-nums"></span></span></div>
  <canvas id="mtp" height="86"></canvas>
</div>

<footer id="foot"></footer>
</div>
<script>
(function(){
"use strict";
var evs=[], next=0, tNow=0, tRef=0, perfRef=performance.now(), cfg={}, nowS={}, win=3600;
var mode='live', hwin=86400, hist=null, sessions=[], selBoot='', storeS={};
var tl=document.getElementById('tl'), mtp=document.getElementById('mtp');
function css(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim();}
function fmt(v,d){return v==null?'–':Number(v).toLocaleString('en-US',
  {minimumFractionDigits:d,maximumFractionDigits:d});}
function clock(ep){var d=new Date(ep*1000);
  return ('0'+d.getHours()).slice(-2)+':'+('0'+d.getMinutes()).slice(-2);}
function stamp(ep){if(!ep)return '–';var d=new Date(ep*1000);
  return (d.getMonth()+1)+'/'+d.getDate()+' '+clock(ep);}

function press(seg,attr,val){
  var hit=null;
  [].forEach.call(seg.querySelectorAll('button'),function(x){
    var on=(x.dataset[attr]==String(val)); if(on)hit=x;
    x.setAttribute('aria-pressed', on?'true':'false');});
  return hit;
}
function setWin(w){ if(press(document.getElementById('win'),'w',w)) win=w; }
function setHwin(h){ if(press(document.getElementById('hwin'),'h',h)){ hwin=h; loadHistory(); } }

function setMode(m){
  mode = (m==='history')?'history':'live';
  press(document.getElementById('mode'),'m',mode);
  var h=(mode==='history');
  document.getElementById('win').hidden=h;
  document.getElementById('hwin').hidden=!h;
  document.getElementById('hbar').hidden=!h;
  if(h){ loadSessions(); loadHistory(); }
}
document.getElementById('win').addEventListener('click',function(e){
  var b=e.target.closest('button'); if(b) setWin(+b.dataset.w);});
document.getElementById('hwin').addEventListener('click',function(e){
  var b=e.target.closest('button'); if(b) setHwin(+b.dataset.h);});
document.getElementById('mode').addEventListener('click',function(e){
  var b=e.target.closest('button'); if(b) setMode(b.dataset.m);});
document.getElementById('sess').addEventListener('change',function(){
  selBoot=this.value; loadHistory();});

function loadSessions(){
  fetch('/metrics?sessions=1',{cache:'no-store'}).then(function(r){return r.json();})
  .then(function(j){
    sessions=j.sessions||[]; storeS=j.store||{};
    var s=document.getElementById('sess'), h='<option value="">전체 구간</option>';
    sessions.forEach(function(x){
      h+='<option value="'+x.boot_id+'">#'+x.boot_id+'  '+stamp(x.started_wall)+
         ' → '+(x.ended_wall?stamp(x.ended_wall):'(진행 중)')+
         '   ·  '+(x.n_events||0).toLocaleString()+' ev</option>';});
    s.innerHTML=h; s.value=selBoot;
  }).catch(function(){});
}

function loadHistory(){
  var to, from;
  if(selBoot){
    var s=null;
    sessions.forEach(function(x){ if(String(x.boot_id)===String(selBoot)) s=x; });
    if(s){ from=(s.first_w||s.started_wall)-2; to=(s.last_w||s.ended_wall||Date.now()/1000)+2; }
  }
  if(from==null){ to=Date.now()/1000; from=to-hwin; }
  fetch('/metrics?from='+from.toFixed(3)+'&to='+to.toFixed(3),{cache:'no-store'})
  .then(function(r){return r.json();}).then(function(j){
    hist=j; hist.from=from; hist.to=to; tiles(); badges();
  }).catch(function(){});
}

// #history / #w=300 / #h=86400 초기 상태(스크린샷·북마크용)
function fromHash(){
  var hs=location.hash||'';
  var w=/(?:^|[#&])w=(\d+)/.exec(hs); if(w) setWin(+w[1]);
  var hh=/(?:^|[#&])h=(\d+)/.exec(hs); if(hh) hwin=+hh[1];
  setMode(/(?:^|[#&])history\b/.test(hs) ? 'history' : 'live');
  if(hh) press(document.getElementById('hwin'),'h',hwin);
}
window.addEventListener('hashchange',fromHash);

function poll(){
  // 라이브 폴링은 인메모리 링버퍼만 읽는다(회고 모드에서도 tile 을 위해 계속 돈다).
  fetch('/metrics?since='+next,{cache:'no-store'}).then(function(r){return r.json();})
  .then(function(j){
    if(j.reset) evs=[];
    if(j.events && j.events.length) evs=evs.concat(j.events);
    next=j.next; tRef=j.t_now; perfRef=performance.now();
    cfg=j.cfg||{}; nowS=j.now||{}; storeS=j.store||storeS;
    var cut=j.t_now-3700;
    if(evs.length && evs[0].t1<cut) evs=evs.filter(function(e){return e.t1>=cut;});
    if(mode==='live'){ tiles(); badges(); }
  }).catch(function(){}).then(function(){setTimeout(poll,1000);});
}

function badges(){
  var b=document.getElementById('badges'), h='';
  function add(t,on){h+='<span class="badge'+(on?' on':'')+'">'+t+'</span>';}
  add('PP2 '+(cfg.pp2?'ON':'off'),cfg.pp2);
  add('HOL 인터리브 '+(cfg.interleave?'ON':'off'),cfg.interleave);
  add('스냅스토어 '+(cfg.snapstore?'ON':'off'),cfg.snapstore);
  if(cfg.chunk) add('chunk '+cfg.chunk);
  if(cfg.min_tokens) add('PP2≥'+cfg.min_tokens+' tok');
  if(cfg.mtp_depth) add('MTP depth '+cfg.mtp_depth);
  add(storeS.enabled?('영속 ON · boot #'+storeS.boot_id):'영속 off', !!storeS.enabled);
  b.innerHTML=h;
  var up=nowS.uptime_s||0;
  document.getElementById('foot').innerHTML=
    (storeS.enabled?('영속: <code>'+storeS.path+'</code> · 기록 '+
      (storeS.written||0).toLocaleString()+'건 · 보존 '+(storeS.retention_days||30)+'일 · '+
      '재기동해도 <b>누적</b> 탭에서 이전 세션이 그대로 보인다.'+
      (storeS.err?(' <b>경고: '+storeS.err+'</b>'):'')+'<br>'):'')+
    'rank0 계측 · 랭크1 하단-슬라이스 프리필은 <b>rank0 이음새 시점으로 근사</b>'+
    '(PP2 는 청크마다 랭크1→랭크0 활성값 전송으로 동기되므로 이음새 간격이 곧 '+
    '양 랭크 합산 청크 시간). 프리필 tok/s 는 이음새 <b>본문</b>(인터리브 디코드 '+
    '스텝)을 뺀 순 프리필 시간 기준 — 디코드 버킷과 이중 계상하지 않는다. '+
    'gauge 갱신 1s · 버퍼 '+Math.round((cfg.win_s||3600)/60)+'분 · uptime '+
    Math.floor(up/60)+'m'+Math.floor(up%60)+'s';
}

function setK(tileId,txt){
  var v=document.getElementById(tileId);
  if(v&&v.parentNode) v.parentNode.querySelector('.k').textContent=txt;
}
function tiles(){
  if(mode==='history'){
    var s=(hist&&hist.summary)||{};
    setK('t_pf','프리필 평균 tok/s'); setK('t_dec','누적 생성 토큰');
    setK('t_act','요청 수'); setK('t_ttft','TTFT p50');
    document.getElementById('s_pf').textContent='구간 전체 가중평균';
    document.getElementById('s_dec').textContent='구간 합계';
    document.getElementById('t_pf').innerHTML=fmt(s.prefill_tps_avg,0)+'<span class="u">tok/s</span>';
    document.getElementById('t_dec').innerHTML=fmt(s.decode_tokens,0)+'<span class="u">tok</span>';
    document.getElementById('t_act').textContent=(s.n_req==null?'–':s.n_req);
    document.getElementById('t_reqs').textContent=
      '프리필 '+fmt(s.prefill_tokens,0)+' tok · 이벤트 '+fmt(s.n_events,0);
    document.getElementById('t_ttft').innerHTML=s.ttft_p50==null?'–':
      fmt(s.ttft_p50,2)+'<span class="u">s</span>';
    document.getElementById('t_snap').textContent=
      s.ttft_max==null?'–':('최대 '+fmt(s.ttft_max,2)+'s');
    document.getElementById('mtpnow').textContent = s.mtp_tpc_mean!=null?
      ('  평균 '+s.mtp_tpc_mean.toFixed(2)) : '  데이터 없음';
    var hi=document.getElementById('hinfo');
    hi.innerHTML = hist? (stamp(hist.from)+' → '+stamp(hist.to)+
      '  ·  세션 '+((hist.sessions||[]).length)+'개  ·  이벤트 '+fmt(s.n_events,0)+
      (hist.truncated?'  ·  <b>상한 초과(구간을 좁히세요)</b>':'')) : '';
    return;
  }
  setK('t_pf','프리필 tok/s'); setK('t_dec','디코드 tok/s');
  setK('t_act','활성 세션'); setK('t_ttft','최근 TTFT');
  document.getElementById('s_pf').textContent='최근 5초 이동';
  document.getElementById('s_dec').textContent='최근 5초 이동';
  document.getElementById('t_pf').innerHTML=fmt(nowS.prefill_tps,0)+'<span class="u">tok/s</span>';
  document.getElementById('t_dec').innerHTML=fmt(nowS.decode_tps,1)+'<span class="u">tok/s</span>';
  document.getElementById('t_act').textContent=nowS.active==null?'–':nowS.active;
  document.getElementById('t_reqs').textContent=
    '요청 '+(nowS.n_req||0)+' 수신 · '+(nowS.n_fin||0)+' 완료';
  document.getElementById('t_ttft').innerHTML=nowS.ttft_ms==null?'–':
    fmt(nowS.ttft_ms/1000,2)+'<span class="u">s</span>';
  document.getElementById('t_snap').textContent=
    '스냅숏 적중 '+(nowS.snap_hits||0)+' · PP2 빌드 '+(nowS.pp2_builds||0);
  var m=nowS.mtp;
  document.getElementById('mtpnow').textContent = m?
    ('  '+m.tpc.toFixed(2)+'   d1/d2/d3 '+(m.d||[]).map(function(x){
      return (x*100).toFixed(0)+'%';}).join(' / ')) : '  대기 중';
}

function fit(c){
  var r=window.devicePixelRatio||1, w=c.clientWidth;
  if(c.width!==Math.round(w*r)||c._h!==c.height){c.width=Math.round(w*r);c._h=c.height;}
  var x=c.getContext('2d'); x.setTransform(r,0,0,r,0,0);
  x.clearRect(0,0,w,c.height); return {x:x,w:w,h:c.height};
}

function draw(){
  tNow = tRef + (performance.now()-perfRef)/1000;
  var C=fit(tl), x=C.x, W=C.w, H=C.h;
  var L=68, R=10, T=10, B=24, iw=W-L-R;
  var gap=26, th=(H-T-B-gap)/2;
  var pfTop=T, pfBase=T+th, decTop=pfBase+gap, decBase=decTop+th;
  var line=css('--line'), muted=css('--muted');

  // ── 축 선택 ────────────────────────────────────────────────────────
  // 라이브: 세션 상대 초(t0/t1), 창은 [now-win, now].
  // 누적  : 절대 epoch(w0/w1) — 재기동으로 상대시각이 리셋돼도 이어진다.
  var HIS=(mode==='history' && hist), src, t0, t1, KA, KB;
  if(HIS){ src=hist.events||[]; t0=hist.from; t1=hist.to; KA='w0'; KB='w1'; }
  else   { src=evs; t1=tNow; t0=tNow-win; KA='t0'; KB='t1'; }
  var span=Math.max(1e-6,t1-t0), sx=iw/span;

  // ── 가시 이벤트 수집 + 픽셀 열 집계 ────────────────────────────────
  var cols=Math.max(1,Math.round(iw));
  var pfCol=new Float32Array(cols), decCol=new Float32Array(cols);
  var pfAct=new Uint8Array(cols), decAct=new Uint8Array(cols);
  var pfMax=1, decMax=1, mtpPts=[];
  for(var i=0;i<src.length;i++){
    var e=src[i];
    if(e[KB]<t0||e[KA]>t1) continue;
    if(e.k==='mtp'){ mtpPts.push(e); continue; }
    if(e.k!=='pf'&&e.k!=='dec') continue;
    var a=Math.max(0,Math.floor((e[KA]-t0)*sx)), b=Math.min(cols-1,Math.ceil((e[KB]-t0)*sx));
    if(b<a) b=a;
    for(var c=a;c<=b;c++){
      if(e.k==='pf'){ if(e.tps>pfCol[c])pfCol[c]=e.tps; pfAct[c]=1; }
      else { if(e.tps>decCol[c])decCol[c]=e.tps; if(e.n>0)decAct[c]=1; }
    }
    if(e.k==='pf'){ if(e.tps>pfMax)pfMax=e.tps; } else if(e.tps>decMax)decMax=e.tps;
  }
  function nice(v){var p=Math.pow(10,Math.floor(Math.log10(v)));var m=v/p;
    return (m<=1?1:m<=2?2:m<=5?5:10)*p;}
  pfMax=nice(pfMax*1.12); decMax=nice(decMax*1.12);
  document.getElementById('scales').innerHTML=
    '상단 축 0–'+pfMax.toLocaleString()+' tok/s  ·  하단 축 0–'+decMax.toLocaleString()+' tok/s';

  // ── 동시 구간 음영(바보다 먼저 = 뒤에) ────────────────────────────
  x.fillStyle=css('--both');
  var run=-1;
  for(var c=0;c<=cols;c++){
    var on = c<cols && pfAct[c] && decAct[c];
    if(on && run<0) run=c;
    if(!on && run>=0){ x.fillRect(L+run,pfTop,Math.max(1,c-run),decBase-pfTop); run=-1; }
  }

  // ── 그리드 + 트랙 베이스라인 ──────────────────────────────────────
  x.strokeStyle=line; x.lineWidth=1; x.font='10.5px ui-monospace,Menlo,monospace';
  [[pfTop,pfBase,pfMax,'프리필'],[decTop,decBase,decMax,'생성']].forEach(function(tr){
    var top=tr[0],base=tr[1],mx=tr[2];
    for(var k=0;k<=2;k++){
      var yy=Math.round(base-(base-top)*k/2)+.5;
      x.globalAlpha=k?.55:1; x.beginPath(); x.moveTo(L,yy); x.lineTo(W-R,yy); x.stroke();
      x.globalAlpha=1; x.fillStyle=muted; x.textAlign='right';
      x.fillText(Math.round(mx*k/2).toLocaleString(),L-7,yy+3.5);
    }
    x.save(); x.translate(14,(top+base)/2); x.rotate(-Math.PI/2);
    x.textAlign='center'; x.fillStyle=muted;
    x.font='11px -apple-system,sans-serif'; x.fillText(tr[3],0,0); x.restore();
    x.font='10.5px ui-monospace,Menlo,monospace';
  });

  // ── 바 ────────────────────────────────────────────────────────────
  x.fillStyle=css('--pf');
  for(var c=0;c<cols;c++) if(pfCol[c]>0){
    var hh=Math.max(1,(pfCol[c]/pfMax)*(pfBase-pfTop));
    x.fillRect(L+c,pfBase-hh,1,hh);
  }
  x.fillStyle=css('--dec');
  for(var c=0;c<cols;c++) if(decCol[c]>0){
    var hh2=Math.max(1,(decCol[c]/decMax)*(decBase-decTop));
    x.fillRect(L+c,decBase-hh2,1,hh2);
  }

  // ── 재기동 경계(누적 뷰) ──────────────────────────────────────────
  if(HIS && hist.sessions){
    x.save(); x.setLineDash([3,3]); x.strokeStyle=css('--mtp'); x.lineWidth=1;
    x.fillStyle=css('--mtp'); x.textAlign='left';
    x.font='10px ui-monospace,Menlo,monospace';
    hist.sessions.forEach(function(s){
      var w=s.started_wall; if(w<t0||w>t1) return;
      var px=Math.round(L+(w-t0)*sx)+.5;
      x.beginPath(); x.moveTo(px,pfTop); x.lineTo(px,decBase); x.stroke();
      x.fillText('#'+s.boot_id, px+3, pfTop+9);
    });
    x.restore();
  }

  // ── 시간축 ────────────────────────────────────────────────────────
  x.fillStyle=muted; x.textAlign='center';
  if(HIS){
    var nT=8, stepS=span/nT;
    var showDate = span>86400*1.5;
    for(var q=0;q<=nT;q++){
      var w=t0+stepS*q, px=L+(w-t0)*sx;
      x.globalAlpha=.5; x.beginPath();
      x.moveTo(Math.round(px)+.5,decBase); x.lineTo(Math.round(px)+.5,decBase+4); x.stroke();
      x.globalAlpha=1;
      var d=new Date(w*1000);
      x.fillText(showDate?((d.getMonth()+1)+'/'+d.getDate()):clock(w),px,decBase+16);
    }
  } else {
    var stepM = win<=300?1:(win<=900?3:10);
    for(var m=0;m*60<=win;m+=stepM){
      var px2=L+iw-m*60*sx;
      if(px2<L-1) break;
      x.globalAlpha=.5; x.beginPath();
      x.moveTo(Math.round(px2)+.5,decBase); x.lineTo(Math.round(px2)+.5,decBase+4); x.stroke();
      x.globalAlpha=1;
      x.fillText(m===0?'지금':('-'+m+'m'),px2,decBase+16);
    }
  }

  drawMtp(mtpPts,t0,sx,HIS?'w0':'t0');
  requestAnimationFrame(draw);
}

function drawMtp(pts,t0,sx,KA){
  KA=KA||'t0';
  var C=fit(mtp), x=C.x, W=C.w, H=C.h, L=54, R=10, T=8, B=14;
  var muted=css('--muted');
  var lo=1, hi=4;
  for(var i=0;i<pts.length;i++){ if(pts[i].tpc>hi) hi=Math.ceil(pts[i].tpc); }
  x.strokeStyle=css('--line'); x.lineWidth=1;
  x.font='10.5px ui-monospace,Menlo,monospace'; x.textAlign='right'; x.fillStyle=muted;
  [lo,hi].forEach(function(v){
    var yy=Math.round(H-B-(v-lo)/(hi-lo)*(H-T-B))+.5;
    x.beginPath(); x.moveTo(L,yy); x.lineTo(W-R,yy); x.stroke();
    x.fillText(v.toFixed(1),L-7,yy+3.5);
  });
  if(!pts.length){ x.textAlign='left';
    x.fillText('MTP 텔레메트리 대기 (시퀀스 종료 시 1건)',L+6,T+13); return; }
  x.strokeStyle=css('--mtp'); x.lineWidth=1.6; x.beginPath();
  pts.forEach(function(p,i){
    var px=L+(p[KA]-t0)*sx, py=H-B-(Math.min(hi,Math.max(lo,p.tpc))-lo)/(hi-lo)*(H-T-B);
    i?x.lineTo(px,py):x.moveTo(px,py);
  });
  x.stroke();
  x.fillStyle=css('--mtp');
  pts.forEach(function(p){
    var px=L+(p[KA]-t0)*sx, py=H-B-(Math.min(hi,Math.max(lo,p.tpc))-lo)/(hi-lo)*(H-T-B);
    x.beginPath(); x.arc(px,py,2.1,0,6.284); x.fill();
  });
}

fromHash(); poll(); requestAnimationFrame(draw);
})();
</script></body></html>
"""


def html_bytes():
    return HTML.encode("utf-8")


def json_bytes(obj):
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")
