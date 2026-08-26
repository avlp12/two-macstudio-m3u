"""dash_metrics 오프라인 재생 서버 — 문서용 스크린샷 전용, GPU/서빙 무관.

목적: `~/dsv4flash/metrics/serving_metrics.sqlite` 의 오늘 검증 런은 95건/71초로
너무 짧아 5분 라이브 창·1시간 누적 창을 채우지 못한다(실측 스크린샷이 빈약한 이유).
이 스크립트는 **dash_metrics.Dash 의 실제 공개 API**(pf_open/pf_close/pf_span/
dec_tick/first_token/req_done/snap/mtp_line)만으로 이벤트를 주입해, 그 진짜
집계·HTML 코드가 렌더한 화면을 재현한다. HTML 문자열도, metrics()/history() 의
집계 로직도 이 스크립트는 전혀 건드리지 않는다 — dash_metrics.py 는 read-only.

시간축 트릭: 각 공개 API 호출은 내부적으로 `self._now() = monotonic() - _t0_mono`
를 읽어 이벤트의 t0/t1 을 찍는다. 실제로 몇 분씩 sleep 하는 대신, 호출 직전마다
`_t0_mono` 를 순간 이동시켜 "지금이 임의의 과거/미래 상대시각이다"를 만든다
(`set_now()`). 세션의 절대시각 위치(`_t0_wall`)도 같은 방식으로 되감는다 —
이 두 필드는 dash_metrics.py 자신의 두 개 사전 검증 하네스
(~/dsv4flash/align/logs/dash_{synth,persist}_harness.py)에서도 이미 쓰인,
이 코드베이스에서 검증된 패턴이다. 다만 그 하네스들은 이벤트 딕셔너리를 직접
`_ev`에 append 했던 반면, 여기서는 시간축만 조작하고 이벤트 생성 자체는 반드시
공개 API 호출로 한다(요청 사항).

데이터 출처 판단: 오늘 실측 트레이스(71초, 이벤트 95건, 세션 2개)를 검토한 결과
(a) 실측 그대로 재생은 라이브 5분 창의 ~24%만 채우고, 누적 뷰는 두 세션이 벽시계
    로 117초 안에 몰려 있어 사용 가능한 최소 창(1시간)에서도 사실상 한 픽셀로
    뭉갠다 → "빈약함" 판정, (b) 로 간다.
(b) 시나리오는 이 실측 트레이스에서 뽑은 상수 그대로 쓴다: PP2 청크 2048×7
    (첫 청크 저속 640-650, 이후 1130-1260 tok/s), TTFT 13.7/13.64s, 스냅숏 히트
    TTFT 0.107s, 단일스트림 디코드 46 tok/s, MTP d1/d2/d3 ≈ .78/.62/.41 →
    tok/cycle = 1+d1+d1·d2+d1·d2·d3 ≈ 2.46(스펙 2.4-2.7 대역). 동시성만 8-스트림
    까지 인위적으로 쌓아 올린다(실측엔 최대 act=6까지만 나옴).
    화면 표식: 대시보드가 이미 그리는 필드 두 곳에 심는다 —
    ① 영속 DB 경로 자체가 `dash_demo_replay.sqlite`(푸터의 <code> 경로에 그대로 노출)
    ② `store.err` 에 DEMO_NOTE 문자열 → 푸터에 **볼드 경고**로 렌더(HTML 수정 없이
       기존 "storeS.err ? 경고: ..." 분기를 그대로 이용).

세션 구성: 같은 데모 DB 파일에 두 번 "기동"한다 — 세션1(과거, 40분 전, 짧고
간단한 재생, close() 로 ended_wall 확정) → 세션2(현재, 30분 전 시작, 8-스트림
피크까지 쌓는 풍부한 시나리오, 계속 "진행 중"인 상태로 라이브 서빙). 누적 탭은
1시간 창에서 두 세션 모두 보이고(재기동 경계 점선), 라이브 탭은 세션2의 링버퍼를
5분 창으로 보여준다.
"""
import http.server
import json
import os
import random
import socket
import sys
import time
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, "/Users/Shared/tp2")
import dash_metrics as dm  # noqa: E402  (read-only import, 소스 미수정)

SCRATCH = "/Users/Shared/tp2"
DEMO_DB = os.path.join(SCRATCH, "dash_demo_replay.sqlite")
for _suf in ("", "-wal", "-shm"):
    try:
        os.remove(DEMO_DB + _suf)
    except FileNotFoundError:
        pass

CFG = {"pp2": True, "interleave": True, "snapstore": True, "chunk": 2048,
       "min_tokens": 4096, "mtp_depth": 3, "model": "deepseek-v4-flash-tp2",
       "win_s": dm.WINDOW_S}

DEMO_NOTE = ("DEMO REPLAY — 합성 시나리오(실측 DSv4-Flash TP2 상수 기반), "
             "실제 서빙 트래픽 아님. 생성: dash_demo_replay.py")


# ───────────────────────────────────────────────── 시간축 조작 + 이벤트 DSL
def set_now(dash, t):
    """dash._now() 가 정확히 t 를 돌려주도록 모노토닉 기준을 순간 이동."""
    dash._t0_mono = time.monotonic() - t


def mtp_msg(tpc, d):
    a = [int(round(x * 1000)) for x in d]
    return (f"MTP tok/cycle={tpc:.2f} depth[d1={a[0]}/1000 "
            f"d2={a[1]}/1000 d3={a[2]}/1000]")


def tpc_from_d(d):
    return 1.0 + d[0] + d[0] * d[1] + d[0] * d[1] * d[2]


def pp2_build(evs, start_t, chunks, src="pp2"):
    """PP2 이음새 청크 열(pf_open/pf_close 쌍)을 start_t 부터 순서대로 쌓는다."""
    t = start_t
    for n, tps in chunks:
        dur = n / tps
        evs.append((t, "pf_open", (), {}))
        t2 = t + dur
        evs.append((t2, "pf_close", (n,), {"src": src}))
        t = t2
    return t  # 마지막 청크의 t1


def bg_prefill(evs, end_t, n, tps, src="bg"):
    dur = n / tps
    evs.append((end_t, "pf_span", (n, dur), {"src": src}))


def req(evs, t, ttft, n_prompt):
    evs.append((t, "first_token", (ttft, n_prompt), {}))


def fin(evs, t, n_tokens, dur_s):
    evs.append((t, "req_done", (int(round(n_tokens)), dur_s), {}))


def snap(evs, t, kind, n=0, tot=0):
    evs.append((t, "snap", (kind,), {"n": n, "tot": tot}))


def mtp(evs, t, d):
    evs.append((t, "mtp_line", (mtp_msg(tpc_from_d(d), d),), {}))


THRU = {0: 0, 1: 46, 2: 64, 3: 78, 4: 90, 5: 99, 6: 106, 7: 111, 8: 115, 9: 117}


def dec_schedule(evs, intervals, t_end, rng):
    """구간 목록(동시 디코드 스트림)에서 초당 활성 개수를 세고, 실측 스케일링
    (act→집계 tok/s)을 적용해 dec_tick 이벤트를 만든다."""
    for sec in range(0, int(t_end) + 1):
        mid = sec + 0.5
        act = sum(1 for (s, e) in intervals if s <= mid < e)
        if act <= 0:
            continue
        base = THRU.get(act, 118)
        n = max(1, round(base * (1 + rng.uniform(-0.06, 0.06))))
        evs.append((mid, "dec_tick", (n, act), {}))


def replay(dash, evs):
    """정렬 후 순서대로 공개 API 호출 — 여기서만 시간을 조작한다."""
    for t, call, args, kwargs in sorted(evs, key=lambda e: e[0]):
        set_now(dash, t)
        getattr(dash, call)(*args, **kwargs)


def build_dash(wall_offset_s, evs, close_after):
    """타임라인 전체를 재생해 영속 DB(boot 세션)에 남긴다. 누적/세션 탭 전용
    (라이브 스냅숏은 build_live_snapshot 참조 — 이 함수로 만든 인스턴스를 그대로
    라이브 서빙에 쓰면 안 되는 이유가 있다, 아래 함수 docstring 참조)."""
    d = dm.Dash(enabled=True, cfg=dict(CFG))
    d._t0_wall = time.time() - wall_offset_s      # 세션 절대시각을 과거로 되감기
    st = d.attach_store(path=DEMO_DB)
    if st is None or not st.ok:
        raise RuntimeError(f"attach_store 실패: {getattr(st, 'err', None)}")
    replay(d, evs)
    t_end = max((t for t, *_ in evs), default=0.0)
    t_freeze = t_end + 2.0
    set_now(d, t_freeze)
    d.metrics(since=0)          # 마지막 dec 버킷을 공개 API 경로로 강제 플러시
    d.set_live(0)
    if close_after:
        d.close()               # 최종 flush + ended_wall 기록
    return d, t_freeze


def build_live_snapshot(wall_offset_s, evs, live_t, live_count, store_src):
    """"지금" = live_t 인 라이브 스냅숏 전용 인스턴스 — live_t 이후(미래) 이벤트는
    아예 주입하지 않는다.

    이유: Dash.metrics() 의 최근-5초 이동평균 집계 루프는 "e[t1] < now-120 이면
    멈추고, e[t1] < now-5 면 건너뛴다"는 하한만 있고 **상한(now 이후 배제) 이
    없다** — 실제 서빙에서는 링버퍼에 미래 이벤트가 존재할 수 없으니 원래
    불필요한 체크였을 뿐인데, 우리처럼 타임라인 전체(0~280초)를 한번에 재생해
    넣는 리플레이에서는 이 가정이 깨진다. 실측: 상한 없이 그대로 live_t=163.68
    로 얼렸더니 디코드 tok/s 가 115 대신 2184(약 19배)로 튀었다 — live_t 이후
    ~270초 분량의 dec 버킷이 전부 "최근 5초"에 합산된 것. dash_metrics.py 는
    수정 금지이므로, 주입 자체를 live_t 이하로 잘라 우회한다.

    store_src: 영속 DB 는 build_dash() 로 만든 인스턴스가 이미 채웠으므로 여기서
    또 attach_store() 하지 않는다(새 boot 세션이 하나 더 생겨 세션 드롭다운을
    오염시킨다). `.store` 참조만 공유해 누적/세션 뱃지·푸터 표기에 쓴다."""
    d = dm.Dash(enabled=True, cfg=dict(CFG))
    d._t0_wall = time.time() - wall_offset_s
    d.store = store_src            # 보고용 참조만 공유(별도 write 경로 없음)
    truncated = [e for e in evs if e[0] <= live_t]
    replay(d, truncated)
    set_now(d, live_t)
    d.set_live(live_count)
    return d


# ============================================================ 세션 1 (과거, 40분 전)
def build_session1():
    evs = []
    rng = random.Random(1)

    chunks_b0 = [(2048, 640.9), (2048, 1131.4), (2048, 1186.0), (2048, 1165.2),
                 (2048, 1172.4), (2048, 1159.6), (1673, 1139.6)]
    end_b0 = pp2_build(evs, 0.0, chunks_b0)
    snap(evs, end_b0 - 0.10, "pp2store", n=13962, tot=0)
    t0 = end_b0 + 0.235
    req(evs, t0, round(t0, 3), 13962)
    fin(evs, 150.0, (150.0 - t0) * 46.0, 150.0 - t0)
    mtp(evs, 149.0, [0.79, 0.61, 0.40])

    smalls = [(20, 0.35, 19, 60), (70, 0.32, 21, 110), (120, 0.30, 18, 160)]
    intervals = [(t0, 150.0)]
    for arr, tt, npm, end in smalls:
        st_t = arr + tt
        bg_prefill(evs, st_t, npm, 175.0)
        snap(evs, st_t - 0.05, "store", n=npm, tot=0)
        req(evs, st_t, tt, npm)
        fin(evs, float(end), (end - st_t) * 46.0, end - st_t)
        mtp(evs, end - 0.5, [0.76 + rng.uniform(-0.03, 0.03),
                              0.60 + rng.uniform(-0.03, 0.03),
                              0.39 + rng.uniform(-0.03, 0.03)])
        intervals.append((st_t, float(end)))

    dec_schedule(evs, intervals, 165, rng)
    return evs


# ============================================================ 세션 2 (라이브, 30분 전 시작)
# 두 번째 PP2 빌드(R_big2)의 정착 구간 청크 tps. 기본값(RB2_CHUNKS_DEFAULT)은
# 실측 트레이스 그대로(최대 1186.9) — 누적/영속(history) 은 이 값으로만 만들어
# 이미 승인된 "1,012 tok/s" 가중평균을 절대 건드리지 않는다. 라이브 탭 캡처는
# 별도로 RB2_CHUNKS_LIVE(990-1040 대역, 이 하드웨어의 실측 PP2 정착 상한
# 1,030-1,044 를 넘지 않게 낮춘 값)로 만든 "다른 인스턴스"를 쓴다 — 같은 R_big2
# 요청이라도 화면(라이브)과 영속(누적)에 서로 다른 숫자가 들어가는 셈이지만,
# 그렇게 하지 않으면 라이브 카드를 고치는 순간 이미 승인된 누적 평균이 함께
# 흔들린다(두 뷰가 같은 evs 리스트를 공유했던 이전 구조의 한계).
RB2_CHUNKS_DEFAULT = [(2048, 645.0), (2048, 1127.9), (2048, 1186.9), (2048, 1168.2),
                       (2048, 1175.7), (2048, 1164.0), (1667, 1183.0)]
RB2_CHUNKS_LIVE = [(2048, 645.0), (2048, 1040.0), (2048, 1038.0), (2048, 1040.0),
                    (2048, 1032.0), (2048, 1038.0), (1667, 1028.0)]


def build_session2(chunks_rb2=None):
    chunks_rb2 = chunks_rb2 if chunks_rb2 is not None else RB2_CHUNKS_DEFAULT
    evs = []
    rng = random.Random(2)

    # R1: 13.9K 콜드 프리필 -> 긴 디코드(요청 램프업 구간 내내 활성 유지)
    chunks_r1 = [(2048, 640.9), (2048, 1131.4), (2048, 1186.0), (2048, 1165.2),
                 (2048, 1172.4), (2048, 1159.6), (1673, 1139.6)]
    end_r1 = pp2_build(evs, 0.0, chunks_r1)
    snap(evs, end_r1 - 0.10, "pp2store", n=13962, tot=0)
    t_r1 = end_r1 + 0.235
    req(evs, t_r1, round(t_r1, 3), 13962)
    fin(evs, 125.0, (125.0 - t_r1) * 46.0, 125.0 - t_r1)
    mtp(evs, 124.0, [0.79, 0.61, 0.40])

    # R2..R9: 8개 단문, 15초 간격 도착 + 150초 디코드 -> 8-스트림 피크로 램프업
    arrivals = [20, 35, 50, 65, 80, 95, 110, 125]
    ttfts = [.35, .36, .30, .32, .33, .34, .31, .29]
    nps = [19, 23, 21, 25, 18, 22, 20, 24]
    small_intervals = []
    for arr, tt, npm in zip(arrivals, ttfts, nps):
        st_t = arr + tt
        bg_prefill(evs, st_t, npm, 180.0)
        snap(evs, st_t - 0.05, "store", n=npm, tot=0)
        req(evs, st_t, tt, npm)
        end_t = st_t + 150.0
        fin(evs, end_t, 150.0 * 46.0, 150.0)
        d = [0.78 + rng.uniform(-0.03, 0.03),
             0.62 + rng.uniform(-0.03, 0.03),
             0.41 + rng.uniform(-0.03, 0.03)]
        mtp(evs, end_t - 1.0, d)
        small_intervals.append((st_t, end_t))

    # R_big2: 두 번째 13.9K PP2 빌드, 8-스트림 피크 한복판에서 겹치게 배치
    # (chunks_rb2 는 함수 인자로 주입 — 기본값은 RB2_CHUNKS_DEFAULT)
    end_rb2 = pp2_build(evs, 150.0, chunks_rb2)
    snap(evs, end_rb2 - 0.10, "pp2store", n=13955, tot=0)
    t_rb2 = end_rb2 + 0.26
    req(evs, t_rb2, round(t_rb2 - 150.0, 3), 13956)
    fin(evs, 225.0, (225.0 - t_rb2) * 46.0, 225.0 - t_rb2)
    mtp(evs, 224.0, [0.80, 0.63, 0.42])

    # R10: 스냅숏 캐시 히트 재요청 -> TTFT 0.107s (실측 스펙과 동일)
    snap(evs, 240.0, "hit", n=13962, tot=13962)
    req(evs, 240.107, 0.107, 13962)
    fin(evs, 270.0, (270.0 - 240.107) * 46.0, 270.0 - 240.107)
    mtp(evs, 269.0, [0.77, 0.60, 0.39])

    intervals = ([(t_r1, 125.0)] + small_intervals
                 + [(t_rb2, 225.0), (240.107, 270.0)])
    dec_schedule(evs, intervals, 280, rng)
    return evs, intervals, end_rb2


# ─────────────────────────────────────────────────────────────── HTTP 서버
DASH = None
T_FREEZE = None


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/dash":
            self._send(dm.html_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/metrics":
            qs = parse_qs(parsed.query)
            if "sessions" in qs:
                st = DASH.store
                resp = {"sessions": st.sessions() if st else [],
                        "store": st.stats() if st else {"enabled": False}}
            elif "from" in qs and "to" in qs:
                resp = DASH.history(float(qs["from"][0]), float(qs["to"][0]))
            else:
                # "지금"을 t_freeze 에 영구 고정한다 — 실제 벽시계가 흐르는 동안
                # 서버가 오래 떠 있어도(스크린샷을 여러 번 다시 찍어도) 라이브
                # 5분 창이 재생 데이터 밖으로 밀려나지 않도록.
                set_now(DASH, T_FREEZE)
                since = int(float(qs.get("since", ["0"])[0]))
                resp = DASH.metrics(since=since)
            self._send(dm.json_bytes(resp), "application/json; charset=utf-8")
            return
        self._send(b"not found", "text/plain; charset=utf-8", code=404)


def find_free_port(start=8899, tries=30):
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("빈 포트를 찾지 못함")


def main():
    ev1 = build_session1()
    # (a) 영속/누적용 — 실측 트레이스 그대로(RB2_CHUNKS_DEFAULT). 이걸로만
    #     dash2_full 을 만들어야 이미 승인된 누적 "1,012 tok/s" 가중평균이
    #     이번 수정으로 조금도 흔들리지 않는다.
    ev2, intervals2, end_rb2 = build_session2()
    # (b) 라이브 탭 전용 — R_big2 정착 구간을 990-1040 대역으로 낮춘 버전
    #     (이 하드웨어 실측 PP2 정착 상한 1,030-1,044 를 넘지 않도록). 콜드
    #     스타트 첫 청크(645.0)는 그대로 낮게 유지 — 자연스러운 램프업.
    ev2_live_src, intervals2_live, end_rb2_live = build_session2(
        chunks_rb2=RB2_CHUNKS_LIVE)

    peak = max((sum(1 for s, e in intervals2 if s <= sec + 0.5 < e)
                for sec in range(0, 280)), default=0)
    # 라이브 탭의 "지금" = 두 번째 PP2 빌드가 막 끝난 직후(0.3초 뒤) — 승인된
    # 캡처 시점 공식 그대로 동결. RB2_CHUNKS_LIVE 의 정착 속도들은 이 값이
    # 디코드 5초 평균의 정수초 버킷 경계(now<=165.0 일 때만 5버킷 온전 포함,
    # 그 이상이면 하나가 통째로 밀려나 90대로 떨어짐 — dash_metrics.metrics()
    # 자체의 특성, 실측 확인함)를 넘지 않도록 역산해 맞춘 것이다.
    t_live = end_rb2_live + 0.3
    live_act = sum(1 for s, e in intervals2_live if s <= t_live < e)
    print(f"[plan] session1 events={len(ev1)}  session2 events={len(ev2)} "
          f"peak_concurrency={peak}  t_live={t_live:.2f}  "
          f"live_act_at_t_live={live_act}", flush=True)

    dash1, _ = build_dash(wall_offset_s=2400.0, evs=ev1, close_after=True)
    print(f"[session1] boot_id={dash1.store.boot_id} "
          f"written={dash1.store._n_written} ended={dash1.store.ok}", flush=True)

    # dash2_full: 타임라인 전체(0~280초, 실측 그대로)를 영속 DB 에 남긴다 —
    # 누적/세션 탭은 이 인스턴스의 store 를 그대로 읽으므로 앞서 승인받은
    # 화면(1,012 tok/s 가중평균 포함)과 완전히 동일하게 유지된다.
    dash2_full, t_end2 = build_dash(wall_offset_s=1800.0, evs=ev2, close_after=False)
    time.sleep(2.5)   # 백그라운드 플러셔(공개 Store.start 타이머)가 한 바퀴 돌게
    dash2_full.store.err = DEMO_NOTE
    print(f"[session2] boot_id={dash2_full.store.boot_id} "
          f"written={dash2_full.store._n_written} t_end={t_end2:.1f}", flush=True)

    # dash2_live: 라이브 탭 서빙 전용 — (b) 소스만 쓰고, live_t 이후 이벤트는
    # 아예 안 넣어서 5초 이동평균이 미래로 새지 않게 한다(위 build_live_snapshot
    # docstring). 영속에는 전혀 관여하지 않으므로 dash2_full/누적 탭과 무관.
    dash2_live = build_live_snapshot(wall_offset_s=1800.0, evs=ev2_live_src,
                                      live_t=t_live, live_count=live_act,
                                      store_src=dash2_full.store)
    m = dash2_live.metrics(since=0)
    print(f"[live-snapshot] t_now={m['t_now']} now={m['now']}", flush=True)

    global DASH, T_FREEZE
    DASH = dash2_live
    T_FREEZE = t_live

    port = find_free_port(8899)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"READY port={port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
