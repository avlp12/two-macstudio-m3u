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
"""

import json
import os
import re
import threading
import time
from collections import deque

_ENV = os.environ.get("DSV4_DASH", "1").strip().lower()
ENABLED = _ENV not in ("0", "", "false", "off")

WINDOW_S = float(os.environ.get("DSV4_DASH_WINDOW_S", "3600"))
MAXLEN = int(os.environ.get("DSV4_DASH_MAXLEN", "20000"))
# 이보다 작은 프리필 조각은 바로 그리지 않는다(PP2 주입 뒤 꼬리 토큰 1개 forward
# 같은 것들 — 20 tok/s 짜리 노이즈 바가 타임라인을 더럽힌다).
PF_MIN_TOKENS = int(os.environ.get("DSV4_DASH_PF_MIN", "16"))

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

    def _flush_bucket_locked(self, upto_b):
        cur = self._cur
        if cur["b"] >= 0 and cur["b"] < upto_b:
            self._seq += 1
            self._ev.append({
                "i": self._seq, "k": "dec",
                "t0": float(cur["b"]), "t1": float(cur["b"] + 1),
                "n": cur["n"], "tps": float(cur["n"]), "act": cur["act"],
            })
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
  <div class="seg" id="win">
    <button data-w="300">5분</button>
    <button data-w="900">15분</button>
    <button data-w="3600" aria-pressed="true">60분</button>
  </div>
</header>

<div class="tiles">
  <div class="tile"><div class="k">프리필 tok/s</div>
    <div class="v" id="t_pf">–</div><div class="s">최근 5초 이동</div></div>
  <div class="tile"><div class="k">디코드 tok/s</div>
    <div class="v" id="t_dec">–</div><div class="s">최근 5초 이동</div></div>
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
var tl=document.getElementById('tl'), mtp=document.getElementById('mtp');
function css(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim();}
function fmt(v,d){return v==null?'–':Number(v).toLocaleString('en-US',
  {minimumFractionDigits:d,maximumFractionDigits:d});}

function setWin(w){
  var seg=document.getElementById('win'), hit=null;
  [].forEach.call(seg.querySelectorAll('button'),function(x){
    var on = (+x.dataset.w===w); if(on) hit=x;
    x.setAttribute('aria-pressed', on?'true':'false');});
  if(hit) win=w;
}
document.getElementById('win').addEventListener('click',function(e){
  var b=e.target.closest('button'); if(!b)return;
  setWin(+b.dataset.w);
});
// #w=300|900|3600 로 초기 시간창 지정(스크린샷·북마크용)
(function(){var m=/(?:^|[#&])w=(\d+)/.exec(location.hash); if(m) setWin(+m[1]);})();
window.addEventListener('hashchange',function(){
  var m=/(?:^|[#&])w=(\d+)/.exec(location.hash); if(m) setWin(+m[1]);});

function poll(){
  fetch('/metrics?since='+next,{cache:'no-store'}).then(function(r){return r.json();})
  .then(function(j){
    if(j.reset) evs=[];
    if(j.events && j.events.length) evs=evs.concat(j.events);
    next=j.next; tRef=j.t_now; perfRef=performance.now();
    cfg=j.cfg||{}; nowS=j.now||{};
    var cut=j.t_now-3700;
    if(evs.length && evs[0].t1<cut) evs=evs.filter(function(e){return e.t1>=cut;});
    tiles(); badges();
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
  b.innerHTML=h;
  var up=nowS.uptime_s||0;
  document.getElementById('foot').innerHTML=
    'rank0 계측 · 랭크1 하단-슬라이스 프리필은 <b>rank0 이음새 시점으로 근사</b>'+
    '(PP2 는 청크마다 랭크1→랭크0 활성값 전송으로 동기되므로 이음새 간격이 곧 '+
    '양 랭크 합산 청크 시간). 프리필 tok/s 는 이음새 <b>본문</b>(인터리브 디코드 '+
    '스텝)을 뺀 순 프리필 시간 기준 — 디코드 버킷과 이중 계상하지 않는다. '+
    'gauge 갱신 1s · 버퍼 '+Math.round((cfg.win_s||3600)/60)+'분 · uptime '+
    Math.floor(up/60)+'m'+Math.floor(up%60)+'s';
}

function tiles(){
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
  var t1=tNow, t0=tNow-win, sx=iw/win;
  var line=css('--line'), muted=css('--muted');

  // ── 가시 이벤트 수집 + 픽셀 열 집계 ────────────────────────────────
  var cols=Math.max(1,Math.round(iw));
  var pfCol=new Float32Array(cols), decCol=new Float32Array(cols);
  var pfAct=new Uint8Array(cols), decAct=new Uint8Array(cols);
  var pfMax=1, decMax=1, mtpPts=[];
  for(var i=0;i<evs.length;i++){
    var e=evs[i];
    if(e.t1<t0) continue;
    if(e.k==='mtp'){ if(e.t0>=t0) mtpPts.push(e); continue; }
    if(e.k!=='pf'&&e.k!=='dec') continue;
    var a=Math.max(0,Math.floor((e.t0-t0)*sx)), b=Math.min(cols-1,Math.ceil((e.t1-t0)*sx));
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

  // ── 시간축(분) ────────────────────────────────────────────────────
  var stepM = win<=300?1:(win<=900?3:10);
  x.fillStyle=muted; x.textAlign='center';
  for(var m=0;m*60<=win;m+=stepM){
    var px=L+iw-m*60*sx;
    if(px<L-1) break;
    x.globalAlpha=.5; x.beginPath();
    x.moveTo(Math.round(px)+.5,decBase); x.lineTo(Math.round(px)+.5,decBase+4); x.stroke();
    x.globalAlpha=1;
    x.fillText(m===0?'지금':('-'+m+'m'),px,decBase+16);
  }

  drawMtp(mtpPts,t0,sx);
  requestAnimationFrame(draw);
}

function drawMtp(pts,t0,sx){
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
    var px=L+(p.t0-t0)*sx, py=H-B-(Math.min(hi,Math.max(lo,p.tpc))-lo)/(hi-lo)*(H-T-B);
    i?x.lineTo(px,py):x.moveTo(px,py);
  });
  x.stroke();
  x.fillStyle=css('--mtp');
  pts.forEach(function(p){
    var px=L+(p.t0-t0)*sx, py=H-B-(Math.min(hi,Math.max(lo,p.tpc))-lo)/(hi-lo)*(H-T-B);
    x.beginPath(); x.arc(px,py,2.1,0,6.284); x.fill();
  });
}

poll(); requestAnimationFrame(draw);
})();
</script></body></html>
"""


def html_bytes():
    return HTML.encode("utf-8")


def json_bytes(obj):
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")
