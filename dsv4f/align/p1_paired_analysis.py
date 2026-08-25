"""[P1] bs1×24토픽 페어드 분석 — per-topic 차이, 부호검정, 풀링 집계.
로그 형식: [p1-topic i/24] <topic> 다음에 MTP[0] 텔레메트리(랭크별 2회 중복)."""
import re, sys
from math import comb

PAT_TOPIC = re.compile(r"\[p1-topic (\d+)/\d+\] (.+)")
PAT_MTP = re.compile(
    r"MTP\[0\] finish=\w+ tokens=(\d+) cycles=(\d+) tok/cycle=([\d.]+) "
    r"accept=(\d+)/(\d+) \(([\d.]+)%\) depth\[d1=(\d+)/(\d+),d2=(\d+)/(\d+),d3=(\d+)/(\d+)\]")


def parse(path):
    topics, cur = {}, None
    for line in open(path):
        m = PAT_TOPIC.search(line)
        if m:
            cur = (int(m.group(1)), m.group(2).strip())
            continue
        m = PAT_MTP.search(line)
        if m and cur is not None and cur not in topics:  # 랭크 중복 첫 줄만
            g = [int(x) if "." not in x else float(x) for x in m.groups()]
            topics[cur] = dict(tokens=g[0], cycles=g[1], tc=g[2],
                               d1n=g[6], d1d=g[7], d2n=g[8], d2d=g[9],
                               d3n=g[10], d3d=g[11])
    return topics


def pooled(ts):
    tok = sum(t["tokens"] for t in ts.values())
    cyc = sum(t["cycles"] for t in ts.values())
    d1n = sum(t["d1n"] for t in ts.values()); d1d = sum(t["d1d"] for t in ts.values())
    d2n = sum(t["d2n"] for t in ts.values()); d2d = sum(t["d2d"] for t in ts.values())
    d3n = sum(t["d3n"] for t in ts.values()); d3d = sum(t["d3d"] for t in ts.values())
    return dict(tc=tok / cyc, d1=d1n / d1d, d2=d2n / d2d, d3=d3n / d3d)


def sign_test(wins, losses):
    """양측 부호검정 p값 (동점 제외)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(p, 1.0)


base = parse(sys.argv[1])
new = parse(sys.argv[2])
keys = sorted(set(base) & set(new))
print(f"페어드 토픽 수: {len(keys)}")
print(f"{'topic':<28}{'base tc':>8}{'6c tc':>8}{'Δtc':>8}  {'base d2':>8}{'6c d2':>7}")
wins = losses = ties = 0
cs_delta, noncs_delta = [], []
CS_IDX = set(range(8))
for k in keys:
    b, n = base[k], new[k]
    d = n["tc"] - b["tc"]
    wins += d > 0; losses += d < 0; ties += d == 0
    (cs_delta if k[0] in CS_IDX else noncs_delta).append(d)
    print(f"{k[1][:26]:<28}{b['tc']:>8.2f}{n['tc']:>8.2f}{d:>+8.2f}  "
          f"{b['d2n']/b['d2d']:>7.1%}{n['d2n']/n['d2d']:>7.1%}")
pb, pn = pooled(base), pooled(new)
print(f"\n풀링: 기준선 tc={pb['tc']:.3f} d1={pb['d1']:.1%} d2={pb['d2']:.1%} d3={pb['d3']:.1%}")
print(f"      6c     tc={pn['tc']:.3f} d1={pn['d1']:.1%} d2={pn['d2']:.1%} d3={pn['d3']:.1%}")
print(f"      Δtc={pn['tc']-pb['tc']:+.3f} ({(pn['tc']/pb['tc']-1)*100:+.2f}%)")
print(f"\n부호검정: 6c 승 {wins} · 패 {losses} · 무 {ties} → p={sign_test(wins, losses):.4f}")
if cs_delta and noncs_delta:
    print(f"도메인 분해: CS(8) 평균Δtc={sum(cs_delta)/len(cs_delta):+.3f} · "
          f"비-CS({len(noncs_delta)}) 평균Δtc={sum(noncs_delta)/len(noncs_delta):+.3f}")
