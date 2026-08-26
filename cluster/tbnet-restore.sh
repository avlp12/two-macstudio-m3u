#!/bin/bash
# TB 직결 서브넷 복원 (재부팅 시 launchd가 root로 실행)
# gesicht: .1 / epsilon: .2 — 호스트명으로 판별
#
# 2026-08-26 개정: **포트 하드코딩 제거.**
#   구판은 PAIRS="en4:10.0.0" 으로 en4 에 무조건 IP 를 붙였다. TB 재-열거로 케이블이
#   en5 로 옮겨진 날, IP 는 캐리어 없는 en4 에 매달렸고 macOS 는 조용히 USB-NCM(100baseTX)
#   으로 강등했다 — 집합 연산이 220배 느려졌는데 아무 오류도 안 났다.
#   이제 **캐리어 있는 포트를 골라 붙이고, 피어 응답으로 검증**한다.
#
# bridge0 멤버는 건드리지 않는다. 게지히트 bridge0(192.168.7.2)에는 우란(Windows/RTX5090)이
# 붙어 있어, 여기에 클러스터 IP 를 얹으면 그 링크가 죽는다.

case "$(scutil --get LocalHostName 2>/dev/null)" in
  *gesicht*|*Gesicht*) OCT=1; PEER=2 ;;
  *) OCT=2; PEER=1 ;;
esac

CAND="${TBNET_CAND:-en4 en5 en3 en2 en6 en7}"
NET="10.0.0"

_bridged() { ifconfig bridge0 2>/dev/null | grep -q "member: $1 "; }
_active()  { [ "$(ifconfig "$1" 2>/dev/null | awk '/status:/{print $2}')" = active ]; }

chosen=""
for tries in $(seq 24); do
  for ifc in $CAND; do
    ifconfig "$ifc" >/dev/null 2>&1 || continue
    _bridged "$ifc" && continue          # 우란/atom 링크 보호
    _active "$ifc"  || continue          # 캐리어 없는 포트에 붙이지 않는다 (구판의 결함)

    cur=$(ifconfig "$ifc" | awk '/inet /{print $2}' | head -1)
    [ "$cur" = "$NET.$OCT" ] || ifconfig "$ifc" inet "$NET.$OCT" netmask 255.255.255.0 alias 2>/dev/null
    sleep 1
    # 피어 응답으로 검증 — 둘 중 늦게 뜬 쪽이 성공하면 양쪽 다 성공이다
    if ping -c 1 -t 2 "$NET.$PEER" >/dev/null 2>&1; then chosen="$ifc"; break 2; fi
    # 실패한 후보는 되돌린다(주소 중복 방지)
    [ "$cur" = "$NET.$OCT" ] || ifconfig "$ifc" inet "$NET.$OCT" -alias 2>/dev/null
  done
  # 아직 피어가 안 떴을 수 있다 — 후보 중 활성인 첫 포트에 붙여두고 다음 라운드에서 재검증
  if [ -z "$chosen" ]; then
    for ifc in $CAND; do
      ifconfig "$ifc" >/dev/null 2>&1 || continue
      _bridged "$ifc" && continue
      _active "$ifc" || continue
      ifconfig "$ifc" | grep -q "inet $NET.$OCT" || \
        ifconfig "$ifc" inet "$NET.$OCT" netmask 255.255.255.0 alias 2>/dev/null
      break
    done
  fi
  sleep 5
done

# 우란/atom 세그먼트 복원 (게지히트 한정) — **클러스터 포트 선택 뒤에** 한다.
# 순서가 반대면, 재열거로 케이블이 브리지 후보 포트에 오는 순간 브리지가 먼저
# 가로채고 클러스터는 영영 못 잡는다. 포트 이름을 다시 하드코딩하지 않기 위해,
# "고른 클러스터 포트를 뺀 나머지 활성 TB 포트"를 멤버로 삼는다.
if [ "$OCT" = 1 ]; then
  ifconfig bridge0 >/dev/null 2>&1 || ifconfig bridge0 create 2>/dev/null
  if ifconfig bridge0 >/dev/null 2>&1; then
    for u in $CAND; do
      [ "$u" = "$chosen" ] && continue
      _active "$u" || continue
      ifconfig bridge0 | grep -q "member: $u " || ifconfig bridge0 addm "$u" 2>/dev/null
    done
    ifconfig bridge0 up 2>/dev/null
    ifconfig bridge0 | grep -q "inet 192.168.7.2" || \
      ifconfig bridge0 inet 192.168.7.2 netmask 255.255.255.0 alias 2>/dev/null
    ping -c 1 -t 2 192.168.7.1 >/dev/null 2>&1 \
      && logger "tbnet-restore: uran 192.168.7.1 확인" \
      || logger "tbnet-restore: uran 미응답(전원/케이블 확인)"
  fi
fi

if [ -n "$chosen" ]; then
  logger "tbnet-restore: $NET.$OCT on $chosen (peer $NET.$PEER 확인)"
  echo "$chosen" > /Users/Shared/tbnet-iface 2>/dev/null
else
  act=$(for i in $CAND; do _active "$i" && ! _bridged "$i" && printf "%s " "$i"; done)
  logger "tbnet-restore: 피어 미확인 — 활성 비-브리지 포트[$act]. jaccl 호스트파일의 rdma_enX 확인 필요"
fi
