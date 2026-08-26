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
# 2026-08-26 저녁 (PA510): 피어 창을 놓치면 세 가지가 겹쳤다.
#   1) chosen 이 빈 채로 "나머지 활성 포트"를 bridge0 에 넣어 클러스터 포트(en5)가
#      브리지에 먹힘. chosen 이 비면 새 멤버를 추가하지 않는다(기존 en2/우란 유지).
#   2) ifconfig alias 는 10.0.0.0/24 연결 라우트를 이전 포트에 남긴다. ifscope 가
#      아니라 일반 연결 라우트라 route delete -ifscope 로는 안 지워진다.
#      붙인 뒤 route delete 10/24 + route add -interface $chosen 으로 강제 정렬.
#   3) 피어 미확인 때 tbnet-iface 를 안 남겨 tb_sync_hostfile 이 근거를 잃음.
#      캐리어 있는 비-브리지 포트가 있으면 기록하고 status=peer-unconfirmed.
#
# 2026-08-26 심야 (26.6.2 승급 후) v3: 네 번째 결함.
#   4) preferences.plist 의 __AUTO__=thunderbolt-bridge 가 **서비스 비활성과 무관하게**
#      부팅마다 모든 TB 포트를 bridge0 에 L2 편입시킨다(OS 소행 — v2 가 규칙을 지키자
#      비로소 드러남). 비-브리지 후보가 0 이 되어 v2 는 굶었다.
#      → 비-브리지 후보가 0 이면 우란(en2)을 뺀 브리지 멤버를 **한 포트씩
#        회수(deletem)해 피어를 시험하고, 실패하면 즉시 원복**한다.
#        시험 순서는 지난 부팅 포트(tbnet-iface) 먼저. 최종 실패 시에도 한 포트를
#        회수 상태로 유지해 IP 를 걸어 둔다 — 늦게 뜬 반대편이 확인하면 성립한다.
#
# bridge0 멤버 중 우란(en2, 192.168.7.x)은 건드리지 않는다.

case "$(scutil --get LocalHostName 2>/dev/null)" in
  *gesicht*|*Gesicht*) OCT=1; PEER=2 ;;
  *) OCT=2; PEER=1 ;;
esac

CAND="${TBNET_CAND:-en4 en5 en3 en2 en6 en7}"
URAN_IF="${TBNET_URAN_IFACE:-en2}"
NET="10.0.0"
SELF_IP="$NET.$OCT"
PEER_IP="$NET.$PEER"
IFACE_FILE="/Users/Shared/tbnet-iface"
IFACE_STATUS="/Users/Shared/tbnet-iface.status"
DRY="${TBNET_DRY:-0}"

_bridged() { ifconfig bridge0 2>/dev/null | grep -q "member: $1 "; }
_active()  { [ "$(ifconfig "$1" 2>/dev/null | awk '/status:/{print $2}')" = active ]; }
_exists()  { ifconfig "$1" >/dev/null 2>&1; }
_is_uran() { [ "$1" = "$URAN_IF" ]; }

# 클러스터 후보: 존재하고, 우란이 아니고, 브리지 멤버가 아니고, 캐리어가 있는 포트.
_cluster_ok() {
  _exists "$1" || return 1
  _is_uran "$1" && return 1
  _bridged "$1" && return 1
  _active "$1"
}

_first_cluster_cand() {
  local ifc
  for ifc in $CAND; do
    _cluster_ok "$ifc" && { printf '%s\n' "$ifc"; return 0; }
  done
  return 1
}

_has_self_ip() { ifconfig "$1" 2>/dev/null | grep -q "inet $SELF_IP "; }

# 모든 후보에서 클러스터 IP 만 벗긴다. 192.168.7.x 는 건드리지 않는다.
_strip_cluster_ip() {
  local ifc
  for ifc in $CAND; do
    _exists "$ifc" || continue
    _has_self_ip "$ifc" || continue
    if [ "$DRY" = 1 ]; then
      echo "[tbnet-dry] would strip $SELF_IP from $ifc"
      continue
    fi
    ifconfig "$ifc" inet "$SELF_IP" -alias 2>/dev/null
  done
}

# 오늘 실측된 유효 절차: /24 를 일반 연결 라우트로 지우고 $1 에 다시 붙인다.
_align_route() {
  local ifc="$1"
  if [ "$DRY" = 1 ]; then
    echo "[tbnet-dry] would route 10.0.0.0/24 -> $ifc"
    return 0
  fi
  route -n delete -net 10.0.0.0 -netmask 255.255.255.0 >/dev/null 2>&1
  route add -net 10.0.0.0 -netmask 255.255.255.0 -interface "$ifc" >/dev/null 2>&1
}

_bind_cluster() {
  local ifc="$1"
  _strip_cluster_ip
  if [ "$DRY" = 1 ]; then
    echo "[tbnet-dry] would bind $SELF_IP alias on $ifc"
    _align_route "$ifc"
    return 0
  fi
  ifconfig "$ifc" inet "$SELF_IP" netmask 255.255.255.0 alias 2>/dev/null
  _align_route "$ifc"
}

# __AUTO__ thunderbolt-bridge 가 모든 포트를 물고 있을 때: 우란을 뺀 브리지 멤버를
# 하나 회수해 피어를 시험한다. 실패하면 IP 를 벗기고 브리지에 원복한다.
_evict_probe() {
  local ifc="$1"
  _exists "$ifc" || return 1
  _is_uran "$ifc" && return 1
  _bridged "$ifc" || return 1
  _active "$ifc" || return 1
  if [ "$DRY" = 1 ]; then
    echo "[tbnet-dry] would evict $ifc from bridge0 and probe $PEER_IP"
    return 1
  fi
  ifconfig bridge0 deletem "$ifc" 2>/dev/null
  _bind_cluster "$ifc"
  sleep 1
  if ping -c 1 -t 2 "$PEER_IP" >/dev/null 2>&1; then
    logger "tbnet-restore: $ifc 를 bridge0 에서 회수 (피어 확인)"
    return 0
  fi
  ifconfig "$ifc" inet "$SELF_IP" -alias 2>/dev/null
  ifconfig bridge0 addm "$ifc" 2>/dev/null
  return 1
}

# 회수 시험 순서: 지난 부팅에 기록된 포트(tbnet-iface) 먼저, 그다음 CAND 순서. 중복 제거.
_evict_order() {
  { cat "$IFACE_FILE" 2>/dev/null; printf '%s\n' $CAND; } | awk 'NF && !seen[$0]++'
}

_write_iface() {
  local ifc="$1" st="$2"
  if [ "$DRY" = 1 ]; then
    echo "[tbnet-dry] would write $IFACE_FILE=$ifc status=$st"
    return 0
  fi
  printf '%s\n' "$ifc" > "$IFACE_FILE" 2>/dev/null
  printf '%s\n' "$st" > "$IFACE_STATUS" 2>/dev/null
}

if [ "$DRY" = 1 ]; then
  echo "[tbnet-dry] host OCT=.$OCT peer=$PEER_IP uran=$URAN_IF"
  echo "[tbnet-dry] candidates:"
  for ifc in $CAND; do
    _exists "$ifc" || { echo "  $ifc missing"; continue; }
    s=$(ifconfig "$ifc" 2>/dev/null | awk '/status:/{print $2}')
    b=""; _bridged "$ifc" && b=" bridged"
    u=""; _is_uran "$ifc" && u=" uran"
    c=""; _cluster_ok "$ifc" && c=" CLUSTER_OK"
    echo "  $ifc status=$s$b$u$c"
  done
  hope=$(_first_cluster_cand || true)
  echo "[tbnet-dry] first cluster candidate: ${hope:-none}"
  echo "[tbnet-dry] evict order (when no clean candidate): $(_evict_order | tr '\n' ' ')"
  echo "[tbnet-dry] if chosen empty: will NOT addm to bridge0"
  exit 0
fi

chosen=""
for tries in $(seq 24); do
  for ifc in $CAND; do
    _cluster_ok "$ifc" || continue
    _bind_cluster "$ifc"
    sleep 1
    if ping -c 1 -t 2 "$PEER_IP" >/dev/null 2>&1; then
      chosen="$ifc"
      break 2
    fi
  done
  # 피어가 아직 없으면 첫 클러스터 후보에 IP+/24 를 유지한 채 다음 라운드에서 재검증.
  # 브리지에는 아직 넣지 않는다.
  if [ -z "$chosen" ]; then
    hope=$(_first_cluster_cand || true)
    if [ -n "$hope" ]; then
      _bind_cluster "$hope"
    else
      # 비-브리지 후보 0 — __AUTO__ thunderbolt-bridge 가 전부 편입한 상태(결함 4).
      # 우란을 뺀 멤버를 한 포트씩 회수 시험한다(실패 시 _evict_probe 가 즉시 원복).
      for evict in $(_evict_order); do
        _evict_probe "$evict" && { chosen="$evict"; break; }
      done
    fi
  fi
  [ -n "$chosen" ] && break
  sleep 5
done

# 우란 세그먼트 복원 (게지히트 한정) — **클러스터 포트가 확정된 뒤에만** 새 멤버 추가.
# chosen 이 비면 기존 멤버(en2)만 유지한다. en2 / 192.168.7.x 는 제거하지 않는다.
if [ "$OCT" = 1 ]; then
  ifconfig bridge0 >/dev/null 2>&1 || ifconfig bridge0 create 2>/dev/null
  if ifconfig bridge0 >/dev/null 2>&1; then
    if [ -n "$chosen" ]; then
      if _bridged "$chosen"; then
        ifconfig bridge0 deletem "$chosen" 2>/dev/null
        logger "tbnet-restore: $chosen 을 bridge0 에서 분리 (클러스터)"
      fi
      for u in $CAND; do
        [ "$u" = "$chosen" ] && continue
        _active "$u" || continue
        # en2(우란) 포함 — 클러스터 포트만 빼고, 기존 멤버는 addm 이 그대로 유지.
        ifconfig bridge0 | grep -q "member: $u " || ifconfig bridge0 addm "$u" 2>/dev/null
      done
    else
      logger "tbnet-restore: 피어 미확인 — bridge0 에 새 멤버를 추가하지 않음"
    fi
    ifconfig bridge0 up 2>/dev/null
    ifconfig bridge0 | grep -q "inet 192.168.7.2" || \
      ifconfig bridge0 inet 192.168.7.2 netmask 255.255.255.0 alias 2>/dev/null
    ping -c 1 -t 2 192.168.7.1 >/dev/null 2>&1 \
      && logger "tbnet-restore: uran 192.168.7.1 확인" \
      || logger "tbnet-restore: uran 미응답(전원/케이블 확인)"
  fi
fi

if [ -n "$chosen" ]; then
  _bind_cluster "$chosen"
  _write_iface "$chosen" "peer-ok"
  logger "tbnet-restore: $SELF_IP on $chosen (peer $PEER_IP 확인)"
else
  hope=$(_first_cluster_cand || true)
  if [ -n "$hope" ]; then
    _bind_cluster "$hope"
    _write_iface "$hope" "peer-unconfirmed"
    logger "tbnet-restore: 피어 미확인 — $SELF_IP on $hope (tbnet-iface 기록, 브리지 추가 없음)"
  else
    # 깨끗한 후보가 없으면(전 포트 브리지 편입) 한 포트를 회수 상태로 유지하며 대기.
    # 원복하지 않는 것이 핵심 — 반대편이 나중에 떠서 이 IP 를 확인하면 링크가 성립한다.
    held=""
    for evict in $(_evict_order); do
      _exists "$evict" || continue
      _is_uran "$evict" && continue
      _bridged "$evict" || continue
      _active "$evict" || continue
      ifconfig bridge0 deletem "$evict" 2>/dev/null
      _bind_cluster "$evict"
      held="$evict"
      break
    done
    if [ -n "$held" ]; then
      _write_iface "$held" "peer-unconfirmed"
      logger "tbnet-restore: 피어 미확인 — $held 를 브리지에서 회수해 $SELF_IP 대기 (반대편 확인 시 성립)"
    else
      act=$(for i in $CAND; do _active "$i" && ! _bridged "$i" && ! _is_uran "$i" && printf "%s " "$i"; done)
      logger "tbnet-restore: 피어 미확인 — 활성 비-브리지 클러스터 후보[$act]. jaccl 호스트파일의 rdma_enX 확인 필요"
    fi
  fi
fi
