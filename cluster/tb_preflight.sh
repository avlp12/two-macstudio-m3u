#!/bin/bash
# TB 링크 프리플라이트 — 발사 전에 인터커넥트가 "진짜" 살아있는지 확인하고,
# 필요하면 클러스터 IP를 캐리어 있는 포트로 옮긴다.
#
# 왜 필요한가 (2026-08-26 사고):
#   클러스터 IP(10.0.0.1/.2)가 en4 에 고정돼 있었는데 TB 재-열거로 케이블이 en5 로
#   옮겨졌다. 인터페이스는 존재하고 IP 도 붙어 있는데 캐리어가 없어서, macOS 가
#   USB-NCM(100baseTX)으로 조용히 강등했다. 겉으로는 "그냥 느린 실행"으로 보였고
#   실제로는 집합 연산이 220배 느려진 상태였다(16MiB all_sum 790ms vs 정상 5.4ms).
#   즉 이 사고의 본질은 케이블이 아니라 **아무도 링크를 확인하지 않았다**는 것이다.
#
# 사용:
#   source tb_preflight.sh && tb_preflight            # 검사만, 실패 시 rc!=0
#   source tb_preflight.sh && tb_preflight --repair   # IP 재배치까지 시도
#
# rc: 0=정상 · 10=피어 불통 · 11=대역폭 미달 · 12=복구 실패
set -uo pipefail

TB_PEER_IP="${TB_PEER_IP:-10.0.0.2}"
TB_SELF_IP="${TB_SELF_IP:-10.0.0.1}"
TB_PEER_SSH="${TB_PEER_SSH:-m3ms@10.0.0.2}"
TB_PEER_ADMIN="${TB_PEER_ADMIN:-m3ms@100.73.68.100}"   # 폴백 경로(Tailscale) — TB가 죽었을 때 복구 지시용
TB_MIN_MBPS="${TB_MIN_MBPS:-300}"                       # ssh 암호화 포함 최소 처리량. USB-NCM(37MB/s)은 여기서 걸린다.
TB_CANDIDATES="${TB_CANDIDATES:-2 3 4 5 6 7}"

_tb_active() { ifconfig "en$1" 2>/dev/null | awk '/status:/{print $2}'; }

# bridge0 멤버는 후보에서 제외한다.
# 게지히트의 bridge0(192.168.7.2)에는 **우란**(Windows/RTX 5090)과 atom 이 붙어 있다.
# 여기에 클러스터 IP 를 붙이거나 멤버를 떼면 그 링크들이 죽는다 — 2026-08-26 에 실제로 냈던 사고.
_tb_bridged() { ifconfig bridge0 2>/dev/null | grep -q "member: en$1 "; }

tb_link_report() {
  echo "[tb] 로컬 포트:"
  for i in $TB_CANDIDATES; do
    local s ip b; s=$(_tb_active "$i"); ip=$(ifconfig "en$i" 2>/dev/null | awk '/inet /{print $2}' | tr '\n' ' ')
    b=""; _tb_bridged "$i" && b="[bridge0]"
    [ -n "$s" ] && printf "[tb]   en%s %-8s %s%s\n" "$i" "$s" "$ip" "$b"
  done
}

# 캐리어 있는 포트로 클러스터 IP 이전. 피어도 함께 옮겨야 하므로 관리 경로로 지시한다.
tb_repair() {
  echo "[tb] 복구 시도 — 캐리어 있는 포트 탐색" >&2
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$TB_PEER_ADMIN" \
    "for i in $TB_CANDIDATES; do sudo -n ifconfig en\$i up 2>/dev/null; done" 2>/dev/null
  for i in $TB_CANDIDATES; do sudo -n ifconfig "en$i" up 2>/dev/null; done
  sleep 4
  # 피어에서 활성 포트 목록을 받아온다
  local peer_act; peer_act=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$TB_PEER_ADMIN" \
    "for i in $TB_CANDIDATES; do s=\$(ifconfig en\$i 2>/dev/null | awk '/status:/{print \$2}'); [ \"\$s\" = active ] && echo \$i; done" 2>/dev/null)
  [ -z "$peer_act" ] && { echo "[tb] 피어에 활성 포트 없음 — 케이블 재삽입 또는 재부팅 필요" >&2; return 12; }

  for x in $TB_CANDIDATES; do
    [ "$(_tb_active "$x")" = active ] || continue
    if _tb_bridged "$x"; then echo "[tb]   en$x 는 bridge0 멤버(우란/atom) — 건너뜀" >&2; continue; fi
    for y in $peer_act; do
      ssh -o BatchMode=yes "$TB_PEER_ADMIN" "sudo -n ifconfig en$y inet $TB_PEER_IP/24 alias 2>/dev/null" 2>/dev/null
      sudo -n ifconfig "en$x" inet "$TB_SELF_IP/24" alias 2>/dev/null
      sleep 2
      if ping -c 1 -t 2 "$TB_PEER_IP" >/dev/null 2>&1; then
        echo "[tb] 복구됨: 로컬 en$x ↔ 피어 en$y" >&2
        echo "[tb] ⚠ ifconfig alias 는 재부팅에 사라진다. jaccl 호스트파일의 rdma_enX 도 en$x 여야 한다." >&2
        return 0
      fi
      sudo -n ifconfig "en$x" inet "$TB_SELF_IP" -alias 2>/dev/null
      ssh -o BatchMode=yes "$TB_PEER_ADMIN" "sudo -n ifconfig en$y inet $TB_PEER_IP -alias 2>/dev/null" 2>/dev/null
    done
  done
  echo "[tb] 모든 조합 불통 — 물리 확인 필요" >&2; return 12
}

tb_preflight() {
  local repair=0; [ "${1:-}" = "--repair" ] && repair=1

  if ! ping -c 2 -t 3 "$TB_PEER_IP" >/dev/null 2>&1; then
    echo "[tb] ✗ 피어 $TB_PEER_IP 불통" >&2; tb_link_report >&2
    [ $repair = 1 ] && { tb_repair || return 12; } || return 10
  fi

  # 대역폭 바닥 검사 — 이게 이 스크립트의 존재 이유다.
  # USB-NCM 강등(37MB/s)과 정상 TB(800MB/s+)를 가르는 유일한 자동 판별자.
  local t0 t1 mbps
  t0=$(python3 -c 'import time;print(time.time())')
  dd if=/dev/zero bs=1m count=128 2>/dev/null | \
    ssh -o BatchMode=yes -c aes128-gcm@openssh.com "$TB_PEER_SSH" 'cat > /dev/null' 2>/dev/null
  t1=$(python3 -c 'import time;print(time.time())')
  mbps=$(python3 -c "print(int(128/($t1-$t0)))")

  if [ "$mbps" -lt "$TB_MIN_MBPS" ]; then
    echo "[tb] ✗ 처리량 ${mbps}MB/s < 하한 ${TB_MIN_MBPS}MB/s — USB-NCM 강등 의심" >&2
    tb_link_report >&2
    echo "[tb]   처방: ① TB 케이블 재삽입 ② tb_preflight --repair ③ 재부팅" >&2
    return 11
  fi
  echo "[tb] ✓ 링크 정상 — $TB_PEER_IP 도달, ${mbps}MB/s (하한 ${TB_MIN_MBPS})"
  return 0
}
