#!/bin/bash
# Restore the direct Thunderbolt subnets at boot (run as root by launchd).
#
# Which box am I? Put a single digit in /Users/Shared/tbnet-octet:
#     box A  ->  echo 1 | sudo tee /Users/Shared/tbnet-octet
#     box B  ->  echo 2 | sudo tee /Users/Shared/tbnet-octet
# (Ours keyed off the hostname; a file is portable and one less thing to get
#  silently wrong after a rename.)
#
# Wiring assumed (adjust the pair list / TBNET_LINKS if your enX numbering
# differs — check `networksetup -listallhardwareports` for the Thunderbolt ports):
#     en4 <-> en4 = 10.0.0.x   (serving link)
#     en5 <-> en5 = 10.0.1.x   (extra link, bandwidth-bound jobs only)
#     en3 <-> en3 = 10.0.2.x   (extra link, bandwidth-bound jobs only)
LINKS="${TBNET_LINKS:-en4:10.0.0 en5:10.0.1 en3:10.0.2}"

OCT=$(tr -dc '12' < /Users/Shared/tbnet-octet 2>/dev/null | head -c1)
case "$OCT" in
  1|2) ;;
  *) logger "tbnet-restore: /Users/Shared/tbnet-octet missing or not 1/2 — aborting"; exit 1 ;;
esac

want=0
for pair in $LINKS; do want=$((want+1)); done

for tries in $(seq 24); do
  ok=0
  for pair in $LINKS; do
    ifc=${pair%%:*}; net=${pair##*:}
    if ifconfig $ifc >/dev/null 2>&1; then
      cur=$(ifconfig $ifc | awk '/inet /{print $2}')
      [ "$cur" = "$net.$OCT" ] || ifconfig $ifc inet $net.$OCT netmask 255.255.255.0
      ok=$((ok+1))
    fi
  done
  [ $ok -ge $want ] && break
  sleep 5
done
logger "tbnet-restore: $ok/$want TB links configured (octet .$OCT)"
