#!/bin/bash
# Shared lifecycle guard for two-box MLX harnesses. Every sweep/launch script
# should `source` this. These five rules are not style preferences — each one
# is a postmortem from a crash that took a box (or both) down.
#
#  R1. trap EXIT — run the cleanup block on every failure path, not just success.
#  R2. NEVER `kill -9` a process that is stuck in a collective.
#      A TB-RDMA (jaccl) rank that dies abnormally mid-collective does not clean
#      up: the macOS 26 TB-RDMA kernel path reacts by corrupting queue-pair/DMA
#      state, which leaks wired memory that no process owns and eventually wedges
#      the whole machine. If TERM is ignored: stop, report, reboot that box.
#  R3. Only chain the next config after shutdown verification passes
#      (zero surviving processes AND wired < 50 GB on BOTH boxes).
#  R4. Never auto-chain a config that changes graph shape (MTP depth, TP degree,
#      kernel switches). Launch those alone — a wedge there is unrecoverable
#      without a reboot, so you do not want it inside an unattended chain.
#  R5. Risky / unvalidated distributed experiments run on the `ring` (TCP)
#      backend only, never jaccl (RDMA). A ring hang is killable without
#      corrupting kernel state; an RDMA hang is not.
#
# Requires: export BOX_B=<user>@10.0.0.2   (ssh target of the second box)

EPS="${BOX_B:?BOX_B not set — export BOX_B=<user>@10.0.0.2 (ssh target for box B)}"
TP2_PAT="serve_batched_tp2|serve_tp4_dspark|jbench|run_tp2"

tp2_real_pids() {
  pgrep -f "${1:-$TP2_PAT}" 2>/dev/null | while read p; do
    case "$(ps -o comm= -p $p 2>/dev/null)" in *[Pp]ython*) echo $p;; esac
  done
}
_e_real() {
  ssh -o BatchMode=yes $EPS 'pgrep -f "serve_batched_tp2|serve_tp4_dspark|jbench" | while read p; do case "$(ps -o comm= -p $p)" in *[Pp]ython*) echo $p;; esac; done' 2>/dev/null
}
_g_wired() { vm_stat | awk '/wired/ {print int($4*16384/1e9)}'; }
_e_wired() { ssh -o BatchMode=yes $EPS "vm_stat | awk '/wired/ {print int(\$4*16384/1e9)}'" 2>/dev/null || echo 999; }

tp2_safe_shutdown() {   # 0 = clean, 1 = stuck (stop and reboot) — never KILLs
  local P0=$(tp2_real_pids | head -1)
  if [ -n "$P0" ]; then
    sleep 20   # quiesce: TERM issued immediately after work lands mid-Metal/RDMA op and corrupts state
    for p in $(tp2_real_pids); do kill -TERM $p 2>/dev/null; done
    ssh -o BatchMode=yes $EPS 'pkill -f "serve_batched_tp2|serve_tp4_dspark|jbench"' 2>/dev/null
    for i in $(seq 1 36); do [ -n "$(tp2_real_pids)" ] || break; sleep 5; done
    if [ -n "$(tp2_real_pids)" ]; then
      echo "[guard] TERM ignored — per R2 do NOT kill. Chain aborted; reboot this box." >&2
      return 1
    fi
  fi
  return 0
}

tp2_chain_ok() {   # R3: shutdown verification before chaining the next config
  [ -z "$(tp2_real_pids)" ] || { echo "[guard] local process still alive" >&2; return 1; }
  [ -z "$(_e_real)" ] || { echo "[guard] box B process still alive" >&2; return 1; }
  local GW=$(_g_wired) EW=$(_e_wired)
  [ "${GW:-999}" -lt 50 ] || { echo "[guard] box A wired ${GW}G >= 50G" >&2; return 1; }
  [ "${EW:-999}" -lt 50 ] || { echo "[guard] box B wired ${EW}G >= 50G" >&2; return 1; }
  return 0
}

tp2_require_ring() {   # R5: call at the entry point of any risky experiment
  case "$*" in *jaccl*) echo "[guard] R5 violation: risky experiments are ring-only (jaccl forbidden)" >&2; return 1;; esac
  return 0
}
