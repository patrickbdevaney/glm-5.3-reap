#!/usr/bin/env bash
# memguard.sh — keep the streaming pass alive on a box where the kernel cannot help.
#
# WHY THE KERNEL NEVER HELPS HERE. Established on this machine by the DSpark project
# (deepseek-v4-flash-0731-cuda/scripts/memguard.sh, measured 2026-08-20): a touched 2048 MiB
# cudaMalloc charges **42 MiB** to the calling cgroup. Tegra unified memory comes from the
# driver's allocator, not the page allocator, so it is invisible to every kernel accounting
# path. The OOM killer scores by RSS and therefore never selects the real consumer -- which is
# why s03 died six times with NO oom-kill line in dmesg. memory.max cannot bound it and this
# kernel has no PSI, so systemd-oomd cannot run at all.
#
# WHAT IS DIFFERENT ABOUT THIS WORKLOAD, AND WHY TIER 1 EXISTS. s03 mmaps 306 GB of source
# shards. That page cache IS reclaimable -- dropping it returned the box from "107 GiB used,
# 350 MB available" to "3 GiB used, 119 GiB available" instantly -- but Tegra under-reports it
# in both Cached and MemAvailable, and the kernel evidently will not reclaim it fast enough to
# satisfy a driver allocation. So the first response is to drop page cache, which costs nothing
# but re-reads and harms no process. Killing is the last resort, not the first.
#
# NARROW LICENSE. Tier 2 may only ever kill this project's own stage children (run_stage.py).
# It never picks "the biggest RSS", because RSS is precisely the number already proven not to
# reflect who holds the memory.
set -u
ROOT="${ROOT:-/home/patrickd/glm-5.3-reap}"
CACHE_MB="${CACHE_MB:-16000}"     # tier 1: drop page cache below this
# MEASURED 2026-08-27: this workload's healthy plateau during the layer sweep is 2-3 GiB
# available, because ~105 GiB is mmap page cache the kernel holds but will release under a
# host allocation. A 4000 MB floor therefore killed perfectly healthy runs. Same reasoning as
# the DSpark memguard: the floor cannot be set above the plateau, so the LEVEL test is set
# below it and the SLOPE test catches genuine runaways.
# MEASURED AGAIN 2026-08-27: a large model load (s07 places 157 GiB across RAM and disk)
# legitimately drives MemAvailable to a few hundred MB, and the box does not die there - the
# earlier real collapse was observed at 317 MB with a 1730 MB/s slope. A 900 MB floor killed
# a healthy s07 at 866 MB. The LEVEL test keeps moving below the observed plateau; the SLOPE
# test is what actually distinguishes a load from a runaway.
FLOOR_MB="${FLOOR_MB:-250}"       # tier 2: kill our stage below this
# A NORMAL MoE layer build legitimately drops ~28 GiB in ~8 s (~3500 MB/s), so a 900 MB/s
# slope threshold fires on healthy work - it killed a run at 11 GB available doing exactly
# that. The slope test only earns its place if it is well above normal operation, and the
# level floor remains the real backstop.
RATE_MB_S="${RATE_MB_S:-6000}"    # tier 2 slope trigger
DANGER_MB="${DANGER_MB:-4000}"    # slope only counts below this
BREACHES="${BREACHES:-3}"
SLOPE_N="${SLOPE_N:-2}"
POLL_S="${POLL_S:-0.5}"
DROP_COOLDOWN_S="${DROP_COOLDOWN_S:-10}"
LOG="${LOG:-$ROOT/logs/memguard.log}"
mkdir -p "$(dirname "$LOG")"
say(){ echo "[memguard $(date -Is)] $*" >> "$LOG"; }

# Spawn-free wait: sleep is not a builtin, and one fork per poll is noise on the thing we watch.
_FIFO=$(mktemp -u); mkfifo "$_FIFO" 2>/dev/null && exec 8<>"$_FIFO" && rm -f "$_FIFO"
napp(){ if [ -e /proc/self/fd/8 ]; then read -t "$POLL_S" -u 8 _x 2>/dev/null; else sleep "$POLL_S"; fi; return 0; }

say "started: cache_drop<${CACHE_MB}MB kill<${FLOOR_MB}MB slope>${RATE_MB_S}MB/s below ${DANGER_MB}MB poll=${POLL_S}s"

bad=0; srate=0; prev=""; last_drop=0; warned=0
while true; do
  avail=""
  while read -r _k _v _; do
    if [ "$_k" = "MemAvailable:" ]; then avail=$((_v/1024)); break; fi
  done < /proc/meminfo 2>/dev/null
  [ -z "$avail" ] && { napp; continue; }

  now=$(date +%s)

  # ---- tier 1: reclaim page cache. Free, reversible, and the actual cause here. ----------
  if [ "$avail" -lt "$CACHE_MB" ] && [ $((now - last_drop)) -ge "$DROP_COOLDOWN_S" ]; then
    sync
    sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches' 2>/dev/null
    last_drop=$now
    after=""
    while read -r _k _v _; do
      if [ "$_k" = "MemAvailable:" ]; then after=$((_v/1024)); break; fi
    done < /proc/meminfo 2>/dev/null
    say "tier1 drop_caches at ${avail}MB -> ${after:-?}MB"
    prev=""; bad=0; srate=0
    napp; continue
  fi

  # ---- tier 2: our stage is genuinely running away. Kill it so it can be retried. --------
  trip=""
  if [ -n "$prev" ]; then
    rate=$(( (prev - avail) * 2 ))    # POLL_S=0.5 -> MB/s
    if [ "$avail" -lt "$DANGER_MB" ] && [ "$rate" -gt "$RATE_MB_S" ]; then srate=$((srate+1)); else srate=0; fi
    [ "$srate" -ge "$SLOPE_N" ] && trip="MemAvailable ${avail}MB falling ${rate}MB/s"
  fi
  prev=$avail
  if [ "$avail" -lt "$FLOOR_MB" ]; then bad=$((bad+1)); else bad=0; fi
  [ "$bad" -ge "$BREACHES" ] && trip="MemAvailable ${avail}MB < floor ${FLOOR_MB}MB x${bad}"

  if [ -n "$trip" ]; then
    # match both absolute and relative launches of our stage runner
    pid=$(pgrep -f "run_stage\.py" | head -1)
    if [ -n "$pid" ]; then
      stg=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | awk '{print $3}')
      say "!!! $trip — killing stage ${stg:-?} pid $pid (orchestrator will retry)"
      kill -9 "$pid" 2>/dev/null
      sync; sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches' 2>/dev/null
      say "killed; MemAvailable now $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB"
      bad=0; srate=0; prev=""; last_drop=$(date +%s)
    elif [ "$warned" = 0 ]; then
      say "WARN $trip — no run_stage.py child running; doing nothing (narrow license)"
      warned=1
    fi
  else
    warned=0
  fi
  napp
done
