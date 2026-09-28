#!/usr/bin/env bash
# Throughput watchdog for teacher-trace extraction.
#
# The chain already survives crashes and reboots, but nothing watched whether it was making
# PROGRESS. A run that quietly fails every chunk, or grinds at a third of the expected rate,
# looks identical to a healthy one until someone reads the log. This closes that.
#
# It is deliberately conservative: it restarts a chain that is genuinely DEAD, and otherwise
# only records health. It never retunes ngl or edits the run - a watchdog that second-guesses
# the config while unattended is how a working job turns into an unexplainable one.
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
STATE=artifacts/overnight_state
LOG=logs/watchdog.log
HEALTH=$STATE/HEALTH
say() { echo "[$(date -Is)] watchdog: $*" >> "$LOG"; }

[ -f "$STATE/ALL_DONE" ] && { echo "ALL_DONE" > "$HEALTH"; exit 0; }
[ -f "$STATE/STOP" ]     && { echo "STOPPED (operator)" > "$HEALTH"; exit 0; }

# --- is anything actually running? --------------------------------------------------------
chain_up=0; work_up=0
[ -f "$STATE/pid" ] && kill -0 "$(cat "$STATE/pid")" 2>/dev/null && chain_up=1
pgrep -x llama-embedding >/dev/null 2>&1 && work_up=1

# --- when did a chunk last complete? ------------------------------------------------------
last=$(grep -a "ok: .* tok / .* seq in" logs/overnight.log 2>/dev/null | tail -1)
last_epoch=0
if [ -n "$last" ]; then
  ts=$(echo "$last" | sed -n 's/^\[\([^]]*\)\].*/\1/p')
  last_epoch=$(date -d "$ts" +%s 2>/dev/null || echo 0)
fi
now=$(date +%s)
idle_min=$(( last_epoch > 0 ? (now - last_epoch)/60 : 9999 ))

# --- recent rate: median tok/s of the last 5 completed chunks -----------------------------
rate=$(grep -a "ok: .* tok / .* seq in" logs/overnight.log 2>/dev/null | tail -5 \
       | sed -n 's/.* = \([0-9]\+\) tok\/s.*/\1/p' | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]; else print 0}')

# --- recent failures ----------------------------------------------------------------------
fails=$(grep -a "FAILED rc=" logs/overnight.log 2>/dev/null | tail -40 \
        | awk -v cutoff="$(date -d '8 hours ago' -Is)" '{gsub(/[][]/,"",$1); if ($1 > cutoff) n++} END{print n+0}')

# A chunk is ~55 min at healthy rate and the hard timeout is 4h, so nothing under ~5h of
# silence is evidence of a stall on its own.
STALL_MIN=300
# Sustained rate this low means thrash, not variance. 54 tok/s dips are normal; 35 is not.
SLOW_TOKS=35

status="ok"
[ "$rate" -gt 0 ] && [ "$rate" -lt "$SLOW_TOKS" ] && status="SLOW"
[ "$fails" -ge 3 ] && status="FAILING"
[ "$idle_min" -ge "$STALL_MIN" ] && status="STALLED"
[ "$chain_up" -eq 0 ] && status="DOWN"

printf 'status=%s chain_up=%s worker_up=%s idle_min=%s rate=%s fails_8h=%s checked=%s\n' \
  "$status" "$chain_up" "$work_up" "$idle_min" "$rate" "$fails" "$(date -Is)" > "$HEALTH"

case "$status" in
  DOWN)
    say "chain not running and no STOP/ALL_DONE - restarting (idle ${idle_min}m, last rate ${rate} tok/s)"
    systemctl --user start glm53-traces.service 2>>"$LOG" \
      || setsid nohup ./scripts/overnight.sh >/dev/null 2>&1 &
    ;;
  STALLED)
    # Alive but producing nothing for longer than the hard timeout allows. The worker is wedged;
    # killing it costs one chunk, which the chain re-runs, and is cheaper than losing a night.
    say "STALLED: ${idle_min}m since a chunk completed (worker_up=$work_up). Killing worker to force a retry."
    pkill -x llama-embedding
    ;;
  SLOW)    say "SLOW: median ${rate} tok/s over last 5 chunks (floor ${SLOW_TOKS}); check memory pressure" ;;
  FAILING) say "FAILING: ${fails} chunk failures in the last 8h; extraction is not advancing cleanly" ;;
esac
exit 0
