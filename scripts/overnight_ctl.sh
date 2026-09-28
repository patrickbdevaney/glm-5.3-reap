#!/usr/bin/env bash
# stop / resume / status for the overnight chain.
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
STATE=artifacts/overnight_state
PIDF=$STATE/pid
case "${1:-status}" in
  stop)
    touch "$STATE/STOP"
    if [ -f "$PIDF" ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then
      echo "stop requested; waiting for pid $(cat $PIDF) to wind down (<=15s)"
      for _ in $(seq 1 15); do kill -0 "$(cat $PIDF)" 2>/dev/null || break; sleep 1; done
    fi
    kill -0 "$(cat $PIDF 2>/dev/null)" 2>/dev/null \
      && echo "still alive - kill -TERM $(cat $PIDF) if you want it gone now" \
      || echo "stopped. completed steps are preserved; resume picks up where it left off."
    ;;
  resume)
    rm -f "$STATE/STOP"
    if [ -f "$PIDF" ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then
      echo "already running as pid $(cat $PIDF)"; exit 0
    fi
    setsid nohup ./scripts/overnight.sh > logs/overnight.err 2>&1 < /dev/null &
    sleep 3; echo "resumed as pid $(cat $PIDF 2>/dev/null)"
    ;;
  status)
    [ -f "$STATE/STOP" ] && echo "STOP flag: set" || echo "STOP flag: clear"
    if [ -f "$PIDF" ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then
      echo "running: pid $(cat $PIDF)"; else echo "running: no"; fi
    echo "completed steps:"; ls "$STATE"/*.done 2>/dev/null | xargs -rn1 basename | sed 's/^/  /' || echo "  (none)"
    echo "last log lines:"; tail -5 logs/overnight.log | sed 's/^/  /'
    ;;
  *) echo "usage: $0 {stop|resume|status}"; exit 2;;
esac
