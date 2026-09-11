#!/usr/bin/env bash
set -u
cd "$HOME/glm-5.3-reap"
LOG=logs/effort_probe.log
GLMDIR="$HOME/glm-5.3-flash-cuda-server"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
port_free() { ! curl -sf -m 3 http://127.0.0.1:8080/health >/dev/null 2>&1; }
if ! port_free; then say "ABORT -- :8080 already held"; exit 1; fi
( cd "$GLMDIR" && exec ./build/glm5-server --ctx 16384 --port 8080 ) > "$GLMDIR/logs_probe_server.log" 2>&1 &
pid=$!
for i in $(seq 1 400); do
  kill -0 "$pid" 2>/dev/null || { say "server died"; exit 1; }
  curl -sf -m 5 http://127.0.0.1:8080/health >/dev/null 2>&1 && break
  sleep 10
done
say "server up (pid $pid) SERVING '$(curl -sf -m5 http://127.0.0.1:8080/health | .venv/bin/python -c 'import sys,json;print(json.load(sys.stdin)["model"])')'"
python3 scripts/effort_probe.py 2>&1 | tee -a "$LOG"
kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
for i in $(seq 1 120); do port_free && break; sleep 5; done
say "PROBE COMPLETE, :8080 $(port_free && echo clear || echo HELD)"
