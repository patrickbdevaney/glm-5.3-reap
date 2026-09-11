#!/usr/bin/env bash
# STEP 1 of the head-to-head redo: find the real max_tokens CEILING before spending 6 h on a
# scored run. The 2026-09-11 run scored GLM 41.5% vs DeepSeek 77.4% and the entire gap was
# budget -- 92 of GLM's 96 failures emitted no `def` at all, and ZERO were wrong answers.
# `eval-budget-choice-protocol`: the needed budget is NOT identifiable from a too-small run,
# so pilot AT the ceiling and read the distribution back.
set -u
cd "$HOME/glm-5.3-reap"
LOG=logs/budget_pilot.log
GLMDIR="$HOME/glm-5.3-flash-cuda-server"
DSDIR="$HOME/deepseek-v4-flash-0731-cuda"
N=8          # problems per arm
MT=8192      # pilot budget = the ceiling, not a guess
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
port_free() { ! curl -sf -m 3 http://127.0.0.1:8080/health >/dev/null 2>&1; }
served()    { curl -sf -m 5 http://127.0.0.1:8080/health 2>/dev/null \
              | .venv/bin/python -c 'import sys,json;print(json.load(sys.stdin).get("model","?"))' 2>/dev/null; }
wait_health() { for i in $(seq 1 400); do
    kill -0 "$1" 2>/dev/null || return 1          # LIVENESS FIRST -- see the orphan-bug note
    curl -sf -m 5 http://127.0.0.1:8080/health >/dev/null 2>&1 && return 0
    sleep 10; done; return 1; }

pilot() {   # pilot DIR LABEL BINARY_AND_ARGS...
  local dir="$1" label="$2"; shift 2
  if ! port_free; then say "  $label ABORT -- :8080 held by '$(served)'"; return 1; fi
  ( cd "$dir" && exec "$@" ) > "$dir/logs_pilot_server.log" 2>&1 &
  local pid=$!
  if wait_health "$pid"; then
    say "  $label up (pid $pid) SERVING '$(served)'  -- $N problems at max_tokens=$MT"
    ( cd "$GLMDIR" && python3 tools/eval_humaneval.py --base http://127.0.0.1:8080 \
        --limit $N --max-tokens $MT --label "$label" \
        --out "$HOME/glm-5.3-reap/artifacts/pilot/he_${label}.json" ) >> "$LOG" 2>&1 \
      && say "  $label pilot DONE" || say "  $label pilot FAILED"
  else
    say "  $label SKIPPED -- never ready (see $dir/logs_pilot_server.log)"
  fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  for i in $(seq 1 120); do port_free && break; sleep 5; done
  port_free && say "  $label torn down, :8080 clear" \
            || { say "  FATAL: :8080 still held after $label"; exit 1; }
}

say "=== BUDGET PILOT: $N problems x max_tokens=$MT, ctx 16384 ==="
pilot "$GLMDIR" glm5 ./build/glm5-server --ctx 16384 --port 8080
pilot "$DSDIR"  dsv4 ./build/dsv4-server --seqmax 16384 --port 8080

say "=== DISTRIBUTION ==="
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, pathlib
for lbl in ("glm5", "dsv4"):
    p = pathlib.Path(f"artifacts/pilot/he_{lbl}.json")
    if not p.exists(): print(f"{lbl}: no result"); continue
    d = json.loads(p.read_text()); ct = sorted(r["completion_tokens"] for r in d["rows"])
    nd = sum(not r["has_def"] for r in d["rows"])
    print(f"{lbl:5} n={d['n']}  pass@1 {d['pass']}/{d['n']}  truncated {d['truncated']}  no-def {nd}")
    print(f"      tokens {ct}")
    print(f"      median {ct[len(ct)//2]}  max {ct[-1]}  -> suggested budget {int(ct[-1]*1.5)//256*256+256}")
PY
say "PILOT COMPLETE"
