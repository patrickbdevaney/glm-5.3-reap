#!/usr/bin/env bash
# UNTRUNCATE the head-to-head. Detached; see CLAUDE.md §1.
#
# WHY THIS EXISTS. The corrected 4096-budget run scored glm5 91.5% and dsv4 78.7%, but dsv4 hit
# the cap on 61 of 164 problems against glm5's 7. Conditioned on NOT truncating, dsv4 is 101/103
# and glm5 is 150/157 -- so at 4096 the gap measures token ECONOMY, not capability, and dsv4's
# true pass@1 is only bracketed to [61.6%, 98.8%]. That bracket is too wide to compare anything.
#
# Greedy decoding (temperature 0.0) means max_tokens can only ever STOP a sequence, never steer
# one, so every row that already terminated below 4096 is bit-identical at a larger budget. Only
# the truncated rows need re-measuring; --only-truncated-in splices them back over the rest.
#
# STAGE A pilots at the CEILING on 6 problems per arm, because the needed budget is not
# identifiable from a run that was itself capped (memory: eval-budget-choice-protocol). STAGE B
# runs the full truncated set at a budget chosen from A -- by a human, after reading A.
set -u
cd "$HOME/glm-5.3-reap"
LOG=logs/untruncate.log
GLMDIR="$HOME/glm-5.3-flash-cuda-server"
DSDIR="$HOME/deepseek-v4-flash-0731-cuda"
PILOT_MT=${PILOT_MT:-16384}
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
port_free() { ! curl -sf -m 3 http://127.0.0.1:8080/health >/dev/null 2>&1; }
served_model() { curl -sf -m 5 http://127.0.0.1:8080/health 2>/dev/null \
  | .venv/bin/python -c 'import sys,json;print(json.load(sys.stdin).get("model","?"))' 2>/dev/null; }
wait_health() {  # $1 = server pid. LIVENESS FIRST -- a stale orphan on :8080 will answer for a
  for i in $(seq 1 400); do   # server that never came up, which voided two arms on 2026-09-10.
    kill -0 "$1" 2>/dev/null || return 1
    curl -sf -m 5 http://127.0.0.1:8080/health >/dev/null 2>&1 && return 0
    sleep 10
  done; return 1
}

# arm LABEL DIR PRIOR_JSON OUT_JSON MAXTOK LIMIT BINARY_AND_ARGS...
arm() {
  local label="$1" dir="$2" prior="$3" out="$4" mt="$5" limit="$6"; shift 6
  if ! port_free; then say "  $label ABORTED -- :8080 held, serving '$(served_model)'"; return 1; fi
  ( cd "$dir" && exec "$@" ) > "$dir/logs_untrunc_server.log" 2>&1 &
  local pid=$!
  if wait_health "$pid"; then
    local served; served=$(served_model)
    say "  $label up (pid $pid) SERVING '$served'  max_tokens=$mt limit=$limit"
    ( cd "$GLMDIR" && python3 tools/eval_humaneval.py --base http://127.0.0.1:8080 \
        --label "$label" --reasoning-effort high --max-tokens "$mt" \
        --only-truncated-in "$HOME/glm-5.3-reap/$prior" ${limit:+--limit $limit} \
        --out "$HOME/glm-5.3-reap/$out" ) >> "$LOG" 2>&1 \
      && say "  $label DONE (served '$served')" || say "  $label FAILED"
  else
    say "  $label SKIPPED -- server never ready (see $dir/logs_untrunc_server.log)"
  fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  for i in $(seq 1 120); do port_free && break; sleep 5; done
  port_free || { say "  FATAL: :8080 still held after $label -- refusing to continue"; exit 1; }
}

say "waiting for the head-to-head to release :8080"
while ! port_free; do sleep 60; done
say "GPU free. RAM $(free -g | awk '/^Mem:/{print $7}') GiB"

# STAGE A -- pilot at the ceiling. --limit takes the FIRST N of the truncated subset, so both
# arms pilot on a deterministic, reproducible slice rather than a sample.
say "STAGE A pilot @ $PILOT_MT"
arm dsv4 "$DSDIR" artifacts/humaneval_dsv4.json artifacts/pilot_untrunc_dsv4.json \
    "$PILOT_MT" 6 ./build/dsv4-server --port 8080 --seqmax 20480
arm glm5 "$GLMDIR" artifacts/humaneval_glm5.json artifacts/pilot_untrunc_glm5.json \
    "$PILOT_MT" 6 ./build/glm5-server --ctx 20480 --port 8080
say "STAGE A COMPLETE -- read the token distributions before launching STAGE B."
