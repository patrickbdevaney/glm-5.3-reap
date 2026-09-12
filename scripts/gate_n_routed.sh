#!/usr/bin/env bash
# GATE for the N_ROUTED discovery change (deepseek-v4-flash-0731-cuda).
#
# The change replaces a compile-time 160 with discover_n_routed() in the three load loops so the
# engine can serve DeepSeek-V4-Flash-Vision-Exp-REAP-145B (K128), whose ONLY architectural
# difference is the expert count. On the 0731 checkpoint the discovered value IS 160, so by
# inspection nothing numeric changed -- but CLAUDE.md §2 is explicit that a kernel is not believed
# because its diff looks safe. This runs the PRE-CHANGE binary and the post-change binary over the
# same greedy prompts and requires the generated text to be IDENTICAL. Any difference voids the
# dsv4 head-to-head arm, which was measured on the post-change binary.
#
# A gate that passes against a dead engine is worse than no gate, so a served-model assertion and
# a non-empty-reply assertion run before the comparison.
set -u
cd "$HOME/glm-5.3-reap"
LOG=logs/gate_n_routed.log
DSDIR="$HOME/deepseek-v4-flash-0731-cuda"
PRE="/tmp/claude-2002/-home-patrickd/3952824c-51a0-4b40-aa8e-acf3b3b02ab9/scratchpad/prechange/dsv4-server.prechange"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
port_free() { ! curl -sf -m 3 http://127.0.0.1:8080/health >/dev/null 2>&1; }
wait_health() { for i in $(seq 1 400); do kill -0 "$1" 2>/dev/null || return 1
  curl -sf -m 5 http://127.0.0.1:8080/health >/dev/null 2>&1 && return 0; sleep 10; done; return 1; }

# capture OUTFILE BINARY...  -- serve, generate the fixed prompt set greedily, tear down
capture() {
  local out="$1"; shift
  port_free || { say "ABORT -- :8080 already held"; exit 1; }
  ( cd "$DSDIR" && exec "$@" --port 8080 ) > "$DSDIR/logs_gate_nr_server.log" 2>&1 &
  local pid=$!
  wait_health "$pid" || { say "FAIL -- server never ready for $out"; kill "$pid" 2>/dev/null; exit 1; }
  say "  up (pid $pid) for $out :: $(grep -m1 'routed experts' "$DSDIR/logs_gate_nr_server.log" || echo 'no expert line')"
  .venv/bin/python scripts/gate_n_routed_probe.py "$out" >> "$LOG" 2>&1 \
    || { say "FAIL -- probe errored for $out"; kill "$pid" 2>/dev/null; exit 1; }
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  for i in $(seq 1 120); do port_free && break; sleep 5; done
  port_free || { say "FATAL -- :8080 still held after $out"; exit 1; }
}

say "waiting for :8080 to clear"
while ! port_free; do sleep 60; done
say "GATE N_ROUTED: pre-change vs post-change, same greedy prompts"
capture artifacts/gate_nr_pre.json  "$PRE"
capture artifacts/gate_nr_post.json ./build/dsv4-server
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json,sys
a=json.load(open("artifacts/gate_nr_pre.json")); b=json.load(open("artifacts/gate_nr_post.json"))
bad=0
for x,y in zip(a,b):
    same = x["text"]==y["text"]
    print(f"  {'ok  ' if same else 'DIFF'}  {x['prompt'][:52]!r}  {len(x['text'])} vs {len(y['text'])} chars")
    if not same:
        bad+=1
        print(f"      pre : {x['text'][:200]!r}")
        print(f"      post: {y['text'][:200]!r}")
print(("GATE N_ROUTED PASS -- 0731 at K160 is bit-identical across the change"
       if not bad else f"GATE N_ROUTED FAIL -- {bad} prompts differ; the dsv4 head-to-head arm is VOID"))
sys.exit(1 if bad else 0)
PY
say "gate rc=$?"
