#!/usr/bin/env bash
# STEP 2: the head-to-head, redone with both arms configured IDENTICALLY.
#
# Why every earlier number is void:
#   1. 2026-09-10 -- `$!` captured a subshell, the server orphaned onto :8080, and two of three
#      arms silently re-measured GLM. See artifacts/void_2026-09-10_server_orphan_bug/.
#   2. 2026-09-11 -- the surviving arm was ALSO invalid. The harness sent no reasoning_effort,
#      and the two servers disagree on the default: dsv4 -> "low", glm5 -> Max. glm5 at Max
#      DEGENERATES (measured repetition fraction 0.61, 8192 tok, empty content, answer stranded
#      in reasoning_content). So 41.5% vs 77.4% was a looping GLM against a low-effort DeepSeek.
#      Raising max_tokens would NOT have fixed it -- the loop is unbounded.
# Both arms now get reasoning_effort=high explicitly, a generous budget, and every result file
# records the effort, the budget and the truncation count so this is auditable after the fact.
#
# vexp is NOT included: build/dsv4-server hardcodes N_ROUTED=160 and VOCAB=129280
# (include/deepseek_v4.h:38), and the Vision-Exp checkpoint is 128 experts / vocab 128000.
# It aborts with `weight not found: layers.0.ffn.experts.128.w1.weight`. Making the engine
# read its shape from config.json is real work, tracked separately.
set -u
cd "$HOME/glm-5.3-reap"
LOG=logs/headtohead.log
GLMDIR="$HOME/glm-5.3-flash-cuda-server"
DSDIR="$HOME/deepseek-v4-flash-0731-cuda"
EFFORT=high
MT_HE=4096
MT_BFCL=2048
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
port_free() { ! curl -sf -m 3 http://127.0.0.1:8080/health >/dev/null 2>&1; }
served()    { curl -sf -m 5 http://127.0.0.1:8080/health 2>/dev/null \
              | .venv/bin/python -c 'import sys,json;print(json.load(sys.stdin).get("model","?"))' 2>/dev/null; }
wait_health() { for i in $(seq 1 400); do
    kill -0 "$1" 2>/dev/null || return 1
    curl -sf -m 5 http://127.0.0.1:8080/health >/dev/null 2>&1 && return 0
    sleep 10; done; return 1; }

arm() {   # arm DIR LABEL BINARY_AND_ARGS...
  local dir="$1" label="$2"; shift 2
  if ! port_free; then say "  $label ABORT -- :8080 held by '$(served)'"; return 1; fi
  ( cd "$dir" && exec "$@" ) > "$dir/logs_h2h_server.log" 2>&1 &
  local pid=$!
  if wait_health "$pid"; then
    local sv; sv=$(served)
    say "  $label up (pid $pid) SERVING '$sv'  effort=$EFFORT"
    ( cd "$GLMDIR" && python3 tools/eval_humaneval.py --base http://127.0.0.1:8080 \
        --max-tokens $MT_HE --reasoning-effort $EFFORT --label "$label" \
        --out "$HOME/glm-5.3-reap/artifacts/humaneval_${label}.json" ) >> "$LOG" 2>&1 \
      && say "  $label HumanEval DONE (served '$sv')" || say "  $label HumanEval FAILED"
    ( cd "$GLMDIR" && python3 tools/eval_bfcl.py --base http://127.0.0.1:8080 --limit 100 \
        --max-tokens $MT_BFCL --reasoning-effort $EFFORT --label "$label" \
        --out "$HOME/glm-5.3-reap/artifacts/bfcl_${label}.json" ) >> "$LOG" 2>&1 \
      && say "  $label BFCL DONE (served '$sv')" || say "  $label BFCL FAILED"
  else
    say "  $label SKIPPED -- never ready (see $dir/logs_h2h_server.log)"
  fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  for i in $(seq 1 120); do port_free && break; sleep 5; done
  port_free && say "  $label torn down, :8080 clear" \
            || { say "  FATAL: :8080 still held after $label"; exit 1; }
}

# The void results are moved aside, not deleted, so a stale file cannot be mistaken for a fresh one.
mkdir -p artifacts/void_2026-09-11_effort_mismatch
for f in humaneval_glm.json humaneval_glm5.json humaneval_dsv4.json bfcl_glm5.json bfcl_dsv4.json; do
  [ -f "artifacts/$f" ] && mv "artifacts/$f" "artifacts/void_2026-09-11_effort_mismatch/$f"
done
say "=== HEAD-TO-HEAD: effort=$EFFORT, HumanEval max_tokens=$MT_HE, BFCL max_tokens=$MT_BFCL ==="
arm "$GLMDIR" glm5 ./build/glm5-server --ctx 16384 --port 8080
arm "$DSDIR"  dsv4 ./build/dsv4-server --seqmax 16384 --port 8080

say "=== RESULT ==="
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, pathlib
META = {"glm5": ("GLM-5.3-Flash-REAP50", "144/288 = 50.0% pruned", "top-8"),
        "dsv4": ("DeepSeek-0731-REAP",   "160/256 = 37.5% pruned", "top-6")}
for kind, key, fmt in (("HumanEval", "pass@1", "humaneval_{}.json"),
                       ("BFCL", "acc", "bfcl_{}.json")):
    for lbl in ("glm5", "dsv4"):
        name, prune, tk = META[lbl]; p = pathlib.Path("artifacts/" + fmt.format(lbl))
        if not p.exists(): print(f"{kind:10} {name:22} (no result)"); continue
        d = json.loads(p.read_text())
        flag = "" if not d.get("truncated") else f"  <-- {d['truncated']} TRUNCATED, not a capability number"
        print(f"{kind:10} {name:22} {d['pass']:3d}/{d['n']:3d} = {d[key]:6.1%}  "
              f"[effort {d.get('reasoning_effort','?')}, budget {d.get('max_tokens','?')}, "
              f"{prune}, {tk}]{flag}")
    print()
print("Prune ratios DIFFER (50.0% vs 37.5%) -- 0731's is fit-driven, not a choice. The")
print("comparison is between the two models AS THEY FIT THIS BOX, not at equal compression.")
print("Greedy, single sample, our own prompt: comparable to EACH OTHER, not to leaderboards.")
PY
say "HEAD-TO-HEAD COMPLETE"
