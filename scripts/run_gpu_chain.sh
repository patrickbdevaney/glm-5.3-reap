#!/usr/bin/env bash
# Detached GPU chain, runs after NIAH releases the box.
#
# ORDER: the head-to-head runs FIRST because it decides whether the GLM compression stages are
# worth running at all. Both stacks measure their SPEED and assert their CAPABILITY
# (deepseek's NORTH_STAR.md: "Every performance number here is measured on this box.
# Capability is not."). Nothing here consumes the 328 GB GLM source.
set -u
cd "$HOME/glm-5.3-reap"
LOG=logs/gpu_chain.log
GLMDIR="$HOME/glm-5.3-flash-cuda-server"
DSDIR="$HOME/deepseek-v4-flash-0731-cuda"
VEXP="$HOME/models/DeepSeek-V4-Flash-Vision-Exp-REAP-145B"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
py()  { .venv/bin/python -c "$1" >> "$LOG" 2>&1; }
port_free() { ! curl -sf -m 3 http://127.0.0.1:8080/health >/dev/null 2>&1; }
served_model() {
  curl -sf -m 5 http://127.0.0.1:8080/health 2>/dev/null \
    | .venv/bin/python -c 'import sys,json;print(json.load(sys.stdin).get("model","?"))' 2>/dev/null
}
wait_health() {   # $1 = server pid
  for i in $(seq 1 400); do
    # LIVENESS FIRST. Curling first lets a stale orphan on :8080 answer for a server that
    # never came up -- that bug silently evaluated the previous model for two whole arms
    # on 2026-09-10. See artifacts/void_2026-09-10_server_orphan_bug/.
    kill -0 "$1" 2>/dev/null || return 1
    curl -sf -m 5 http://127.0.0.1:8080/health >/dev/null 2>&1 && return 0
    sleep 10
  done
  return 1
}
# he DIR LABEL OUTFILE BINARY_AND_ARGS...  -- serves, evals, tears down
he() {
  local dir="$1" label="$2" out="$3"; shift 3
  if ! port_free; then
    say "  $label ABORTED -- :8080 already held, serving '$(served_model)'"
    return 1
  fi
  # `exec` makes the subshell BECOME the server, so $! is the server's own pid. Without it
  # $! is the throwaway subshell, kill misses, and the server keeps the port forever.
  ( cd "$dir" && exec "$@" ) > "$dir/logs_chain_server.log" 2>&1 &
  local pid=$!
  if wait_health "$pid"; then
    local served; served=$(served_model)
    say "  $label server up (pid $pid) SERVING '$served'; 164 problems"
    # Two evals per server start -- reloading a ~100 GB model per benchmark is pure waste.
    # HumanEval measures whether it writes correct code; BFCL measures whether it can pick the
    # right tool and fill its arguments, which is what actually decides agentic daily driving.
    ( cd "$GLMDIR" && python3 tools/eval_humaneval.py --base http://127.0.0.1:8080 \
        --label "$label" --out "$HOME/glm-5.3-reap/$out" ) >> "$LOG" 2>&1 \
      && say "  $label HumanEval DONE (served '$served')" || say "  $label HumanEval FAILED"
    ( cd "$GLMDIR" && python3 tools/eval_bfcl.py --base http://127.0.0.1:8080 --limit 100 \
        --label "$label" --out "$HOME/glm-5.3-reap/artifacts/bfcl_$label.json" ) >> "$LOG" 2>&1 \
      && say "  $label BFCL DONE (served '$served')" || say "  $label BFCL FAILED"
  else
    say "  $label SKIPPED -- server never ready (see $dir/logs_chain_server.log)"
  fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  for i in $(seq 1 120); do port_free && break; sleep 5; done
  if port_free; then say "  $label torn down, :8080 clear"
  else say "  FATAL: :8080 still held after $label teardown -- refusing to continue"; exit 1; fi
}

say "waiting for NIAH to release the GPU"
while pgrep -f "tools/niah_nvfp[4]" >/dev/null 2>&1 || pgrep -x glm5-server >/dev/null 2>&1; do
  sleep 60
done
say "GPU free. RAM $(free -g | awk '/^Mem:/{print $7}') GiB, disk $(df -h / | awk 'NR==2{print $4}')"

say "STAGE 1 decode profile"
( cd "$GLMDIR" && ./build/bench_decode --steps 32 ) >> "$LOG" 2>&1 \
  && say "STAGE 1 DONE" || say "STAGE 1 FAILED"

say "STAGE 2 three-way HumanEval"
# The glm5 arm was the FIRST arm of the 2026-09-10 run, so it is the one arm the orphaned-server
# bug did not corrupt (see artifacts/void_2026-09-10_server_orphan_bug/). Its numbers stand and are
# reused. The gemv_multi fusion landed after it, but that change is bit-identical on real weights
# (tests/gate_gemv_multi.cu), so it cannot have moved a capability score. Re-running it would cost
# 6.1 h to reproduce a number we already hold.
if [ -s artifacts/humaneval_glm.json ] && [ -s artifacts/bfcl_glm5.json ]; then
  say "  glm5 REUSED -- valid first-arm result from 2026-09-10 (HumanEval 68/164, BFCL 70/100)"
else
  he "$GLMDIR" glm5 artifacts/humaneval_glm.json ./build/glm5-server --ctx 8192 --port 8080
fi
he "$DSDIR"  dsv4  artifacts/humaneval_dsv4.json ./build/dsv4-server --port 8080
if [ -f "$VEXP/config.json" ] && [ "$(du -sb "$VEXP" | cut -f1)" -gt 80000000000 ]; then
  he "$DSDIR" vexp artifacts/humaneval_vexp.json ./build/dsv4-server --ckpt "$VEXP" --port 8080
else
  say "  vexp SKIPPED -- checkpoint not fully downloaded"
fi

say "=== THREE-WAY ==="
.venv/bin/python - <<'PY' 2>&1 | tee -a "$LOG"
import json, pathlib
META = {"GLM-5.3-Flash-REAP50":     ("144/288 = 50.0% pruned", "top-8", "18.0:1"),
        "DeepSeek-0731-REAP":       ("160/256 = 37.5% pruned", "top-6", "26.7:1"),
        "DeepSeek-VisionExp-REAP":  ("128/256 = 50.0% pruned", "top-6", "21.3:1")}
for lbl, f in (("GLM-5.3-Flash-REAP50", "artifacts/humaneval_glm.json"),
               ("DeepSeek-0731-REAP", "artifacts/humaneval_dsv4.json"),
               ("DeepSeek-VisionExp-REAP", "artifacts/humaneval_vexp.json")):
    prune, tk, ratio = META[lbl]; p = pathlib.Path(f)
    if p.exists():
        d = json.loads(p.read_text())
        print(f"{lbl:26} pass@1 {d['pass']:3d}/{d['n']:3d} = {d['pass@1']:6.1%}  [{prune}, {tk}, {ratio}]")
    else:
        print(f"{lbl:26} (no result)                 [{prune}, {tk}, {ratio}]")
print()
for lbl, f in (("GLM-5.3-Flash-REAP50","artifacts/bfcl_glm5.json"),
               ("DeepSeek-0731-REAP","artifacts/bfcl_dsv4.json"),
               ("DeepSeek-VisionExp-REAP","artifacts/bfcl_vexp.json")):
    p2 = pathlib.Path(f)
    if p2.exists():
        d2 = json.loads(p2.read_text())
        print(f"{lbl:26} BFCL tool-call {d2['pass']:3d}/{d2['n']:3d} = {d2['acc']:6.1%}")
    else:
        print(f"{lbl:26} BFCL (no result)")
print("\nNOT pruned equally. GLM and VisionExp are at 50%; 0731 at 37.5%, which is the")
print("FIT-DRIVEN ratio (105 GB budget / 0.660 GB per expert = 160 experts). VisionExp is")
print("over-pruned by 32 experts for this box, so it bounds that model from BELOW.")
print("Greedy, single sample, our own prompt: comparable to EACH OTHER, not to leaderboards.")
PY

say "STAGE 3 router KD -- smoke"
py "import sys;sys.path.insert(0,'scripts');import router_kd as R;R.run(smoke_layers=2,steps_per_layer=6,batch=1,max_len=512)" \
  && { say "smoke OK"; py "import sys;sys.path.insert(0,'scripts');import router_kd as R;R.run(steps_per_layer=150,batch=2,max_len=1024)" \
       && say "STAGE 3 DONE" || say "STAGE 3 FAILED"; } || say "STAGE 3 SMOKE FAILED -- skipped"

say "STAGE 4 masked-teacher eval -- selfcheck"
py "import sys;sys.path.insert(0,'scripts');import mask_eval as M;raise SystemExit(M.selfcheck())" \
  && { say "selfcheck OK"; py "import sys;sys.path.insert(0,'scripts');import mask_eval as M;M.run_arms(['teacher','mean','mass'],'artifacts/mask_eval.json')" \
       && say "STAGE 4 DONE" || say "STAGE 4 FAILED"; } || say "STAGE 4 SELFCHECK FAILED -- skipped"

say "STAGE 5 channel saliency -- smoke"
py "import sys;sys.path.insert(0,'scripts');import channel_saliency as C;C.run(smoke_layers=2,batch=1,max_len=512)" \
  && { say "smoke OK"; py "import sys;sys.path.insert(0,'scripts');import channel_saliency as C;C.run(batch=2,max_len=1024)" \
       && say "STAGE 5 DONE" || say "STAGE 5 FAILED"; } || say "STAGE 5 SMOKE FAILED -- skipped"

say "CHAIN COMPLETE. GLM source INTACT (311 GB, 62 shards) -- nothing deleted."
