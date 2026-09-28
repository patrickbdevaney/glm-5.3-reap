#!/usr/bin/env bash
# 1) ship the two missing vision processor configs to FP8-v2 and NVFP4-v2
# 2) re-verify fp8-pass2 and reclaim it only on a clean pass
# 3) build llama-bench (never built) and run the -ngl offload sweep
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
LOG=logs/fix_and_sweep.log
say() { echo "[$(date -Is)] fix: $*" >> "$LOG"; }
SRC=output/glm-5.3-flash-reap50-fp8-pass2

say "=== step 1: missing vision processor configs ==="
cp -n "$SRC/preprocessor_config.json" "$SRC/processor_config.json" \
      output/glm-5.3-flash-reap50-nvfp4-pass2/ 2>>"$LOG" && say "copied into nvfp4-pass2"
for REPO in GLM-5.3-Flash-REAP50-FP8-v2 GLM-5.3-Flash-REAP50-NVFP4-v2; do
  for F in preprocessor_config.json processor_config.json; do
    ./.venv/bin/hf upload "patrickbdevaney/$REPO" "$SRC/$F" "$F" >> "$LOG" 2>&1 \
      && say "uploaded $F -> $REPO" || say "FAILED $F -> $REPO"
  done
done

say "=== step 2: re-verify fp8-pass2 ==="
if ./.venv/bin/python scripts/flush_verify_fp8.py >> "$LOG" 2>&1; then
  rm -rf "$SRC"
  say "verified clean; removed fp8-pass2. free $(df -h / | awk 'NR==2{print $4}')"
else
  say "still does NOT verify - keeping it (see detail above)"
fi

say "=== step 3: build llama-bench ==="
cmake --build "$HOME/glm5-llama.cpp/build-cuda" --target llama-bench -j "$(nproc)" >> "$LOG" 2>&1
BIN=$HOME/glm5-llama.cpp/build-cuda/bin/llama-bench
if [ ! -x "$BIN" ]; then say "FATAL: llama-bench still missing after build"; exit 1; fi
say "llama-bench built"

say "=== step 4: offload sweep ==="
exec ./scripts/probe_offload.sh
